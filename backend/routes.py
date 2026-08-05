import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File
from backend.core.logging import logger
from backend.core.config import settings
from backend.core.exceptions import PipelineError, ModelError
from backend.services.preprocess import GradingPreprocessor, DoubtPreprocessor
from backend.services.grading_pipeline import GradingModelTrainer
from backend.services.triage_pipeline import NLPDoubtTriage
from backend.schemas import (
    GradingPredictionRequest, GradingPredictionResponse,
    DoubtPredictionRequest, DoubtPredictionResponse,
    TrainRequest, TrainResponse,
    MetricsResponse, FeatureImportanceResponse, ModelInfoResponse,
)

router = APIRouter()

grading_trainer = GradingModelTrainer()
triage = NLPDoubtTriage()
grading_preprocessor = GradingPreprocessor()
doubt_preprocessor = DoubtPreprocessor()

_loaded = {"grading": False, "triage": False}


def _ensure_loaded():
    if not _loaded["grading"]:
        try:
            grading_trainer.load_model()
            _loaded["grading"] = True
        except Exception as e:
            logger.warning(f"Could not load grading model: {e}")
    if not _loaded["triage"]:
        try:
            triage.load_models()
            _loaded["triage"] = True
        except Exception as e:
            logger.warning(f"Could not load triage models: {e}")


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    _ensure_loaded()
    grading = grading_trainer.results.get(grading_trainer.best_model_name, {})
    return MetricsResponse(
        grading=grading,
        triage_topic=grading_trainer.cv_results,
        triage_urgency=triage.urgency_results,
        threshold_analysis=triage.threshold_analysis,
    )


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
def get_feature_importance():
    _ensure_loaded()
    try:
        fi = grading_trainer.feature_importance_
        features = []
        if fi is not None and not fi.empty:
            for n, r in fi.iterrows():
                val = r.get("mean_importance", 0)
                if pd.notna(val):
                    features.append({"name": str(n), "importance": float(val)})
        shap = grading_trainer.get_shap_summary()
        return FeatureImportanceResponse(
            features=features[:15],
            shap_top_features=shap.get("top_features"),
        )
    except Exception as e:
        logger.error(f"Feature importance error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    _ensure_loaded()
    return ModelInfoResponse(
        grading_model={
            "name": grading_trainer.best_model_name,
            "type": type(grading_trainer.best_model).__name__,
            "results": grading_trainer.results,
            "feature_count": len(grading_trainer.feature_selector.get_selected_features()),
        },
        triage_models={
            "topic": triage.topic_report,
            "urgency": triage.urgency_report,
        },
        threshold=triage.optimal_threshold,
        timestamp=grading_trainer.results.get("timestamp", ""),
    )


@router.post("/train", response_model=TrainResponse)
def train_models(request: TrainRequest):
    try:
        data_path = Path(request.dataset_path)
        if not data_path.exists():
            raise HTTPException(status_code=400, detail=f"Dataset not found: {data_path}")

        df = pd.read_csv(data_path)
        logger.info(f"Training on dataset: {df.shape}")

        # Grading pipeline
        gX_train, gX_val, gX_test, gy_train, gy_val, gy_test, g_report = (
            grading_preprocessor.fit_transform(
                df, target_col=request.target_column,
                handle_imbalance=request.handle_imbalance,
            )
        )

        comparison = grading_trainer.train_all_models(
            gX_train, gy_train, gX_val, gy_val,
            model_names=request.model_types,
            cv_folds=request.cv_folds,
        )

        grading_trainer.target_classes_ = grading_preprocessor.target_classes_

        test_metrics = grading_trainer.evaluate_on_test(gX_test, gy_test)
        fi = grading_trainer.compute_feature_importance()
        grading_trainer.compute_shap(gX_test)
        grading_trainer.save_model()

        # Triage pipeline
        if all(c in df.columns for c in ["question", "topic", "urgency"]):
            _, topic_labels, urg_labels, t_report = doubt_preprocessor.fit_transform(df)
            triage.train_topic_model(df["question"], topic_labels)
            triage.train_urgency_model(df["question"], urg_labels)
            triage.find_optimal_threshold(df["question"], urg_labels)
            triage.save_models()

        fi_list = [
            {"name": n, "importance": float(r["mean_importance"])}
            for n, r in fi.iterrows()
        ] if not fi.empty else []

        return TrainResponse(
            status="success",
            model_comparison=comparison,
            best_model=grading_trainer.best_model_name,
            feature_importance=fi_list[:15],
            cv_results=grading_trainer.cv_results,
            test_metrics=test_metrics,
            training_report=g_report,
        )
    except PipelineError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-grading", response_model=GradingPredictionResponse)
def predict_grading(request: GradingPredictionRequest):
    _ensure_loaded()
    try:
        df = pd.DataFrame([request.model_dump()])
        result = grading_trainer.predict(df)
        pred_idx = result["predictions"][0]
        target_classes = grading_trainer.target_classes_

        model = grading_trainer.best_model
        if hasattr(model, "classes_"):
            model_classes = model.classes_
            pred_label = target_classes[pred_idx] if pred_idx < len(target_classes) else str(pred_idx)
            probs = {}
            prob_arr = result["probabilities"][0] if result["probabilities"] else []
            for i, mc in enumerate(model_classes):
                if i < len(prob_arr):
                    label = target_classes[mc] if mc < len(target_classes) else str(mc)
                    val = float(prob_arr[i])
                    probs[label] = round(val, 4) if not np.isnan(val) else 0.0
        else:
            pred_label = target_classes[pred_idx] if pred_idx < len(target_classes) else str(pred_idx)
            probs = {}
            if result["probabilities"]:
                for i, c in enumerate(target_classes):
                    if i < len(result["probabilities"][0]):
                        val = float(result["probabilities"][0][i])
                        probs[c] = round(val, 4) if not np.isnan(val) else 0.0

        confidence = max(probs.values()) if probs else 0.0
        return GradingPredictionResponse(
            prediction=pred_label,
            confidence=confidence,
            probabilities=probs,
            model_used=result["model_used"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-doubt", response_model=DoubtPredictionResponse)
def predict_doubt(request: DoubtPredictionRequest):
    _ensure_loaded()
    try:
        texts = pd.Series([request.question])
        result = triage.predict(texts, return_confidence=True)
        pred = result["predictions"][0]
        return DoubtPredictionResponse(**pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-triage")
def train_triage(request: TrainRequest):
    try:
        data_path = Path(request.dataset_path)
        if not data_path.exists():
            raise HTTPException(status_code=400, detail=f"Dataset not found: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Training triage on dataset: {df.shape}")

        if not all(c in df.columns for c in ["question", "topic", "urgency"]):
            raise HTTPException(status_code=400, detail="Dataset must contain 'question', 'topic', 'urgency' columns")

        _, topic_labels, urg_labels, t_report = doubt_preprocessor.fit_transform(df)
        topic_results = triage.train_topic_model(df["question"], df["topic"])
        urgency_results = triage.train_urgency_model(df["question"], df["urgency"])
        threshold = triage.find_optimal_threshold(df["question"], df["urgency"])
        triage.save_models()
        _loaded["triage"] = True

        return {
            "status": "success",
            "topic_results": topic_results,
            "urgency_results": urgency_results,
            "optimal_threshold": threshold,
            "threshold_justification": triage.threshold_analysis.get("justification", ""),
            "report": t_report,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Triage training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        data_dir = settings.data_dir / "uploads"
        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        if file.filename.endswith(".json"):
            df = pd.read_json(file_path)
        else:
            df = pd.read_csv(file_path)
        return {
            "status": "success",
            "file_path": str(file_path),
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "preview": df.head(5).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
