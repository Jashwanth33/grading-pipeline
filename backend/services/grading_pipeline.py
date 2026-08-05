import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
)
from sklearn.calibration import CalibratedClassifierCV

import lightgbm as lgb
import xgboost as xgb
import shap

from backend.core.config import settings
from backend.core.logging import logger
from backend.core.exceptions import ModelError
from backend.services.features import GradingFeatureEngineer, FeatureSelector


MODEL_REGISTRY = {
    "random_forest": {
        "class": RandomForestClassifier,
        "default_params": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced",
        },
    },
    "logistic_regression": {
        "class": LogisticRegression,
        "default_params": {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
            "class_weight": "balanced",
        },
    },
    "lightgbm": {
        "class": lgb.LGBMClassifier,
        "default_params": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced",
            "verbose": -1,
        },
    },
    "xgboost": {
        "class": xgb.XGBClassifier,
        "default_params": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
        },
    },
}


class GradingModelTrainer:
    """Train, evaluate, and store grading models."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.feature_engineer = GradingFeatureEngineer()
        self.feature_selector = FeatureSelector(k=15)
        self.best_model_name: Optional[str] = None
        self.best_model = None
        self.shap_explainer: Optional[Any] = None
        self.shap_values: Optional[Any] = None
        self.cv_results: Dict[str, Any] = {}
        self.feature_importance_: Optional[pd.DataFrame] = None

    def _create_model(self, model_name: str, custom_params: Optional[Dict] = None):
        if model_name not in MODEL_REGISTRY:
            raise ModelError(f"Unknown model: {model_name}")
        config = MODEL_REGISTRY[model_name]
        params = {**config["default_params"]}
        if custom_params:
            params.update(custom_params)
        return config["class"](**params)

    def cross_validate_model(
        self, model, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, float]:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted",
            "roc_auc": "roc_auc_ovr_weighted",
        }
        cv_results = cross_validate(
            model, X, y, cv=skf, scoring=scoring, return_train_score=True
        )
        return {
            f"train_{k}": float(np.mean(v))
            for k, v in cv_results.items()
            if k.startswith("train_")
        } | {
            f"test_{k}": float(np.mean(v))
            for k, v in cv_results.items()
            if k.startswith("test_")
        } | {
            f"std_{k}": float(np.std(v))
            for k, v in cv_results.items()
            if k.startswith("test_")
        }

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_names: Optional[List[str]] = None,
        custom_params: Optional[Dict[str, Dict]] = None,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Train all models and return comparison results."""
        if model_names is None:
            model_names = list(MODEL_REGISTRY.keys())
        if custom_params is None:
            custom_params = {}

        logger.info(f"Training models: {model_names}")

        # Feature engineering
        X_train_eng = self.feature_engineer.fit_transform(X_train, y_train)
        X_val_eng = self.feature_engineer.transform(X_val)

        # Feature selection
        X_train_sel = self.feature_selector.fit_transform(X_train_eng, y_train)
        X_val_sel = self.feature_selector.transform(X_val_eng)

        feature_names = self.feature_selector.get_selected_features()
        logger.info(f"Selected features: {feature_names}")

        comparison = {}
        for name in model_names:
            logger.info(f"Training {name}...")
            model = self._create_model(name, custom_params.get(name))

            # Cross validation
            cv_scores = self.cross_validate_model(
                model, X_train_sel.values, y_train.values, cv_folds
            )
            self.cv_results[name] = cv_scores

            # Fit on full train
            if name == "lightgbm":
                model.fit(
                    X_train_sel, y_train,
                    eval_set=[(X_val_sel, y_val)],
                )
            elif name == "xgboost":
                model.fit(
                    X_train_sel, y_train,
                    eval_set=[(X_val_sel, y_val)],
                    verbose=False,
                )
            else:
                model.fit(X_train_sel, y_train)

            # Evaluate on val
            y_pred = model.predict(X_val_sel)
            y_prob = model.predict_proba(X_val_sel) if hasattr(model, "predict_proba") else None

            metrics = {
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "precision": float(precision_score(y_val, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_val, y_pred, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_val, y_pred, average="weighted", zero_division=0)),
                "cv_scores": cv_scores,
            }
            if y_prob is not None and len(np.unique(y_val)) > 1:
                try:
                    n_classes_prob = y_prob.shape[1]
                    n_classes_true = len(np.unique(y_val))
                    if n_classes_prob == n_classes_true:
                        metrics["roc_auc"] = float(roc_auc_score(y_val, y_prob, multi_class="ovr", average="weighted"))
                except Exception as e:
                    logger.warning(f"ROC AUC failed for {name}: {e}")

            self.models[name] = model
            self.results[name] = metrics
            comparison[name] = metrics
            logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        # Select best model — prefer LightGBM as default
        if "lightgbm" in comparison:
            self.best_model_name = "lightgbm"
        else:
            self.best_model_name = max(comparison, key=lambda k: comparison[k]["f1"])
        self.best_model = self.models[self.best_model_name]
        logger.info(f"Best model: {self.best_model_name}")

        return comparison

    def evaluate_on_test(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate best model on test set."""
        if self.best_model is None:
            raise ModelError("No model trained yet")

        X_test_eng = self.feature_engineer.transform(X_test)
        X_test_sel = self.feature_selector.transform(X_test_eng)

        y_pred = self.best_model.predict(X_test_sel)
        y_prob = self.best_model.predict_proba(X_test_sel) if hasattr(self.best_model, "predict_proba") else None

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

        if y_prob is not None and len(np.unique(y_test)) > 1:
            try:
                n_classes_prob = y_prob.shape[1]
                n_classes_true = len(np.unique(y_test))
                if n_classes_prob == n_classes_true:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"))
                    fpr_tpr = {}
                    if y_prob.shape[1] == 2:
                        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                        fpr_tpr = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                    else:
                        from sklearn.preprocessing import label_binarize
                        y_bin = label_binarize(y_test, classes=np.unique(y_test))
                        for i in range(y_prob.shape[1]):
                            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                            fpr_tpr[f"class_{i}"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                    metrics["roc_curves"] = fpr_tpr
            except Exception as e:
                logger.warning(f"ROC AUC on test failed: {e}")

        self.test_results = metrics
        return metrics

    def compute_feature_importance(self) -> pd.DataFrame:
        """Compute and aggregate feature importance across models."""
        feature_names = self.feature_selector.get_selected_features()
        importances = {}
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importances[name] = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances[name] = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)

        if not importances:
            return pd.DataFrame()

        df = pd.DataFrame(importances, index=feature_names)
        df["mean_importance"] = df.mean(axis=1)
        df = df.sort_values("mean_importance", ascending=False)
        self.feature_importance_ = df
        return df

    def compute_shap(self, X_sample: pd.DataFrame, max_samples: int = 100):
        """Compute SHAP values for the best model."""
        if self.best_model is None:
            raise ModelError("No model trained")

        X_eng = self.feature_engineer.transform(X_sample)
        X_sel = self.feature_selector.transform(X_eng)

        sample = X_sel.head(max_samples)

        if isinstance(self.best_model, (lgb.LGBMClassifier, xgb.XGBClassifier)):
            self.shap_explainer = shap.TreeExplainer(self.best_model)
        else:
            self.shap_explainer = shap.KernelExplainer(
                self.best_model.predict_proba, shap.sample(sample, 50)
            )
        self.shap_values = self.shap_explainer.shap_values(sample)

        return self.shap_values

    def get_shap_summary(self) -> Dict[str, Any]:
        """Return SHAP feature importance summary."""
        if self.shap_values is None:
            return {}
        feature_names = self.feature_selector.get_selected_features()
        if isinstance(self.shap_values, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in self.shap_values], axis=0)
        else:
            mean_shap = np.abs(self.shap_values).mean(axis=0)
        if hasattr(mean_shap, 'ndim') and mean_shap.ndim > 1:
            mean_shap = mean_shap.mean(axis=0)
        importance = dict(zip(feature_names, [float(x) for x in mean_shap.tolist()]))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return {
            "shap_importance": sorted_imp,
            "top_features": [{"name": n, "shap_value": round(float(v), 4)} for n, v in sorted_imp[:10]],
        }

    def save_model(self, path: Optional[Path] = None):
        path = path or settings.model_dir / "grading_model.pkl"
        artifact = {
            "model": self.best_model,
            "feature_engineer": self.feature_engineer,
            "feature_selector": self.feature_selector,
            "preprocessor": getattr(self, "preprocessor", None),
            "best_model_name": self.best_model_name,
            "results": self.results,
            "cv_results": self.cv_results,
            "feature_importance": self.feature_importance_,
            "test_results": getattr(self, "test_results", {}),
            "target_classes": getattr(self, "target_classes_", []),
            "timestamp": datetime.now().isoformat(),
        }
        joblib.dump(artifact, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Optional[Path] = None) -> Dict[str, Any]:
        path = path or settings.model_dir / "grading_model.pkl"
        if not path.exists():
            raise ModelError(f"Model not found: {path}")
        artifact = joblib.load(path)
        self.best_model = artifact["model"]
        self.feature_engineer = artifact["feature_engineer"]
        self.feature_selector = artifact["feature_selector"]
        self.preprocessor = artifact.get("preprocessor")
        self.best_model_name = artifact["best_model_name"]
        self.results = artifact["results"]
        self.cv_results = artifact.get("cv_results", {})
        self.feature_importance_ = artifact.get("feature_importance")
        self.test_results = artifact.get("test_results", {})
        self.target_classes_ = artifact.get("target_classes", [])
        logger.info(f"Model loaded from {path}")
        return artifact

    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        if self.best_model is None:
            raise ModelError("No model loaded")
        if self.preprocessor is not None and self.preprocessor.is_fitted:
            X = self.preprocessor.transform_new(X)
        X_eng = self.feature_engineer.transform(X)
        X_sel = self.feature_selector.transform(X_eng)
        preds = self.best_model.predict(X_sel)
        probs = self.best_model.predict_proba(X_sel) if hasattr(self.best_model, "predict_proba") else None
        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist() if probs is not None else None,
            "model_used": self.best_model_name,
        }
