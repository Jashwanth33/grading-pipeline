import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

import warnings
warnings.filterwarnings("ignore")

from backend.core.config import settings
from backend.core.logging import logger
from backend.core.exceptions import ModelError


class NLPDoubtTriage:
    """NLP pipeline for student doubt classification."""

    def __init__(self):
        self.topic_pipeline: Optional[Pipeline] = None
        self.urgency_pipeline: Optional[Pipeline] = None
        self.topic_results: Dict[str, Any] = {}
        self.urgency_results: Dict[str, Any] = {}
        self.optimal_threshold: float = settings.confidence_threshold
        self.threshold_analysis: Dict[str, Any] = {}
        self.topic_report: Dict[str, Any] = {}
        self.urgency_report: Dict[str, Any] = {}

    def _build_pipeline(self, model_type: str = "tfidf_logreg"):
        """Build sklearn pipeline with vectorizer + classifier."""
        pipelines = {
            "tfidf_nb": Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=10000, ngram_range=(1, 2),
                    min_df=2, max_df=0.95, sublinear_tf=True,
                )),
                ("clf", ComplementNB(alpha=0.5)),
            ]),
            "tfidf_logreg": Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=10000, ngram_range=(1, 2),
                    min_df=2, max_df=0.95, sublinear_tf=True,
                )),
                ("clf", LogisticRegression(
                    C=1.0, max_iter=1000, class_weight="balanced",
                    random_state=42,
                )),
            ]),
            "count_nb": Pipeline([
                ("count", CountVectorizer(
                    max_features=10000, ngram_range=(1, 2),
                    min_df=2, max_df=0.95,
                )),
                ("clf", ComplementNB(alpha=0.5)),
            ]),
        }
        return pipelines.get(model_type, pipelines["tfidf_logreg"])

    def train_topic_model(
        self,
        texts: pd.Series,
        labels: pd.Series,
        model_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Train topic classification models and compare."""
        if model_types is None:
            model_types = ["tfidf_nb", "tfidf_logreg", "count_nb"]

        logger.info(f"Training topic models: {model_types}")
        results = {}

        for mt in model_types:
            pipeline = self._build_pipeline(mt)
            cv = StratifiedKFold(n_splits=settings.n_splits, shuffle=True, random_state=42)

            scoring = {
                "accuracy": "accuracy",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
                "f1": "f1_weighted",
            }
            cv_results = cross_validate(pipeline, texts, labels, cv=cv, scoring=scoring)

            results[mt] = {
                "accuracy": float(np.mean(cv_results["test_accuracy"])),
                "precision": float(np.mean(cv_results["test_precision"])),
                "recall": float(np.mean(cv_results["test_recall"])),
                "f1": float(np.mean(cv_results["test_f1"])),
                "std_accuracy": float(np.std(cv_results["test_accuracy"])),
            }
            logger.info(f"  {mt}: Acc={results[mt]['accuracy']:.4f}, F1={results[mt]['f1']:.4f}")

        best_mt = max(results, key=lambda k: results[k]["f1"])
        logger.info(f"Best topic model: {best_mt}")

        self.topic_pipeline = self._build_pipeline(best_mt)
        self.topic_pipeline.fit(texts, labels)
        self.topic_results = results
        self.topic_report = {
            "best_model": best_mt,
            "model_comparison": results,
        }
        return results

    def train_urgency_model(
        self,
        texts: pd.Series,
        labels: pd.Series,
        model_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Train urgency classification models."""
        if model_types is None:
            model_types = ["tfidf_nb", "tfidf_logreg"]

        logger.info(f"Training urgency models: {model_types}")
        results = {}

        for mt in model_types:
            pipeline = self._build_pipeline(mt)
            cv = StratifiedKFold(n_splits=settings.n_splits, shuffle=True, random_state=42)

            scoring = {
                "accuracy": "accuracy",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
                "f1": "f1_weighted",
            }
            cv_results = cross_validate(pipeline, texts, labels, cv=cv, scoring=scoring)

            results[mt] = {
                "accuracy": float(np.mean(cv_results["test_accuracy"])),
                "precision": float(np.mean(cv_results["test_precision"])),
                "recall": float(np.mean(cv_results["test_recall"])),
                "f1": float(np.mean(cv_results["test_f1"])),
                "std_accuracy": float(np.std(cv_results["test_accuracy"])),
            }
            logger.info(f"  {mt}: Acc={results[mt]['accuracy']:.4f}, F1={results[mt]['f1']:.4f}")

        best_mt = max(results, key=lambda k: results[k]["f1"])
        logger.info(f"Best urgency model: {best_mt}")

        self.urgency_pipeline = self._build_pipeline(best_mt)
        self.urgency_pipeline.fit(texts, labels)
        self.urgency_results = results
        self.urgency_report = {
            "best_model": best_mt,
            "model_comparison": results,
        }
        return results

    def find_optimal_threshold(
        self,
        texts: pd.Series,
        labels: pd.Series,
        metric: str = "f1",
        threshold_range: Optional[List[float]] = None,
    ) -> float:
        """Find optimal confidence threshold using validation data."""
        if threshold_range is None:
            threshold_range = np.arange(0.3, 0.95, 0.05).tolist()

        logger.info(f"Finding optimal threshold (metric={metric})")

        calibrated = CalibratedClassifierCV(
            self.urgency_pipeline, cv=3, method="isotonic"
        )
        calibrated.fit(texts, labels)

        y_prob = calibrated.predict_proba(texts)
        max_probs = y_prob.max(axis=1)
        y_pred = calibrated.predict(texts)

        results = []
        for thresh in threshold_range:
            mask = max_probs >= thresh
            if mask.sum() == 0:
                continue
            y_thresh = y_pred[mask]
            y_true_thresh = labels.values[mask]
            if len(np.unique(y_true_thresh)) < 2:
                continue

            f1 = f1_score(y_true_thresh, y_thresh, average="weighted", zero_division=0)
            coverage = mask.mean()
            auto_approve = mask.sum()
            manual_review = (~mask).sum()

            results.append({
                "threshold": float(thresh),
                "f1": float(f1),
                "coverage": float(coverage),
                "auto_approve_count": int(auto_approve),
                "manual_review_count": int(manual_review),
            })

        if not results:
            self.optimal_threshold = settings.confidence_threshold
            return self.optimal_threshold

        best = max(results, key=lambda x: x["f1"])
        self.optimal_threshold = best["threshold"]
        self.threshold_analysis = {
            "results": results,
            "optimal_threshold": best["threshold"],
            "optimal_f1": best["f1"],
            "optimal_coverage": best["coverage"],
            "metric_used": metric,
            "justification": (
                f"Threshold {best['threshold']:.2f} selected because it maximizes "
                f"weighted F1-score ({best['f1']:.4f}) while maintaining "
                f"{best['coverage']*100:.1f}% coverage. "
                f"At this threshold, {best['auto_approve_count']} questions can be "
                f"auto-approved and {best['manual_review_count']} need teacher review. "
                "The threshold balances prediction quality with operational efficiency - "
                "too low increases false positives (poor auto-approval), "
                "too high wastes capacity on trivial questions requiring review."
            ),
        }
        logger.info(f"Optimal threshold: {self.optimal_threshold:.2f}")
        return self.optimal_threshold

    def predict(
        self, texts: pd.Series, return_confidence: bool = True
    ) -> Dict[str, Any]:
        """Predict topic and urgency with confidence scores."""
        if self.topic_pipeline is None or self.urgency_pipeline is None:
            raise ModelError("Models not trained yet")

        topic_probs = self.topic_pipeline.predict_proba(texts)
        urgency_probs = self.urgency_pipeline.predict_proba(texts)

        topic_preds = self.topic_pipeline.predict(texts)
        urgency_preds = self.urgency_pipeline.predict(texts)

        topic_classes = self.topic_pipeline.classes_
        urgency_classes = self.urgency_pipeline.classes_

        results = []
        for i in range(len(texts)):
            topic_conf = float(topic_probs[i].max())
            urgency_conf = float(urgency_probs[i].max())
            auto_approve = urgency_conf >= self.optimal_threshold

            entry = {
                "question": texts.iloc[i],
                "predicted_topic": topic_preds[i],
                "topic_confidence": topic_conf,
                "predicted_urgency": urgency_preds[i],
                "urgency_confidence": urgency_conf,
                "auto_approve": auto_approve,
                "route": "auto_approve" if auto_approve else "teacher_review",
            }
            if return_confidence:
                entry["topic_probabilities"] = {
                    c: float(p) for c, p in zip(topic_classes, topic_probs[i])
                }
                entry["urgency_probabilities"] = {
                    c: float(p) for c, p in zip(urgency_classes, urgency_probs[i])
                }
            results.append(entry)

        return {
            "predictions": results,
            "optimal_threshold": self.optimal_threshold,
            "summary": {
                "total": len(texts),
                "auto_approved": sum(1 for r in results if r["auto_approve"]),
                "teacher_review": sum(1 for r in results if not r["auto_approve"]),
            },
        }

    def save_models(self, path: Optional[Path] = None):
        path = path or settings.model_dir / "triage_model.pkl"
        artifact = {
            "topic_pipeline": self.topic_pipeline,
            "urgency_pipeline": self.urgency_pipeline,
            "topic_results": self.topic_results,
            "urgency_results": self.urgency_results,
            "optimal_threshold": self.optimal_threshold,
            "threshold_analysis": self.threshold_analysis,
            "topic_report": self.topic_report,
            "urgency_report": self.urgency_report,
            "timestamp": datetime.now().isoformat(),
        }
        joblib.dump(artifact, path)
        logger.info(f"Triage models saved to {path}")

    def load_models(self, path: Optional[Path] = None):
        path = path or settings.model_dir / "triage_model.pkl"
        if not path.exists():
            raise ModelError(f"Model not found: {path}")
        artifact = joblib.load(path)
        self.topic_pipeline = artifact["topic_pipeline"]
        self.urgency_pipeline = artifact["urgency_pipeline"]
        self.topic_results = artifact["topic_results"]
        self.urgency_results = artifact["urgency_results"]
        self.optimal_threshold = artifact["optimal_threshold"]
        self.threshold_analysis = artifact.get("threshold_analysis", {})
        self.topic_report = artifact.get("topic_report", {})
        self.urgency_report = artifact.get("urgency_report", {})
        logger.info(f"Triage models loaded from {path}")
