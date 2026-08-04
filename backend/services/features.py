import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from backend.core.logging import logger
from backend.core.config import settings


class GradingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineer features from raw grading metrics."""

    def __init__(self, add_interactions: bool = True, add_bins: bool = True):
        self.add_interactions = add_interactions
        self.add_bins = add_bins
        self.feature_names_: List[str] = []
        self.feature_importance_: Optional[Dict[str, float]] = None

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_ = list(X.columns)
        if y is not None and X.shape[0] == y.shape[0]:
            mi = mutual_info_classif(X.fillna(0), y, random_state=42)
            self.feature_importance_ = dict(zip(X.columns, mi))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if self.add_interactions:
            if "test_pass_rate" in X.columns and "lint_score" in X.columns:
                X["quality_composite"] = X["test_pass_rate"] * X["lint_score"]
            if "cyclomatic_complexity" in X.columns and "lines_of_code" in X.columns:
                X["complexity_density"] = np.where(
                    X["lines_of_code"] > 0,
                    X["cyclomatic_complexity"] / X["lines_of_code"],
                    0,
                )
            if "runtime_ms" in X.columns and "memory_usage_mb" in X.columns:
                X["resource_score"] = np.log1p(X["runtime_ms"]) + np.log1p(X["memory_usage_mb"])
            if "num_warnings" in X.columns and "documentation_score" in X.columns:
                X["code_health"] = X["documentation_score"] - X["num_warnings"]
            if "test_pass_rate" in X.columns and "num_failed_tests" in X.columns:
                X["test_consistency"] = X["test_pass_rate"] - (X["num_failed_tests"] / (X["num_failed_tests"] + 1))

        if self.add_bins:
            for col in ["cyclomatic_complexity", "lines_of_code", "runtime_ms"]:
                if col in X.columns:
                    X[f"{col}_bin"] = pd.qcut(X[col], q=5, labels=False, duplicates="drop")

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def get_feature_report(self) -> Dict[str, Any]:
        report = {
            "total_features": len(self.feature_names_),
            "feature_importance": self.feature_importance_ or {},
        }
        if self.feature_importance_:
            sorted_imp = sorted(
                self.feature_importance_.items(), key=lambda x: x[1], reverse=True
            )
            report["top_features"] = [
                {"name": n, "importance": round(v, 4)} for n, v in sorted_imp[:10]
            ]
        return report


class NLPFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract NLP features from text."""

    def __init__(self):
        self.vocabulary_: Optional[Dict[str, int]] = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def extract_text_features(self, text: str) -> Dict[str, float]:
        words = text.split()
        sentences = text.split(".")
        return {
            "word_count": len(words),
            "char_count": len(text),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "sentence_count": len([s for s in sentences if s.strip()]),
            "has_question_mark": float("?" in text),
            "exclamation_count": float(text.count("!")),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "technical_keyword_count": sum(
                1 for w in words
                if w.lower() in {
                    "error", "bug", "exception", "runtime", "compile", "test",
                    "function", "class", "loop", "variable", "import", "api",
                    "database", "query", "null", "undefined", "syntax",
                }
            ),
        }

    def build_features(self, texts: pd.Series) -> pd.DataFrame:
        features = texts.apply(lambda t: self.extract_text_features(str(t)))
        return pd.DataFrame(features.tolist())


class FeatureSelector:
    """Select most relevant features using statistical tests."""

    def __init__(self, k: int = 15):
        self.k = k
        self.selector: Optional[SelectKBest] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        actual_k = min(self.k, X.shape[1])
        self.selector = SelectKBest(mutual_info_classif, k=actual_k)
        self.selector.fit(X.fillna(0), y)
        self.selected_features_ = [
            X.columns[i] for i in self.selector.get_support(indices=True)
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selector is None:
            raise ValueError("Selector not fitted")
        return pd.DataFrame(
            self.selector.transform(X.fillna(0)),
            columns=self.selected_features_,
            index=X.index,
        )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def get_selected_features(self) -> List[str]:
        return self.selected_features_
