import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from backend.core.config import settings
from backend.core.logging import logger
from backend.core.exceptions import DataError, PreprocessingError


class OutlierDetector(BaseEstimator, TransformerMixin):
    """Detect and cap outliers using IQR method per column."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None
        self.columns_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        Q1 = X[self.columns_].quantile(0.25)
        Q3 = X[self.columns_].quantile(0.75)
        IQR = Q3 - Q1
        self.lower_bounds_ = (Q1 - self.factor * IQR).values
        self.upper_bounds_ = (Q3 + self.factor * IQR).values
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for i, col in enumerate(self.columns_):
            X[col] = X[col].clip(self.lower_bounds_[i], self.upper_bounds_[i])
        return X


class FeatureValidator(BaseEstimator, TransformerMixin):
    """Validate feature ranges and types."""

    def __init__(self, expected_features: Optional[List[str]] = None):
        self.expected_features = expected_features

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.expected_features:
            missing = set(self.expected_features) - set(X.columns)
            if missing:
                logger.warning(f"Missing features: {missing}. Adding with defaults.")
                for f in missing:
                    X[f] = 0.0
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return X


class GradingPreprocessor:
    """End-to-end preprocessing pipeline for grading data."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.outlier_detector = OutlierDetector(factor=1.5)
        self.feature_validator = FeatureValidator(
            expected_features=settings.grading_features
        )
        self.column_transformer: Optional[ColumnTransformer] = None
        self.is_fitted = False

    def _build_column_transformer(self, numeric_features: List[str]):
        numeric_pipeline = SKPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        self.column_transformer = ColumnTransformer(
            transformers=[("num", numeric_pipeline, numeric_features)],
            remainder="passthrough",
        )
        return self.column_transformer

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str,
        handle_imbalance: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Full preprocessing: clean, validate, encode, split, scale."""
        logger.info(f"Preprocessing grading data: {df.shape[0]} rows, {df.shape[1]} cols")
        report: Dict[str, Any] = {}

        if target_col not in df.columns:
            raise DataError(f"Target column '{target_col}' not found")

        # Drop rows with NaN target before splitting
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        report["after_drop_target_nan"] = df.shape[0]

        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        report["original_shape"] = df.shape

        # Encode target
        label_enc = LabelEncoder()
        y = pd.Series(label_enc.fit_transform(y), name=target_col)
        self.target_classes_ = list(label_enc.classes_)
        self.target_encoder_ = label_enc

        # Clean — align y with X after dropping duplicates
        X = self._clean(X)
        y = y.loc[X.index].reset_index(drop=True)
        X = X.reset_index(drop=True)
        report["after_clean"] = X.shape

        # Validate
        X = self.feature_validator.fit_transform(X)
        report["after_validate"] = X.shape

        # Outlier detection
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            X = self.outlier_detector.fit_transform(X)
            report["outliers_capped"] = True

        # Split first to prevent leakage
        min_class_count = y.value_counts().min()
        can_stratify = len(y.unique()) > 1 and min_class_count >= 2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=y if can_stratify else None,
        )
        min_train_count = y_train.value_counts().min()
        can_stratify_train = len(y_train.unique()) > 1 and min_train_count >= 2
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=settings.val_size / (1 - settings.test_size),
            random_state=settings.random_state,
            stratify=y_train if can_stratify_train else None,
        )

        report["train_size"] = len(X_train)
        report["val_size"] = len(X_val)
        report["test_size"] = len(X_test)

        # Fit transform on train only
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        self._build_column_transformer(numeric_features)

        X_train_processed = pd.DataFrame(
            self.column_transformer.fit_transform(X_train),
            columns=numeric_features,
            index=X_train.index,
        )
        X_val_processed = pd.DataFrame(
            self.column_transformer.transform(X_val),
            columns=numeric_features,
            index=X_val.index,
        )
        X_test_processed = pd.DataFrame(
            self.column_transformer.transform(X_test),
            columns=numeric_features,
            index=X_test.index,
        )

        # Handle class imbalance
        if handle_imbalance and len(y_train.unique()) > 1:
            class_counts = y_train.value_counts()
            min_class_count = class_counts.min()
            majority_count = int(class_counts.max())
            minority_ratio = min_class_count / majority_count
            if minority_ratio < 0.3 and min_class_count >= 6:
                logger.info("Applying SMOTE for class imbalance")
                sampling = {cls: majority_count for cls in y_train.unique() if class_counts[cls] < majority_count}
                if sampling:
                    smote = SMOTE(random_state=settings.random_state, sampling_strategy=sampling)
                    X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
                    report["smote_applied"] = True
            else:
                report["smote_applied"] = False
                report["smote_reason"] = "skipped: classes balanced or too few samples"

        self.is_fitted = True
        logger.info("Preprocessing complete")
        return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, report

    def transform_new(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline."""
        if not self.is_fitted:
            raise PreprocessingError("Preprocessor not fitted yet")
        X = self._clean(X)
        X = self.feature_validator.transform(X)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            X = self.outlier_detector.transform(X)
        processed = pd.DataFrame(
            self.column_transformer.transform(X),
            columns=numeric_cols,
            index=X.index,
        )
        return processed

    def _clean(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.drop_duplicates()
        X = X.replace([np.inf, -np.inf], np.nan)
        return X


class DoubtPreprocessor:
    """Preprocessing for NLP doubt triage data."""

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted = False

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        """Preprocess doubt data for topic/urgency prediction."""
        logger.info(f"Preprocessing doubt data: {df.shape}")
        report: Dict[str, Any] = {}

        if "question" not in df.columns:
            raise DataError("Missing 'question' column")
        if "topic" not in df.columns:
            raise DataError("Missing 'topic' column")
        if "urgency" not in df.columns:
            raise DataError("Missing 'urgency' column")

        df = df.dropna(subset=["question"]).copy()
        df["question"] = df["question"].str.strip()
        df = df[df["question"].str.len() > 0]

        topic_encoder = LabelEncoder()
        urgency_encoder = LabelEncoder()
        df["topic_encoded"] = topic_encoder.fit_transform(df["topic"])
        df["urgency_encoded"] = urgency_encoder.fit_transform(df["urgency"])

        self.label_encoders["topic"] = topic_encoder
        self.label_encoders["urgency"] = urgency_encoder
        self.is_fitted = True

        report["num_topics"] = len(topic_encoder.classes_)
        report["num_urgencies"] = len(urgency_encoder.classes_)
        report["topic_classes"] = list(topic_encoder.classes_)
        report["urgency_classes"] = list(urgency_encoder.classes_)

        return df, df["topic_encoded"], df["urgency_encoded"], report

    def encode_new(self, series: pd.Series, label_type: str) -> pd.Series:
        if label_type not in self.label_encoders:
            raise PreprocessingError(f"No encoder for '{label_type}'")
        le = self.label_encoders[label_type]
        known = set(le.classes_)
        encoded = series.apply(lambda x: le.transform([x])[0] if x in known else -1)
        return encoded

    def decode_new(self, series: pd.Series, label_type: str) -> pd.Series:
        if label_type not in self.label_encoders:
            raise PreprocessingError(f"No decoder for '{label_type}'")
        le = self.label_encoders[label_type]
        return series.apply(
            lambda x: le.inverse_transform([int(x)])[0]
            if 0 <= int(x) < len(le.classes_)
            else "unknown"
        )
