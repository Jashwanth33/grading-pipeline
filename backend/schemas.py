from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"


class GradingPredictionRequest(BaseModel):
    test_pass_rate: float = Field(..., ge=0, le=1)
    cyclomatic_complexity: float = Field(..., ge=0)
    num_functions: int = Field(..., ge=0)
    lines_of_code: int = Field(..., ge=0)
    runtime_ms: float = Field(..., ge=0)
    memory_usage_mb: float = Field(..., ge=0)
    num_failed_tests: int = Field(..., ge=0)
    num_warnings: int = Field(..., ge=0)
    lint_score: float = Field(..., ge=0, le=1)
    documentation_score: float = Field(..., ge=0, le=1)


class GradingPredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str


class DoubtPredictionRequest(BaseModel):
    question: str = Field(..., min_length=1)


class DoubtPredictionResponse(BaseModel):
    question: str
    predicted_topic: str
    topic_confidence: float
    predicted_urgency: str
    urgency_confidence: float
    auto_approve: bool
    route: str
    topic_probabilities: Optional[Dict[str, float]] = None
    urgency_probabilities: Optional[Dict[str, float]] = None


class TrainRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to training dataset")
    target_column: str = Field(default="quality_label")
    model_types: Optional[List[str]] = None
    handle_imbalance: bool = True
    cv_folds: int = Field(default=5, ge=2, le=20)


class TrainResponse(BaseModel):
    status: str
    model_comparison: Dict[str, Any]
    best_model: str
    feature_importance: List[Dict[str, Any]]
    cv_results: Dict[str, Any]
    test_metrics: Dict[str, Any]
    training_report: Dict[str, Any]


class MetricsResponse(BaseModel):
    grading: Dict[str, Any]
    triage_topic: Dict[str, Any]
    triage_urgency: Dict[str, Any]
    threshold_analysis: Dict[str, Any]


class FeatureImportanceResponse(BaseModel):
    features: List[Dict[str, Any]]
    shap_top_features: Optional[List[Dict[str, Any]]] = None


class ModelInfoResponse(BaseModel):
    grading_model: Dict[str, Any]
    triage_models: Dict[str, Any]
    threshold: float
    timestamp: str
