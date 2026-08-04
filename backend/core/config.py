import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    app_name: str = Field(default="GradingPipeline", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    model_dir: Path = Field(default=BASE_DIR / "models", env="MODEL_DIR")
    data_dir: Path = Field(default=BASE_DIR / "data", env="DATA_DIR")
    reports_dir: Path = Field(default=BASE_DIR / "reports", env="REPORTS_DIR")

    test_size: float = Field(default=0.2, env="TEST_SIZE")
    val_size: float = Field(default=0.1, env="VAL_SIZE")
    random_state: int = Field(default=42, env="RANDOM_STATE")
    n_splits: int = Field(default=5, env="N_SPLITS")

    confidence_threshold: float = Field(default=0.75, env="CONFIDENCE_THRESHOLD")
    max_sequence_length: int = Field(default=512, env="MAX_SEQUENCE_LENGTH")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")

    grading_features: list = [
        "test_pass_rate",
        "cyclomatic_complexity",
        "num_functions",
        "lines_of_code",
        "runtime_ms",
        "memory_usage_mb",
        "num_failed_tests",
        "num_warnings",
        "lint_score",
        "documentation_score",
    ]

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"

    def ensure_directories(self):
        for d in [self.model_dir, self.data_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
