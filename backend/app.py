import os
import threading
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.core.config import settings
from backend.core.logging import logger
from backend.routes import router

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="ML-Based Grading & Doubt Triage Pipeline for LMS",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

BUILD_DIR = Path(__file__).resolve().parent.parent / "frontend" / "build"
if not BUILD_DIR.exists():
    BUILD_DIR = Path.cwd() / "frontend" / "build"
logger.info(f"BUILD_DIR: {BUILD_DIR} exists: {BUILD_DIR.exists()}")
if BUILD_DIR.exists():
    logger.info(f"BUILD_DIR contents: {list(BUILD_DIR.iterdir())}")

_startup_done = False


def _train_models():
    """Background thread: train or load models so the server port opens immediately."""
    global _startup_done
    try:
        import pandas as pd
        from backend.services.preprocess import GradingPreprocessor, DoubtPreprocessor
        from backend.services.grading_pipeline import GradingModelTrainer
        from backend.services.triage_pipeline import NLPDoubtTriage
        from backend.routes import grading_trainer, triage, grading_preprocessor, doubt_preprocessor, _loaded

        grading_model_path = settings.model_dir / "grading_model.pkl"
        triage_model_path = settings.model_dir / "triage_model.pkl"

        if not grading_model_path.exists() or not triage_model_path.exists():
            logger.info("No saved models found. Training from scratch...")
            data_dir = settings.data_dir
            grading_csv = data_dir / "grading_dataset.csv"
            doubt_csv = data_dir / "doubt_dataset.csv"

            if not grading_csv.exists() or not doubt_csv.exists():
                logger.info("Generating sample data...")
                from scripts.generate_data import generate_grading_dataset, generate_doubt_dataset
                data_dir.mkdir(parents=True, exist_ok=True)
                generate_grading_dataset(2000).to_csv(grading_csv, index=False)
                generate_doubt_dataset(1500).to_csv(doubt_csv, index=False)

            if not grading_model_path.exists():
                logger.info("Training grading models...")
                df = pd.read_csv(grading_csv)
                gX_train, gX_val, gX_test, gy_train, gy_val, gy_test, _ = (
                    grading_preprocessor.fit_transform(df, target_col="quality_label")
                )
                grading_trainer.train_all_models(gX_train, gy_train, gX_val, gy_val, cv_folds=3)
                grading_trainer.target_classes_ = grading_preprocessor.target_classes_
                grading_trainer.evaluate_on_test(gX_test, gy_test)
                grading_trainer.compute_feature_importance()
                grading_trainer.save_model()
                _loaded["grading"] = True
                logger.info("Grading models trained and saved.")

            if not triage_model_path.exists():
                logger.info("Training triage models...")
                df = pd.read_csv(doubt_csv)
                if all(c in df.columns for c in ["question", "topic", "urgency"]):
                    doubt_preprocessor.fit_transform(df)
                    triage.train_topic_model(df["question"], df["topic"])
                    triage.train_urgency_model(df["question"], df["urgency"])
                    triage.find_optimal_threshold(df["question"], df["urgency"])
                    triage.save_models()
                    _loaded["triage"] = True
                    logger.info("Triage models trained and saved.")
        else:
            logger.info("Found saved models. Loading...")
            from backend.routes import _ensure_loaded
            _ensure_loaded()

        _startup_done = True
        logger.info("Startup training complete.")
    except Exception as e:
        logger.error(f"Startup training failed: {e}", exc_info=True)


@app.on_event("startup")
async def startup():
    global _startup_done
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    settings.ensure_directories()

    if not _startup_done:
        thread = threading.Thread(target=_train_models, daemon=True)
        thread.start()
        logger.info("Model training started in background thread.")


@app.get("/health")
def health():
    return {"status": "healthy", "version": settings.app_version}


if BUILD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(BUILD_DIR / "static")), name="static")

    @app.get("/{full_path:path}")
    async def serve_react(request: Request, full_path: str):
        file_path = BUILD_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(BUILD_DIR / "index.html"))
else:
    @app.get("/")
    def root():
        return {
            "app": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "message": "Frontend build not found. Access API docs at /docs",
        }
