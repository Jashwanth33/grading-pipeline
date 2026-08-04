# ML-Based Grading & Doubt Triage Pipeline

Production-ready end-to-end ML pipeline for automated code submission grading and student doubt classification/routing in an LMS.

## Architecture

```
grading_pipeline/
├── backend/
│   ├── app.py                    # FastAPI application entry
│   ├── routes.py                 # API endpoints
│   ├── schemas.py                # Pydantic request/response models
│   ├── core/
│   │   ├── config.py             # Settings & environment config
│   │   ├── logging.py            # Centralized logging
│   │   └── exceptions.py         # Custom exception hierarchy
│   └── services/
│       ├── preprocess.py         # Data cleaning, outlier detection, encoding
│       ├── features.py           # Feature engineering & selection
│       ├── grading_pipeline.py   # Model training, evaluation, SHAP
│       └── triage_pipeline.py    # NLP classification & threshold routing
├── frontend/
│   └── src/
│       ├── App.js                # React router + sidebar
│       ├── api.js                # Axios API client
│       └── pages/                # Dashboard, Upload, Train, Eval, etc.
├── scripts/
│   └── generate_data.py          # Synthetic dataset generator
├── notebooks/
│   └── training_pipeline.ipynb   # Full walkthrough notebook
├── data/                         # Generated/uploaded datasets
├── models/                       # Saved model artifacts (.pkl)
├── reports/                      # Logs and generated reports
├── requirements.txt
├── .env
└── run.py                        # Server entry point
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
python -m scripts.generate_data
```

### 3. Start Backend

```bash
python run.py
```

Backend runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### 4. Start Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs at `http://localhost:3000`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/train` | Train all models on uploaded dataset |
| POST | `/api/predict-grading` | Predict submission quality |
| POST | `/api/predict-doubt` | Classify student doubt |
| POST | `/api/upload-dataset` | Upload CSV dataset |
| GET | `/api/metrics` | Get all model metrics |
| GET | `/api/feature-importance` | Feature importance + SHAP |
| GET | `/api/model-info` | Model metadata and info |
| GET | `/health` | Health check |

## ML Pipeline Details

### Grading Pipeline

**Features engineered:**
- `test_pass_rate`, `cyclomatic_complexity`, `num_functions`, `lines_of_code`
- `runtime_ms`, `memory_usage_mb`, `num_failed_tests`, `num_warnings`
- `lint_score`, `documentation_score`
- Derived: `quality_composite`, `complexity_density`, `resource_score`, `code_health`, `test_consistency`

**Models trained:**
1. Random Forest (baseline, class-weighted)
2. Logistic Regression (baseline, class-weighted)
3. LightGBM (primary, early stopping)
4. XGBoost (comparison)

**Evaluation metrics:** Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix

**Engineering practices:**
- Stratified train/val/test split (70/10/20)
- 5-fold stratified cross-validation
- SMOTE for class imbalance
- Outlier capping (IQR method)
- Feature scaling via ColumnTransformer
- SHAP TreeExplainer for model explainability
- Data leakage prevention (fit on train only)

### Doubt Triage Pipeline

**Vectorizers:**
- TF-IDF (unigram + bigram, sublinear TF)
- CountVectorizer (unigram + bigram)

**Models:**
- Complement Naive Bayes
- Logistic Regression (class-weighted)

**Routing logic:**
- Confidence >= threshold → auto-approve
- Confidence < threshold → teacher review
- Optimal threshold found via F1 maximization on validation data
- Justification provided with each analysis

## Frontend Pages

1. **Dashboard** — System overview, model status, quick charts
2. **Upload Dataset** — Drag-and-drop CSV upload with preview
3. **Train Model** — Configure and launch training, view results
4. **Evaluation** — Accuracy/Precision/Recall/F1, confusion matrix, radar chart
5. **Student Doubts** — Submit questions, view classification + probabilities
6. **Prediction** — Input submission metrics, get quality prediction
7. **Confidence Routing** — Threshold analysis, auto-approve vs teacher review
8. **Explainability** — Feature importance, SHAP values, feature descriptions

## Engineering Requirements Met

- Clean architecture (core/services/routers separation)
- Centralized logging with file rotation
- Environment-based configuration (.env)
- Pydantic schemas for API validation
- Custom exception hierarchy
- sklearn Pipeline + ColumnTransformer
- joblib for model serialization
- Class imbalance handling (SMOTE / class weights)
- Feature importance + SHAP explanations
- Confidence-based routing with threshold optimization

## Limitations

- Sample data is synthetic; real data may require additional preprocessing
- Sentence Transformers optional dependency (requires torch)
- SHAP computation can be slow on large datasets
- Frontend assumes local backend at localhost:8000
- No authentication/authorization on API endpoints

## Future Improvements

- Add user authentication and role-based access
- Integrate real LMS data connectors
- Add Sentence Transformer embeddings for richer NLP features
- Implement model drift monitoring and retraining triggers
- Add A/B testing framework for model comparison
- Deploy with Docker + Kubernetes
- Add CI/CD pipeline with automated testing
- Implement real-time WebSocket predictions
