# Scotland 2026 Scottish Parliament Election Forecast

An end-to-end MLOps pipeline that generates synthetic voter micro-data from
YouGov MRP polling priors and trains a stacking ensemble to forecast the
Scotland 2026 Scottish Parliament election result.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Data layer                                │
│  scripts/generate_data.py → src/data/generate_voters.py         │
│  12,500 synthetic voters · Dirichlet noise · YouGov MRP priors  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                    Feature pipeline                               │
│  src/features/pipeline.py                                        │
│  ColumnTransformer: scaler + OrdinalEncoder + OneHotEncoder      │
│  Engineered: tactical_swing_index, indep_economy_interaction,    │
│              nhs_dissatisfaction                                  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                   Stacking ensemble                               │
│  src/models/ensemble.py + src/models/base_models.py              │
│  Base: XGBoost · LightGBM · CatBoost · RandomForest             │
│  Meta: LogisticRegression   Tuning: Optuna (50 trials each)     │
│  Tracking: MLflow                                                 │
└──────────┬──────────────────────┬────────────────────────────────┘
           │                      │
┌──────────▼──────────┐  ┌────────▼───────────────────────────────┐
│  D'Hondt allocation │  │        SHAP explainability              │
│  src/models/dhondt  │  │  src/models/explainability.py          │
│  73 con + 56 list   │  │  TreeExplainer → mean |SHAP|           │
└──────────┬──────────┘  └────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                    Serving layer                                  │
│  FastAPI (src/api/main.py)                                       │
│  GET  /health  ·  GET  /model/info  ·  GET /seats/projected     │
│  POST /predict  ·  POST /predict/batch                           │
└──────────┬──────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                   Streamlit dashboard                             │
│  streamlit_app/Home.py  (KPIs, polling bars, seat chart)        │
│  Page 1: Voter Simulator     Page 2: Seat Projections           │
│  Page 3: Model Performance   Page 4: SHAP Explainability        │
└─────────────────────────────────────────────────────────────────┘
```

## Branch table

| Branch | Contents |
|---|---|
| `main` | Stable releases |
| `develop` | Integration branch — all features merged here |
| `feature/data-generation` | `src/data/`, `tests/unit/test_data_generation.py` |
| `feature/feature-engineering` | `src/features/` |
| `feature/model-development` | `src/models/`, `src/orchestration/` |
| `feature/inference-api` | `src/api/`, `tests/unit/test_api.py` |
| `feature/streamlit-dashboard` | `streamlit_app/` |
| `feature/mlops-infra` | `docker/`, `.github/`, `scripts/`, `tests/unit/test_metrics.py` |

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate voter data
python scripts/generate_data.py --n-voters 12500

# 3. Start MLflow (optional)
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# 4. Train the ensemble
python scripts/train_models.py --n-trials 50

# 5. Start the API
uvicorn src.api.main:app --reload --port 8000

# 6. Launch the dashboard
streamlit run streamlit_app/Home.py

# 7. Run all tests
pytest tests/ -v
```

### Docker (all services)

```bash
cd docker
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| MLflow | http://localhost:5000 |

## Model performance targets

| Model | F1-macro (val) | Brier score | Log-loss |
|---|---|---|---|
| XGBoost | 0.58 | 0.121 | 0.93 |
| LightGBM | 0.57 | 0.124 | 0.95 |
| CatBoost | 0.56 | 0.127 | 0.97 |
| Random Forest | 0.54 | 0.135 | 1.05 |
| Logistic Regression | 0.49 | 0.148 | 1.12 |
| **Stacking Ensemble** | **0.62** | **0.112** | **0.88** |

## Tech stack

| Layer | Technology |
|---|---|
| Data generation | NumPy Dirichlet, Pandas |
| Feature engineering | scikit-learn ColumnTransformer |
| ML | XGBoost, LightGBM, CatBoost, scikit-learn |
| HPO | Optuna (TPE sampler) |
| Experiment tracking | MLflow |
| Seat allocation | D'Hondt AMS algorithm |
| Explainability | SHAP (TreeExplainer) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker + Compose |
| CI | GitHub Actions (lint, unit tests, smoke tests) |

## Polling priors

YouGov MRP · April 2026 · n = 3,925

| Party | Constituency | Regional list | Proj. seats |
|---|---|---|---|
| SNP | 35.6% | 29.1% | 67 |
| Reform | 17.9% | 18.0% | 20 |
| Labour | 16.3% | 15.3% | 15 |
| Green | 6.0% | 8.5% | 11 |
| Conservative | 10.3% | 11.1% | 9 |
| Lib Dem | 8.0% | 9.0% | 7 |

Independence: Yes 46.6% · No 44.6% · Undecided 8.9%
