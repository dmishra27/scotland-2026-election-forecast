# Scotland 2026 Election Forecast

[![CI](https://github.com/dmishra27/scotland-2026-election-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/dmishra27/scotland-2026-election-forecast/actions/workflows/ci.yml)
[![Deploy](https://github.com/dmishra27/scotland-2026-election-forecast/actions/workflows/deploy.yml/badge.svg)](https://github.com/dmishra27/scotland-2026-election-forecast/actions/workflows/deploy.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg?logo=docker&logoColor=white)](deploy/docker-compose.prod.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tag](https://img.shields.io/badge/tag-v1.5.0-lightgrey.svg)](https://github.com/dmishra27/scotland-2026-election-forecast/releases/tag/v1.5.0)

A production-grade MLOps pipeline that generates synthetic voter micro-data from YouGov MRP polling priors (April 2026, n=3,925), trains a stacking ensemble classifier across six Scottish political parties, allocates seats via the D'Hondt Additional Member System algorithm, and serves live forecasts through a containerised FastAPI + Streamlit stack on Oracle Cloud — secured with HTTPS via Let's Encrypt and continuously deployed via GitHub Actions.

---

## Live Deployment

| Service | URL |
|---------|-----|
| Streamlit dashboard | `https://scotland-2026-election-forecast.duckdns.org/` |
| FastAPI Swagger UI | `https://scotland-2026-election-forecast.duckdns.org/api/docs` |
| Health check | `https://scotland-2026-election-forecast.duckdns.org/api/health` |
| Marginal seats JSON | `https://scotland-2026-election-forecast.duckdns.org/api/seats/marginals` |
| MLflow experiment tracker | `https://scotland-2026-election-forecast.duckdns.org/mlflow/` |

---

## Architecture

```
Synthetic voter data (12,500 voters · Dirichlet noise · YouGov MRP priors)
        |
Feature pipeline (ColumnTransformer · 18 features)
  - Numerical:  age, income, tactical_swing_index, nhs_dissatisfaction
  - Ordinal:    education, indep_economy_interaction
  - One-hot:    region, gender, employment
        |
Stacking Ensemble (Optuna HPO · 50 trials per base learner)
  Base:  XGBoost · LightGBM · CatBoost · RandomForest
  Meta:  LogisticRegression (class_weight=balanced)
        |
D'Hondt AMS seat allocation (73 constituency + 56 regional list seats)
        |
SHAP explainability (TreeExplainer · mean |SHAP| per feature)
        |
FastAPI (REST) + Streamlit (dashboard)
        |
nginx reverse proxy → Oracle Cloud VM (ARM64, Ubuntu 22.04)
```

---

## MLOps Features

| # | Feature | Tool | Status |
|---|---------|------|--------|
| 1 | Data versioning — `generate → featurise → train` pipeline with content-addressed caching | DVC | Done |
| 2 | Drift monitoring — feature distribution shift across 18 inputs, HTML reports at `/reports/` | Evidently AI | Live |
| 3 | Reverse proxy — routes `/api/`, `/mlflow/`, `/` with WebSocket + gzip support | Nginx | Live |
| 4 | HTTPS / SSL — TLSv1.2/1.3, auto-renewal cron, HTTP → HTTPS 301 redirect | Let's Encrypt | Live |
| 5 | CI/CD — lint (flake8), 150+ unit tests (pytest), SSH deploy on push to `main` | GitHub Actions | Live |
| 6 | Marginal seat analysis — 56 marginals identified, swing probabilities, tactical vote recommendations | Custom D'Hondt | Live |

---

## Constituency Explorer

The **Constituency Explorer** page (Streamlit page 6) lets users search any of Scotland's 73 constituency seats by name. Each result shows:

- Predicted vote share for all 6 parties rendered as a horizontal bar chart sorted by descending share (percentages, not raw probabilities)
- Leading party, majority margin (pp), and tactical vote recommendation
- Full-text search with partial match — e.g. "Glasgow" returns all Glasgow constituencies instantly

The explorer is backed by `GET /api/seats/constituencies?search=<term>` and falls back to YouGov MRP priors in demo mode (no trained model on disk).

---

## Model Performance

Trained on Oracle Cloud VM.Standard.E2.1.Micro · Ubuntu 20.04 · 2026-05-05

| Metric | Value |
|--------|-------|
| F1 macro | 0.3060 |
| Accuracy | 0.4032 |
| F1 — SNP | 0.6156 |
| F1 — Labour | 0.3623 |
| F1 — Conservative | 0.3130 |
| F1 — Reform | 0.2522 |
| F1 — Green | 0.1598 |
| F1 — LibDem | 0.1333 |
| Training samples | 10,000 |
| Validation samples | 2,500 |
| Features | 18 |
| Model type | StackingClassifier |
| Base learners | XGBoost, LightGBM, CatBoost, RandomForest |
| Meta learner | LogisticRegression |
| Optuna trials | 50 |

> Low macro F1 reflects genuine 6-class imbalance — SNP dominates the synthetic distribution. Per-class F1 scores are more informative than the macro average for this task.

---

## Seat Projection (v1.5.0)

| Party | Constituency | Regional List | Total |
|-------|:-----------:|:------------:|:-----:|
| **SNP** | **73** | 0 | **73** |
| Reform | 0 | 16 | 16 |
| Labour | 0 | 16 | 16 |
| Conservative | 0 | 8 | 8 |
| LibDem | 0 | 8 | 8 |
| Green | 0 | 8 | 8 |

**SNP majority** — threshold: 65 seats ✓

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data generation | NumPy Dirichlet, Pandas |
| Feature engineering | scikit-learn ColumnTransformer |
| ML models | XGBoost, LightGBM, CatBoost, scikit-learn |
| Hyperparameter optimisation | Optuna (TPE sampler, 50 trials) |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| Drift monitoring | Evidently AI |
| Explainability | SHAP (TreeExplainer) |
| Seat allocation | D'Hondt AMS algorithm |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |
| Reverse proxy | Nginx (stable-alpine) |
| SSL | Let's Encrypt (certbot standalone) |
| DNS | DuckDNS |
| CI/CD | GitHub Actions (ci.yml, deploy.yml, drift.yml) |
| Cloud | Oracle Cloud VM.Standard.E2.1.Micro |
| OS | Ubuntu 22.04 LTS |

---

## Quickstart (Local)

**Prerequisites:** Python 3.11, Docker Desktop

```bash
# 1. Clone and enter the repo
git clone https://github.com/dmishra27/scotland-2026-election-forecast.git
cd scotland-2026-election-forecast

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic voter data
python scripts/generate_data.py

# 5. Build the feature pipeline
python scripts/featurise.py

# 6. Train the ensemble (--n-trials 5 for a fast local run)
python scripts/train_models.py --n-trials 5

# 7. Start the API and dashboard
uvicorn src.api.main:app --reload --port 8000 &
streamlit run streamlit_app/Home.py
```

Dashboard: `http://localhost:8501` · API docs: `http://localhost:8000/docs`

---

## Production Deployment (Docker)

```bash
# Copy environment template and fill in secrets
cp deploy/.env.prod.example deploy/.env.prod

# Pull and start all services (nginx, api, streamlit, mlflow)
docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env.prod up -d --build

# Verify all containers are healthy
docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env.prod ps
```

See [`deploy/README.md`](deploy/README.md) for Oracle Cloud VM provisioning, SSL setup, and GitHub Actions CD secrets.

---

## DVC Pipeline

```bash
# Run the full pipeline (skips unchanged stages)
dvc repro

# Compare metrics across runs
dvc metrics show
dvc metrics diff

# Push data and models to remote storage
dvc push
```

Pipeline stages defined in `dvc.yaml`:

| Stage | Command | Outputs |
|-------|---------|---------|
| `generate` | `python scripts/generate_data.py` | `data/raw/voters.parquet` |
| `featurise` | `python scripts/featurise.py` | `data/processed/features.parquet`, `models/pipeline.pkl` |
| `train` | `python scripts/train_models.py --n-trials 20` | `models/ensemble.pkl`, `metrics/train_metrics.json` |

---

## API Endpoint Reference

Base URL: `https://scotland-2026-election-forecast.duckdns.org/api`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health — model loaded flag, version |
| `POST` | `/predict` | Single-voter vote-intention prediction with party probabilities |
| `POST` | `/predict/batch` | Batch prediction (up to configured max batch size) |
| `GET` | `/seats/projected` | D'Hondt seat allocation — constituency + regional + total per party |
| `GET` | `/seats/marginals` | Top 20 tightest marginal seats with swing probability and tactical vote |
| `GET` | `/seats/constituencies` | All 73 constituency results; `?search=<term>` filters by name |
| `GET` | `/model/info` | Model type, base learners, feature count, MLflow run ID |
| `POST` | `/model/retrain` | Trigger background retrain (202 Accepted) |
| `GET` | `/model/retrain/status` | Poll retrain job status (queued / running / complete / failed) |

Full interactive docs at `/api/docs` (Swagger UI) and `/api/redoc`.

---

## Polling Priors (YouGov MRP, April 2026, n=3,925)

These priors seed the synthetic voter generation and the demo-mode API fallback.

| Party | Constituency vote | Regional list vote |
|-------|:-----------------:|:-----------------:|
| SNP | 35.6% | 29.1% |
| Reform | 17.9% | 18.0% |
| Labour | 16.3% | 15.3% |
| Conservative | 10.3% | 11.1% |
| LibDem | 8.0% | 9.0% |
| Green | 6.0% | 8.5% |

Independence polling (April 2026): Yes 46.6% · No 44.6% · Undecided 8.9%

---

## Post-Election Validation (8 May 2026)

The Scottish Parliament election took place on 8 May 2026. Results below allow direct comparison against this model's v1.5.0 projections.

| Party | Projected seats | Actual seats | Delta |
|-------|:--------------:|:------------:|:-----:|
| SNP | 73 | — | — |
| Reform | 16 | — | — |
| Labour | 16 | — | — |
| Conservative | 8 | — | — |
| LibDem | 8 | — | — |
| Green | 8 | — | — |

> Actual results will be filled in once the official count is certified. The model's constituency-level projections were built entirely from synthetic data seeded by pre-election polling — no outcome data was used in training.

---

## Known Limitations

- **Synthetic training data** — the model learns from Dirichlet-noised polling priors, not real voter records. Patterns in the synthetic data may not reflect actual voter behaviour.
- **Six-class imbalance** — SNP dominates the synthetic distribution (≈36% constituency share), suppressing per-class F1 for smaller parties (LibDem: 0.13, Green: 0.16).
- **No temporal dynamics** — the model is a static snapshot trained on April 2026 priors; late campaign swings or events after the data cut-off are not captured.
- **Constituency boundary assumptions** — D'Hondt allocation uses the 73 + 56 seat AMS structure. Any boundary changes or by-election effects are not modelled.
- **Free-tier hardware constraints** — training runs on Oracle Cloud VM.Standard.E2.1.Micro (1 OCPU, 1 GB RAM). Optuna trials are capped at 50 per base learner; a larger search budget would likely improve macro F1.

---

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Production — auto-deploys to Oracle Cloud on push |
| `develop` | Integration branch — merge feature branches here before `main` |
| `feature/data-generation` | Synthetic voter generation pipeline |
| `feature/feature-engineering` | ColumnTransformer feature pipeline |
| `feature/model-development` | Stacking ensemble and Optuna HPO |
| `feature/inference-api` | FastAPI endpoints and schemas |
| `feature/streamlit-dashboard` | Multi-page Streamlit UI |
| `feature/mlops-infra` | DVC, Evidently, MLflow, Docker, GitHub Actions |

---

## Author

**Debabrata Mishra** — Senior Data Scientist / ML Engineer, Glasgow, Scotland

- GitHub: [dmishra27](https://github.com/dmishra27)
- LinkedIn: [linkedin.com/in/debabrata-mishra](https://linkedin.com/in/debabrata-mishra)
- Live project: [scotland-2026-election-forecast.duckdns.org](https://scotland-2026-election-forecast.duckdns.org)
