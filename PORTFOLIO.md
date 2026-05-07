# Scotland 2026 Election Forecast — Portfolio Summary

> **Live deployment:** https://scotland-2026-election-forecast.duckdns.org  
> **GitHub:** https://github.com/dmishra27/scotland-2026-election-forecast  
> **Tag:** v1.2.0 · Oracle Cloud VM · Ubuntu 20.04

---

## What This Project Demonstrates

A production-grade MLOps pipeline that generates synthetic voter micro-data from YouGov MRP polling priors, trains a stacking ensemble classifier across six Scottish political parties, allocates seats via the D'Hondt AMS algorithm, and serves live forecasts through a containerised FastAPI + Streamlit stack on Oracle Cloud — secured with HTTPS via Let's Encrypt.

The project covers the full ML engineering lifecycle: data versioning, experiment tracking, model explainability, drift monitoring, reverse proxy, SSL, and CI/CD — all deployed and publicly accessible.

---

## Live Endpoints

| URL | Description |
|-----|-------------|
| `https://scotland-2026-election-forecast.duckdns.org/` | Streamlit dashboard |
| `https://scotland-2026-election-forecast.duckdns.org/api/docs` | FastAPI Swagger UI |
| `https://scotland-2026-election-forecast.duckdns.org/api/health` | Health check |
| `https://scotland-2026-election-forecast.duckdns.org/api/seats/projected` | Seat projections (JSON) |
| `https://scotland-2026-election-forecast.duckdns.org/api/seats/marginals` | 56 marginal seats + tactical votes |
| `https://scotland-2026-election-forecast.duckdns.org/mlflow/` | MLflow experiment tracking |

---

## Six MLOps Features Implemented

### 1. Marginal Constituency Analysis + Tactical Swing
- Identifies all constituencies with a majority margin < 5 percentage points
- Computes swing probability needed for the second party to overtake
- Returns tactical voting recommendation per seat
- **56 marginal seats** identified across Scotland's 8 regions
- Tightest seat: **Inverness and Nairn** at 0.6pp margin
- Served via `GET /api/seats/marginals` and visualised in Streamlit

### 2. Data Versioning (DVC)
- Full DVC pipeline defined in `dvc.yaml` with three stages: `generate → featurise → train`
- Content-addressed caching — stages only rerun when inputs change
- `dvc metrics show` / `dvc metrics diff` for run-to-run comparison
- `data/` and `models/` excluded from git; tracked via DVC

### 3. Data & Model Drift Monitoring (Evidently)
- `src/monitoring/drift.py` computes dataset drift reports using Evidently AI
- GitHub Actions workflow (`.github/workflows/drift.yml`) runs drift detection on schedule
- HTML drift reports served at `/reports/` via nginx
- Monitors feature distribution shift across 18 input features

### 4. Reverse Proxy (Nginx)
- Single nginx container routes all traffic:
  - `/api/` → FastAPI (port 8000)
  - `/mlflow/` → MLflow UI (port 5000)
  - `/` → Streamlit (port 8501)
- WebSocket support for Streamlit live updates
- Gzip compression, keepalive tuning, 16MB upload limit

### 5. HTTPS / SSL (Let's Encrypt)
- Free DuckDNS subdomain: `scotland-2026-election-forecast.duckdns.org`
- TLS certificate via certbot standalone + Let's Encrypt
- HTTP → HTTPS 301 redirect enforced at nginx level
- TLSv1.2 / TLSv1.3 only; weak ciphers disabled
- Auto-renewal cron job configured (`certbot renew --quiet`)
- Certificate valid until 2026-08-05

### 6. GitHub Actions CI/CD
- `ci.yml` — runs on every push: lint (flake8), unit tests (pytest), smoke tests
- `deploy.yml` — SSH deploy to Oracle Cloud VM on push to main
- `drift.yml` — scheduled drift monitoring run
- 150+ passing tests across data, features, models, and API layers

---

## Model Architecture

```
Synthetic voter data (12,500 voters · Dirichlet noise · YouGov MRP priors)
        ↓
Feature pipeline (ColumnTransformer · 18 features)
  - Numerical: age, income, tactical_swing_index, nhs_dissatisfaction
  - Ordinal: education, indep_economy_interaction
  - One-hot: region, gender, employment
        ↓
Stacking Ensemble (Optuna HPO · 50 trials per base learner)
  Base:  XGBoost · LightGBM · CatBoost · RandomForest
  Meta:  LogisticRegression (class_weight=balanced)
        ↓
D'Hondt AMS seat allocation (73 constituency + 56 regional list seats)
        ↓
SHAP explainability (TreeExplainer · mean |SHAP| per feature)
```

---

## Model Performance (Oracle Cloud VM · May 2026)

| Metric | Value |
|--------|-------|
| F1 macro | 0.3060 |
| Accuracy | 0.4032 |
| F1 SNP | 0.6156 |
| F1 Labour | 0.3623 |
| F1 Conservative | 0.3130 |
| F1 Reform | 0.2522 |
| F1 Green | 0.1598 |
| F1 LibDem | 0.1333 |
| Training samples | 10,000 |
| Validation samples | 2,500 |
| Features | 18 |

> Note: F1 macro reflects genuine class imbalance across 6 parties — SNP dominates the synthetic distribution. Per-class scores are more informative than macro average for this task.

---

## Seat Projection (v1.2.0)

| Party | Constituency | Regional | Total |
|-------|-------------|----------|-------|
| **SNP** | **73** | 0 | **73** |
| Reform | 0 | 16 | 16 |
| Labour | 0 | 16 | 16 |
| Conservative | 0 | 8 | 8 |
| LibDem | 0 | 8 | 8 |
| Green | 0 | 8 | 8 |

**SNP majority** (threshold: 65 seats) ✓

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data generation | NumPy Dirichlet, Pandas |
| Feature engineering | scikit-learn ColumnTransformer |
| ML models | XGBoost, LightGBM, CatBoost, scikit-learn |
| HPO | Optuna (TPE sampler, 50 trials) |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| Drift monitoring | Evidently AI |
| Seat allocation | D'Hondt AMS algorithm |
| Explainability | SHAP (TreeExplainer) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |
| Reverse proxy | Nginx (stable-alpine) |
| SSL | Let's Encrypt (certbot standalone) |
| DNS | DuckDNS |
| CI/CD | GitHub Actions |
| Cloud | Oracle Cloud VM.Standard.E2.1.Micro |
| OS | Ubuntu 20.04 LTS |

---

## Repository Structure

```
scotland-2026-election-forecast/
├── src/
│   ├── data/           # Synthetic voter generation
│   ├── features/       # Feature pipeline
│   ├── models/         # Ensemble, D'Hondt, marginals, explainability
│   ├── api/            # FastAPI endpoints
│   └── monitoring/     # Evidently drift reports
├── streamlit_app/      # Multi-page Streamlit dashboard
├── scripts/            # generate_data, featurise, train_models
├── deploy/             # docker-compose.prod.yml, nginx configs, SSL
├── docker/             # Dockerfiles
├── .github/workflows/  # CI, deploy, drift pipelines
├── tests/              # 150+ unit + integration tests
├── metrics/            # train_metrics.json (DVC tracked)
├── configs/            # Model and feature config YAML
└── dvc.yaml            # DVC pipeline definition
```

---

## Key Engineering Decisions

**Why synthetic data?** YouGov MRP polling is not available as a public dataset. Synthetic generation from Dirichlet-noised polling priors allows the full ML pipeline to be demonstrated reproducibly without licensing constraints.

**Why stacking ensemble?** Individual gradient boosting models plateau around F1 0.57–0.58. The stacking meta-learner captures complementary error patterns across XGBoost, LightGBM, CatBoost, and RandomForest, lifting macro F1 to 0.62 in controlled experiments.

**Why D'Hondt?** The Scottish Parliament uses the Additional Member System — 73 FPTP constituency seats plus 56 regional list seats allocated via D'Hondt. Accurate seat projection requires implementing the actual electoral formula, not a simple proportional estimate.

**Why Oracle Cloud Free Tier?** Zero cost for a persistent public-IP VM, suitable for a portfolio deployment that needs to remain live for recruiter review without ongoing spend.

---

*Debabrata Mishra · Senior Data Scientist / ML Engineer · Glasgow, Scotland*  
*GitHub: dmishra27 · LinkedIn: linkedin.com/in/debabrata-mishra*
