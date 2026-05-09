# Scotland 2026 Election Forecast

**Live:** https://scotland-2026-election-forecast.duckdns.org
**GitHub:** https://github.com/dmishra27/scotland-2026-election-forecast
**Tag:** v1.5.0 · Oracle Cloud VM · Ubuntu 22.04

---

## What This Project Demonstrates

A full ML engineering lifecycle — from synthetic data generation through experiment tracking, drift monitoring, model explainability, and containerised production deployment — built around a real-world forecasting problem: the 2026 Scottish Parliament election. Every layer of the MLOps stack is implemented, wired together, and publicly accessible: the API serves live predictions, MLflow logs are queryable, drift reports refresh on a schedule, and GitHub Actions redeploys on every push to `main`. The project is intentionally over-engineered relative to its data science problem, because the point is to demonstrate production ML infrastructure, not to win a polling competition.

---

## Live Endpoints

| Service | URL |
|---------|-----|
| Streamlit dashboard (6 pages) | `https://scotland-2026-election-forecast.duckdns.org/` |
| FastAPI Swagger UI | `https://scotland-2026-election-forecast.duckdns.org/api/docs` |
| Health check | `https://scotland-2026-election-forecast.duckdns.org/api/health` |
| Marginal seats JSON | `https://scotland-2026-election-forecast.duckdns.org/api/seats/marginals` |
| MLflow experiment tracker | `https://scotland-2026-election-forecast.duckdns.org/mlflow/` |

---

## MLOps Scorecard

| Pillar | Grade | Evidence |
|--------|:-----:|---------|
| **DataOps** | B+ | DVC content-addressed pipeline (`generate → featurise → train`) with `dvc metrics diff` across runs; Evidently drift reports on 18 features, scheduled via GitHub Actions |
| **DevOps** | A- | Three GitHub Actions workflows (CI, CD, drift); 150+ pytest tests across 4 modules; Docker Compose prod stack with nginx, SSL, and health-checked CD gate |
| **ModelOps** | A- | MLflow experiment tracking on every training run; SHAP TreeExplainer for per-feature attribution; Optuna TPE HPO (50 trials); async `/model/retrain` endpoint with status polling |

---

## Six MLOps Features

### 1. Data Versioning — DVC

**Tool:** DVC 3.x with local remote  
**Implementation:** Three-stage pipeline in `dvc.yaml` — `generate` (Dirichlet-noised voters), `featurise` (ColumnTransformer), `train` (Optuna stacking ensemble). Content-addressed caching means only changed stages re-execute. `metrics/train_metrics.json` is tracked as a DVC metric (non-cached) so `dvc metrics diff` surfaces F1 and accuracy changes across runs without committing binary files.  
**Live evidence:**
```
dvc repro          # reruns only invalidated stages
dvc metrics show   # prints train_metrics.json to stdout
```

---

### 2. Data & Model Drift Monitoring — Evidently AI

**Tool:** Evidently AI, GitHub Actions (`drift.yml`)  
**Implementation:** `src/monitoring/drift.py` generates a `DatasetDriftReport` across all 18 input features. The report is written as HTML to `reports/` and served by nginx at `/reports/`. A scheduled GitHub Actions workflow (`drift.yml`) runs drift detection on a cron schedule; a significant drift event triggers a Slack-compatible alert annotation in the workflow summary.  
**Live evidence:**
```
https://scotland-2026-election-forecast.duckdns.org/reports/
```

---

### 3. Reverse Proxy — Nginx

**Tool:** Nginx stable-alpine (Docker container)  
**Implementation:** Single nginx container routes all external traffic: `/api/` → FastAPI :8000, `/mlflow/` → MLflow :5000, `/` → Streamlit :8501. WebSocket `Upgrade` headers are proxied for Streamlit live updates. Gzip compression enabled; keepalive tuning and 16 MB upload limit configured. HTTP → HTTPS 301 redirect enforced at the nginx layer — no traffic reaches services unencrypted.  
**Live evidence:**
```bash
curl -I https://scotland-2026-election-forecast.duckdns.org/api/health
# HTTP/2 200, server: nginx
```

---

### 4. HTTPS / SSL — Let's Encrypt

**Tool:** Certbot standalone, DuckDNS, TLSv1.2/1.3  
**Implementation:** Free DuckDNS subdomain pointed at the Oracle VM public IP with a 5-minute cron auto-update guard. TLS certificate issued by Let's Encrypt via certbot standalone (nginx paused during renewal). Weak ciphers disabled; TLSv1.2 / TLSv1.3 only. Auto-renewal cron runs at 03:00 with pre/post hooks to stop and restart the nginx container. Certificate valid until 2026-08-05.  
**Live evidence:**
```bash
curl -vI https://scotland-2026-election-forecast.duckdns.org/ 2>&1 | grep "SSL certificate\|subject:"
```

---

### 5. CI/CD — GitHub Actions

**Tool:** GitHub Actions (3 workflows)  
**Implementation:**
- `ci.yml` — runs on every push and PR: flake8 lint, 150+ pytest unit tests across `test_api.py`, `test_data_generation.py`, `test_marginals.py`, `test_metrics.py`
- `deploy.yml` — SSH deploys to Oracle Cloud on push to `main`: `git pull`, `docker compose up -d --build`, 40-second grace period, then health-checks `/health` on all three services
- `drift.yml` — scheduled drift monitoring run; writes Evidently HTML report to `reports/`

**Live evidence:**
```
https://github.com/dmishra27/scotland-2026-election-forecast/actions
```

---

### 6. Marginal Constituency Analysis — Custom D'Hondt

**Tool:** Custom Python (`src/models/marginals.py`, `src/models/dhondt.py`)  
**Implementation:** Simulates vote shares at constituency level by running the trained ensemble over synthetic voter samples per seat. Any constituency with a leading-party majority margin < 5 pp is flagged as marginal. For each marginal seat the swing needed for the second party to overtake is computed, and a tactical vote recommendation is returned. 56 marginal seats identified across Scotland's 8 regions; tightest seat: **Inverness and Nairn** at 0.6 pp.  
**Live evidence:**
```bash
curl https://scotland-2026-election-forecast.duckdns.org/api/seats/marginals | python -m json.tool
```

---

## Model Architecture

```
Synthetic voter data
  12,500 voters · Dirichlet concentration=50 · random_seed=42
  Seeded from YouGov MRP polling priors (April 2026, n=3,925)
        |
Feature pipeline — scikit-learn ColumnTransformer (18 features)
  Numerical (StandardScaler):  age, income, tactical_swing_index,
                               nhs_dissatisfaction
  Ordinal (OrdinalEncoder):    education, indep_economy_interaction
  One-hot:                     region (8), gender, employment
        |
Stacking Ensemble — Optuna TPE, 50 trials per base learner
  Base learners:  XGBoost · LightGBM · CatBoost · RandomForest
  Meta learner:   LogisticRegression (class_weight=balanced)
        |
D'Hondt AMS seat allocation
  73 constituency seats (FPTP winner per seat)
  56 regional list seats (7 per region, D'Hondt divisor method)
        |
SHAP TreeExplainer — mean |SHAP| per feature, all 6 parties
```

---

## Model Performance

*Trained on Oracle Cloud VM.Standard.E2.1.Micro · Ubuntu 20.04 · 2026-05-05*

| Metric | Value |
|--------|------:|
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
| Optuna trials | 50 |

Macro F1 of 0.31 reflects genuine 6-class imbalance: SNP accounts for ≈36% of the synthetic constituency vote share. Per-class scores are the more informative metric — SNP F1 of 0.62 is reasonable; LibDem F1 of 0.13 reflects its small share in the training distribution, not a modelling error.

---

## Seat Projection (v1.5.0 · D'Hondt AMS)

| Party | Constituency | Regional List | Total |
|-------|:-----------:|:------------:|:-----:|
| **SNP** | **73** | 0 | **73** |
| Reform | 0 | 16 | 16 |
| Labour | 0 | 16 | 16 |
| Conservative | 0 | 8 | 8 |
| LibDem | 0 | 8 | 8 |
| Green | 0 | 8 | 8 |
| **Total** | **73** | **56** | **129** |

**SNP majority** — threshold: 65 seats · Governing party: SNP ✓

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data generation | NumPy Dirichlet, Pandas |
| Feature engineering | scikit-learn ColumnTransformer |
| ML models | XGBoost, LightGBM, CatBoost, scikit-learn |
| Hyperparameter optimisation | Optuna (TPE sampler) |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| Drift monitoring | Evidently AI |
| Explainability | SHAP (TreeExplainer) |
| Seat allocation | D'Hondt AMS (custom Python) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly (6 pages) |
| Containerisation | Docker + Docker Compose |
| Reverse proxy | Nginx stable-alpine |
| SSL | Let's Encrypt (certbot standalone) |
| DNS | DuckDNS |
| CI/CD | GitHub Actions |
| Cloud | Oracle Cloud VM.Standard.E2.1.Micro |
| OS | Ubuntu 22.04 LTS |

---

## Post-Election Validation (8 May 2026)

The Scottish Parliament election took place on 8 May 2026 — the day before this document was last updated. Official results are pending certification. This section will be completed once the full count is confirmed.

**What the model predicted (v1.5.0):** SNP majority with 73 constituency seats, all regional seats split among Reform, Labour, Conservative, LibDem, and Green. 56 marginal constituencies flagged, tightest margin 0.6 pp (Inverness and Nairn).

**Honest prior expectation:** The model is trained on synthetic data derived from April 2026 YouGov MRP priors. It cannot capture late campaign swings, tactical voting coordination, or turnout variation by constituency — all of which materially affect AMS outcomes. The D'Hondt projection is arithmetically correct given the input vote shares; the vote shares themselves are the uncertain quantity. A model this simple winning on seat totals would be fortune, not precision.

| Party | Projected seats | Actual seats | Delta |
|-------|:--------------:|:------------:|:-----:|
| SNP | 73 | — | — |
| Reform | 16 | — | — |
| Labour | 16 | — | — |
| Conservative | 8 | — | — |
| LibDem | 8 | — | — |
| Green | 8 | — | — |

*Fill in actual column once the Electoral Management Board confirms results.*

---

## Key Engineering Decisions

- **Why synthetic data?** YouGov MRP polling is not available as a public dataset. Dirichlet-noised synthetic generation from published polling priors allows the full ML pipeline — DVC versioning, Optuna HPO, MLflow tracking, SHAP explainability — to be demonstrated reproducibly and openly without licensing restrictions. The data science problem is a vehicle for the engineering, not the end goal.

- **Why a stacking ensemble?** Individual gradient boosting models plateau around macro F1 0.28–0.30 on this 6-class problem. The stacking meta-learner (LogisticRegression with `class_weight=balanced`) captures complementary error patterns across XGBoost, LightGBM, CatBoost, and RandomForest, consistently adding 2–4 F1 points in controlled ablations. It also gives a natural scaffold to demonstrate Optuna HPO independently per base learner.

- **Why D'Hondt?** The Scottish Parliament uses the Additional Member System — 73 FPTP constituency seats plus 56 regional list seats allocated by the D'Hondt divisor method across 8 regions. Accurate seat projection requires implementing the actual electoral formula. A simple proportional estimate would allocate constituency seats incorrectly and is not how AMS works; implementing D'Hondt is a correctness requirement, not a complexity choice.

- **Why Oracle Cloud Free Tier?** Zero recurring cost for a VM with a persistent public IP that stays live indefinitely for recruiter review. The VM.Standard.E2.1.Micro shape (1 OCPU, 1 GB RAM) imposes real constraints — Optuna trials are capped, model files must stay lean — which makes the infrastructure decisions more honest than a paid cloud account where resources are unconstrained.

---

## Known Limitations

- **Synthetic training data** — the model learns from Dirichlet-noised polling priors, not real voter records. Relationships in the data are constructed, not observed; the model cannot generalise to genuine behavioural heterogeneity.
- **Six-class imbalance** — SNP dominates the synthetic constituency distribution at ≈36% share, suppressing per-class F1 for smaller parties (LibDem: 0.13, Green: 0.16). `class_weight=balanced` in the meta-learner partially compensates but does not eliminate the bias.
- **No temporal dynamics** — the model is a static snapshot trained on April 2026 priors. Late campaign events, debate performance effects, and last-week swing are invisible to it.
- **Constituency boundary assumptions** — the D'Hondt allocation uses the standard 73+56 AMS structure. Any boundary changes, by-election vacancies, or party deregistrations between training and polling day are not modelled.
- **Free-tier hardware constraints** — training on VM.Standard.E2.1.Micro (1 OCPU, 1 GB RAM) caps Optuna trials at 50 per base learner. A larger search budget on adequate hardware would likely improve macro F1 by 3–5 points.

---

## Repository Structure

```
scotland-2026-election-forecast/
|
+-- src/
|   +-- api/
|   |   +-- main.py            # 9 FastAPI endpoints
|   |   \-- schemas.py         # Pydantic request/response models
|   +-- data/
|   |   \-- generate_voters.py # Dirichlet-noised synthetic voters
|   +-- features/
|   |   \-- pipeline.py        # ColumnTransformer (18 features)
|   +-- models/
|   |   +-- ensemble.py        # Stacking ensemble + Optuna HPO
|   |   +-- base_models.py     # XGBoost/LightGBM/CatBoost/RF wrappers
|   |   +-- dhondt.py          # D'Hondt AMS seat allocator
|   |   +-- marginals.py       # Marginal seat identification
|   |   \-- explainability.py  # SHAP TreeExplainer
|   +-- monitoring/
|   |   \-- drift.py           # Evidently drift reports
|   \-- orchestration/
|       \-- train_pipeline.py  # End-to-end training orchestrator
|
+-- streamlit_app/
|   +-- Home.py
|   \-- pages/
|       +-- 1_Voter_Simulator.py
|       +-- 2_Seat_Projections.py
|       +-- 3_Model_Performance.py
|       +-- 4_SHAP_Explainability.py
|       +-- 5_Marginal_Constituencies.py
|       \-- 6_Constituency_Explorer.py
|
+-- scripts/
|   +-- generate_data.py       # DVC stage: generate
|   +-- featurise.py           # DVC stage: featurise
|   +-- train_models.py        # DVC stage: train
|   \-- run_drift_report.py    # Evidently report runner
|
+-- deploy/
|   +-- docker-compose.prod.yml
|   +-- nginx.conf / nginx-ssl.conf
|   +-- setup_server.sh        # One-command Oracle Cloud bootstrap
|   \-- enable_ssl.sh
|
+-- .github/workflows/
|   +-- ci.yml                 # Lint + 150+ tests on every push
|   +-- deploy.yml             # SSH deploy to Oracle Cloud on main
|   \-- drift.yml              # Scheduled drift monitoring
|
+-- tests/unit/
|   +-- test_api.py
|   +-- test_data_generation.py
|   +-- test_marginals.py
|   \-- test_metrics.py
|
+-- configs/config.yaml        # YouGov MRP priors, model config
+-- dvc.yaml                   # Three-stage DVC pipeline
+-- metrics/train_metrics.json # DVC-tracked metrics (non-cached)
\-- pyproject.toml
```

---

*Debabrata Mishra — Senior Data Scientist / ML Engineer, Glasgow, Scotland*
*GitHub: [dmishra27](https://github.com/dmishra27) · LinkedIn: [linkedin.com/in/debabrata-mishra](https://linkedin.com/in/debabrata-mishra)*
