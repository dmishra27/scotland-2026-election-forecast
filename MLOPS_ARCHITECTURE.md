# MLOps Architecture

Scotland 2026 Election Forecast — system design reference.

---

## MLOps = DataOps + DevOps + ModelOps

The three pillars each own a distinct part of the lifecycle. Failures in one cascade into the others: dirty data corrupts model signal, broken CI ships bad code, an unmonitored model silently degrades.

| Pillar | Responsibility | Primary tools | Grade |
|--------|---------------|---------------|:-----:|
| **DataOps** | Data generation, versioning, schema validation, drift detection | DVC, Evidently AI, Parquet, JSONL | B+ |
| **DevOps** | Containerisation, routing, SSL, CI/CD, semantic versioning | Docker Compose, Nginx, GitHub Actions, certbot | A- |
| **ModelOps** | Experiment tracking, HPO, explainability, serving, retraining | MLflow, Optuna, SHAP, FastAPI, StackingClassifier | A- |

---

## DataOps

### DVC Pipeline

Content-addressed three-stage pipeline. DVC fingerprints each stage's command, inputs, and outputs; a stage only re-executes when its dependency hash changes. This means `dvc repro` is a no-op if the data and code have not changed — deterministic and cache-efficient.

```yaml
# dvc.yaml
stages:

  generate:
    cmd: python scripts/generate_data.py
    deps:
      - scripts/generate_data.py
      - src/data/generate_voters.py
    outs:
      - data/raw/voters.parquet

  featurise:
    cmd: python scripts/featurise.py
    deps:
      - scripts/featurise.py
      - src/features/pipeline.py
      - data/raw/voters.parquet
    outs:
      - data/processed/features.parquet
      - models/pipeline.pkl

  train:
    cmd: python scripts/train_models.py --n-trials 20
    deps:
      - scripts/train_models.py
      - src/models/ensemble.py
      - src/orchestration/train_pipeline.py
      - data/processed/features.parquet
    outs:
      - models/ensemble.pkl
    metrics:
      - metrics/train_metrics.json:
          cache: false
```

`metrics/train_metrics.json` is declared `cache: false` — it is committed to git and tracked by DVC metrics, so `dvc metrics diff` can compare it across commits without storing it in the DVC cache.

```bash
dvc repro                  # re-run invalidated stages only
dvc metrics show           # print train_metrics.json
dvc metrics diff HEAD~1    # compare current vs previous run
dvc push                   # push cached data/models to remote
```

---

### Evidently Drift Monitoring

**Architecture:** every call to `POST /predict` appends the request features plus prediction to `data/monitoring/predictions.jsonl` via `log_prediction()` in `src/monitoring/drift.py`. This log is the "current" dataset for drift comparison. The training parquet (`data/raw/voters.parquet`) is the reference.

```python
# src/monitoring/drift.py — core logic
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)
report.save_html(str(output_path))

result = report.as_dict()["metrics"][0]["result"]
drift_detected = result["dataset_drift"]   # True when majority of features drift
n_drifted      = result["number_of_drifted_columns"]
```

**Exit codes from `scripts/run_drift_report.py`:**

| Exit code | Meaning |
|:---------:|---------|
| 0 | No significant drift detected |
| 1 | Script error (missing reference data, import failure) |
| 2 | Drift detected — triggers GitHub issue creation in `drift.yml` |

**Scheduled execution:** `drift.yml` runs at `0 2 * * *` (daily 02:00 UTC). When `run_drift_report.py` exits with code 2, a GitHub Actions Script step automatically creates a labelled issue (`drift`, `model-ops`) with a remediation checklist linking to the workflow run and recommending `dvc repro` + `dvc metrics diff`.

**Report serving:** the HTML report is written to `reports/drift_report.html` and served by nginx at `/reports/` via an `autoindex on` static alias — no additional service required.

---

### Schema Validation

Point-in-time validation runs at the start of the training pipeline before any model code executes. It is not a streaming validator; it checks the batch of generated data once.

```python
# src/orchestration/train_pipeline.py
def validate_dataframe(df: pd.DataFrame) -> None:
    required = [
        "voter_id", "region", "age", "constituency_vote", "list_vote",
        "independence_stance", "economic_concern", "health_concern",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    unknown_parties = set(df["constituency_vote"].unique()) - set(PARTIES)
    if unknown_parties:
        raise ValueError(f"Unknown parties in data: {unknown_parties}")

    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Null values found in columns: {null_cols}")
```

API-level validation is handled by Pydantic in `src/api/schemas.py`. `VoterFeatures` enforces bounds (`age` ge=18 le=85, `economic_concern` ge=0 le=10), validates `independence_stance` against an explicit regex (`^(Strong Yes|Lean Yes|Undecided|Lean No|Strong No)$`), and auto-derives `age_group` from `age` if omitted — a `model_validator` runs post-init.

---

### DataOps Gaps and Future Tools

| Gap | Impact | Future tool |
|-----|--------|-------------|
| No data lineage UI | Hard to audit which model was trained on which data version | DVC Studio (free tier) or MLflow datasets API |
| Single local DVC remote | Data and models are not backed up off-machine | S3-compatible remote (Oracle Object Storage is free-tier eligible) |
| No streaming schema validation on `/predict` | A malformed but schema-passing request can silently inject bad data into the drift log | Great Expectations or Pandera in an API middleware layer |
| Vote share assertion is a range check only | Generation bugs that shift distribution within [0.20, 0.55] are invisible | Statistical test (KS test vs. prior) as a DVC `check` stage |
| No adversarial drift testing | Drift monitoring only triggers on organic distribution shift | Synthetic drift injection test — deliberately shift one feature's distribution and assert exit code 2 |

---

## DevOps

### Docker Compose Services

All four services are defined in `deploy/docker-compose.prod.yml`. Health checks cascade: `api` waits for `mlflow` to be healthy; `streamlit` waits for `api`; `nginx` waits for both `api` and `streamlit`.

| Service | Image | Port | Memory limit | Health check |
|---------|-------|:----:|:------------:|-------------|
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.10.2` | 5000 | 1 GB | HTTP GET `localhost:5000/health` |
| `api` | Custom `deploy/Dockerfile.api.prod` | 8000 | 8 GB | HTTP GET `localhost:8000/health` |
| `streamlit` | Custom `deploy/Dockerfile.streamlit.prod` | 8501 | 2 GB | HTTP GET `localhost:8501/_stcore/health` |
| `nginx` | `nginx:stable-alpine` | 80, 443 | 256 MB | None (reverse proxy; health depends on upstreams) |

MLflow uses a SQLite backend (`sqlite:///mlflow/mlflow.db`) with a `/mlflow/artifacts` volume for artefact storage. The `models_data` volume is bind-mounted from `/home/ubuntu/.../models` on the host so the API container can read pre-trained model files without rebuilding the image.

---

### Nginx Routing

```
Internet : 443 (HTTPS)
    |
nginx:stable-alpine
    |
    +-- /api/       → proxy_pass http://api:8000/
    |                  proxy_read_timeout 120s
    |                  X-Forwarded-Prefix: /api
    |
    +-- /mlflow/    → proxy_pass http://mlflow:5000/
    |                  sub_filter href="/" href="/mlflow/"
    |                  proxy_redirect http://mlflow/ /mlflow/
    |
    +-- /reports/   → alias /app/reports/  (static HTML, autoindex on)
    |
    +-- /           → proxy_pass http://streamlit:8501
                       Upgrade: $http_upgrade
                       Connection: $connection_upgrade
                       proxy_read_timeout 86400s  (WebSocket keepalive)
                       proxy_buffering off
```

HTTP (port 80) serves only `/.well-known/acme-challenge/` (Let's Encrypt ACME webroot) and redirects everything else via `return 301 https://$host$request_uri`. Gzip is enabled at level 6 for `text/`, `application/json`, `application/javascript`, and SVG. `client_max_body_size 16M` permits large batch prediction payloads.

---

### GitHub Actions Workflows

| Workflow | File | Trigger | Jobs |
|----------|------|---------|------|
| CI | `ci.yml` | Push to `main`, `develop`, `feature/**`; PR to `main` or `develop` | `lint` (ruff + black) → `test` (pytest) + `data-pipeline-smoke` → `api-smoke` (FastAPI TestClient) → `docker-build` (main only) |
| CD | `deploy.yml` | `workflow_run: CI completed` on `main` branch, `conclusion == success` | `deploy` — SSH into Oracle VM, `git pull`, `docker compose up -d --build`, 40 s grace, health checks on all three services, external curl via public IP |
| Drift | `drift.yml` | Cron `0 2 * * *` (daily 02:00 UTC) + `workflow_dispatch` | `drift-report` — run `scripts/run_drift_report.py`, upload HTML artifact, auto-create GitHub issue on exit code 2 |

**CI job dependency graph:**

```
lint
 ├── test               (unit tests, pytest, JUnit XML upload)
 │    └── api-smoke     (FastAPI TestClient: health, predict, seats/projected, model/info)
 │         └── docker-build  [main branch only]
 └── data-pipeline-smoke  (generates 500 voters, validates nulls, checks engineered features)
```

The `deploy` job runs in the `production` GitHub environment, which can be configured with required reviewers and protection rules. CD is gated on CI success — it cannot trigger from a failed CI run.

---

### SSL / HTTPS Configuration

| Setting | Value |
|---------|-------|
| DNS provider | DuckDNS (free subdomain, 5-minute cron IP refresh on VM) |
| Certificate authority | Let's Encrypt (certbot standalone) |
| Certificate validity | 90 days, auto-renewed via root crontab at 03:00 |
| TLS protocols | TLSv1.2, TLSv1.3 only |
| Cipher suite | `HIGH:!aNULL:!MD5` |
| Session cache | `shared:SSL:10m`, timeout 10 m |
| HTTP redirect | `return 301 https://$host$request_uri` (port 80 → 443) |
| ACME challenge | `/.well-known/acme-challenge/` served via certbot webroot volume |
| Renewal hooks | `--pre-hook`: stop nginx container; `--post-hook`: start nginx container |

Renewal uses `certbot renew --quiet` with pre/post Docker Compose hooks rather than certbot standalone mode during renewal, so nginx does not need to be stopped for the full renewal duration — only briefly while certbot takes port 80.

---

### Semantic Version History

| Tag | Commits | Features added |
|-----|---------|---------------|
| `v1.0.0` | `32a96b4` | Initial complete pipeline — synthetic data, stacking ensemble, D'Hondt, FastAPI, Streamlit (4 pages), Oracle Cloud ARM64 deployment |
| `v1.1.0` | `c2e8e01` | GitHub Actions CI/CD, DVC data versioning, Nginx reverse proxy, Let's Encrypt SSL preparation |
| `v1.2.0` | `d0a7748` | Marginal constituency analysis, tactical swing probability, Evidently drift monitoring (Phase 2) |
| `v1.3.0` | `43d1d49` | Async retrain endpoint (`POST /model/retrain` + status polling), MLflow info panel in Streamlit |
| `v1.4.0` | `a3f5eb4` | Constituency Explorer — per-seat search, cached projections, `GET /seats/constituencies` |
| `v1.5.0` | `0c525aa` | Constituency vote-share chart fix — percentages, descending sort |
| `v1.6.0` | — | Post-election results page (`7_Actual_Scottish_Election_Results_2026.py`) — certified 129-seat count, regional heatmap, candidate names |
| `v1.7.0` | — | Seat Forecast vs Actual — page renamed from Seat Projections; four-bar chart and three-way comparison table added; Constituency Explorer enhanced with actual results columns and deviation table |
| `v1.8.0` | — | Documentation sync — all docs updated to 8-page dashboard; Feature Importance page (renamed from SHAP Explainability); LLMOps integration roadmap formalised (current production tag) |

---

### DevOps Gaps and Future Tools

| Gap | Impact | Future tool |
|-----|--------|-------------|
| No blue/green deployment | CD restarts containers in place; there is a brief (~40 s) downtime window during `docker compose up --build` | Traefik with weighted routing between two Compose stacks |
| No rollback automation | A broken deploy requires a manual SSH session and `git revert` | Store the previous image digest in CD; auto-rollback on failed post-deploy health check |
| Single VM, no HA | Oracle Cloud VM is a single point of failure | Oracle Cloud Load Balancer + two Always Free VMs (requires upgrading account type) |
| No container image scanning | Vulnerabilities in `nginx:stable-alpine` or Python dependencies go undetected | Trivy or Grype in CI after `docker-build` job |
| No alerting on health check failure | A service crash is invisible until a user notices | Prometheus + Grafana Cloud free tier; alert on `/health` returning non-200 |

---

## ModelOps

### MLflow Experiment Tracking

**Server:** `ghcr.io/mlflow/mlflow:v2.10.2`, SQLite backend, artefact root at `/mlflow/artifacts` (Docker volume). Exposed at `/mlflow/` through nginx with `sub_filter` rewriting asset paths.

**Experiment name:** `scotland-2026-forecast`
**Run name prefix:** `stacking-ensemble`

Two MLflow runs are created per training pipeline execution:

| Run suffix | Logged params | Logged metrics | Logged artefacts |
|-----------|--------------|----------------|-----------------|
| `-base-models` | `n_voters`, `n_trials`, `seed`, `{name}___{k}` per learner | `{name}_val_f1` per base learner | — |
| `-ensemble` | Inherited from base-model run | `test_accuracy`, `test_f1_macro`, `test_log_loss`, `test_brier_score`, `test_f1_{party}` × 6 | `ensemble.pkl`, `pipeline.pkl`, `shap_importance.csv` |

The ensemble artefacts are also written to `models/` on the host via the bind-mounted `models_data` volume, so the API container can load them without pulling from the MLflow artefact store.

---

### Optuna Hyperparameter Optimisation

**Sampler:** TPESampler (Tree-structured Parzen Estimator), `seed=42`
**Direction:** `minimize` log_loss
**Inner CV:** `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` inside each trial objective
**Outer CV (stacking):** `StratifiedKFold(n_splits=5)` used by `StackingClassifier`
**Trials in production:** 50 (config: `model.optuna_trials`)

Each base learner is tuned independently in its own Optuna study. Search spaces:

| Learner | Hyperparameters searched | Objective |
|---------|--------------------------|-----------|
| XGBoost | `n_estimators` [200,800], `max_depth` [3,8], `learning_rate` [0.01,0.3 log], `subsample` [0.6,1], `colsample_bytree` [0.6,1], `min_child_weight` [1,10], `gamma` [0,5] | OOF log_loss |
| LightGBM | `n_estimators` [200,800], `num_leaves` [20,150], `learning_rate` [0.01,0.3 log], `subsample` [0.6,1], `colsample_bytree` [0.6,1], `min_child_samples` [5,50] | OOF log_loss |
| CatBoost | `iterations` [200,800], `depth` [3,8], `learning_rate` [0.01,0.3 log], `l2_leaf_reg` [1,10] | OOF log_loss |
| RandomForest | `n_estimators` [200,600], `max_depth` [5,20], `min_samples_split` [2,20], `min_samples_leaf` [1,10], `max_features` {sqrt, log2} | OOF log_loss |

Best params from each study are logged to MLflow with the prefix `{learner_name}__` so all four learners' params appear in a single run without key collisions.

---

### Stacking Ensemble Performance

| Level | Component | Configuration |
|-------|-----------|--------------|
| Base (level-0) | XGBoostClassifier | Optuna-tuned, `eval_metric=mlogloss`, `verbosity=0` |
| Base (level-0) | LGBMClassifier | Optuna-tuned, `class_weight=balanced`, `verbose=-1` |
| Base (level-0) | CatBoostClassifier | Optuna-tuned, `allow_writing_files=False` |
| Base (level-0) | RandomForestClassifier | Optuna-tuned, `class_weight=balanced`, `n_jobs=-1` |
| Meta (level-1) | LogisticRegression | `C=1.0`, `class_weight=balanced`, `max_iter=1000` |
| Stacker | `StackingClassifier` | `stack_method=predict_proba`, `cv=StratifiedKFold(5)`, `n_jobs=-1` |

String class labels (`"SNP"`, `"Reform"`, …) are encoded to integers via a fitted `LabelEncoder` before being passed to XGBoost and other strict classifiers, then decoded back to strings in `predict()`. `predict_proba()` returns the raw probability matrix indexed by the integer-encoded class order.

**Test-set metrics (Oracle Cloud VM, 2026-05-05):**

| Metric | Value |
|--------|------:|
| Accuracy | 0.4032 |
| F1 macro | 0.3060 |
| Log-loss | — |
| F1 — SNP | 0.6156 |
| F1 — Labour | 0.3623 |
| F1 — Conservative | 0.3130 |
| F1 — Reform | 0.2522 |
| F1 — Green | 0.1598 |
| F1 — LibDem | 0.1333 |

---

### FastAPI Endpoint Reference

Base path: `/api/` (nginx proxy strips the prefix before forwarding to FastAPI on port 8000)

| Method | Path | Tag | Request | Response | Demo mode fallback |
|--------|------|-----|---------|----------|--------------------|
| `GET` | `/health` | meta | — | `{status, model_loaded, version}` | Always returns `ok`; `model_loaded: false` |
| `POST` | `/predict` | prediction | `VoterFeatures` (15 fields, Pydantic-validated) | `{predicted_party, probabilities[6], tactical_swing_index, indep_economy_interaction, nhs_dissatisfaction}` | Rule-based adjustments on MRP priors |
| `POST` | `/predict/batch` | prediction | `{voters: [VoterFeatures], max_length: 1000}` | `{predictions: [...], n_voters}` | Same rule-based fallback per voter |
| `GET` | `/seats/projected` | seats | — | `{constituency, regional, total, majority_threshold, governing_party, has_majority}` | D'Hondt from MRP priors directly |
| `GET` | `/seats/marginals` | seats | — | `{seats[20], n_marginal, threshold_pp, demo_mode}` | Dirichlet-noised MRP priors, 56 marginals |
| `GET` | `/seats/constituencies` | seats | `?search=<term>` (optional) | `{constituencies[73], n_results, search_term, demo_mode}` | Seeded Dirichlet noise (seed=42) |
| `GET` | `/model/info` | meta | — | `{model_type, base_learners, meta_learner, classes, n_features, is_loaded, mlflow_run_id}` | Returns `is_loaded: false` |
| `POST` | `/model/retrain` | meta | — | `{status: queued, started_at, finished_at, metrics, error}` | 202 Accepted; background thread |
| `GET` | `/model/retrain/status` | meta | — | `{status, started_at, finished_at, metrics, error}` | Reflects current `_retrain_job` dict |

CORS is open (`allow_origins=["*"]`) for dashboard-to-API communication inside the Docker network and for external browser access via nginx.

---

### SHAP Explainability

`src/models/explainability.py` computes TreeExplainer SHAP values for each base learner independently, then averages the mean absolute SHAP matrices across all four learners to produce an ensemble-level feature importance.

```python
for name, estimator in model.model_.named_estimators_.items():
    try:
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(X_sample)        # list[array] for multi-class
        sv = np.stack(sv, axis=-1)                  # (n, features, classes)
        all_shap.append(np.abs(sv).mean(axis=-1))   # mean over classes
    except Exception:
        # KernelExplainer fallback — uses k-means background, capped at 50 samples
        background = shap.kmeans(X_sample, 10)
        explainer  = shap.KernelExplainer(estimator.predict_proba, background)
        sv = explainer.shap_values(X_sample[:50])
        all_shap.append(np.abs(sv).mean(axis=-1))

shap_values = np.mean(all_shap, axis=0)    # average over base learners
```

`max_samples=500` caps the computation: a random subsample of 500 rows is drawn from the test set. The output `shap_importance.csv` (top 20 features by mean |SHAP|) is logged as an MLflow artefact and served in the Streamlit `4_SHAP_Explainability.py` page as a horizontal bar chart.

The SHAP values reflect the average base-learner importance, not the meta-learner's weighting. Meta-learner coefficients (LogisticRegression weights on the OOF probability matrix) are not currently visualised.

---

### Retrain Endpoint — State Machine

`POST /model/retrain` triggers an asynchronous background training run in a daemon thread. `GET /model/retrain/status` polls the shared `_retrain_job` dict (protected by `_retrain_lock`).

```
         POST /model/retrain
                 |
         [lock acquired]
                 |
    ┌────────────▼────────────┐
    │  status == "queued"      │
    └────────────┬────────────┘
                 │  [daemon thread starts]
    ┌────────────▼────────────┐
    │  status == "running"     │
    │  python scripts/         │
    │    train_models.py       │
    │    --n-trials 1          │
    │    --n-voters 5000       │
    │    --model-dir models/   │
    │                latest    │
    └──────┬──────────┬───────┘
           │ success  │ failure
    ┌──────▼──────┐  ┌▼──────────────────────┐
    │ status ==   │  │ status == "failed"     │
    │ "complete"  │  │ error: <stderr>        │
    │ metrics: {  │  │ finished_at: <UTC iso> │
    │   f1_macro, │  └────────────────────────┘
    │   accuracy  │
    │ }           │
    └─────────────┘
```

On success, the retrained `models/latest/ensemble.pkl` and `models/latest/pipeline.pkl` are hot-swapped into `_state["ensemble"]` and `_state["pipeline"]`, and both caches (`marginals_cache`, `constituencies_cache`) are invalidated so the next seat-related request recomputes from the new model. Requesting `POST /model/retrain` while a job is already `queued` or `running` returns `409 Conflict`.

---

### Constituency Explorer

`GET /seats/constituencies` returns predicted vote shares for all 73 Scottish Parliament constituency seats. An optional `?search=<term>` parameter applies a case-insensitive partial string match on the constituency name.

**Caching strategy:** constituency results are computed once per model lifecycle (or once per process start in demo mode) and stored in `_state["constituencies_cache"]`. The cache is invalidated on retrain completion. This avoids re-running 73 × model inference calls on every page render.

**Demo mode:** when no trained model is present, `_demo_constituency_results()` generates per-constituency vote shares by drawing from YouGov MRP regional priors with Dirichlet noise (`seed=42`), producing a deterministic and consistent set of results for every demo session.

**Streamlit page (`6_Constituency_Explorer.py`):** free-text search box → API call → horizontal Plotly bar chart of vote shares (converted to %, sorted descending), plus a data table with leading party, margin (pp), and tactical vote recommendation. Results update on every keystroke via Streamlit session state.

---

### ModelOps Gaps and Future Tools

| Gap | Impact | Future tool |
|-----|--------|-------------|
| Retrain is manually triggered, not automated | Evidently detects drift but does not automatically kick off a retrain | Wire `run_drift_report.py` exit code 2 to `POST /model/retrain` in the drift workflow |
| No model registry | Models are tracked as MLflow artefacts, not registered versions with staging/production lifecycle | MLflow Model Registry (`mlflow.register_model`) with `Staging → Production` promotion gates |
| Meta-learner not explained | SHAP covers the four base learners; the LogisticRegression meta-learner's OOF weighting is invisible | Log LR coefficients as a `meta_weights` artefact in MLflow; add a `5b_Meta_Weights` Streamlit page |
| No prediction calibration | `predict_proba` outputs are raw classifier probabilities, not calibrated | Platt scaling or isotonic regression via `sklearn.calibration.CalibratedClassifierCV` on the meta-learner |
| No A/B testing | All traffic hits the same model; comparing two model versions requires sequential deployment | Shadow mode endpoint: route 10% of `/predict` traffic to a challenger model, log both predictions, compare F1 offline |

---

## Integrated Architecture (16 Layers)

| Layer | Component | Technology | Location |
|-------|-----------|-----------|---------|
| 1 | Synthetic voter generation | NumPy Dirichlet, Pandas | `src/data/generate_voters.py` |
| 2 | Raw data storage | Parquet (columnar, Snappy-compressed) | `data/raw/voters.parquet` |
| 3 | Data versioning | DVC 3.x, content-addressed cache | `dvc.yaml`, `.dvc/` |
| 4 | Schema validation | Pydantic v2 (API), custom `validate_dataframe` (training) | `src/api/schemas.py`, `src/orchestration/train_pipeline.py` |
| 5 | Feature engineering | scikit-learn ColumnTransformer (18 features, 3 engineered) | `src/features/pipeline.py` |
| 6 | Hyperparameter optimisation | Optuna TPE, 50 trials × 4 learners, 3-fold inner CV | `src/models/ensemble.py` |
| 7 | Model training | StackingClassifier — XGBoost, LightGBM, CatBoost, RF → LR meta | `src/models/ensemble.py` |
| 8 | Experiment tracking | MLflow 2.10.2, SQLite backend, artefact volume | `src/orchestration/train_pipeline.py` |
| 9 | Model explainability | SHAP TreeExplainer, averaged over base learners, top-20 CSV | `src/models/explainability.py` |
| 10 | Seat allocation | D'Hondt AMS — 73 constituency + 56 regional list seats | `src/models/dhondt.py` |
| 11 | Drift monitoring | Evidently `DataDriftPreset`, JSONL prediction log, HTML reports | `src/monitoring/drift.py` |
| 12 | REST API | FastAPI + Uvicorn, 9 endpoints, demo-mode fallback, async retrain | `src/api/main.py` |
| 13 | Interactive dashboard | Streamlit 8-page app + Plotly charts | `streamlit_app/` |
| 14 | Containerisation | Docker Compose — 4 services, health-check cascade, bind volumes | `deploy/docker-compose.prod.yml` |
| 15 | Reverse proxy | Nginx stable-alpine — routing, WebSocket, gzip, HTTPS | `deploy/nginx.conf` |
| 16 | CI/CD | GitHub Actions — 3 workflows, gated deploy, drift alerting | `.github/workflows/` |

---

## LLMOps Integration Roadmap

The current system has no LLM components. The following roadmap describes where AI-native capabilities would add the most value, ordered by the MLOps pillar they belong to. Implementation of any item below would extend the stack beyond classical MLOps into LLMOps — closing the loop between live election data, automated reasoning, and model governance.

---

### DataOps: LLM-Powered Data Quality and Generation

**Semantic data quality checks**
Replace the current boolean column-presence and null-count validation with a language-model quality assessor. An LLM evaluator could be prompted with the generation config and a sample of the output data and asked to flag statistical anomalies, implausible combinations (e.g. a voter with `independence_stance=Strong Yes` and `previous_vote=Conservative` in 2021), or distribution shifts that pass numeric tests but are substantively wrong. This is especially valuable for synthetic data where the reference distribution is not observed reality but a design specification.

**Real-survey grounding via Claude API**
The multiplier adjustment table in `generate_voters.py` (SNP × 2.5 for Strong Yes, Reform × 2.0 for top-priority Immigration, etc.) is currently set by hand. A Claude API call could be used at generation time to propose or revise multipliers by reasoning over published British Election Study marginals and YouGov Scotland cross-breaks, producing calibrated adjustments rather than subjective constants.

**Automated prior updating**
When a new YouGov MRP poll is published, a scheduled agent could parse the PDF or HTML table, extract party vote shares, update `configs/config.yaml`, and trigger `dvc repro` — closing the loop between polling evidence and synthetic data distribution with no manual intervention.

---

### DevOps: AI-Assisted Operations

**Natural language deployment review**
Before the CD workflow SSHs into the Oracle VM, a Claude-powered step could summarise the diff between the outgoing and incoming Docker images — listing changed dependencies, new endpoints, removed endpoints, and any security-relevant changes — and post the summary as a GitHub Actions step annotation. This replaces reading raw image diffs with a scannable English summary.

**Log anomaly detection**
nginx access logs and Docker container stdout are currently not analysed. An LLM-based anomaly detector (running on a cron or streaming from a log aggregator) could flag unusual request patterns — a sudden spike in `/predict/batch` calls, repeated 422 validation errors from a single IP, or a shift in the distribution of `independence_stance` values in incoming requests that suggests a non-human caller.

**Dependency update triage**
Pinned versions in `requirements.txt` and `deploy/docker-compose.prod.yml` will accumulate security debt. An AI agent scheduled weekly could evaluate pending dependency updates by reading the changelogs and CVE descriptions, classify each update as security-critical / breaking / routine, and open labelled GitHub issues with recommended actions rather than a flat Dependabot PR.

---

### ModelOps: Explainability, Monitoring, and Self-Improving Retraining

**Natural language SHAP explanations**
The SHAP page currently shows a bar chart of mean absolute feature importance. A Claude API call using the `shap_importance.csv` artefact could generate a two-paragraph plain-English explanation of which voter attributes drive party predictions — calibrated for a non-technical audience (e.g. a journalist or politician reading the dashboard) rather than a data scientist. The explanation could be cached per MLflow run ID so it is computed once and served from MLflow metadata.

**Automated model card generation**
After each retrain, a Claude API call could generate a structured model card — intended use, performance metrics in plain English, known limitations, ethical considerations for using synthetic voter predictions — and commit it to `models/MODEL_CARD.md` as part of the CD pipeline. This closes the documentation gap between experiment tracking (MLflow logs) and human-readable model governance.

**Drift-to-retrain reasoning agent**
Currently, drift detection exits with code 2 and creates a GitHub issue. A more capable loop would have a reasoning agent read the Evidently HTML report, identify which features drifted most and in which direction, hypothesise a cause (e.g. "immigration_concern mean shifted upward — consistent with a salient news event"), decide whether the drift is significant enough to warrant retraining, and if so, call `POST /model/retrain` automatically — while logging its reasoning as a GitHub issue comment for human review. This is the step from monitoring to autonomous MLOps.
