# Research Centre Quality Classification

## Overview

Unsupervised machine learning pipeline that classifies UK research centres
into three quality tiers — **Premium**, **Standard**, and **Basic** — based
on internal infrastructure and external healthcare access features.

| Item | Detail |
|---|---|
| Dataset | 50 synthetic UK research centres across 5 cities |
| Algorithm | K-Means clustering (k=3, silhouette score = 0.5519) |
| Features | `internalFacilitiesCount`, `hospitals_10km`, `pharmacies_10km`, `facilityDiversity_10km`, `facilityDensity_10km` |
| Deployment | FastAPI endpoint — `POST /predict` |
| Bonus | Dockerfile · docker-compose · Geospatial map · Plotly dashboard · Claude AI assistant |

---

## Notebooks

| File | Description |
|---|---|
| `EDA_and_Model.ipynb` | **Primary submission notebook** — required filename per assignment spec |
| `Debabrata_Mishra_Research_Center_Quality_Classification.ipynb` | Same notebook — full named copy |

> Both files are identical. Open either one —
> `EDA_and_Model.ipynb` is the assignment-required filename.

---

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Install all dependencies
pip install -r requirements.txt
```

---

## Run

**Step 1 — Execute the notebook** to train and save `model_bundle.pkl`:

Open `EDA_and_Model.ipynb` in Jupyter and run all cells:

```bash
jupyter notebook
```

**Step 2 — Start the API** (from the project folder with venv active):

```bash
python -m uvicorn app:app --reload
```

> Use `python -m uvicorn` rather than `uvicorn` directly to ensure
> the correct virtual environment is used.

---

## API Usage

### Health check

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "ok",
  "model": "K-Means (k=3)",
  "features": ["internalFacilitiesCount", "hospitals_10km", "pharmacies_10km",
               "facilityDiversity_10km", "facilityDensity_10km"],
  "tiers": ["Premium", "Standard", "Basic"]
}
```

### Predict quality tier

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "internalFacilitiesCount": 9,
           "hospitals_10km": 3,
           "pharmacies_10km": 2,
           "facilityDiversity_10km": 0.82,
           "facilityDensity_10km": 0.45
         }'
```

**Response:**
```json
{
  "predictedCategory": "Premium"
}
```

### Swagger UI

Interactive API documentation available at:
```
http://127.0.0.1:8000/docs
```

---

## Docker Deployment (Bonus)

```bash
# Build and run with docker-compose
docker-compose up --build

# API will be available at http://localhost:8000
```

---

## File Structure

```
research-center-assignment/
│
├── data/
│   └── research_centers.csv
│
├── EDA_and_Model.ipynb                  ← primary submission notebook
├── Debabrata_Mishra_Research_Center_Quality_Classification.ipynb
├── app.py                               ← FastAPI endpoint
├── model_bundle.pkl                     ← trained model (scaler + KMeans + tier_map)
├── research_centers.csv                 ← dataset (root copy)
├── research_centers_clustered.csv       ← enriched output with cluster + qualityTier
├── requirements.txt
├── README.md
├── Dockerfile
├── docker-compose.yaml
├── .env.draft
├── .gitignore
└── .dockerignore
```

---

## Requirements

Key dependencies (see `requirements.txt` for full list):

```
fastapi            uvicorn[standard]    pandas
numpy              scikit-learn         matplotlib
seaborn            plotly               joblib
anthropic          nbformat>=4.2.0      requests
```

---

## Author

**Debabrata Mishra**  
GitHub: [dmishra27](https://github.com/dmishra27)  
Repository: [Debabrata_Research_Center_Classification_Assignment](https://github.com/dmishra27/Debabrata_Research_Center_Classification_Assignment)
