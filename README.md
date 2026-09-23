# Project Sentinel (AXIOM)

Independent progress verification for public infrastructure projects.

Government progress reports (MoSPI PAIMANA flash reports) are self-reported by implementing agencies. Sentinel checks them against evidence the agency doesn't control, which today means site photographs and satellite imagery. It combines the gap between reported and observed progress with budget, schedule, sensor and rainfall signals to produce an explainable 0–100 risk score for each project.

This repository contains the FastAPI backend and ML pipeline.

## What's distinctive

**1. Self-trained YOLOv8 pipeline for site-photo verification** (`app/ml/photo_verifier.py`)
- **Classifier (`yolov8_construction_cls.pt`):** fine-tuned from `yolov8n-cls` to label a site photo as `completed` / `incomplete`. From that label it derives a photo-verified progress estimate and a `photo_gap`, the difference from reported progress.
- **Detector (`yolov8_construction_detect.pt`):** trained on the Roboflow *Construction Site Safety* dataset (CC BY 4.0). It detects machinery, workers and structural elements as supporting evidence.
- **Endpoint:** `POST /api/photos/classify` accepts a photo plus a project ID and returns the label, confidence, verified-progress estimate, and gap against reported progress, with a detection summary in `heuristic_notes`.

**2. Copernicus / Sentinel Hub satellite integration** (`app/services/satellite_imagery_service.py`)
- **Data source:** the official `sentinelhub` SDK against the Copernicus Data Space Ecosystem (CDSE).
- **Before/after search:** the Sentinel-2 L2A catalog is searched for the lowest-cloud scene near the project start date and again in the last 14 days. If nothing is under 30% cloud cover, the search window widens to ±30 days.
- **Output:** true-colour before/after images for each project's area of interest. These are cached to `data/satellite_cache/` and indexed in the `satellite_image_cache` table.
- **Endpoint:** `GET /api/satellite/{project_id}/change-detection` serves the cached imagery and never calls CDSE during a request.

**Also included:**
- **PAIMANA PDF ingestion** (`pdfplumber`): `POST /api/pipeline/ingest-paimana`
- **Transparent weighted risk formula** with a per-factor breakdown: `GET /api/projects/{id}/risk-breakdown`
- **XGBoost delay and cost-overrun model with SHAP explanations**
- **K-means risk clustering**
- **Open-Meteo rainfall exposure**

## Current status and known gaps

Please read this before a demo:

- **Model weights are not in git.** `data/models/` is gitignored. Copy `yolov8_construction_cls.pt` and `yolov8_construction_detect.pt` into `project-sentinel/backend/data/models/`. Without them the classifier falls back to the generic `yolov8n-cls.pt`, which does not produce meaningful completed/incomplete labels.
- **The photo classifier training set is small.** It has 30 training and 10 validation images, trained for 5 epochs.
- **Satellite imagery fetching is not wired into the app yet.** `fetch_before_after_images()` exists, but its caller (`scripts/refresh_satellite_cache.py`, the job behind `ENABLE_SATELLITE_SCHEDULER`) is not in this repo. With an empty cache the endpoint serves static fallback tiles.
- **The satellite *progress %* used in the risk score is simulated.** `app/services/satellite_service.py` produces a deterministic heuristic, not real NDBI change detection. It is labelled as simulated in API responses.
- **The predictive risk model is trained on 600 synthetic samples** calibrated to plausible ranges, not on historical outcomes.
- **Sensor telemetry is simulated.** `data/sensors.csv` is not included, so asset health returns a default healthy reading.

## Running it

Requires Python 3.11 (tested on Windows).

```bash
cd project-sentinel/backend
python -m venv .venv
.venv/Scripts/activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional: add CDSE_CLIENT_ID / CDSE_CLIENT_SECRET
uvicorn app.main:app --reload --port 8000
```

On first start the app creates `data/sentinel_v2.db` (SQLite) and seeds 10 monitored projects. Interactive API docs are at http://127.0.0.1:8000/docs.

The app reads the CDSE credentials with `os.getenv`, so either export them in your shell or load `.env` with your process manager. Credentials are free from https://dataspace.copernicus.eu/.

## Main endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/projects`, `/api/projects/{id}` | Project register |
| GET | `/api/projects/{id}/risk-breakdown` | Weighted risk score, discrepancy details, XGBoost + SHAP |
| GET | `/api/model-validation` | Clustering / model validation summary |
| POST | `/api/photos/classify` | YOLOv8 site-photo verification |
| GET | `/api/photos/{project_id}` | Photo history for a project |
| GET | `/api/satellite/{id}/change-detection` | Sentinel-2 before/after imagery + change metadata |
| GET | `/api/weather/{project_id}` | Rainfall exposure (Open-Meteo) |
| GET | `/api/assets/{asset_id}/health`, POST `/api/sensors/ingest` | Asset telemetry |
| POST | `/api/pipeline/ingest-paimana` | Parse a PAIMANA flash report PDF |

## Data credits

- MoSPI PAIMANA flash reports (Government of India)
- Copernicus Sentinel-2 data, provided through the Copernicus Data Space Ecosystem
- Roboflow Universe construction datasets (CC BY 4.0)
- Open-Meteo weather API
