# Context-Aware GPS Trajectory Anomaly Detection — Backend (v1)

Research/academic backend for analyzing vehicle GPS trajectories and flagging **potential** anomalies using an unsupervised baseline. Weather and traffic context are planned for a later phase; this version implements the GPS pipeline only.

## Project purpose

Support B.Tech Data Science research on whether route deviations are unusual relative to trajectory dynamics, with future integration of weather and traffic to reduce false positives. **Isolation Forest scores are not proof of suspicious behavior** — labels are `Normal` or `Potential Anomaly` only.

## Backend architecture

```
backend/
├── app/
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Paths and settings
│   ├── api/                    # HTTP routes
│   ├── services/               # Preprocessing, trajectory, features
│   │   ├── weather.py          # Placeholder
│   │   ├── traffic.py          # Placeholder
│   │   └── context_integration.py
│   ├── models/anomaly_model.py # Isolation Forest + Joblib
│   └── schemas/                # Pydantic response models
├── data/raw|processed|sample/
├── trained_models/
└── tests/
```

Pipeline flow: **upload CSV → preprocess → generate features → train model → predict**.

## Installation

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` and adjust log level or Isolation Forest parameters.

## Run FastAPI

From the `backend` directory:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open interactive docs: http://127.0.0.1:8000/docs

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Service status |
| POST | `/api/data/upload` | Upload GPS CSV |
| POST | `/api/data/preprocess` | Clean and save processed GPS |
| GET | `/api/data/summary` | Pipeline artifact summary |
| POST | `/api/features/generate` | Build trajectory features |
| POST | `/api/model/train` | Train Isolation Forest |
| POST | `/api/predict` | Scores on feature dataset |

## Input CSV schema

**Required columns:** `vehicle_id`, `timestamp`, `latitude`, `longitude`  

**Optional:** `speed`, `heading` (derived from GPS when missing)

Preprocessing reports counts of removed invalid, duplicate, or missing-required rows.

## Feature generation (v1)

Per-vehicle segments between consecutive points:

- `speed`, `acceleration`, `heading`, `distance`, `stop_duration`, `time_gap`

**Route deviation** is not computed in v1 (depends on a future reference-route dataset).

## Isolation Forest baseline

Unsupervised model on the numerical features above, with median imputation in a sklearn `Pipeline`. Model saved under `trained_models/isolation_forest.joblib`.

## Sample data

`data/sample/sample_gps.csv` contains **synthetic SAMPLE DATA** for local testing only — not real-world GPS.

Example workflow with sample file (use `/docs` or curl):

1. POST `/api/data/upload` with `sample_gps.csv`
2. POST `/api/data/preprocess`
3. POST `/api/features/generate`
4. POST `/api/model/train`
5. POST `/api/predict`

## Tests

```bash
cd backend
pytest -q
```

## Current limitations

- No frontend, database, live weather, or live traffic APIs
- CSV/file storage only
- No route-deviation or context-aware features yet
- No ground-truth anomaly labels; Isolation Forest is an exploratory baseline
- Final Indian GPS dataset and external sources are not fixed

## Future weather/traffic integration

Planned modules: `weather.py`, `traffic.py`, `context_integration.py` will join context features onto trajectory features before detection, without rewriting the v1 GPS pipeline.
