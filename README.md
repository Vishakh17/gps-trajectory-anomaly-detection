# Context-Aware GPS Trajectory Anomaly Detection

Research backend for detecting potential anomalies in vehicle GPS trajectories. The project implements a FastAPI service that accepts GPS CSV files, preprocesses trajectory data, generates movement and context-ready features, trains an Isolation Forest baseline, and returns anomaly scores.

The model output is intended for research exploration only. A prediction of `Potential Anomaly` is not proof of suspicious behavior.

## Features

- FastAPI backend with interactive OpenAPI docs
- CSV upload and validation for GPS trajectory data
- GPS preprocessing for timestamps, coordinates, duplicates, and required fields
- Trajectory feature generation for speed, acceleration, heading, distance, stops, and time gaps
- Route, weather, and traffic integration modules prepared for context-aware experiments
- Isolation Forest training and prediction pipeline
- Pytest test suite for API and service behavior

## Project Structure

```text
backend/
├── app/
│   ├── api/              # FastAPI route handlers
│   ├── models/           # Anomaly detection model wrapper
│   ├── schemas/          # Pydantic request/response schemas
│   ├── services/         # Preprocessing, feature, route, weather, traffic logic
│   ├── config.py         # Paths, feature contracts, and settings
│   └── main.py           # FastAPI app entry point
├── data/
│   └── sample/           # Synthetic sample GPS data
├── tests/                # Pytest suite
└── requirements.txt
```

Runtime uploads, processed CSVs, and trained model files are generated locally and are intentionally excluded from git.

## Tech Stack

- Python
- FastAPI
- Pandas and NumPy
- scikit-learn
- Pydantic
- Pytest

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Vishakh17/gps-trajectory-anomaly-detection.git
cd gps-trajectory-anomaly-detection/backend
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open the API docs at:

```text
http://127.0.0.1:8000/docs
```

## API Workflow

Typical local workflow:

1. Upload a GPS CSV with `POST /api/data/upload`
2. Preprocess uploaded data with `POST /api/data/preprocess`
3. Generate trajectory features with `POST /api/features/generate`
4. Train the Isolation Forest model with `POST /api/model/train`
5. Predict anomaly scores with `POST /api/predict`

Useful endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Check service status |
| `POST` | `/api/data/upload` | Upload a GPS CSV file |
| `POST` | `/api/data/preprocess` | Clean and normalize uploaded GPS data |
| `GET` | `/api/data/summary` | View pipeline artifact status |
| `POST` | `/api/features/generate` | Generate trajectory features |
| `POST` | `/api/model/train` | Train and save the Isolation Forest model |
| `POST` | `/api/predict` | Return anomaly scores and labels |

## Input Data Format

CSV files must include:

| Column | Description |
| --- | --- |
| `vehicle_id` | Vehicle identifier |
| `timestamp` | GPS timestamp |
| `latitude` | Latitude coordinate |
| `longitude` | Longitude coordinate |

Optional columns:

| Column | Description |
| --- | --- |
| `speed` | Vehicle speed, if available |
| `heading` | Heading angle, if available |
| `route_id`, `pattern_id`, `trip_id` | Route or trip context columns |

The repository includes synthetic sample data at `backend/data/sample/sample_gps.csv`.

## Testing

Run tests from the `backend` directory:

```bash
pytest -q
```

## Notes

- This is a research prototype, not a production monitoring system.
- The current backend stores artifacts on the local filesystem.
- Generated runtime data is excluded from the repository.
- Weather and traffic files are treated as prepared local artifacts, not live API integrations.
