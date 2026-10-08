"""Feature generation, model training, and prediction endpoints."""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.config import BASE_FEATURE_COLUMNS, get_settings
from app.models.anomaly_model import TrajectoryAnomalyDetector
from app.schemas.trajectory_schema import (
    FeatureGenerateResponse,
    ModelTrainResponse,
    PredictResponse,
    PredictionRecord,
)
from app.services.feature_engineering import (
    feature_matrix_for_model,
    generate_trajectory_features,
)
from app.services.pipeline_state import load_state, save_state

logger = logging.getLogger(__name__)
router = APIRouter(tags=["features", "model", "predict"])


@router.post("/features/generate", response_model=FeatureGenerateResponse)
def generate_features() -> FeatureGenerateResponse:
    settings = get_settings()
    if not settings.processed_gps_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Processed GPS not found. Run POST /api/data/preprocess first.",
        )

    processed = pd.read_csv(settings.processed_gps_path)
    if "timestamp" in processed.columns:
        processed["timestamp"] = pd.to_datetime(processed["timestamp"], utc=True)

    features = generate_trajectory_features(processed)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(settings.features_path, index=False)
    logger.info("Saved features to %s", settings.features_path)

    state = load_state(settings)
    state.feature_columns = [c for c in features.columns if c not in ("vehicle_id", "timestamp")]
    if "vehicle_id" in features.columns:
        state.vehicles = sorted(features["vehicle_id"].unique().tolist())
    save_state(state, settings)

    feature_names = [c for c in features.columns if c in BASE_FEATURE_COLUMNS]
    return FeatureGenerateResponse(
        number_of_records=len(features),
        feature_names=feature_names,
        number_of_vehicles=features["vehicle_id"].nunique() if len(features) else 0,
        output_path=str(settings.features_path.relative_to(settings.backend_root)),
    )


@router.post("/model/train", response_model=ModelTrainResponse)
def train_model() -> ModelTrainResponse:
    settings = get_settings()
    if not settings.features_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Feature dataset not found. Run POST /api/features/generate first.",
        )

    features = pd.read_csv(settings.features_path)
    if "timestamp" in features.columns:
        features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)

    _, X = feature_matrix_for_model(features)
    if len(X) < 2:
        raise HTTPException(
            status_code=400,
            detail="Insufficient complete feature rows for training (need at least 2).",
        )

    detector = TrajectoryAnomalyDetector(settings)
    try:
        detector.train(X)
        model_path = detector.save()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Model training failed")
        raise HTTPException(status_code=500, detail="Model training failed.") from exc

    return ModelTrainResponse(
        number_of_training_records=len(X),
        features_used=detector.feature_names,
        model_status="trained_and_saved",
        model_path=str(model_path.relative_to(settings.backend_root)),
    )


@router.post("/predict", response_model=PredictResponse)
def predict() -> PredictResponse:
    settings = get_settings()
    if not settings.features_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Feature dataset not found. Run POST /api/features/generate first.",
        )
    if not settings.model_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Trained model not found. Run POST /api/model/train first.",
        )

    features = pd.read_csv(settings.features_path)
    if "timestamp" in features.columns:
        features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)

    meta, X = feature_matrix_for_model(features)
    if len(X) == 0:
        raise HTTPException(status_code=400, detail="No complete feature rows for prediction.")

    detector = TrajectoryAnomalyDetector(settings)
    try:
        detector.load()
        scores = detector.predict(X)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    records: list[PredictionRecord] = []
    for i in range(len(meta)):
        records.append(
            PredictionRecord(
                vehicle_id=str(meta.iloc[i]["vehicle_id"]),
                timestamp=meta.iloc[i]["timestamp"].to_pydatetime(),
                anomaly_score=float(scores.iloc[i]["anomaly_score"]),
                prediction=str(scores.iloc[i]["prediction"]),
            )
        )

    return PredictResponse(
        predictions=records,
        total_records=len(records),
        model_loaded=True,
    )
