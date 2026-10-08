"""Join Phase-2 trajectory features with prepared weather and traffic context."""

from __future__ import annotations

import logging

import pandas as pd

from app.config import (
    BASE_FEATURE_COLUMNS,
    PHASE2_FEATURE_COLUMNS,
    Settings,
    TRAFFIC_FEATURES,
    WEATHER_FEATURES,
    get_settings,
)
from app.services.traffic import TrafficContextProvider, get_traffic_service
from app.services.weather import WeatherContextProvider, get_weather_service

logger = logging.getLogger(__name__)

MODEL_D_ID_COLUMNS = ("vehicle_id", "timestamp")


def _lat_lon_frame(
    trajectory_features: pd.DataFrame,
    observations: pd.DataFrame | None,
) -> pd.DataFrame:
    """Ensure latitude/longitude exist for traffic cell lookup."""
    frame = trajectory_features.copy()
    if {"latitude", "longitude"}.issubset(frame.columns):
        return frame
    if observations is None or not {"vehicle_id", "timestamp", "latitude", "longitude"}.issubset(
        observations.columns
    ):
        frame["latitude"] = pd.NA
        frame["longitude"] = pd.NA
        return frame
    obs = observations[["vehicle_id", "timestamp", "latitude", "longitude"]].copy()
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], utc=True, errors="coerce")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    obs = obs.drop_duplicates(subset=["vehicle_id", "timestamp"], keep="first")
    return frame.merge(obs, on=["vehicle_id", "timestamp"], how="left")


def integrate_context_features(
    trajectory_features: pd.DataFrame,
    weather_features: pd.DataFrame | None = None,
    traffic_features: pd.DataFrame | None = None,
    observations: pd.DataFrame | None = None,
    weather_service: WeatherContextProvider | None = None,
    traffic_service: TrafficContextProvider | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """
    Attach weather and traffic columns onto Phase-2 features.

    Does not run anomaly detection. Missing context stays NaN (traffic_available=0).
    """
    settings = settings or get_settings()
    if trajectory_features.empty:
        cols = list(MODEL_D_ID_COLUMNS) + list(BASE_FEATURE_COLUMNS)
        existing = [c for c in MODEL_D_ID_COLUMNS if c in trajectory_features.columns]
        out = pd.DataFrame(columns=existing + [c for c in BASE_FEATURE_COLUMNS])
        return out.reindex(columns=cols)

    geo = _lat_lon_frame(trajectory_features, observations)

    if weather_features is None:
        if weather_service is None:
            weather_service = get_weather_service(settings)
        weather_features = weather_service.attach(geo)
    weather_features = weather_features.reindex(geo.index)

    if traffic_features is None:
        if traffic_service is None:
            traffic_service = get_traffic_service(settings)
        traffic_features = traffic_service.attach(geo)
    traffic_features = traffic_features.reindex(geo.index)

    combined = geo.copy()
    for col in WEATHER_FEATURES:
        combined[col] = weather_features[col] if col in weather_features.columns else pd.NA
    for col in TRAFFIC_FEATURES:
        combined[col] = traffic_features[col] if col in traffic_features.columns else pd.NA

    if "traffic_available" in combined.columns:
        combined["traffic_available"] = combined["traffic_available"].fillna(0.0)

    ordered = [c for c in MODEL_D_ID_COLUMNS if c in combined.columns]
    ordered.extend([c for c in BASE_FEATURE_COLUMNS if c in combined.columns])
    extras = [c for c in combined.columns if c not in ordered and c not in ("latitude", "longitude")]
    result = combined.loc[:, ordered + extras]
    logger.info(
        "Context integration produced %s rows with weather/traffic columns",
        len(result),
    )
    return result
