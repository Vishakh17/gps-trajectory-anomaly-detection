"""Trajectory feature generation (movement, heading, route-pattern deviation)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.config import (
    BASE_FEATURE_COLUMNS,
    OBSERVATION_PATTERN_FALLBACK_COLUMNS,
    PHASE2_FEATURE_COLUMNS,
    Settings,
    get_settings,
)
from app.services import trajectory as traj
from app.services.context_integration import integrate_context_features
from app.services.route_profile import RouteProfileService, get_route_profile_service
from app.services.traffic import TrafficContextProvider
from app.services.weather import WeatherContextProvider

logger = logging.getLogger(__name__)

PHASE2_OUTPUT_COLUMNS = ("vehicle_id", "timestamp", *PHASE2_FEATURE_COLUMNS)
MODEL_D_OUTPUT_COLUMNS = ("vehicle_id", "timestamp", *BASE_FEATURE_COLUMNS)


def _empty_feature_frame(*, include_context: bool) -> pd.DataFrame:
    cols = MODEL_D_OUTPUT_COLUMNS if include_context else PHASE2_OUTPUT_COLUMNS
    return pd.DataFrame(columns=list(cols))


def resolve_observation_pattern_column(
    df: pd.DataFrame, settings: Settings
) -> str | None:
    """Return the observation pattern-id column if present (configurable, with fallbacks)."""
    preferred = settings.observation_pattern_column.strip().lower()
    columns = {str(c).strip().lower(): c for c in df.columns}
    if preferred in columns:
        return columns[preferred]
    for name in OBSERVATION_PATTERN_FALLBACK_COLUMNS:
        if name in columns:
            return columns[name]
    return None


def compute_pattern_route_deviation(
    observations: pd.DataFrame,
    service: RouteProfileService,
    pattern_column: str | None,
) -> tuple[pd.Series, pd.Series]:
    """
    Distance in meters to the nearest historical route-pattern profile point.

    If the artifact, pattern id, or match is missing: deviation is NaN and
    route_deviation_available is 0. Zero is never used as a stand-in for missing.
    """
    if pattern_column is None:
        return (
            pd.Series(np.nan, index=observations.index, dtype=float),
            pd.Series(0.0, index=observations.index, dtype=float),
        )
    return service.deviations_for_observations(observations, pattern_column)


def generate_trajectory_features(
    processed_gps: pd.DataFrame,
    settings: Settings | None = None,
    route_service: RouteProfileService | None = None,
    weather_service: WeatherContextProvider | None = None,
    traffic_service: TrafficContextProvider | None = None,
    include_context: bool = True,
) -> pd.DataFrame:
    """
    Build per-segment features.

    Phase 2: movement, heading, route-pattern deviation.
    Phase 3 (include_context=True): plus prepared weather and traffic → Model D.
    """
    settings = settings or get_settings()
    if processed_gps.empty:
        return _empty_feature_frame(include_context=include_context)

    required = {"vehicle_id", "timestamp", "latitude", "longitude"}
    missing = required - set(processed_gps.columns)
    if missing:
        raise ValueError(f"Processed GPS missing columns: {missing}")

    if route_service is None:
        route_service = get_route_profile_service(settings)
    if not route_service.is_loaded:
        logger.warning("Route-profile status: %s", route_service.status.message)

    pattern_column = resolve_observation_pattern_column(processed_gps, settings)
    deviation_m, deviation_available = compute_pattern_route_deviation(
        processed_gps, route_service, pattern_column
    )

    frames: list[pd.DataFrame] = []
    for vehicle_id, group in processed_gps.groupby("vehicle_id", sort=False):
        g = group.sort_values("timestamp")
        orig_index = g.index
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue

        lat = g["latitude"]
        lon = g["longitude"]
        ts = g["timestamp"]
        lat_next = lat.shift(-1)
        lon_next = lon.shift(-1)
        ts_next = ts.shift(-1)

        distance_km = traj.haversine_series(lat, lon, lat_next, lon_next)
        time_gap_sec = traj.compute_time_gap_seconds(ts, ts_next)

        derived_speed = traj.derive_speed_kmh(distance_km, time_gap_sec)
        if "speed" in g.columns:
            speed_kmh = pd.to_numeric(g["speed"], errors="coerce").fillna(derived_speed)
        else:
            speed_kmh = derived_speed

        acceleration_kmh_s = traj.derive_acceleration_kmh_s(speed_kmh, time_gap_sec)
        stop_duration_sec = traj.compute_stop_duration_seconds(speed_kmh, time_gap_sec)

        bearing = traj.bearing_series(lat, lon, lat_next, lon_next)
        if "heading" in g.columns:
            explicit = traj.normalize_heading_degrees(g["heading"])
            heading_deg = explicit.where(explicit.notna(), bearing)
        else:
            heading_deg = bearing
        heading_deg = traj.normalize_heading_degrees(heading_deg)

        heading_sin, heading_cos = traj.heading_sin_cos(heading_deg)
        heading_available = heading_deg.notna().astype(float)
        heading_next = heading_deg.shift(-1)
        heading_change_deg = traj.smallest_heading_change_deg(heading_deg, heading_next)
        heading_change_available = (heading_deg.notna() & heading_next.notna()).astype(
            float
        )
        heading_change_deg = heading_change_deg.where(heading_change_available == 1.0)

        aligned_dev = deviation_m.loc[orig_index].reset_index(drop=True)
        aligned_avail = deviation_available.loc[orig_index].reset_index(drop=True)

        segment = pd.DataFrame(
            {
                "vehicle_id": vehicle_id,
                "timestamp": ts,
                "latitude": lat,
                "longitude": lon,
                "speed_kmh": speed_kmh,
                "acceleration_kmh_s": acceleration_kmh_s,
                "distance_km": distance_km,
                "stop_duration_sec": stop_duration_sec,
                "time_gap_sec": time_gap_sec,
                "heading_sin": heading_sin,
                "heading_cos": heading_cos,
                "heading_change_deg": heading_change_deg,
                "heading_available": heading_available,
                "heading_change_available": heading_change_available,
                "pattern_route_deviation_m": aligned_dev,
                "route_deviation_available": aligned_avail,
            }
        )
        segment = segment.iloc[:-1].copy()
        frames.append(segment)

    if not frames:
        logger.warning("No segments generated (need >= 2 points per vehicle).")
        return _empty_feature_frame(include_context=include_context)

    features = pd.concat(frames, ignore_index=True)
    logger.info(
        "Generated %s feature rows for %s vehicles",
        len(features),
        features["vehicle_id"].nunique(),
    )

    if include_context:
        features = integrate_context_features(
            features,
            observations=processed_gps,
            weather_service=weather_service,
            traffic_service=traffic_service,
            settings=settings,
        )
        keep = [c for c in MODEL_D_OUTPUT_COLUMNS if c in features.columns]
        features = features.loc[:, keep]
    else:
        features = features.loc[:, [c for c in PHASE2_OUTPUT_COLUMNS if c in features.columns]]

    return features


def feature_matrix_for_model(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (metadata with vehicle_id/timestamp, numeric matrix for sklearn).

    Rows with any NaN in Model D feature columns are dropped for training/prediction.
    Weather/traffic columns must already be present for a full Model D matrix.
    """
    from app.config import BASE_FEATURE_COLUMNS

    meta = features[["vehicle_id", "timestamp"]].copy()
    missing = [c for c in BASE_FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(
            f"Feature matrix missing Model D columns: {missing}."
        )
    X = features[list(BASE_FEATURE_COLUMNS)].copy()
    valid = X.notna().all(axis=1)
    return meta.loc[valid].reset_index(drop=True), X.loc[valid].reset_index(drop=True)
