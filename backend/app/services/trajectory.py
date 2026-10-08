"""Trajectory geometry and kinematics (Haversine, speed, heading, stops)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
# Speed below this (km/h) is treated as stopped when deriving stop duration
STOP_SPEED_THRESHOLD_KMH = 1.0


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Great-circle distance between two WGS84 points in kilometers.

    Uses the Haversine formula.
    """
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_RADIUS_KM * c


def haversine_series(
    lat: pd.Series,
    lon: pd.Series,
    lat_next: pd.Series,
    lon_next: pd.Series,
) -> pd.Series:
    """Vectorized Haversine distance (km) between consecutive point pairs."""
    lat1 = np.radians(lat.astype(float))
    lon1 = np.radians(lon.astype(float))
    lat2 = np.radians(lat_next.astype(float))
    lon2 = np.radians(lon_next.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return pd.Series(EARTH_RADIUS_KM * c, index=lat.index)


def bearing_degrees(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Initial bearing from point 1 to point 2 in degrees [0, 360)."""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(
        lat2_r
    ) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360.0) % 360.0


def bearing_series(
    lat: pd.Series,
    lon: pd.Series,
    lat_next: pd.Series,
    lon_next: pd.Series,
) -> pd.Series:
    """Vectorized initial bearing in degrees [0, 360). NaN where a point is missing."""
    valid = lat.notna() & lon.notna() & lat_next.notna() & lon_next.notna()
    lat1 = np.radians(lat.astype(float))
    lon1 = np.radians(lon.astype(float))
    lat2 = np.radians(lat_next.astype(float))
    lon2 = np.radians(lon_next.astype(float))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    out = pd.Series(bearing, index=lat.index)
    return out.where(valid)


def compute_time_gap_seconds(ts: pd.Series, ts_next: pd.Series) -> pd.Series:
    """Seconds between consecutive timestamps."""
    delta = ts_next - ts
    return delta.dt.total_seconds()


def derive_speed_kmh(distance_km: pd.Series, time_gap_s: pd.Series) -> pd.Series:
    """Speed in km/h from distance and elapsed time; NaN where time_gap <= 0."""
    hours = time_gap_s / 3600.0
    speed = distance_km / hours.replace(0, np.nan)
    return speed


def derive_acceleration_kmh_s(
    speed_kmh: pd.Series, time_gap_s: pd.Series
) -> pd.Series:
    """Approximate acceleration (km/h per second) from speed change over time_gap."""
    speed_next = speed_kmh.shift(-1)
    dv = speed_next - speed_kmh
    acc = dv / time_gap_s.replace(0, np.nan)
    return acc


def compute_stop_duration_seconds(
    speed_kmh: pd.Series,
    time_gap_s: pd.Series,
    threshold_kmh: float = STOP_SPEED_THRESHOLD_KMH,
) -> pd.Series:
    """
    Duration attributed to a stop at each segment end when speed is at/below threshold.

    Uses the time gap to the next point when the current speed indicates stopped.
    """
    stopped = speed_kmh <= threshold_kmh
    duration = time_gap_s.where(stopped, 0.0)
    return duration.fillna(0.0)


def min_haversine_m(
    lat: float,
    lon: float,
    profile_lats: np.ndarray,
    profile_lons: np.ndarray,
) -> float:
    """Minimum Haversine distance in meters from one point to a set of profile points."""
    if profile_lats.size == 0:
        return float("nan")
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(profile_lats.astype(float))
    lon2 = np.radians(profile_lons.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return float(np.min(EARTH_RADIUS_KM * c) * 1000.0)


def normalize_heading_degrees(heading: pd.Series) -> pd.Series:
    """Normalize heading to [0, 360). Invalid values remain NaN."""
    h = pd.to_numeric(heading, errors="coerce")
    return (h % 360.0 + 360.0) % 360.0


def heading_sin_cos(heading_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Sine and cosine of heading in radians. NaN heading yields NaN components."""
    radians = np.radians(heading_deg.astype(float))
    return pd.Series(np.sin(radians), index=heading_deg.index), pd.Series(
        np.cos(radians), index=heading_deg.index
    )


def smallest_heading_change_deg(heading_from: pd.Series, heading_to: pd.Series) -> pd.Series:
    """
    Signed smallest angular difference in degrees, range (-180, 180].

    Wrap-around is handled so 359° → 1° is +2°, not 358°.
    Unavailability of either heading yields NaN (callers must not fill this).
    """
    delta = (heading_to.astype(float) - heading_from.astype(float) + 180.0) % 360.0 - 180.0
    return pd.Series(delta, index=heading_from.index)
