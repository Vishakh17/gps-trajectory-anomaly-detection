"""Historical route-pattern profiles for pattern_route_deviation_m.

Deviation is the Haversine distance in meters from an observation to the
nearest prepared profile point of the same pattern identifier.

This is not road-network routing (OSM/Google Maps) and is not regenerated
from raw GTFS on each prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.services.trajectory import min_haversine_m

logger = logging.getLogger(__name__)

_CACHE: dict[str, RouteProfileService] = {}


def normalize_route_id(value: object) -> str | None:
    """Stable string key for route_id (e.g. 100 and 100.0 both become '100')."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    try:
        as_float = float(text)
        if as_float.is_integer():
            return str(int(as_float))
    except ValueError:
        pass
    return text


@dataclass
class RouteProfileStatus:
    loaded: bool
    path: str | None
    pattern_count: int = 0
    point_count: int = 0
    message: str = ""


class RouteProfileService:
    """Load a prepared route-pattern CSV once and compute nearest-point distances."""

    def __init__(self, settings: Settings | None = None, path: Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(path) if path is not None else self.settings.route_profile_path
        self._points: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.status = RouteProfileStatus(
            loaded=False,
            path=str(self.path),
            message="Route profile has not been loaded.",
        )

    def load(self) -> RouteProfileStatus:
        """Load profile points from CSV. Missing/invalid files do not invent data."""
        if not self.path.is_file():
            self._points = {}
            self.status = RouteProfileStatus(
                loaded=False,
                path=str(self.path),
                message=(
                    f"Route-profile artifact not found at {self.path}. "
                    "pattern_route_deviation_m will be NaN; route_deviation_available=0."
                ),
            )
            logger.warning(self.status.message)
            return self.status

        df = pd.read_csv(self.path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        id_col = self.settings.route_profile_id_column.strip().lower()
        lat_col = self.settings.route_profile_lat_column.strip().lower()
        lon_col = self.settings.route_profile_lon_column.strip().lower()
        missing = [c for c in (id_col, lat_col, lon_col) if c not in df.columns]
        if missing:
            self._points = {}
            self.status = RouteProfileStatus(
                loaded=False,
                path=str(self.path),
                message=f"Route-profile CSV missing columns: {missing}.",
            )
            logger.warning(self.status.message)
            return self.status

        work = df[[id_col, lat_col, lon_col]].copy()
        work[id_col] = work[id_col].map(normalize_route_id)
        work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
        work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
        work = work.dropna(subset=[id_col, lat_col, lon_col])
        work = work[work[id_col] != ""]

        points: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for pattern_id, group in work.groupby(id_col, sort=False):
            points[str(pattern_id)] = (
                group[lat_col].to_numpy(dtype=float),
                group[lon_col].to_numpy(dtype=float),
            )

        self._points = points
        self.status = RouteProfileStatus(
            loaded=True,
            path=str(self.path),
            pattern_count=len(points),
            point_count=int(len(work)),
            message=f"Loaded {len(work)} profile points across {len(points)} patterns.",
        )
        logger.info(self.status.message)
        return self.status

    @property
    def is_loaded(self) -> bool:
        return self.status.loaded

    def deviation_m(self, pattern_id: str | None, lat: float, lon: float) -> float:
        """Nearest profile-point distance in meters, or NaN if unavailable."""
        if not self.status.loaded or pattern_id is None or str(pattern_id).strip() == "":
            return float("nan")
        key = normalize_route_id(pattern_id)
        if key is None:
            return float("nan")
        coords = self._points.get(key)
        if coords is None:
            return float("nan")
        profile_lats, profile_lons = coords
        return min_haversine_m(lat, lon, profile_lats, profile_lons)

    def deviations_for_observations(
        self,
        observations: pd.DataFrame,
        pattern_column: str,
        lat_column: str = "latitude",
        lon_column: str = "longitude",
    ) -> tuple[pd.Series, pd.Series]:
        """
        Return (pattern_route_deviation_m, route_deviation_available).

        Availability is 1 only when a loaded profile matches the observation id
        and a finite distance was computed. Missing profiles are NaN, not 0.
        """
        deviation = pd.Series(np.nan, index=observations.index, dtype=float)
        available = pd.Series(0.0, index=observations.index, dtype=float)
        if not self.status.loaded or pattern_column not in observations.columns:
            return deviation, available

        ids = observations[pattern_column]
        lats = pd.to_numeric(observations[lat_column], errors="coerce")
        lons = pd.to_numeric(observations[lon_column], errors="coerce")
        values: list[float] = []
        flags: list[float] = []
        for pattern_id, lat, lon in zip(ids, lats, lons, strict=True):
            if pd.isna(lat) or pd.isna(lon) or pd.isna(pattern_id):
                values.append(float("nan"))
                flags.append(0.0)
                continue
            if normalize_route_id(pattern_id) is None:
                values.append(float("nan"))
                flags.append(0.0)
                continue
            dist = self.deviation_m(pattern_id, float(lat), float(lon))
            if np.isfinite(dist):
                values.append(dist)
                flags.append(1.0)
            else:
                values.append(float("nan"))
                flags.append(0.0)
        return pd.Series(values, index=observations.index), pd.Series(
            flags, index=observations.index
        )


def get_route_profile_service(
    settings: Settings | None = None,
    path: Path | None = None,
) -> RouteProfileService:
    """Return a cached service for the resolved artifact path (loaded once)."""
    settings = settings or get_settings()
    resolved = Path(path) if path is not None else settings.route_profile_path
    key = str(resolved.resolve()) if resolved.exists() else str(resolved)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    service = RouteProfileService(settings=settings, path=resolved)
    service.load()
    _CACHE[key] = service
    return service


def clear_route_profile_cache() -> None:
    """Clear the in-process profile cache (used by tests)."""
    _CACHE.clear()
