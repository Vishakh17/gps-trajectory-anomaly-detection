"""Prepared-file PMPML traffic context (30-minute, 0.001° cells). No live APIs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import Settings, TRAFFIC_FEATURES, get_settings
from app.services.context_time import to_local_naive

logger = logging.getLogger(__name__)

_CACHE: dict[str, TrafficContextProvider] = {}


@dataclass
class TrafficStatus:
    loaded: bool
    path: str | None
    row_count: int = 0
    message: str = ""


def spatial_cell_keys(
    lat: pd.Series | np.ndarray,
    lon: pd.Series | np.ndarray,
    decimals: int = 3,
    method: str = "floor",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integer millidegree keys for a 0.001° spatial cell.

    Default method is floor (cell origin), matching a 0.001° bin:
    lat=18.4473 → lat_cell 18.447 → key 18447.
    """
    scale = 10**decimals
    lat_v = np.asarray(lat, dtype=float)
    lon_v = np.asarray(lon, dtype=float)
    if method == "round":
        lat_k = np.round(lat_v * scale)
        lon_k = np.round(lon_v * scale)
    else:
        lat_k = np.floor(lat_v * scale)
        lon_k = np.floor(lon_v * scale)
    lat_k = np.where(np.isfinite(lat_v), lat_k, np.nan)
    lon_k = np.where(np.isfinite(lon_v), lon_k, np.nan)
    return lat_k, lon_k


def stored_cell_keys(
    lat_cell: pd.Series | np.ndarray,
    lon_cell: pd.Series | np.ndarray,
    decimals: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover millidegree keys from already-quantized artifact cell values.

    Prepared cells such as 18.447 are identifiers, not raw GPS. Rounding
    avoids float error from floor(18.447 * 1000).
    """
    scale = 10**decimals
    lat_v = np.asarray(lat_cell, dtype=float)
    lon_v = np.asarray(lon_cell, dtype=float)
    lat_k = np.round(lat_v * scale)
    lon_k = np.round(lon_v * scale)
    lat_k = np.where(np.isfinite(lat_v), lat_k, np.nan)
    lon_k = np.where(np.isfinite(lon_v), lon_k, np.nan)
    return lat_k, lon_k


class TrafficContextProvider:
    """Load pmpml_traffic_30min.csv once and join on 30-min bucket + spatial cell."""

    def __init__(self, settings: Settings | None = None, path: Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(path) if path is not None else self.settings.traffic_path
        self._table: pd.DataFrame | None = None
        self.status = TrafficStatus(
            loaded=False,
            path=str(self.path),
            message="Traffic artifact has not been loaded.",
        )

    def load(self) -> TrafficStatus:
        if not self.path.is_file():
            self._table = None
            self.status = TrafficStatus(
                loaded=False,
                path=str(self.path),
                message=(
                    f"Traffic artifact not found at {self.path}. "
                    "Traffic features will be NaN; traffic_available=0."
                ),
            )
            logger.warning(self.status.message)
            return self.status

        df = pd.read_csv(self.path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        time_col = self.settings.traffic_time_column.strip().lower()
        lat_col = self.settings.traffic_lat_column.strip().lower()
        lon_col = self.settings.traffic_lon_column.strip().lower()
        veh_col = self.settings.traffic_vehicles_column.strip().lower()
        avg_col = self.settings.traffic_avg_speed_column.strip().lower()
        med_col = self.settings.traffic_median_speed_column.strip().lower()
        required = [time_col, lat_col, lon_col, veh_col, avg_col, med_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            self._table = None
            self.status = TrafficStatus(
                loaded=False,
                path=str(self.path),
                message=f"Traffic CSV missing columns: {missing}.",
            )
            logger.warning(self.status.message)
            return self.status

        local_ts = to_local_naive(df[time_col], self.settings.traffic_timezone)
        lat_k, lon_k = stored_cell_keys(
            pd.to_numeric(df[lat_col], errors="coerce"),
            pd.to_numeric(df[lon_col], errors="coerce"),
            decimals=self.settings.traffic_cell_decimals,
        )
        table = pd.DataFrame(
            {
                "time_30min": local_ts.dt.floor("30min"),
                "lat_key": lat_k,
                "lon_key": lon_k,
                "traffic_vehicle_count": pd.to_numeric(df[veh_col], errors="coerce"),
                "traffic_avg_speed_kmh": pd.to_numeric(df[avg_col], errors="coerce"),
                "traffic_median_speed_kmh": pd.to_numeric(df[med_col], errors="coerce"),
            }
        ).dropna(subset=["time_30min", "lat_key", "lon_key"])
        table["lat_key"] = table["lat_key"].astype("int64")
        table["lon_key"] = table["lon_key"].astype("int64")
        table = table.drop_duplicates(
            subset=["time_30min", "lat_key", "lon_key"], keep="first"
        )
        self._table = table.reset_index(drop=True)
        self.status = TrafficStatus(
            loaded=True,
            path=str(self.path),
            row_count=len(self._table),
            message=f"Loaded {len(self._table)} traffic cells from {self.path}.",
        )
        logger.info(self.status.message)
        return self.status

    @property
    def is_loaded(self) -> bool:
        return self.status.loaded

    def attach(
        self,
        observations: pd.DataFrame,
        timestamp_column: str = "timestamp",
        lat_column: str = "latitude",
        lon_column: str = "longitude",
    ) -> pd.DataFrame:
        """Join 30-min + 0.001° cell. Unmatched rows stay NaN with traffic_available=0."""
        n_index = observations.index
        out = pd.DataFrame(
            {
                "traffic_vehicle_count": np.nan,
                "traffic_avg_speed_kmh": np.nan,
                "traffic_median_speed_kmh": np.nan,
                "traffic_available": 0.0,
            },
            index=n_index,
        )
        if not self.status.loaded or self._table is None or observations.empty:
            return out
        needed = {timestamp_column, lat_column, lon_column}
        if not needed.issubset(observations.columns):
            return out

        local_ts = to_local_naive(
            observations[timestamp_column], self.settings.traffic_timezone
        )
        lat_k, lon_k = spatial_cell_keys(
            pd.to_numeric(observations[lat_column], errors="coerce"),
            pd.to_numeric(observations[lon_column], errors="coerce"),
            decimals=self.settings.traffic_cell_decimals,
            method=self.settings.traffic_cell_method,
        )
        left = pd.DataFrame(
            {
                "time_30min": local_ts.dt.floor("30min"),
                "lat_key": lat_k,
                "lon_key": lon_k,
            },
            index=n_index,
        )
        valid = left.notna().all(axis=1)
        if not valid.any():
            return out
        keyed = left.loc[valid].copy()
        keyed["lat_key"] = keyed["lat_key"].astype("int64")
        keyed["lon_key"] = keyed["lon_key"].astype("int64")
        merged = keyed.merge(
            self._table,
            on=["time_30min", "lat_key", "lon_key"],
            how="left",
            sort=False,
        )
        matched = merged["traffic_vehicle_count"].notna()
        hit_index = merged.index[matched]
        for col in (
            "traffic_vehicle_count",
            "traffic_avg_speed_kmh",
            "traffic_median_speed_kmh",
        ):
            out.loc[hit_index, col] = merged.loc[matched, col].to_numpy()
        out.loc[hit_index, "traffic_available"] = 1.0
        return out[list(TRAFFIC_FEATURES)]

    def get_context_for_trajectory(self, observations: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        return self.attach(
            observations,
            timestamp_column=str(kwargs.get("timestamp_column", "timestamp")),
            lat_column=str(kwargs.get("lat_column", "latitude")),
            lon_column=str(kwargs.get("lon_column", "longitude")),
        )


def get_traffic_service(
    settings: Settings | None = None,
    path: Path | None = None,
) -> TrafficContextProvider:
    settings = settings or get_settings()
    resolved = Path(path) if path is not None else settings.traffic_path
    key = str(resolved.resolve()) if resolved.exists() else str(resolved)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    service = TrafficContextProvider(settings=settings, path=resolved)
    service.load()
    _CACHE[key] = service
    return service


def clear_traffic_cache() -> None:
    _CACHE.clear()
