"""Prepared-file weather context (hourly Pune / Asia-Kolkata). No live APIs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import Settings, WEATHER_FEATURES, get_settings
from app.services.context_time import to_local_naive

logger = logging.getLogger(__name__)

_CACHE: dict[str, WeatherContextProvider] = {}

TIME_CANDIDATES = ("time", "timestamp", "datetime", "date", "valid_time")
TEMP_CANDIDATES = ("temperature", "temperature_2m", "temp", "temp_c", "t2m")
HUM_CANDIDATES = (
    "humidity",
    "relative_humidity_2m",
    "relativehumidity_2m",
    "relative_humidity",
    "rh",
)
WIND_CANDIDATES = (
    "wind_speed",
    "wind_speed_10m",
    "windspeed_10m",
    "wind_speed_kmh",
    "ws10m",
)


@dataclass
class WeatherStatus:
    loaded: bool
    path: str | None
    row_count: int = 0
    column_map: dict[str, str] | None = None
    message: str = ""


def _pick_column(columns: list[str], configured: str, candidates: tuple[str, ...]) -> str | None:
    names = {c.strip().lower(): c for c in columns}
    if configured and configured.strip().lower() in names:
        return names[configured.strip().lower()]
    for name in candidates:
        if name in names:
            return names[name]
    return None


class WeatherContextProvider:
    """Load a prepared hourly weather CSV once and match GPS times (≤30 min)."""

    def __init__(self, settings: Settings | None = None, path: Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(path) if path is not None else self.settings.weather_path
        self._table: pd.DataFrame | None = None
        self.status = WeatherStatus(
            loaded=False,
            path=str(self.path),
            message="Weather artifact has not been loaded.",
        )

    def load(self) -> WeatherStatus:
        if not self.path.is_file():
            self._table = None
            self.status = WeatherStatus(
                loaded=False,
                path=str(self.path),
                message=(
                    f"Weather artifact not found at {self.path}. "
                    "temperature/humidity/wind_speed will be NaN."
                ),
            )
            logger.warning(self.status.message)
            return self.status

        df = pd.read_csv(self.path)
        df.columns = [str(c).strip() for c in df.columns]
        lowered = [c.lower() for c in df.columns]
        time_col = _pick_column(
            list(df.columns), self.settings.weather_time_column, TIME_CANDIDATES
        )
        temp_col = _pick_column(
            list(df.columns), self.settings.weather_temperature_column, TEMP_CANDIDATES
        )
        hum_col = _pick_column(
            list(df.columns), self.settings.weather_humidity_column, HUM_CANDIDATES
        )
        wind_col = _pick_column(
            list(df.columns), self.settings.weather_wind_column, WIND_CANDIDATES
        )
        missing = [
            label
            for label, col in (
                ("time", time_col),
                ("temperature", temp_col),
                ("humidity", hum_col),
                ("wind_speed", wind_col),
            )
            if col is None
        ]
        if missing:
            self._table = None
            self.status = WeatherStatus(
                loaded=False,
                path=str(self.path),
                message=f"Weather CSV missing required fields: {missing}. Detected columns: {lowered}.",
            )
            logger.warning(self.status.message)
            return self.status

        assert time_col and temp_col and hum_col and wind_col
        table = pd.DataFrame(
            {
                "_weather_ts": to_local_naive(
                    df[time_col], self.settings.weather_timezone
                ),
                "temperature": pd.to_numeric(df[temp_col], errors="coerce"),
                "humidity": pd.to_numeric(df[hum_col], errors="coerce"),
                "wind_speed": pd.to_numeric(df[wind_col], errors="coerce"),
            }
        ).dropna(subset=["_weather_ts"])
        table = table.sort_values("_weather_ts").drop_duplicates(
            subset=["_weather_ts"], keep="first"
        )
        self._table = table.reset_index(drop=True)
        column_map = {
            "time": time_col,
            "temperature": temp_col,
            "humidity": hum_col,
            "wind_speed": wind_col,
        }
        self.status = WeatherStatus(
            loaded=True,
            path=str(self.path),
            row_count=len(self._table),
            column_map=column_map,
            message=f"Loaded {len(self._table)} hourly weather rows from {self.path}.",
        )
        logger.info("%s Mapped columns: %s", self.status.message, column_map)
        return self.status

    @property
    def is_loaded(self) -> bool:
        return self.status.loaded

    def attach(self, observations: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
        """
        Return temperature, humidity, wind_speed aligned to observations.

        Nearest hourly record within weather_max_offset_minutes; otherwise NaN.
        """
        n = len(observations)
        empty = pd.DataFrame(
            {name: np.nan for name in WEATHER_FEATURES},
            index=observations.index,
        )
        if not self.status.loaded or self._table is None or n == 0:
            return empty
        if timestamp_column not in observations.columns:
            return empty

        local_ts = to_local_naive(
            observations[timestamp_column], self.settings.weather_timezone
        )
        work = pd.DataFrame({"_weather_ts": local_ts}, index=observations.index)
        valid = work["_weather_ts"].notna()
        unmatched = empty.copy()
        if not valid.any():
            return unmatched

        left = work.loc[valid].sort_values("_weather_ts")
        right = self._table
        merged = pd.merge_asof(
            left,
            right,
            on="_weather_ts",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=self.settings.weather_max_offset_minutes),
        )
        merged.index = left.index
        out = unmatched.copy()
        out.loc[merged.index, "temperature"] = merged["temperature"].to_numpy()
        out.loc[merged.index, "humidity"] = merged["humidity"].to_numpy()
        out.loc[merged.index, "wind_speed"] = merged["wind_speed"].to_numpy()
        return out

    def get_context_for_trajectory(self, observations: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        timestamp_column = str(kwargs.get("timestamp_column", "timestamp"))
        return self.attach(observations, timestamp_column=timestamp_column)


def get_weather_service(
    settings: Settings | None = None,
    path: Path | None = None,
) -> WeatherContextProvider:
    settings = settings or get_settings()
    resolved = Path(path) if path is not None else settings.weather_path
    key = str(resolved.resolve()) if resolved.exists() else str(resolved)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    service = WeatherContextProvider(settings=settings, path=resolved)
    service.load()
    _CACHE[key] = service
    return service


def clear_weather_cache() -> None:
    _CACHE.clear()
