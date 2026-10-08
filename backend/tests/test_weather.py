from pathlib import Path

import pandas as pd

from app.config import Settings, WEATHER_FEATURES
from app.services.weather import (
    WeatherContextProvider,
    clear_weather_cache,
    get_weather_service,
)


def _weather_csv_openmeteo_names(tmp_path: Path) -> Path:
    """Fixture using Open-Meteo-style names (no project artifact is present)."""
    path = tmp_path / "pune_weather_hourly.csv"
    pd.DataFrame(
        {
            "time": ["2019-02-08 18:00:00", "2019-02-08 19:00:00"],
            "temperature_2m": [28.4, 27.1],
            "relative_humidity_2m": [41.0, 45.0],
            "wind_speed_10m": [3.2, 4.0],
        }
    ).to_csv(path, index=False)
    return path


def test_weather_artifact_loads_and_maps_columns(tmp_path: Path):
    path = _weather_csv_openmeteo_names(tmp_path)
    settings = Settings(weather_filename=str(path), weather_timezone="Asia/Kolkata")
    service = WeatherContextProvider(settings=settings, path=path)
    status = service.load()
    assert status.loaded is True
    assert status.row_count == 2
    assert status.column_map is not None
    assert status.column_map["temperature"] == "temperature_2m"
    assert status.column_map["humidity"] == "relative_humidity_2m"
    assert status.column_map["wind_speed"] == "wind_speed_10m"
    assert status.column_map["time"] == "time"


def test_weather_valid_timestamp_match(tmp_path: Path):
    path = _weather_csv_openmeteo_names(tmp_path)
    settings = Settings(weather_filename=str(path), weather_max_offset_minutes=30)
    service = WeatherContextProvider(settings=settings, path=path)
    service.load()
    obs = pd.DataFrame({"timestamp": [pd.Timestamp("2019-02-08 18:20:00")]})
    out = service.attach(obs)
    assert list(out.columns) == list(WEATHER_FEATURES)
    assert abs(out["temperature"].iloc[0] - 28.4) < 1e-9
    assert abs(out["humidity"].iloc[0] - 41.0) < 1e-9
    assert abs(out["wind_speed"].iloc[0] - 3.2) < 1e-9


def test_weather_unmatched_timestamp_is_nan(tmp_path: Path):
    path = _weather_csv_openmeteo_names(tmp_path)
    settings = Settings(weather_filename=str(path), weather_max_offset_minutes=30)
    service = WeatherContextProvider(settings=settings, path=path)
    service.load()
    obs = pd.DataFrame({"timestamp": [pd.Timestamp("2019-03-01 00:00:00")]})
    out = service.attach(obs)
    assert out[list(WEATHER_FEATURES)].isna().all(axis=None)
    assert not (out[list(WEATHER_FEATURES)] == 0).any(axis=None)


def test_missing_weather_artifact(tmp_path: Path):
    clear_weather_cache()
    missing = tmp_path / "pune_weather_hourly.csv"
    settings = Settings(weather_filename=str(missing))
    service = get_weather_service(settings=settings, path=missing)
    assert service.is_loaded is False
    obs = pd.DataFrame({"timestamp": [pd.Timestamp("2019-02-08 18:00:00")]})
    out = service.attach(obs)
    assert out[list(WEATHER_FEATURES)].isna().all(axis=None)
    clear_weather_cache()
