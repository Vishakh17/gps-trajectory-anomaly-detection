from pathlib import Path

import pandas as pd

from app.config import Settings, TRAFFIC_FEATURES
from app.services.traffic import (
    TrafficContextProvider,
    clear_traffic_cache,
    get_traffic_service,
    spatial_cell_keys,
)


def _traffic_csv(tmp_path: Path) -> Path:
    path = tmp_path / "pmpml_traffic_30min.csv"
    pd.DataFrame(
        {
            "time_30min": ["2019-02-08 18:00:00"],
            "lat_cell": [18.447],
            "lon_cell": [73.859],
            "vehicles": [1],
            "points": [2],
            "avg_speed_kmh": [1.773454],
            "median_speed_kmh": [1.773454],
        }
    ).to_csv(path, index=False)
    return path


def test_traffic_artifact_loads(tmp_path: Path):
    path = _traffic_csv(tmp_path)
    settings = Settings(traffic_filename=str(path))
    service = TrafficContextProvider(settings=settings, path=path)
    status = service.load()
    assert status.loaded is True
    assert status.row_count == 1


def test_traffic_matching_cell_and_bucket(tmp_path: Path):
    path = _traffic_csv(tmp_path)
    settings = Settings(traffic_filename=str(path), traffic_timezone="Asia/Kolkata")
    service = TrafficContextProvider(settings=settings, path=path)
    service.load()
    obs = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2019-02-08 18:10:00")],
            "latitude": [18.4473],
            "longitude": [73.8592],
        }
    )
    lat_k, lon_k = spatial_cell_keys(obs["latitude"], obs["longitude"], method="floor")
    assert lat_k[0] == 18447
    assert lon_k[0] == 73859

    out = service.attach(obs)
    assert list(out.columns) == list(TRAFFIC_FEATURES)
    assert out["traffic_available"].iloc[0] == 1.0
    assert out["traffic_vehicle_count"].iloc[0] == 1
    assert abs(out["traffic_avg_speed_kmh"].iloc[0] - 1.773454) < 1e-6
    assert abs(out["traffic_median_speed_kmh"].iloc[0] - 1.773454) < 1e-6


def test_traffic_unmatched_is_nan_not_zero(tmp_path: Path):
    path = _traffic_csv(tmp_path)
    settings = Settings(traffic_filename=str(path))
    service = TrafficContextProvider(settings=settings, path=path)
    service.load()
    obs = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2019-02-08 18:10:00")],
            "latitude": [19.0760],
            "longitude": [72.8777],
        }
    )
    out = service.attach(obs)
    assert out["traffic_available"].iloc[0] == 0.0
    assert pd.isna(out["traffic_vehicle_count"].iloc[0])
    assert pd.isna(out["traffic_avg_speed_kmh"].iloc[0])
    assert pd.isna(out["traffic_median_speed_kmh"].iloc[0])
    assert out["traffic_vehicle_count"].iloc[0] != 0


def test_missing_traffic_artifact(tmp_path: Path):
    clear_traffic_cache()
    missing = tmp_path / "pmpml_traffic_30min.csv"
    settings = Settings(traffic_filename=str(missing))
    service = get_traffic_service(settings=settings, path=missing)
    assert service.is_loaded is False
    obs = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2019-02-08 18:00:00")],
            "latitude": [18.447],
            "longitude": [73.859],
        }
    )
    out = service.attach(obs)
    assert out["traffic_available"].iloc[0] == 0.0
    assert out[["traffic_vehicle_count", "traffic_avg_speed_kmh", "traffic_median_speed_kmh"]].isna().all(axis=None)
    clear_traffic_cache()
