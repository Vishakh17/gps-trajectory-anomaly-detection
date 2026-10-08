from pathlib import Path

import pandas as pd

from app.config import Settings
from app.services.route_profile import (
    RouteProfileService,
    clear_route_profile_cache,
    get_route_profile_service,
)
from app.services.trajectory import haversine_km


def _generic_profile_settings(path: Path) -> Settings:
    """Keep legacy pattern_id/lat/lon tests working after PMPML defaults."""
    return Settings(
        route_profile_filename=str(path),
        route_profile_id_column="pattern_id",
        route_profile_lat_column="latitude",
        route_profile_lon_column="longitude",
        observation_pattern_column="pattern_id",
    )


def _sample_profile_csv(tmp_path: Path) -> Path:
    path = tmp_path / "route_profiles.csv"
    pd.DataFrame(
        {
            "pattern_id": ["P1", "P1", "P2"],
            "latitude": [18.5204, 18.5210, 18.5300],
            "longitude": [73.8567, 73.8575, 73.8600],
        }
    ).to_csv(path, index=False)
    return path


def _pmpml_profile_csv(tmp_path: Path) -> Path:
    path = tmp_path / "pmpml_route_profiles.csv"
    pd.DataFrame(
        {
            "route_id": [100, 100, 217],
            "lat_grid": [18.5222, 18.5223, 18.4540],
            "lon_grid": [73.8517, 73.8521, 73.8521],
            "observations": [4, 3, 2],
            "vehicles": [1, 2, 1],
            "trips": [1, 2, 1],
        }
    ).to_csv(path, index=False)
    return path


def test_route_profile_loading(tmp_path: Path):
    path = _sample_profile_csv(tmp_path)
    settings = _generic_profile_settings(path)
    service = RouteProfileService(settings=settings, path=path)
    status = service.load()
    assert status.loaded is True
    assert status.pattern_count == 2
    assert status.point_count == 3


def test_route_deviation_distance_meters(tmp_path: Path):
    path = _sample_profile_csv(tmp_path)
    settings = _generic_profile_settings(path)
    service = RouteProfileService(settings=settings, path=path)
    service.load()
    dist = service.deviation_m("P1", 18.5204, 73.8567)
    assert dist == 0.0
    offset = service.deviation_m("P1", 18.5204, 73.8577)
    expected_m = haversine_km(18.5204, 73.8577, 18.5204, 73.8567) * 1000.0
    assert offset > 0
    assert abs(offset - min(expected_m, haversine_km(18.5204, 73.8577, 18.5210, 73.8575) * 1000.0)) < 1.0


def test_route_deviation_available_flag(tmp_path: Path):
    path = _sample_profile_csv(tmp_path)
    settings = _generic_profile_settings(path)
    service = RouteProfileService(settings=settings, path=path)
    service.load()
    obs = pd.DataFrame(
        {
            "pattern_id": ["P1", "UNKNOWN", None],
            "latitude": [18.5204, 18.5204, 18.5204],
            "longitude": [73.8567, 73.8567, 73.8567],
        }
    )
    deviation, available = service.deviations_for_observations(obs, "pattern_id")
    assert available.iloc[0] == 1.0
    assert deviation.iloc[0] == 0.0
    assert available.iloc[1] == 0.0
    assert pd.isna(deviation.iloc[1])
    assert available.iloc[2] == 0.0
    assert pd.isna(deviation.iloc[2])


def test_missing_route_profile_artifact(tmp_path: Path):
    clear_route_profile_cache()
    missing = tmp_path / "does_not_exist.csv"
    settings = _generic_profile_settings(missing)
    service = get_route_profile_service(settings=settings, path=missing)
    assert service.is_loaded is False
    assert "not found" in service.status.message.lower()
    obs = pd.DataFrame(
        {"pattern_id": ["P1"], "latitude": [18.52], "longitude": [73.85]}
    )
    deviation, available = service.deviations_for_observations(obs, "pattern_id")
    assert available.iloc[0] == 0.0
    assert pd.isna(deviation.iloc[0])
    clear_route_profile_cache()


def test_pmpml_route_profile_loads_route_id_lat_lon_grid(tmp_path: Path):
    path = _pmpml_profile_csv(tmp_path)
    settings = Settings(route_profile_filename=str(path))
    service = RouteProfileService(settings=settings, path=path)
    status = service.load()
    assert status.loaded is True
    assert settings.route_profile_id_column == "route_id"
    assert settings.route_profile_lat_column == "lat_grid"
    assert settings.route_profile_lon_column == "lon_grid"
    assert status.pattern_count == 2
    assert status.point_count == 3


def test_pmpml_route_id_match_distance_meters(tmp_path: Path):
    path = _pmpml_profile_csv(tmp_path)
    settings = Settings(route_profile_filename=str(path))
    service = RouteProfileService(settings=settings, path=path)
    service.load()
    dist = service.deviation_m("100", 18.5222, 73.8517)
    assert dist == 0.0
    offset = service.deviation_m(100, 18.5222, 73.8520)
    assert offset > 0
    assert offset < 100.0
    obs = pd.DataFrame(
        {
            "route_id": [100, 999, None],
            "latitude": [18.5222, 18.5222, 18.5222],
            "longitude": [73.8517, 73.8517, 73.8517],
        }
    )
    deviation, available = service.deviations_for_observations(obs, "route_id")
    assert available.iloc[0] == 1.0
    assert deviation.iloc[0] == 0.0
    assert available.iloc[1] == 0.0
    assert pd.isna(deviation.iloc[1])
    assert available.iloc[2] == 0.0
    assert pd.isna(deviation.iloc[2])
