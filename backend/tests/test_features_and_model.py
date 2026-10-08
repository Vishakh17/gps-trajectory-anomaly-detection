from pathlib import Path

import pandas as pd
import pytest

from app.config import (
    BASE_FEATURE_COLUMNS,
    FULL_CONTEXT_FEATURE_COLUMNS,
    HEADING_FEATURES,
    MOVEMENT_FEATURES,
    PHASE2_FEATURE_COLUMNS,
    ROUTE_FEATURES,
    TRAFFIC_FEATURES,
    WEATHER_FEATURES,
    get_settings,
)
from app.models.anomaly_model import (
    PREDICTION_NORMAL,
    PREDICTION_POTENTIAL_ANOMALY,
    TrajectoryAnomalyDetector,
)
from app.services.feature_engineering import generate_trajectory_features


SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_gps.csv"


def test_feature_generation_from_sample():
    df = pd.read_csv(SAMPLE_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    from app.services.preprocessing import preprocess_gps_dataframe
    from app.services.route_profile import RouteProfileService, clear_route_profile_cache

    processed, _ = preprocess_gps_dataframe(df)
    missing_profile = Path("_no_such_route_profiles.csv")
    service = RouteProfileService(path=missing_profile)
    service.load()
    features = generate_trajectory_features(
        processed, route_service=service, include_context=True
    )
    assert len(features) > 0
    for col in PHASE2_FEATURE_COLUMNS:
        assert col in features.columns
    for col in ("temperature", "humidity", "wind_speed"):
        assert col in features.columns
        assert features[col].isna().all()
    assert features["traffic_available"].eq(0.0).all()
    assert features["traffic_vehicle_count"].isna().all()
    assert "temperature_2m" not in features.columns
    assert features["route_deviation_available"].eq(0.0).all()
    assert features["pattern_route_deviation_m"].isna().all()
    assert set(features["heading_available"].unique()).issubset({0.0, 1.0})
    clear_route_profile_cache()


def _model_d_feature_frame(n_normal: int = 20) -> pd.DataFrame:
    """Synthetic 19-feature matrix for Isolation Forest tests (not real-world data)."""
    rows = []
    for i in range(n_normal):
        rows.append(
            {
                "speed_kmh": 30.0 + i * 0.5,
                "acceleration_kmh_s": 0.1 * (i % 3),
                "distance_km": 0.5,
                "stop_duration_sec": 0.0,
                "time_gap_sec": 120.0,
                "heading_sin": 1.0,
                "heading_cos": 0.0,
                "heading_change_deg": 2.0 + i * 0.1,
                "heading_available": 1.0,
                "heading_change_available": 1.0,
                "pattern_route_deviation_m": 15.0 + i,
                "route_deviation_available": 1.0,
                "temperature": 28.0,
                "humidity": 60.0,
                "wind_speed": 4.0,
                "traffic_vehicle_count": 20.0,
                "traffic_avg_speed_kmh": 32.0,
                "traffic_median_speed_kmh": 30.0,
                "traffic_available": 1.0,
            }
        )
    rows.append(
        {
            "speed_kmh": 200.0,
            "acceleration_kmh_s": 50.0,
            "distance_km": 50.0,
            "stop_duration_sec": 600.0,
            "time_gap_sec": 10.0,
            "heading_sin": -1.0,
            "heading_cos": 0.0,
            "heading_change_deg": 175.0,
            "heading_available": 1.0,
            "heading_change_available": 1.0,
            "pattern_route_deviation_m": 5000.0,
            "route_deviation_available": 1.0,
            "temperature": 5.0,
            "humidity": 95.0,
            "wind_speed": 40.0,
            "traffic_vehicle_count": 1.0,
            "traffic_avg_speed_kmh": 5.0,
            "traffic_median_speed_kmh": 4.0,
            "traffic_available": 1.0,
        }
    )
    return pd.DataFrame(rows)


def test_full_context_feature_contract():
    expected = (
        "speed_kmh",
        "acceleration_kmh_s",
        "distance_km",
        "stop_duration_sec",
        "time_gap_sec",
        "heading_sin",
        "heading_cos",
        "heading_change_deg",
        "heading_available",
        "heading_change_available",
        "pattern_route_deviation_m",
        "route_deviation_available",
        "temperature",
        "humidity",
        "wind_speed",
        "traffic_vehicle_count",
        "traffic_avg_speed_kmh",
        "traffic_median_speed_kmh",
        "traffic_available",
    )
    assert (
        MOVEMENT_FEATURES
        + HEADING_FEATURES
        + ROUTE_FEATURES
        + WEATHER_FEATURES
        + TRAFFIC_FEATURES
        == expected
    )
    assert FULL_CONTEXT_FEATURE_COLUMNS == expected
    assert BASE_FEATURE_COLUMNS == expected
    assert len(BASE_FEATURE_COLUMNS) == 19
    assert len(set(BASE_FEATURE_COLUMNS)) == 19


def test_isolation_forest_research_defaults():
    settings = get_settings()
    assert settings.isolation_forest_n_estimators == 200
    assert settings.isolation_forest_random_state == 42
    assert settings.isolation_forest_contamination == "auto"


def test_model_train_and_predict(tmp_path):
    settings = get_settings()
    X = _model_d_feature_frame()
    model_path = tmp_path / "model.joblib"
    detector = TrajectoryAnomalyDetector(settings)
    detector.train(X)
    saved = detector.save(model_path)
    assert saved == model_path

    detector2 = TrajectoryAnomalyDetector(settings)
    detector2.load(model_path)
    assert detector2.feature_names == list(BASE_FEATURE_COLUMNS)
    assert detector2._scaler is not None
    assert detector2._model is not None
    out = detector2.predict(X)
    assert len(out) == len(X)
    assert set(out["prediction"].unique()).issubset(
        {PREDICTION_NORMAL, PREDICTION_POTENTIAL_ANOMALY}
    )
    assert "anomaly_score" in out.columns


def test_feature_generation_heading_wraparound_and_route_profile(tmp_path: Path):
    from app.config import Settings
    from app.services.preprocessing import preprocess_gps_dataframe
    from app.services.route_profile import RouteProfileService

    gps = pd.DataFrame(
        {
            "vehicle_id": ["v1", "v1", "v1"],
            "timestamp": [
                "2025-01-01T10:00:00+00:00",
                "2025-01-01T10:01:00+00:00",
                "2025-01-01T10:02:00+00:00",
            ],
            "latitude": [18.5204, 18.5208, 18.5212],
            "longitude": [73.8567, 73.8571, 73.8575],
            "heading": [359.0, 1.0, 3.0],
            "route_id": ["P1", "P1", "P1"],
        }
    )
    processed, _ = preprocess_gps_dataframe(gps)

    profile_path = tmp_path / "route_profiles.csv"
    pd.DataFrame(
        {
            "route_id": ["P1", "P1"],
            "lat_grid": [18.5204, 18.5212],
            "lon_grid": [73.8567, 73.8575],
        }
    ).to_csv(profile_path, index=False)
    settings = Settings(route_profile_filename=str(profile_path))
    service = RouteProfileService(settings=settings, path=profile_path)
    service.load()

    features = generate_trajectory_features(
        processed, settings=settings, route_service=service, include_context=False
    )
    assert list(PHASE2_FEATURE_COLUMNS) == [
        c for c in features.columns if c not in ("vehicle_id", "timestamp")
    ]
    assert abs(float(features["heading_change_deg"].iloc[0]) - 2.0) < 1e-6
    assert features["heading_available"].eq(1.0).all()
    assert features["heading_change_available"].iloc[0] == 1.0
    assert features["route_deviation_available"].eq(1.0).all()
    assert features["pattern_route_deviation_m"].notna().all()
    assert (features["pattern_route_deviation_m"] >= 0).all()
    for col in MOVEMENT_FEATURES:
        assert col in features.columns
    assert "speed" not in features.columns


def test_model_missing_columns_raise():
    settings = get_settings()
    X = _model_d_feature_frame()
    incomplete = X.drop(columns=["speed_kmh"])
    detector = TrajectoryAnomalyDetector(settings)
    with pytest.raises(ValueError, match="Missing required feature columns"):
        detector.train(incomplete)
