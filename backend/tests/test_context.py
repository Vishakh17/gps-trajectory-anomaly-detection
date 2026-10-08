from pathlib import Path

import pandas as pd

from app.config import BASE_FEATURE_COLUMNS, Settings
from app.services.context_integration import integrate_context_features
from app.services.feature_engineering import generate_trajectory_features
from app.services.preprocessing import preprocess_gps_dataframe
from app.services.route_profile import RouteProfileService
from app.services.traffic import TrafficContextProvider
from app.services.weather import WeatherContextProvider


def test_context_produces_model_d_order(tmp_path: Path):
    weather_path = tmp_path / "pune_weather_hourly.csv"
    pd.DataFrame(
        {
            "time": ["2019-02-08 18:00:00"],
            "temperature_2m": [28.4],
            "relative_humidity_2m": [41.0],
            "wind_speed_10m": [3.2],
        }
    ).to_csv(weather_path, index=False)

    traffic_path = tmp_path / "pmpml_traffic_30min.csv"
    pd.DataFrame(
        {
            "time_30min": ["2019-02-08 18:00:00"],
            "lat_cell": [18.447],
            "lon_cell": [73.859],
            "vehicles": [4],
            "points": [8],
            "avg_speed_kmh": [22.5],
            "median_speed_kmh": [21.0],
        }
    ).to_csv(traffic_path, index=False)

    settings = Settings(
        weather_filename=str(weather_path),
        traffic_filename=str(traffic_path),
        weather_timezone="Asia/Kolkata",
        traffic_timezone="Asia/Kolkata",
    )
    weather = WeatherContextProvider(settings=settings, path=weather_path)
    traffic = TrafficContextProvider(settings=settings, path=traffic_path)
    weather.load()
    traffic.load()

    gps = pd.DataFrame(
        {
            "vehicle_id": ["v1", "v1"],
            "timestamp": [
                "2019-02-08T18:10:00+05:30",
                "2019-02-08T18:12:00+05:30",
            ],
            "latitude": [18.4473, 18.4474],
            "longitude": [73.8591, 73.8592],
        }
    )
    processed, _ = preprocess_gps_dataframe(gps)
    route = RouteProfileService(path=tmp_path / "no_profiles.csv")
    route.load()

    features = generate_trajectory_features(
        processed,
        settings=settings,
        route_service=route,
        weather_service=weather,
        traffic_service=traffic,
        include_context=True,
    )
    assert list(features.columns) == ["vehicle_id", "timestamp", *BASE_FEATURE_COLUMNS]
    assert len(BASE_FEATURE_COLUMNS) == 19
    assert features["traffic_available"].iloc[0] == 1.0
    assert abs(features["temperature"].iloc[0] - 28.4) < 1e-9
    assert abs(features["traffic_vehicle_count"].iloc[0] - 4) < 1e-9
    assert features["route_deviation_available"].eq(0.0).all()
    assert features["pattern_route_deviation_m"].isna().all()


def test_missing_context_not_converted_to_zero():
    phase2 = pd.DataFrame(
        {
            "vehicle_id": ["v1"],
            "timestamp": [pd.Timestamp("2019-02-08 18:10:00")],
            "latitude": [18.52],
            "longitude": [73.85],
            "speed_kmh": [30.0],
            "acceleration_kmh_s": [0.1],
            "distance_km": [0.5],
            "stop_duration_sec": [0.0],
            "time_gap_sec": [120.0],
            "heading_sin": [1.0],
            "heading_cos": [0.0],
            "heading_change_deg": [2.0],
            "heading_available": [1.0],
            "heading_change_available": [1.0],
            "pattern_route_deviation_m": [float("nan")],
            "route_deviation_available": [0.0],
        }
    )
    weather = pd.DataFrame(
        {
            "temperature": [float("nan")],
            "humidity": [float("nan")],
            "wind_speed": [float("nan")],
        },
        index=phase2.index,
    )
    traffic = pd.DataFrame(
        {
            "traffic_vehicle_count": [float("nan")],
            "traffic_avg_speed_kmh": [float("nan")],
            "traffic_median_speed_kmh": [float("nan")],
            "traffic_available": [0.0],
        },
        index=phase2.index,
    )
    out = integrate_context_features(
        phase2, weather_features=weather, traffic_features=traffic
    )
    assert out["traffic_available"].iloc[0] == 0.0
    for col in (
        "temperature",
        "humidity",
        "wind_speed",
        "traffic_vehicle_count",
        "traffic_avg_speed_kmh",
        "traffic_median_speed_kmh",
        "pattern_route_deviation_m",
    ):
        assert pd.isna(out[col].iloc[0])
    assert list(out.columns)[:21] == ["vehicle_id", "timestamp", *BASE_FEATURE_COLUMNS]
