"""Central configuration for the GPS trajectory anomaly detection backend."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


def _backend_root() -> Path:
    """Return the backend/ directory (parent of app/)."""
    return Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "GPS Trajectory Anomaly Detection API"
    debug: bool = False
    log_level: str = "INFO"

    # Relative to backend/
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    data_sample_dir: str = "data/sample"
    trained_models_dir: str = "trained_models"

    # Filenames for pipeline artifacts
    state_filename: str = "pipeline_state.json"
    processed_gps_filename: str = "processed_gps.csv"
    features_filename: str = "trajectory_features.csv"
    model_filename: str = "isolation_forest.joblib"

    # Prepared PMPML historical route-pattern profiles (not regenerated at prediction time).
    route_profile_filename: str = "route_profiles.csv"
    route_profile_id_column: str = "route_id"
    route_profile_lat_column: str = "lat_grid"
    route_profile_lon_column: str = "lon_grid"
    observation_pattern_column: str = "route_id"

    # Prepared weather (hourly Pune / Asia-Kolkata). Not a live API.
    weather_filename: str = "pune_weather_feb2019.csv"
    weather_timezone: str = "Asia/Kolkata"
    weather_max_offset_minutes: int = 30
    weather_time_column: str = ""
    weather_temperature_column: str = ""
    weather_humidity_column: str = ""
    weather_wind_column: str = ""

    # Prepared PMPML 30-minute traffic cells. Not a live API.
    traffic_filename: str = "pmpml_traffic_30min.csv"
    traffic_timezone: str = "Asia/Kolkata"
    traffic_cell_decimals: int = 3
    traffic_cell_method: str = "floor"  # floor | round; 0.001° cells
    traffic_time_column: str = "time_30min"
    traffic_lat_column: str = "lat_cell"
    traffic_lon_column: str = "lon_cell"
    traffic_vehicles_column: str = "vehicles"
    traffic_avg_speed_column: str = "avg_speed_kmh"
    traffic_median_speed_column: str = "median_speed_kmh"

    # Isolation Forest — finalized research experiment (Model D)
    isolation_forest_contamination: str | float = "auto"
    isolation_forest_random_state: int = 42
    isolation_forest_n_estimators: int = 200

    @property
    def backend_root(self) -> Path:
        return _backend_root()

    def path(self, relative: str) -> Path:
        return self.backend_root / relative

    @property
    def raw_dir(self) -> Path:
        return self.path(self.data_raw_dir)

    @property
    def processed_dir(self) -> Path:
        return self.path(self.data_processed_dir)

    @property
    def sample_dir(self) -> Path:
        return self.path(self.data_sample_dir)

    @property
    def models_dir(self) -> Path:
        return self.path(self.trained_models_dir)

    @property
    def state_path(self) -> Path:
        return self.processed_dir / self.state_filename

    @property
    def processed_gps_path(self) -> Path:
        return self.processed_dir / self.processed_gps_filename

    @property
    def features_path(self) -> Path:
        return self.processed_dir / self.features_filename

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename

    def resolve_artifact_path(self, filename: str) -> Path:
        """Absolute path, otherwise under data/processed/."""
        candidate = Path(filename)
        if candidate.is_absolute():
            return candidate
        return self.processed_dir / filename

    @property
    def route_profile_path(self) -> Path:
        """Prepared route-pattern profile CSV; not regenerated at prediction time."""
        return self.resolve_artifact_path(self.route_profile_filename)

    @property
    def weather_path(self) -> Path:
        return self.resolve_artifact_path(self.weather_filename)

    @property
    def traffic_path(self) -> Path:
        return self.resolve_artifact_path(self.traffic_filename)


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Standard CSV column names
REQUIRED_GPS_COLUMNS = frozenset({"vehicle_id", "timestamp", "latitude", "longitude"})
OPTIONAL_GPS_COLUMNS = frozenset({"speed", "heading", "pattern_id", "route_id", "trip_id"})
# Do not fall back to trip_id; that would invent a trip→route mapping.
OBSERVATION_PATTERN_FALLBACK_COLUMNS = ("route_id", "pattern_id")
ALL_GPS_COLUMNS = REQUIRED_GPS_COLUMNS | OPTIONAL_GPS_COLUMNS

# v1 GPS pipeline column names (Phase 2 will map these onto the frozen names).
V1_GPS_FEATURE_COLUMNS = (
    "speed",
    "acceleration",
    "heading",
    "distance",
    "stop_duration",
    "time_gap",
)

# Frozen Model D feature groups (GPS + Route + Weather + Traffic).
# Route/weather/traffic values are not generated in Phase 1.
MOVEMENT_FEATURES = (
    "speed_kmh",
    "acceleration_kmh_s",
    "distance_km",
    "stop_duration_sec",
    "time_gap_sec",
)
HEADING_FEATURES = (
    "heading_sin",
    "heading_cos",
    "heading_change_deg",
    "heading_available",
    "heading_change_available",
)
ROUTE_FEATURES = (
    "pattern_route_deviation_m",
    "route_deviation_available",
)

# Phase 2 output (weather/traffic are added in later phases).
PHASE2_FEATURE_COLUMNS = (
    *MOVEMENT_FEATURES,
    *HEADING_FEATURES,
    *ROUTE_FEATURES,
)
WEATHER_FEATURES = (
    "temperature",
    "humidity",
    "wind_speed",
)
TRAFFIC_FEATURES = (
    "traffic_vehicle_count",
    "traffic_avg_speed_kmh",
    "traffic_median_speed_kmh",
    "traffic_available",
)

FULL_CONTEXT_FEATURE_COLUMNS = (
    *MOVEMENT_FEATURES,
    *HEADING_FEATURES,
    *ROUTE_FEATURES,
    *WEATHER_FEATURES,
    *TRAFFIC_FEATURES,
)

# Final application model feature list (Model D). Order is part of the contract.
BASE_FEATURE_COLUMNS = FULL_CONTEXT_FEATURE_COLUMNS

APPLICATION_MODEL_NAME = "Model D"
MODEL_ARTIFACT_VERSION = "model_d_v1"
