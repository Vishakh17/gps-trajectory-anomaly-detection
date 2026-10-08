import pandas as pd

from app.services.preprocessing import (
    preprocess_gps_dataframe,
    validate_coordinate,
    validate_csv_columns,
)
from app.services.preprocessing import parse_timestamps


def test_validate_csv_columns_ok():
    result = validate_csv_columns(
        ["vehicle_id", "timestamp", "latitude", "longitude", "speed"]
    )
    assert result.valid is True


def test_validate_csv_columns_missing():
    result = validate_csv_columns(["vehicle_id", "latitude"])
    assert result.valid is False
    assert "timestamp" in result.errors[0]


def test_validate_coordinate():
    assert validate_coordinate(28.6, 77.2) is True
    assert validate_coordinate(91.0, 0.0) is False
    assert validate_coordinate(0.0, 181.0) is False


def test_timestamp_validation_in_preprocess():
    df = pd.DataFrame(
        {
            "vehicle_id": ["v1", "v1"],
            "timestamp": ["2025-01-01T10:00:00+00:00", "not-a-date"],
            "latitude": [28.6, 28.61],
            "longitude": [77.2, 77.21],
        }
    )
    out, report = preprocess_gps_dataframe(df)
    assert report.removed_invalid_timestamp == 1
    assert len(out) == 1


def test_parse_timestamps():
    ts = parse_timestamps(pd.Series(["2025-01-01T10:00:00+00:00", "bad"]))
    assert ts.notna().sum() == 1
