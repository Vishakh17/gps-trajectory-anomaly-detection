"""GPS CSV validation, cleaning, and preprocessing with explicit reporting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.config import REQUIRED_GPS_COLUMNS

logger = logging.getLogger(__name__)

LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    column_check: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessReport:
    original_rows: int
    processed_rows: int
    removed_rows: int
    removed_missing_required: int
    removed_invalid_coords: int
    removed_invalid_timestamp: int
    duplicates_removed: int
    vehicles: int
    time_min: pd.Timestamp | None
    time_max: pd.Timestamp | None
    missing_value_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_rows": self.original_rows,
            "processed_rows": self.processed_rows,
            "removed_rows": self.removed_rows,
            "removed_missing_required": self.removed_missing_required,
            "removed_invalid_coords": self.removed_invalid_coords,
            "removed_invalid_timestamp": self.removed_invalid_timestamp,
            "duplicates_removed": self.duplicates_removed,
            "vehicles": self.vehicles,
            "time_range": {
                "start": self.time_min.isoformat() if self.time_min is not None else None,
                "end": self.time_max.isoformat() if self.time_max is not None else None,
            },
            "missing_value_summary": self.missing_value_summary,
            "warnings": self.warnings,
        }


def validate_csv_columns(columns: list[str]) -> ValidationResult:
    """Check that required GPS columns are present."""
    col_set = {c.strip().lower() for c in columns}
    normalized = {c.strip().lower(): c for c in columns}
    missing = REQUIRED_GPS_COLUMNS - col_set
    result = ValidationResult(valid=len(missing) == 0)
    result.column_check = {
        "normalized_columns": list(col_set),
        "missing_required": sorted(missing),
    }
    if missing:
        result.errors.append(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )
    result.column_check["original_mapping"] = normalized
    return result


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip column names for consistent processing."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def validate_coordinate(lat: float, lon: float) -> bool:
    """Return True if latitude/longitude are within valid WGS84 ranges."""
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return LAT_MIN <= la <= LAT_MAX and LON_MIN <= lo <= LON_MAX


def parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse timestamps; invalid values become NaT.

    Integer/float values in the typical Unix-seconds range are treated as
    seconds since epoch (PMPML ``timestamp`` convention).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    valid_numeric = numeric.dropna()
    if (
        len(valid_numeric) > 0
        and len(valid_numeric) == series.notna().sum()
        and bool(valid_numeric.between(1_000_000_000, 20_000_000_000).all())
    ):
        return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def preprocess_gps_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessReport]:
    """
    Clean GPS records: required fields, coords, timestamps, duplicates.

    Does not silently drop large fractions without recording counts in the report.
    """
    df = normalize_column_names(df)
    original_rows = len(df)
    warnings: list[str] = []

    for col in REQUIRED_GPS_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Required column missing after normalization: {col}")

    # Missing required
    req_mask = df[list(REQUIRED_GPS_COLUMNS)].notna().all(axis=1)
    removed_missing = int((~req_mask).sum())
    df = df.loc[req_mask].copy()

    # Timestamps
    df["timestamp"] = parse_timestamps(df["timestamp"])
    invalid_ts_mask = df["timestamp"].isna()
    removed_invalid_ts = int(invalid_ts_mask.sum())
    df = df.loc[~invalid_ts_mask].copy()

    # Coordinates
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    valid_coords = (
        lat.between(LAT_MIN, LAT_MAX)
        & lon.between(LON_MIN, LON_MAX)
        & lat.notna()
        & lon.notna()
    )
    removed_invalid_coords = int((~valid_coords).sum())
    df = df.loc[valid_coords].copy()
    df["latitude"] = lat.loc[df.index].astype(float)
    df["longitude"] = lon.loc[df.index].astype(float)

    # Optional numeric columns
    for opt in ("speed", "heading"):
        if opt in df.columns:
            df[opt] = pd.to_numeric(df[opt], errors="coerce")

    df["vehicle_id"] = df["vehicle_id"].astype(str).str.strip()
    df = df[df["vehicle_id"] != ""]

    # Duplicates: same vehicle, timestamp, lat, lon
    dup_mask = df.duplicated(
        subset=["vehicle_id", "timestamp", "latitude", "longitude"],
        keep="first",
    )
    duplicates_removed = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()

    df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)

    processed_rows = len(df)
    removed_total = original_rows - processed_rows

    if original_rows > 0 and removed_total / original_rows > 0.5:
        msg = (
            f"More than 50% of rows removed during preprocessing "
            f"({removed_total}/{original_rows}). Review input data quality."
        )
        warnings.append(msg)
        logger.warning(msg)

    missing_summary: dict[str, Any] = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            missing_summary[col] = {"missing_count": n_missing, "pct": round(
                100.0 * n_missing / max(processed_rows, 1), 2
            )}

    vehicles = df["vehicle_id"].nunique() if processed_rows else 0
    time_min = df["timestamp"].min() if processed_rows else None
    time_max = df["timestamp"].max() if processed_rows else None

    report = PreprocessReport(
        original_rows=original_rows,
        processed_rows=processed_rows,
        removed_rows=removed_total,
        removed_missing_required=removed_missing,
        removed_invalid_coords=removed_invalid_coords,
        removed_invalid_timestamp=removed_invalid_ts,
        duplicates_removed=duplicates_removed,
        vehicles=vehicles,
        time_min=time_min,
        time_max=time_max,
        missing_value_summary=missing_summary,
        warnings=warnings,
    )
    logger.info(
        "Preprocessing complete: %s -> %s rows (%s removed)",
        original_rows,
        processed_rows,
        removed_total,
    )
    return df, report
