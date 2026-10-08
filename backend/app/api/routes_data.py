"""Data upload, preprocessing, and summary endpoints."""

from __future__ import annotations

import logging
import uuid
from io import BytesIO

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from app.schemas.trajectory_schema import (
    DataSummaryResponse,
    PreprocessResponse,
    UploadResponse,
)
from app.services.pipeline_state import load_state, resolve_raw_path, save_state
from app.services.preprocessing import (
    preprocess_gps_dataframe,
    validate_csv_columns,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data", tags=["data"])


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload must be a CSV file.")

    settings = get_settings()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:
        logger.exception("CSV parse error")
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc

    col_validation = validate_csv_columns(list(df.columns))
    if not col_validation.valid:
        raise HTTPException(
            status_code=422,
            detail="; ".join(col_validation.errors),
        )

    safe_name = f"{uuid.uuid4().hex}_{file.filename.replace(' ', '_')}"
    dest = settings.raw_dir / safe_name
    dest.write_bytes(content)
    logger.info("Uploaded raw GPS file: %s (%s rows)", safe_name, len(df))

    state = load_state(settings)
    state.raw_filename = safe_name
    state.raw_path = str(dest)
    state.raw_rows = len(df)
    state.raw_columns = [c.strip().lower() for c in df.columns]
    save_state(state, settings)

    return UploadResponse(
        filename=safe_name,
        number_of_rows=len(df),
        columns=list(df.columns),
        validation_status="valid",
        validation_details={
            "warnings": col_validation.warnings,
            "column_check": col_validation.column_check,
        },
    )


@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess_data() -> PreprocessResponse:
    settings = get_settings()
    state = load_state(settings)
    raw_path = resolve_raw_path(state, settings)
    if raw_path is None:
        raise HTTPException(
            status_code=404,
            detail="No uploaded raw GPS file. POST /api/data/upload first.",
        )

    try:
        df = pd.read_csv(raw_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read raw file: {exc}") from exc

    col_validation = validate_csv_columns(list(df.columns))
    if not col_validation.valid:
        raise HTTPException(status_code=422, detail="; ".join(col_validation.errors))

    try:
        processed, report = preprocess_gps_dataframe(df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    processed.to_csv(settings.processed_gps_path, index=False)
    logger.info("Saved processed GPS to %s", settings.processed_gps_path)

    report_dict = report.to_dict()
    state.last_preprocess_report = report_dict
    state.vehicles = sorted(processed["vehicle_id"].unique().tolist()) if len(processed) else []
    save_state(state, settings)

    return PreprocessResponse(
        original_rows=report.original_rows,
        processed_rows=report.processed_rows,
        removed_rows=report.removed_rows,
        vehicles=report.vehicles,
        time_range=report_dict["time_range"],
        missing_value_summary=report.missing_value_summary,
        preprocessing_report={
            "removed_missing_required": report.removed_missing_required,
            "removed_invalid_coords": report.removed_invalid_coords,
            "removed_invalid_timestamp": report.removed_invalid_timestamp,
            "duplicates_removed": report.duplicates_removed,
            "warnings": report.warnings,
        },
    )


def _vehicle_ids_as_strings(values: object) -> list[str] | None:
    """Return unique vehicle IDs as strings (Pydantic list[str]).

    Pipeline state may store numeric IDs such as 1000; never treat the
    preprocess vehicle *count* as this list.
    """
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        text = str(values).strip()
        return [text] if text else None
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        # Scalar count/id is not a vehicle-id list.
        return None
    try:
        sequence = list(values)
    except TypeError:
        return None
    ids: list[str] = []
    seen: set[str] = set()
    for value in sequence:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if isinstance(value, bool):
            continue
        text = str(value)
        if text not in seen:
            seen.add(text)
            ids.append(text)
    return ids or None


@router.get("/summary", response_model=DataSummaryResponse)
def data_summary() -> DataSummaryResponse:
    settings = get_settings()
    state = load_state(settings)

    has_raw = resolve_raw_path(state, settings) is not None
    has_processed = settings.processed_gps_path.is_file()
    has_features = settings.features_path.is_file()
    has_model = settings.model_path.is_file()

    processed_rows = None
    feature_rows = None
    time_range = None
    vehicles = _vehicle_ids_as_strings(state.vehicles)

    if state.last_preprocess_report:
        time_range = state.last_preprocess_report.get("time_range")

    if has_processed:
        try:
            processed_df = pd.read_csv(settings.processed_gps_path)
            processed_rows = len(processed_df)
            if "vehicle_id" in processed_df.columns:
                vehicles = _vehicle_ids_as_strings(
                    processed_df["vehicle_id"].tolist()
                )
        except Exception:
            processed_rows = None

    if has_features:
        try:
            feat_df = pd.read_csv(settings.features_path)
            feature_rows = len(feat_df)
            if not vehicles and "vehicle_id" in feat_df.columns:
                vehicles = _vehicle_ids_as_strings(feat_df["vehicle_id"].tolist())
        except Exception:
            feature_rows = None

    return DataSummaryResponse(
        has_raw_upload=has_raw,
        raw_filename=state.raw_filename,
        raw_rows=state.raw_rows,
        has_processed_gps=has_processed,
        processed_rows=processed_rows,
        has_features=has_features,
        feature_rows=feature_rows,
        has_trained_model=has_model,
        vehicles=vehicles,
        time_range=time_range,
    )
