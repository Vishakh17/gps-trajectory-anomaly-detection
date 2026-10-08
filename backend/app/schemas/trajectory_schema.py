"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str


class UploadResponse(BaseModel):
    filename: str
    number_of_rows: int
    columns: list[str]
    validation_status: str
    validation_details: dict[str, Any] = Field(default_factory=dict)


class PreprocessResponse(BaseModel):
    original_rows: int
    processed_rows: int
    removed_rows: int
    vehicles: int
    time_range: dict[str, str | None]
    missing_value_summary: dict[str, Any]
    preprocessing_report: dict[str, Any] = Field(default_factory=dict)


class FeatureGenerateResponse(BaseModel):
    number_of_records: int
    feature_names: list[str]
    number_of_vehicles: int
    output_path: str


class ModelTrainResponse(BaseModel):
    number_of_training_records: int
    features_used: list[str]
    model_status: str
    model_path: str


class PredictionRecord(BaseModel):
    vehicle_id: str
    timestamp: datetime
    anomaly_score: float
    prediction: str  # "Normal" or "Potential Anomaly"


class PredictResponse(BaseModel):
    predictions: list[PredictionRecord]
    total_records: int
    model_loaded: bool


class DataSummaryResponse(BaseModel):
    has_raw_upload: bool
    raw_filename: str | None
    raw_rows: int | None
    has_processed_gps: bool
    processed_rows: int | None
    has_features: bool
    feature_rows: int | None
    has_trained_model: bool
    vehicles: list[str] | None
    time_range: dict[str, str | None] | None


class ErrorDetail(BaseModel):
    detail: str
