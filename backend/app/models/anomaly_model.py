"""Isolation Forest detector for unsupervised trajectory anomaly scoring."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import (
    APPLICATION_MODEL_NAME,
    BASE_FEATURE_COLUMNS,
    MODEL_ARTIFACT_VERSION,
    Settings,
    get_settings,
)

logger = logging.getLogger(__name__)

PREDICTION_NORMAL = "Normal"
PREDICTION_POTENTIAL_ANOMALY = "Potential Anomaly"


def _resolve_contamination(value: str | float) -> str | float:
    """Return sklearn-compatible contamination ('auto' or float in (0, 0.5])."""
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned == "auto":
            return "auto"
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid contamination value: {value!r}. Use 'auto' or a float."
            ) from exc
    return float(value)


class TrajectoryAnomalyDetector:
    """
    Unsupervised Model D detector using StandardScaler + Isolation Forest.

    Outputs are indicative only — not proof of suspicious behavior.
    Labels are "Normal" or "Potential Anomaly".
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.feature_names: list[str] = list(BASE_FEATURE_COLUMNS)
        self.model_version: str = MODEL_ARTIFACT_VERSION
        self.application_model: str = APPLICATION_MODEL_NAME
        self._scaler: StandardScaler | None = None
        self._model: IsolationForest | None = None
        self._pipeline: Pipeline | None = None

    def _build_estimator(self) -> IsolationForest:
        return IsolationForest(
            n_estimators=self.settings.isolation_forest_n_estimators,
            contamination=_resolve_contamination(
                self.settings.isolation_forest_contamination
            ),
            random_state=self.settings.isolation_forest_random_state,
        )

    def _select_features(
        self, X: pd.DataFrame, feature_names: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Select configured columns; raise if any required column is missing."""
        names = list(feature_names) if feature_names is not None else self.feature_names
        missing = [name for name in names if name not in X.columns]
        if missing:
            raise ValueError(
                f"Missing required feature columns: {missing}. "
                "Do not invent placeholder values; provide the full feature matrix."
            )
        return X.loc[:, list(names)]

    def train(self, X: pd.DataFrame, feature_names: Sequence[str] | None = None) -> None:
        """Fit StandardScaler and Isolation Forest on numerical trajectory features."""
        if feature_names is not None:
            self.feature_names = list(feature_names)
        matrix = self._select_features(X, self.feature_names)
        if len(matrix) < 2:
            raise ValueError("Need at least 2 training records for Isolation Forest.")
        if matrix.isna().any().any():
            raise ValueError(
                "Training features contain missing values. "
                "Required columns must be fully populated; values are not imputed."
            )

        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(matrix)
        self._model = self._build_estimator()
        self._model.fit(scaled)
        self._pipeline = Pipeline(
            steps=[
                ("scaler", self._scaler),
                ("model", self._model),
            ]
        )
        logger.info(
            "Trained %s Isolation Forest on %s records using %s features",
            self.application_model,
            len(matrix),
            len(self.feature_names),
        )

    def save(self, path: Path | None = None) -> Path:
        """Persist scaler, Isolation Forest, feature list, and metadata with Joblib."""
        if self._scaler is None or self._model is None:
            raise RuntimeError("Model not trained; call train() first.")
        out = path or self.settings.model_path
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "scaler": self._scaler,
            "model": self._model,
            "pipeline": self._pipeline,
            "feature_names": self.feature_names,
            "model_version": self.model_version,
            "application_model": self.application_model,
            "n_estimators": self.settings.isolation_forest_n_estimators,
            "contamination": _resolve_contamination(
                self.settings.isolation_forest_contamination
            ),
            "random_state": self.settings.isolation_forest_random_state,
        }
        joblib.dump(payload, out)
        logger.info("Saved model to %s", out)
        return out

    def load(self, path: Path | None = None) -> None:
        """Load a previously saved model, scaler, and feature metadata."""
        src = path or self.settings.model_path
        if not src.is_file():
            raise FileNotFoundError(f"Model file not found: {src}")
        payload = joblib.load(src)
        self.feature_names = list(payload["feature_names"])
        self.model_version = str(payload.get("model_version", MODEL_ARTIFACT_VERSION))
        self.application_model = str(
            payload.get("application_model", APPLICATION_MODEL_NAME)
        )

        if payload.get("scaler") is not None and payload.get("model") is not None:
            self._scaler = payload["scaler"]
            self._model = payload["model"]
            self._pipeline = payload.get("pipeline")
            if self._pipeline is None:
                self._pipeline = Pipeline(
                    steps=[
                        ("scaler", self._scaler),
                        ("model", self._model),
                    ]
                )
        elif payload.get("pipeline") is not None:
            self._pipeline = payload["pipeline"]
            self._scaler = self._pipeline.named_steps.get("scaler")
            self._model = self._pipeline.named_steps.get("model")
        else:
            raise ValueError("Saved artifact is missing scaler/model (and pipeline).")

        logger.info("Loaded model from %s", src)

    @property
    def is_fitted(self) -> bool:
        return self._scaler is not None and self._model is not None

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return anomaly_score and prediction label per row.

        sklearn IsolationForest: decision_function higher => more normal.
        anomaly_score here is -decision_function so higher => more anomalous.
        """
        if self._scaler is None or self._model is None:
            raise RuntimeError("Model not loaded or trained.")

        matrix = self._select_features(X, self.feature_names)
        if matrix.isna().any().any():
            raise ValueError(
                "Prediction features contain missing values. "
                "Required columns must be fully populated; values are not imputed."
            )

        scaled = self._scaler.transform(matrix)
        raw_scores = self._model.decision_function(scaled)
        labels = self._model.predict(scaled)

        anomaly_score = -np.asarray(raw_scores, dtype=float)
        prediction = np.where(
            labels == -1,
            PREDICTION_POTENTIAL_ANOMALY,
            PREDICTION_NORMAL,
        )
        return pd.DataFrame(
            {
                "anomaly_score": anomaly_score,
                "prediction": prediction,
            }
        )
