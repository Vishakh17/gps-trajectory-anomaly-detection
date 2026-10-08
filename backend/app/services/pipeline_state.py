"""JSON-backed pipeline state for uploaded and processed artifacts."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    raw_filename: str | None = None
    raw_path: str | None = None
    raw_rows: int | None = None
    raw_columns: list[str] = field(default_factory=list)
    last_preprocess_report: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    vehicles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineState:
        return cls(
            raw_filename=data.get("raw_filename"),
            raw_path=data.get("raw_path"),
            raw_rows=data.get("raw_rows"),
            raw_columns=list(data.get("raw_columns") or []),
            last_preprocess_report=dict(data.get("last_preprocess_report") or {}),
            feature_columns=list(data.get("feature_columns") or []),
            vehicles=list(data.get("vehicles") or []),
        )


def load_state(settings: Settings | None = None) -> PipelineState:
    settings = settings or get_settings()
    path = settings.state_path
    if not path.is_file():
        return PipelineState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PipelineState.from_dict(data)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load pipeline state: %s", exc)
        return PipelineState()


def save_state(state: PipelineState, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.state_path.write_text(
        json.dumps(state.to_dict(), indent=2),
        encoding="utf-8",
    )


def resolve_raw_path(state: PipelineState, settings: Settings) -> Path | None:
    if state.raw_path:
        p = Path(state.raw_path)
        if p.is_file():
            return p
    if state.raw_filename:
        candidate = settings.raw_dir / state.raw_filename
        if candidate.is_file():
            return candidate
    return None
