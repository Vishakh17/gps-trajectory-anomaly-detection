"""Timezone helpers for prepared weather/traffic artifacts (Asia/Kolkata)."""

from __future__ import annotations

import pandas as pd


def to_local_naive(series: pd.Series, timezone: str = "Asia/Kolkata") -> pd.Series:
    """
    Convert timestamps to timezone-naive local time.

    Timezone-aware values are converted to ``timezone``.
    Naive values are treated as already local (prepared artifact convention).
    """
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert(timezone).dt.tz_localize(None)
    return ts
