import numpy as np
import pandas as pd

from app.services.trajectory import (
    haversine_km,
    heading_sin_cos,
    min_haversine_m,
    normalize_heading_degrees,
    smallest_heading_change_deg,
)


def test_haversine_known_distance():
    # Same point -> 0 km
    assert haversine_km(28.6139, 77.2090, 28.6139, 77.2090) == 0.0

    # Rough check: ~1 degree latitude ~111 km at equator; use small delta
    d = haversine_km(0.0, 0.0, 0.1, 0.0)
    assert 10.0 < d < 12.0


def test_heading_sin_cos():
    heading = pd.Series([0.0, 90.0, 180.0])
    sin_h, cos_h = heading_sin_cos(heading)
    assert abs(sin_h.iloc[0]) < 1e-9
    assert abs(cos_h.iloc[0] - 1.0) < 1e-9
    assert abs(sin_h.iloc[1] - 1.0) < 1e-9
    assert abs(cos_h.iloc[1]) < 1e-9
    assert abs(sin_h.iloc[2]) < 1e-9
    assert abs(cos_h.iloc[2] + 1.0) < 1e-9


def test_heading_wraparound_359_to_1():
    change = smallest_heading_change_deg(pd.Series([359.0]), pd.Series([1.0]))
    assert abs(float(change.iloc[0]) - 2.0) < 1e-9


def test_heading_wraparound_1_to_359():
    change = smallest_heading_change_deg(pd.Series([1.0]), pd.Series([359.0]))
    assert abs(float(change.iloc[0]) + 2.0) < 1e-9


def test_heading_normalize_and_missing():
    h = normalize_heading_degrees(pd.Series([361.0, -10.0, np.nan, "bad"]))
    assert abs(float(h.iloc[0]) - 1.0) < 1e-9
    assert abs(float(h.iloc[1]) - 350.0) < 1e-9
    assert pd.isna(h.iloc[2])
    assert pd.isna(h.iloc[3])


def test_min_haversine_m_same_point():
    dist = min_haversine_m(18.52, 73.85, np.array([18.52]), np.array([73.85]))
    assert dist == 0.0
