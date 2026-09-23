"""Derive canonical interval-level wearable-sensor features."""

from __future__ import annotations

import pandas as pd

from ..features.enmo import compute_enmo
from ..time import floor_15min

_AGGREGATIONS = {
    "ENMO": ["mean", "std", "min", "max", "median", "nunique"],
    "HEARTBEAT": ["mean", "std", "min", "max", "median", "nunique", "count"],
    "STEP": ["max"],
    "CALORIES": ["max"],
    "DISTANCE": ["max"],
    "SLEEP": ["max"],
    "BATTERY": ["mean"],
}
_HEART_RATE_COLUMNS = {
    f"HEARTBEAT_{statistic}": f"HR_{statistic}"
    for statistic in ("mean", "std", "min", "max", "median", "nunique")
}


def aggregate_sensor(frame: pd.DataFrame, institution: str) -> pd.DataFrame:
    """Aggregate a canonical raw-sensor frame onto the 15-minute grid."""
    work = frame.copy()
    work["time"] = floor_15min(work["time"])
    work["ENMO"] = compute_enmo(
        work["ACCELER_X_AXIS"], work["ACCELER_Y_AXIS"], work["ACCELER_Z_AXIS"]
    )
    result = work.groupby(["key", "time"]).agg(_AGGREGATIONS)
    result.columns = [f"{column}_{statistic}" for column, statistic in result.columns]
    result = result.reset_index().rename(columns=_HEART_RATE_COLUMNS)
    result["institution"] = institution
    return result


def calculate_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert cumulative maxima to per-step deltas."""
    result = frame.sort_values(["key", "time"]).copy()
    cumulative = ["STEP_max", "CALORIES_max", "DISTANCE_max", "SLEEP_max"]
    present = [column for column in cumulative if column in result]
    for column in present:
        delta = f"{column.removesuffix('_max')}_delta"
        result[delta] = result.groupby("key")[column].diff().fillna(0).clip(lower=0)
    return result.drop(columns=present)


def detect_non_wearing(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark intervals whose canonical heart-rate mean is zero."""
    result = frame.copy()
    result["nonwearing"] = result["HR_mean"] == 0
    return result


def calculate_sensor_features(frame: pd.DataFrame, institution: str) -> pd.DataFrame:
    result = aggregate_sensor(frame, institution)
    result = calculate_deltas(result)
    return detect_non_wearing(result)
