"""Imputation for the preprocessing pipeline."""
from __future__ import annotations

import pandas as pd

from ..config import ImputeSpec

LOCATION_COLS_TO_IMPUTE = [
    "Latitude", "Longitude", "Entropy_Day", "Entropy_Day_Norm",
    "Distance_Traveled", "Location_Variability",
]
SENSOR_COLS = [
    "ENMO_mean", "ENMO_std", "ENMO_min", "ENMO_max", "ENMO_median", "ENMO_nunique",
    "HR_mean", "HR_std", "HR_min", "HR_max", "HR_median", "HR_nunique",
    "HEARTBEAT_count",
    "BATTERY_mean", "STEP_delta", "CALORIES_delta", "DISTANCE_delta", "SLEEP_delta",
]
COSINOR_COLS = [
    "ENMO_Cos_MESOR", "ENMO_Cos_Amplitude", "ENMO_Cos_Phase",
    "ENMO_Phase_unwrapped", "HR_Cos_MESOR", "HR_Cos_Amplitude",
    "HR_Cos_Phase", "HEARTBEAT_Phase_unwrapped",
]
ZONE_COLS = ["place_room", "Place_hallway", "Place_other"]
LABEL_HOURS = [1, 3, 6, 12, 18, 24]


def require_softimpute():
    try:
        from fancyimpute import SoftImpute
    except ImportError as exc:
        raise RuntimeError("SoftImpute is required for location imputation") from exc
    return SoftImpute


def remove_invalid_controls(df, ratio_threshold):
    df = df.copy()
    control_mask = df["is_case"] == 0
    removed = []
    for ev in df[control_mask]["event_index"].unique():
        ev_df = df[df["event_index"] == ev]
        nan_ratio = ev_df["ENMO_mean"].isnull().sum() / len(ev_df)
        if nan_ratio >= ratio_threshold:
            removed.append(ev)
    if removed:
        df = df[~df["event_index"].isin(removed)]
    return df


def fill_zero_padded_with_zeros(df):
    df = df.copy()
    zp = df["is_zero_padded"] == True  # noqa: E712
    if zp.sum() == 0:
        return df
    for col in SENSOR_COLS + LOCATION_COLS_TO_IMPUTE + COSINOR_COLS + ZONE_COLS:
        if col in df.columns:
            df.loc[zp, col] = df.loc[zp, col].fillna(0)
    if "nonwearing" in df.columns:
        df.loc[zp, "nonwearing"] = df.loc[zp, "nonwearing"].fillna(True)
    return df


def fill_sensor_nan_as_nonwearing(df):
    df = df.copy()
    mask = df["ENMO_mean"].isnull() & (df["is_zero_padded"] == False)  # noqa: E712
    n = int(mask.sum())
    if n > 0:
        for col in SENSOR_COLS:
            if col in df.columns:
                df.loc[mask, col] = 0
        df.loc[mask, "nonwearing"] = True
    return df


def apply_location_imputation(df):
    df = df.copy()
    is_zp = df["is_zero_padded"] == True  # noqa: E712
    missing = df[LOCATION_COLS_TO_IMPUTE].isnull().any(axis=1) & ~is_zp
    df["is_imputed_location"] = False
    df.loc[missing, "is_imputed_location"] = True
    if missing.sum() == 0:
        return df
    idx = df[~is_zp].index
    matrix = df.loc[idx, LOCATION_COLS_TO_IMPUTE].values
    SoftImpute = require_softimpute()
    try:
        imputed = SoftImpute(verbose=False).fit_transform(matrix)
    except Exception as exc:
        raise RuntimeError("SoftImpute failed during location imputation") from exc
    df.loc[idx, LOCATION_COLS_TO_IMPUTE] = imputed
    return df


def _fill_zeros(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)
    return df


def fill_std_nan_with_zero(df):
    return _fill_zeros(df, ["ENMO_std", "HR_std"])


def add_calendar_day(frame):
    result = frame.copy()
    if "time" in result.columns:
        result["Day_of_week"] = pd.to_datetime(result["time"]).dt.dayofweek
    return result


def fill_label_nan_with_cbi_based(df):
    df = df.copy()
    label_cols = [f"y_{h}hr_true" for h in LABEL_HOURS]
    case_events = df[df["is_case"] == 1]["event_index"].unique()
    for ev in case_events:
        ev_mask = df["event_index"] == ev
        cbi_time = df[ev_mask]["time"].max() + pd.Timedelta(minutes=15)
        for col, h in zip(label_cols, LABEL_HOURS):
            nan_mask = ev_mask & df[col].isnull()
            if nan_mask.sum() > 0:
                start = cbi_time - pd.Timedelta(hours=h)
                df.loc[nan_mask, col] = 0
                pos = nan_mask & (df["time"] >= start) & (df["time"] < cbi_time)
                df.loc[pos, col] = 1
    control_mask = df["is_case"] == 0
    for col in label_cols:
        m = control_mask & df[col].isnull()
        if m.sum() > 0:
            df.loc[m, col] = 0
    for col in label_cols:
        df[col] = df[col].astype(int)
    return df


def fill_remaining_nan(df):
    df = df.copy()
    for col in ["institution", "Sex", "Age"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df.groupby("event_index")[col].transform(lambda x: x.ffill().bfill())
    if "cluster_label" in df.columns:
        df["cluster_label"] = df["cluster_label"].fillna(-1)
    return df


def impute_windows(frame: pd.DataFrame, spec: ImputeSpec) -> pd.DataFrame:
    """Apply the accepted imputation sequence to an in-memory window frame."""
    result = remove_invalid_controls(
        frame,
        spec.invalid_control_sensor_nan_ratio,
    )
    result = fill_zero_padded_with_zeros(result)
    result = fill_sensor_nan_as_nonwearing(result)
    result = apply_location_imputation(result)
    result = _fill_zeros(result, COSINOR_COLS)
    result = _fill_zeros(result, ZONE_COLS)
    result = fill_std_nan_with_zero(result)
    result = fill_label_nan_with_cbi_based(result)
    result = fill_remaining_nan(result)
    return add_calendar_day(result)
