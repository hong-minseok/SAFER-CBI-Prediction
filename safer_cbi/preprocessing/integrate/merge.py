"""Merge canonical sensor, location, cosinor, demographic, and event frames."""

from __future__ import annotations

import pandas as pd

from safer_cbi.contracts import HORIZON_HOURS

from ..time import floor_15min

COSINOR_COLUMNS = (
    "ENMO_Cos_MESOR",
    "ENMO_Cos_Amplitude",
    "ENMO_Cos_Phase",
    "ENMO_Phase_unwrapped",
    "HR_Cos_MESOR",
    "HR_Cos_Amplitude",
    "HR_Cos_Phase",
    "HEARTBEAT_Phase_unwrapped",
)


def normalize_cosinor(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["time"] = floor_15min(result["time"])
    result = result.drop_duplicates(subset=["key", "time"], keep="first")
    present = [column for column in COSINOR_COLUMNS if column in result]
    return result.loc[:, ["key", "time", *present]]


def add_cbi_labels(
    frame: pd.DataFrame,
    event_frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    for hours in HORIZON_HOURS:
        result[f"y_{hours}hr_true"] = 0
    if event_frame is None or event_frame.empty:
        return result
    common_keys = set(event_frame["key"].unique()) & set(result["key"].unique())
    for key in common_keys:
        events = (
            event_frame.loc[event_frame["key"] == key, "event_time"]
            .sort_values()
            .tolist()
        )
        key_mask = result["key"] == key
        for event_time in events:
            for hours in HORIZON_HOURS:
                start = event_time - pd.Timedelta(hours=hours)
                mask = key_mask & (result["time"] >= start) & (result["time"] < event_time)
                result.loc[mask, f"y_{hours}hr_true"] = 1
    return result


def merge_feature_frames(
    sensor: pd.DataFrame,
    location: pd.DataFrame,
    cosinor: pd.DataFrame,
    demographics: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Merge already canonicalized frames without repository access."""
    result = pd.merge(
        sensor,
        location,
        on=["key", "time", "institution"],
        how="outer",
    )
    result = pd.merge(result, normalize_cosinor(cosinor), on=["key", "time"], how="left")
    result = pd.merge(result, demographics, on="key", how="left")
    result = add_cbi_labels(result, events)
    return result.sort_values(["key", "time"])
