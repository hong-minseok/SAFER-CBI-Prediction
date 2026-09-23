"""Build the 15-minute wearing-only series used for cosinor fitting.

This aggregation is separate from the model-feature aggregation in
``preprocess/sensor.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CosinorSpec
from ..features.enmo import compute_enmo
from ..time import floor_15min


def detect_nonwearing(heartbeat, normal_threshold: int = 30) -> pd.Series:
    """Non-wear if HR is 0, NaN, or below the threshold (cosinor rule; HR<30)."""
    hb = pd.Series(heartbeat).fillna(0)
    return (hb == 0) | (hb < normal_threshold)


def aggregate_15min(frame: pd.DataFrame, spec: CosinorSpec) -> pd.DataFrame:
    """Aggregate raw sensor rows to 15-min bins for cosinor fitting.

    Output columns: key, time, ENMO_mean, HR_mean. Bins whose
    non-wear ratio exceeds the configured limit have both means set to NaN.
    """
    work = frame.copy()
    work["time"] = pd.to_datetime(work["time"])
    work["ENMO"] = compute_enmo(
        work["ACCELER_X_AXIS"], work["ACCELER_Y_AXIS"], work["ACCELER_Z_AXIS"]
    )
    work["nonwearing"] = detect_nonwearing(
        work["HEARTBEAT"], spec.nonwear_hr_threshold
    )
    work["time_group"] = floor_15min(work["time"])

    grp = work.groupby(["key", "time_group"])
    counts = grp["nonwearing"].agg(
        data_count_total="size", nonwear_count="sum", nonwearing_ratio="mean"
    )
    wearing = work[~work["nonwearing"]]
    means = wearing.groupby(["key", "time_group"])[["ENMO", "HEARTBEAT"]].mean()
    means = means.rename(columns={"ENMO": "ENMO_mean", "HEARTBEAT": "HR_mean"})

    out = counts.join(means)  # bins with no wearing rows -> NaN means
    out = out.reset_index().rename(columns={"time_group": "time"})

    drop = out["nonwearing_ratio"] > spec.nonwear_ratio_drop
    out.loc[drop, ["ENMO_mean", "HR_mean"]] = np.nan

    out = out.sort_values(["key", "time"]).reset_index(drop=True)
    return out[["key", "time", "ENMO_mean", "HR_mean"]]
