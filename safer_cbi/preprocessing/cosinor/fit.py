"""Cosinor feature calculation for sliding windows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import medfilt

from ..config import CosinorSpec

# Output columns for fitted phase features.
FINAL_COLS = [
    "key", "time",
    "ENMO_Cos_MESOR", "ENMO_Cos_Amplitude", "ENMO_Cos_Phase",
    "ENMO_Phase_unwrapped", "HR_Cos_MESOR", "HR_Cos_Amplitude",
    "HR_Cos_Phase", "HEARTBEAT_Phase_unwrapped",
]


def cosinor_model(t, M, A, phi, period_hours: float = 24.0):
    return M + A * np.cos(2 * np.pi * t / period_hours + phi)


def single_component_cosinor(times, values):
    """Least-squares fit. Returns (MESOR, Amplitude>=0, Acrophase_rad in [-pi, pi])."""
    mask = ~np.isnan(values)
    times_clean = times[mask]
    values_clean = values[mask]
    if len(times_clean) < 3:
        return 0.0, 0.0, 0.0
    M_init = np.mean(values_clean)
    A_init = (np.max(values_clean) - np.min(values_clean)) / 2
    try:
        popt, _ = curve_fit(
            cosinor_model, times_clean, values_clean,
            p0=[M_init, A_init, 0.0],
            bounds=([0, 0, -2 * np.pi], [np.inf, np.inf, 2 * np.pi]),
        )
    except Exception:
        return 0.0, 0.0, 0.0
    mesor, amplitude, acro = popt[0], abs(popt[1]), popt[2]
    while acro > np.pi:
        acro -= 2 * np.pi
    while acro < -np.pi:
        acro += 2 * np.pi
    return mesor, amplitude, acro


def unwrap_phases(phases_hours, max_jump: float = 6.0):
    if len(phases_hours) < 2:
        return phases_hours
    unwrapped = np.zeros_like(phases_hours)
    unwrapped[0] = phases_hours[0]
    for i in range(1, len(phases_hours)):
        diff = phases_hours[i] - phases_hours[i - 1]
        if diff > max_jump:
            diff -= 24
        elif diff < -max_jump:
            diff += 24
        unwrapped[i] = unwrapped[i - 1] + diff
    return unwrapped


def analyze_circadian_rhythm(
    frame: pd.DataFrame,
    variable: str,
    spec: CosinorSpec,
) -> pd.DataFrame:
    """Per-subject 7-day trailing-window cosinor fit at every 15-min timestamp."""
    window_days = spec.trailing_window_days
    period = spec.period_hours
    min_fraction = spec.min_window_fraction
    max_missing = spec.max_missing_ratio
    max_jump = spec.phase_unwrap_max_jump_hours
    kernel = spec.median_filter_kernel
    expected_points = window_days * 24 * 4  # 96 per day

    results = []
    for key in frame["key"].unique():
        sdf = frame[frame["key"] == key].sort_values("time").copy()
        sdf["hours_from_start"] = (
            (sdf["time"] - sdf["time"].iloc[0]).dt.total_seconds() / 3600
        )
        subj_phases, subj_results = [], []
        for i in range(len(sdf)):
            current_time = sdf["time"].iloc[i]
            window_start = current_time - pd.Timedelta(days=window_days)
            wmask = (sdf["time"] > window_start) & (sdf["time"] <= current_time)
            wdata = sdf[wmask]
            if len(wdata) < (expected_points * min_fraction):
                continue
            missing_ratio = wdata[variable].isna().sum() / len(wdata)
            if missing_ratio >= max_missing:
                result = {
                    "key": key, "time": current_time,
                    f"{variable}_MESOR": 0, f"{variable}_Amplitude": 0,
                    f"{variable}_Acrophase_wrapped": 0, f"{variable}_Acrophase_unwrapped": 0,
                }
                results.append(result)
                continue
            times = wdata["hours_from_start"].values % period
            values = wdata[variable].values
            mesor, amplitude, acro_rad = single_component_cosinor(times, values)
            acro_hours = (acro_rad / (2 * np.pi)) * period
            if acro_hours > 12:
                acro_hours -= 24
            elif acro_hours < -12:
                acro_hours += 24
            result = {
                "key": key, "time": current_time,
                f"{variable}_MESOR": mesor, f"{variable}_Amplitude": amplitude,
                f"{variable}_Acrophase_wrapped": acro_hours,
            }
            subj_phases.append(acro_hours)
            subj_results.append(result)

        if subj_phases:
            unwrapped = unwrap_phases(np.array(subj_phases), max_jump=max_jump)
            if len(unwrapped) > 5:
                unwrapped = medfilt(unwrapped, kernel_size=kernel)
            for j, res in enumerate(subj_results):
                res[f"{variable}_Acrophase_unwrapped"] = unwrapped[j]
                results.append(res)

    return pd.DataFrame(results)


def perform_cosinor_analysis(
    aggregate_frame: pd.DataFrame,
    spec: CosinorSpec,
) -> pd.DataFrame:
    """Fit ENMO_mean and HR_mean, then join the estimates to the full time grid."""
    enmo = analyze_circadian_rhythm(aggregate_frame, "ENMO_mean", spec)
    hr = analyze_circadian_rhythm(aggregate_frame, "HR_mean", spec)

    if len(enmo) and len(hr):
        res = pd.merge(enmo, hr, on=["key", "time"], how="outer")
    elif len(enmo):
        res = enmo
    elif len(hr):
        res = hr
    else:
        res = pd.DataFrame()

    # Left-join back to the full (key, time) grid, fill missing fits with 0.
    grid = aggregate_frame[["key", "time"]].copy()
    res = pd.merge(grid, res, on=["key", "time"], how="left").fillna(0)

    # Stable output column order.
    rename_order = [
        "key", "time",
        "ENMO_mean_MESOR", "ENMO_mean_Amplitude",
        "ENMO_mean_Acrophase_wrapped", "ENMO_mean_Acrophase_unwrapped",
        "HR_mean_MESOR", "HR_mean_Amplitude",
        "HR_mean_Acrophase_wrapped", "HR_mean_Acrophase_unwrapped",
    ]
    for col in rename_order:
        if col not in res.columns:
            res[col] = 0
    res = res[rename_order]
    res.columns = FINAL_COLS
    return res
