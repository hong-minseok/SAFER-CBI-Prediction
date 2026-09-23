"""Extract fixed-length case and control windows from canonical frames."""

from __future__ import annotations

from collections.abc import Mapping, Set

import numpy as np
import pandas as pd

from ..config import WindowSpec
from ..time import floor_15min


def extract_case_windows(
    merged_frame: pd.DataFrame,
    event_frame: pd.DataFrame,
    data_start_times: Mapping[object, pd.Timestamp],
    spec: WindowSpec,
) -> list[pd.DataFrame]:
    windows = []
    available = set(merged_frame["key"].unique())
    events = event_frame[event_frame["key"].isin(available)]
    for index, event in events.iterrows():
        key = event["key"]
        event_time = floor_15min(pd.Series([event["event_time"]]))[0]
        event_index = event.get("event_index", index)
        start_time = event_time - pd.Timedelta(hours=spec.observation_hours)
        actual_start = data_start_times.get(key)
        if actual_start is None:
            continue
        final_start = start_time
        if start_time < actual_start:
            padding = (actual_start - start_time).total_seconds() / 3600
            if padding > spec.zero_padding_max_hours:
                final_start = actual_start - pd.Timedelta(
                    hours=spec.zero_padding_max_hours
                )
        grid = pd.DataFrame(
            {
                "time": pd.date_range(
                    start=final_start,
                    end=event_time,
                    freq="15min",
                    inclusive="left",
                )
            }
        )
        if len(grid) < spec.min_case_timesteps:
            continue
        sliced = pd.merge(
            grid,
            merged_frame.loc[merged_frame["key"] == key],
            on="time",
            how="left",
        )
        sliced["is_zero_padded"] = sliced["time"] < actual_start
        sliced["key"] = key
        sliced["event_index"] = str(event_index)
        sliced["is_case"] = 1
        sliced["window_length"] = len(grid)
        windows.append(sliced)
    return windows


def extract_control_windows(
    merged_frame: pd.DataFrame,
    valid_control_keys: Set[object],
    spec: WindowSpec,
    rng: np.random.Generator,
) -> list[pd.DataFrame]:
    all_keys = set(merged_frame["key"].unique())
    control_keys = sorted(all_keys & valid_control_keys)
    windows = []
    for key in control_keys:
        participant = (
            merged_frame.loc[merged_frame["key"] == key]
            .sort_values("time")
            .reset_index(drop=True)
        )
        if len(participant) < spec.control_timesteps:
            continue
        found, attempts, used = 0, 0, []
        while (
            found < spec.control_samples_per_patient
            and attempts < spec.control_max_attempts
        ):
            attempts += 1
            max_index = len(participant) - spec.control_timesteps
            if max_index <= 0:
                break
            start_index = int(rng.integers(0, max_index, endpoint=True))
            candidate = participant.iloc[
                start_index : start_index + spec.control_timesteps
            ].copy()
            if not all(
                candidate["time"].diff().dropna() == pd.Timedelta(minutes=15)
            ):
                continue
            start, end = candidate["time"].iloc[0], candidate["time"].iloc[-1]
            if any(start <= used_end and end >= used_start for used_start, used_end in used):
                continue
            if (
                "nonwearing" in candidate
                and candidate["nonwearing"].sum() / spec.control_timesteps
                > spec.control_max_nonwear_ratio
            ):
                continue
            required = [
                column
                for column in ("ENMO_mean", "HR_mean", "Latitude")
                if column in candidate
            ]
            if candidate[required].isnull().any().any():
                continue
            candidate["event_index"] = f"control_{key}_{start_index}"
            candidate["is_case"] = 0
            candidate["is_zero_padded"] = False
            candidate["window_length"] = spec.control_timesteps
            windows.append(candidate)
            used.append((start, end))
            found += 1
    return windows


def extract_windows(
    merged_frame: pd.DataFrame,
    event_frame: pd.DataFrame,
    valid_case_keys: Set[object],
    valid_control_keys: Set[object],
    spec: WindowSpec,
    rng: np.random.Generator,
) -> pd.DataFrame:
    starts = merged_frame.groupby("key")["time"].min().to_dict()
    events = event_frame[event_frame["key"].isin(valid_case_keys)]
    windows = extract_case_windows(merged_frame, events, starts, spec)
    windows.extend(extract_control_windows(merged_frame, valid_control_keys, spec, rng))
    if not windows:
        raise ValueError("No valid case or control windows were extracted")
    return pd.concat(windows, ignore_index=True)
