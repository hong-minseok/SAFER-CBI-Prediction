"""Deterministic episode-level control sampling calculations."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd

from ..schema.feature_schema import EVENT_ID_COL, IS_CASE_COL, KEY_COL


def select_control_episodes(
    pool_event_ids,
    n_target: int,
    rng: np.random.Generator,
) -> tuple[list[str], int]:
    pool = sorted(map(str, set(pool_event_ids)))
    if len(pool) < n_target:
        raise ValueError(f"control pool ({len(pool)}) smaller than target ({n_target})")
    chosen = rng.choice(np.array(pool, dtype=object), size=n_target, replace=False)
    return sorted(chosen.tolist()), n_target


def assert_no_case_control_overlap(frame: pd.DataFrame) -> None:
    case_keys = set(frame.loc[frame[IS_CASE_COL] == 1, KEY_COL])
    control_keys = set(frame.loc[frame[IS_CASE_COL] == 0, KEY_COL])
    overlap = case_keys & control_keys
    if overlap:
        raise ValueError(
            f"{len(overlap)} patients are both case and control: "
            f"{sorted(overlap)[:10]}"
        )


def sample_cohort(
    combined: pd.DataFrame,
    *,
    case_keys: Collection[object],
    control_keys: Collection[object],
    ratio: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return sampled case/control frames and aggregate sampling metadata."""
    case_frame = combined[
        (combined[IS_CASE_COL] == 1) & combined[KEY_COL].isin(case_keys)
    ]
    control_pool = combined[
        (combined[IS_CASE_COL] == 0) & combined[KEY_COL].isin(control_keys)
    ]
    case_episodes = sorted(case_frame[EVENT_ID_COL].astype(str).unique())
    target = len(case_episodes) * ratio
    chosen, achieved = select_control_episodes(
        control_pool[EVENT_ID_COL],
        target,
        rng,
    )
    control_frame = control_pool[
        control_pool[EVENT_ID_COL].astype(str).isin(chosen)
    ]
    metadata = {
        "ratio_requested": ratio,
        "n_case_episodes": len(case_episodes),
        "control_pool_size": int(control_pool[EVENT_ID_COL].nunique()),
        "n_control_target": target,
        "n_control_achieved": achieved,
        "achieved_ratio": achieved / len(case_episodes) if case_episodes else None,
        "case_rows": int(len(case_frame)),
        "control_rows": int(len(control_frame)),
        "chosen_control_event_index": chosen,
    }
    return case_frame, control_frame, metadata
