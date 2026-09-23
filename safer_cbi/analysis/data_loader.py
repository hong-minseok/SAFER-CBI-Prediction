"""Validate aligned, canonical prediction frames for numeric analysis."""

import hashlib
import numpy as np
import pandas as pd
from typing import Dict

from ..contracts import HORIZONS


# Aligned multi/single/lgbm contract
def _assert_aligned(frames: Dict[str, pd.DataFrame], split: str, horizon: str) -> None:
    """Fail-fast: every model must be positionally aligned within one horizon.

    Checks (cross-model, within this horizon): row count, identity columns,
    y_true, and — internal only — fold. Uses explicit ``raise`` (not ``assert``)
    so integrity holds under ``python -O``. y_true is NEVER compared across
    horizons (it legitimately differs: 6hr has fewer positives than 24hr).
    """
    models = list(frames)
    ref = frames[models[0]]
    n = len(ref)
    for m in models[1:]:
        f = frames[m]
        if len(f) != n:
            raise ValueError(
                f"[{split}/{horizon}] {m} row count {len(f)} != {n} ({models[0]}) — regenerate aligned inputs")
        for col in ("key", "event_index", "time"):
            if not (f[col].values == ref[col].values).all():
                raise ValueError(
                    f"[{split}/{horizon}] {m} identity column '{col}' misaligned — regenerate aligned inputs")
        if not (f["y_true"].values == ref["y_true"].values).all():
            raise ValueError(
                f"[{split}/{horizon}] {m} y_true differs from {models[0]} — alignment broken")
    if split == "internal":  # external CSVs carry no fold column
        for m in models[1:]:
            if not (frames[m]["fold"].values == ref["fold"].values).all():
                raise ValueError(f"[{split}/{horizon}] {m} fold differs — CV alignment broken")


def _assert_cross_horizon_identity(
    frames_by_h: Dict[str, pd.DataFrame], split: str
) -> None:
    """Fail-fast: identity columns must be identical across every horizon.

    One cluster bootstrap plan per split is built from the primary horizon and
    shared by every model and horizon, so a horizon whose rows differ in key,
    event_index, or time would silently resample a different sample. Combined
    with the per-horizon cross-model check in ``_assert_aligned``, verifying the
    primary model here closes the whole grid — and both cluster domains
    (participant and episode) follow from these three columns.
    """
    reference_horizon = HORIZONS[0]
    ref = frames_by_h[reference_horizon]
    for horizon in HORIZONS[1:]:
        frame = frames_by_h[horizon]
        if len(frame) != len(ref):
            raise ValueError(
                f"[{split}] horizon {horizon} has {len(frame)} rows, "
                f"{reference_horizon} has {len(ref)} — one plan cannot span both")
        for col in ("key", "event_index", "time"):
            if not (frame[col].values == ref[col].values).all():
                raise ValueError(
                    f"[{split}] identity column '{col}' differs between horizons "
                    f"{reference_horizon} and {horizon} — the shared bootstrap plan "
                    "would resample a different sample")


def _fingerprint(frames_by_h: Dict[str, pd.DataFrame]) -> str:
    """Stable hash of identity columns + y_true across all horizons.

    Model-invariant (excludes y_prob) so it gates plan regeneration: the same
    plan must regenerate identically regardless of which model's predictions
    changed. Identity is shared across horizons — enforced by
    ``_assert_cross_horizon_identity`` — while y_true is folded in per horizon
    so any change to a single-horizon input is caught.
    """
    ref = frames_by_h[HORIZONS[0]]                       # identity shared across horizons
    ident = (ref["key"].astype(str) + "|" + ref["event_index"].astype(str)
             + "|" + ref["time"].astype(str)).values
    h = hashlib.sha1()
    h.update("\n".join(ident).encode())
    for hz in HORIZONS:
        h.update(np.ascontiguousarray(frames_by_h[hz]["y_true"].values.astype(np.int8)))
    return h.hexdigest()[:16]


def _prob_fingerprint(frame: pd.DataFrame) -> str:
    """Stable hash of y_prob — gates store/threshold validity (per split×model×horizon).

    Complements ``_fingerprint``: that one stays y_prob-invariant for plan
    determinism, while this one detects a retrained model whose predictions
    changed even though episodes/labels/row order did not.
    """
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(frame["y_prob"].values.astype(np.float64)))
    return h.hexdigest()[:16]


def align_prediction_frames(
    frames: Dict[str, Dict[str, pd.DataFrame]],
    split: str,
) -> dict:
    """Validate injected model frames and return the aligned numeric input.

    Returns
    -------
    {
      "frames": {model: {horizon: DataFrame}},   # unified schema, identical row order
      "episode_id": (n_rows,) str,               # "key|event_index"
      "key_id":     (n_rows,) str,
      "fold_index": (n_rows,) int | None,         # 0-based held-out fold; None externally
      "fingerprint": str,                         # identity+y_true hash (model-invariant)
    }
    """
    for h in HORIZONS:
        comparable = {model: values[h] for model, values in frames.items() if h in values}
        if not comparable:
            raise ValueError(f"No prediction frames supplied for horizon {h}")
        _assert_aligned(comparable, split, h)
    if "multitask" not in frames or "24hr" not in frames["multitask"]:
        raise ValueError("Primary prediction frame is missing")
    # fold has served the cross-model alignment check; downstream numeric code
    # reads only key/event_index/time/y_true/y_prob, so the held-out fold is
    # lifted out beside episode_id/key_id rather than kept in every frame. It
    # stays 0-based as stored; the emit layer owns the 1-based report label.
    primary = frames["multitask"]["24hr"]
    fold_index = (
        primary["fold"].to_numpy(dtype=int)
        if "fold" in primary.columns else None      # external carries no fold
    )
    for by_horizon in frames.values():
        for h, frame in by_horizon.items():
            by_horizon[h] = frame.drop(columns="fold", errors="ignore")
    _assert_cross_horizon_identity(frames["multitask"], split)
    ref = frames["multitask"]["24hr"]
    episode_id = (ref["key"].astype(str) + "|" + ref["event_index"].astype(str)).values
    return {
        "frames": frames,
        "episode_id": episode_id,
        "key_id": ref["key"].astype(str).values,
        "fold_index": fold_index,
        "fingerprint": _fingerprint(frames["multitask"]),
    }


# Episode tagging (episode_id = key|event_index)
def identify_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with episode_id (= key|event_index) and episode_type ('cbi'/'control').

    A CBI episode has at least one y_true==1 within its episode_id group.
    """
    df = df.copy()
    df["episode_id"] = df["key"].astype(str) + "|" + df["event_index"].astype(str)
    has_positive = df.groupby("episode_id")["y_true"].transform("max")
    df["episode_type"] = np.where(has_positive > 0, "cbi", "control")
    return df


def clean_cbi_episode(ep: pd.DataFrame) -> pd.DataFrame:
    """Remove early contamination from a prior CBI within a single episode.

    Finds the final contiguous y_true=1 block at the episode end, and if any
    y_true=1 exists before that block, trims the rows up to (and including) it.
    Used by episode-level detection.
    """
    y = ep["y_true"].values
    if len(y) == 0 or y[-1] != 1:
        return ep
    # Find start of final contiguous block of 1s
    final_start = len(y) - 1
    while final_start > 0 and y[final_start - 1] == 1:
        final_start -= 1
    if final_start == 0:
        return ep  # entire episode is positive or single block
    # Check for contamination before the final block
    early = y[:final_start]
    if early.max() == 0:
        return ep  # no contamination
    contam_end = np.where(early == 1)[0][-1]
    return ep.iloc[contam_end + 1:]
