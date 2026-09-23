"""Stable task and label contracts shared by the modeling entrypoints."""

from __future__ import annotations

from typing import Literal

from ...contracts import (
    BENCHMARK_HORIZON,
    HORIZON_HOURS as HORIZONS,
    LABEL_COLUMNS as LABEL_COLS,
    PRIMARY_HORIZON,
)

ModelKind = Literal["multitask", "gru", "singletask", "lgbm"]

TASK_NAMES = tuple(f"{hours}hr" for hours in HORIZONS)


def normalize_horizon(value: object) -> str:
    """Return a supported horizon in ``Nhr`` form."""
    if value is None:
        raise ValueError("horizon is empty")
    text = str(value).strip().lower().replace(" ", "")
    if text.endswith("hr"):
        text = text[:-2]
    elif text.endswith("h"):
        text = text[:-1]
    if not text.isdigit() or int(text) not in HORIZONS:
        allowed = "/".join(str(hours) for hours in HORIZONS)
        raise ValueError(f"Unsupported horizon: {value!r}. Allowed: {allowed} (h|hr optional)")
    return f"{int(text)}hr"


def label_for_horizon(value: object) -> str:
    """Return the canonical label column for a supported horizon."""
    return f"y_{normalize_horizon(value)}_true"


def horizon_for_label(label_col: str) -> str:
    """Return the canonical horizon encoded by a label column."""
    if not isinstance(label_col, str) or not label_col.startswith("y_") or not label_col.endswith("_true"):
        raise ValueError(f"Unsupported label column: {label_col!r}")
    return normalize_horizon(label_col[2:-5])
