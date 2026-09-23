"""Time-grid calculations shared by preprocessing stages."""

from __future__ import annotations

import pandas as pd


def floor_15min(values: pd.Series) -> pd.Series:
    """Floor datetime-like values to the canonical 15-minute grid."""
    return pd.to_datetime(values).dt.floor("15min")
