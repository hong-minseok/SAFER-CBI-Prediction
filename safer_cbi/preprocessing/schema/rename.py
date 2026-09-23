"""Assemble institution frames into the exact canonical output schema."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from safer_cbi.contracts import OUTPUT_COLUMNS

from .feature_schema import validate_schema


def assemble_canonical_output(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("At least one imputed frame is required")
    combined = pd.concat(frames, ignore_index=True)
    combined["event_index"] = combined["event_index"].astype(str)
    validate_schema(combined)
    return combined.loc[:, OUTPUT_COLUMNS].copy()
