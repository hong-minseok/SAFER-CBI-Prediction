"""Validation helpers for the shared preprocessing output contract."""

from __future__ import annotations

import pandas as pd

from safer_cbi.contracts import CATEGORY_MAP, FEATURE_COLUMNS, OUTPUT_COLUMNS

MODEL_FEATURES = FEATURE_COLUMNS
CANONICAL_COLUMNS = OUTPUT_COLUMNS
EVENT_ID_COL = "event_index"
IS_CASE_COL = "is_case"
KEY_COL = "key"


def validate_schema(frame: pd.DataFrame) -> None:
    """Require every model feature and every final output column."""
    missing_features = [column for column in FEATURE_COLUMNS if column not in frame]
    if missing_features:
        raise ValueError(
            f"Missing {len(missing_features)} model feature columns: {missing_features}"
        )
    missing_output = [column for column in OUTPUT_COLUMNS if column not in frame]
    if missing_output:
        raise ValueError(f"Missing canonical output columns: {missing_output}")
