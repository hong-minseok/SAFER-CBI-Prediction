"""Patient-grouped fold creation, persistence, and validation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .validation import validate_fold_splits


def create_global_fold_splits(
    groups: np.ndarray,
    stratify_y: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int,
) -> list[dict[str, Any]]:
    """Create one seeded patient-grouped validation partition."""
    groups = np.asarray(groups).astype(str)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [
        {"fold": fold, "train_indices": train.tolist(), "val_indices": val.tolist()}
        for fold, (train, val) in enumerate(
            splitter.split(np.zeros_like(stratify_y), stratify_y, groups)
        )
    ]
    validate_fold_splits(folds, len(groups), groups, n_folds)
    return folds


def inner_fold_splits(
    groups: np.ndarray,
    stratify_y: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create patient-grouped inner folds for nested OOF predictions."""
    groups = np.asarray(groups).astype(str)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros_like(stratify_y), stratify_y, groups))
