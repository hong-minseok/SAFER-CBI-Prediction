"""Pure validators for persisted modeling boundary contracts."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_fold_splits(
    fold_splits: list[dict[str, Any]],
    n_samples: int,
    groups: np.ndarray | None = None,
    n_folds: int = 5,
) -> None:
    """Validate five-fold range, partition, coverage, and group-disjoint invariants."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if len(fold_splits) != n_folds:
        raise ValueError(f"expected {n_folds} folds; got {len(fold_splits)}")
    if groups is not None:
        groups = np.asarray(groups).astype(str)
        if len(groups) != n_samples:
            raise ValueError("groups length must equal n_samples")
    expected = set(range(n_samples))
    counts = np.zeros(n_samples, dtype=np.int64)
    seen: set[int] = set()
    for expected_fold, fold in enumerate(fold_splits):
        if set(fold) != {"fold", "train_indices", "val_indices"}:
            raise ValueError("each fold must contain fold, train_indices, and val_indices")
        fold_id = fold["fold"]
        if not isinstance(fold_id, int) or isinstance(fold_id, bool):
            raise ValueError("fold identifiers must be integers")
        if fold_id in seen or fold_id != expected_fold:
            raise ValueError("fold identifiers must be unique and ordered from zero")
        seen.add(fold_id)
        train = _indices(fold["train_indices"], n_samples, "train_indices")
        val = _indices(fold["val_indices"], n_samples, "val_indices")
        train_set, val_set = set(train), set(val)
        if train_set & val_set:
            raise ValueError(f"fold {fold_id} train and validation indices overlap")
        if train_set | val_set != expected:
            raise ValueError(f"fold {fold_id} train/validation do not partition samples")
        counts[val] += 1
        if groups is not None and set(groups[train]) & set(groups[val]):
            raise ValueError(f"fold {fold_id} leaks groups across train/validation")
    if not np.all(counts == 1):
        raise ValueError("every sample must appear in validation exactly once")


def _indices(values: Any, n_samples: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"{name} must contain integers")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"{name} contains duplicate indices")
    if np.any(array < 0) or np.any(array >= n_samples):
        raise ValueError(f"{name} contains out-of-range indices")
    return array.astype(np.int64, copy=False)
