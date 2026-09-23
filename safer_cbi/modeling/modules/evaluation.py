"""Pure evaluation helpers shared by the modeling entry points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .utils import (
    THRESHOLD_GRID_MAX,
    THRESHOLD_GRID_MIN,
    THRESHOLD_GRID_POINTS,
    compute_classification_metrics,
)

FOLD_METRICS = ("auroc", "auprc", "f1", "precision", "recall", "accuracy")
MACRO_METRICS = FOLD_METRICS + ("positive_class_proportion",)


def as_task_matrix(values: Any, *, name: str = "values") -> np.ndarray:
    """Return binary-task values in canonical ``(samples, tasks)`` shape."""
    array = np.asarray(values)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D; got shape {array.shape}")
    return array


def normalize_evaluation_inputs(y_true: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = as_task_matrix(y_true, name="y_true")
    probs = as_task_matrix(probabilities, name="probabilities")
    if labels.shape != probs.shape:
        raise ValueError(f"y_true and probabilities must have the same shape; got {labels.shape} and {probs.shape}")
    if labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError("evaluation inputs must contain at least one sample and task")
    return labels, probs


def tune_f1_thresholds(y_true: Any, probabilities: Any, grid: np.ndarray | None = None) -> list[float]:
    """Tune one F1 threshold per task using a deterministic first-tie rule."""
    labels, probs = normalize_evaluation_inputs(y_true, probabilities)
    if grid is None:
        grid = np.linspace(THRESHOLD_GRID_MIN, THRESHOLD_GRID_MAX, THRESHOLD_GRID_POINTS)
    grid = np.asarray(grid)
    if grid.ndim != 1 or len(grid) == 0:
        raise ValueError("grid must be a non-empty 1D array")
    thresholds: list[float] = []
    for index in range(labels.shape[1]):
        if len(np.unique(labels[:, index])) <= 1:
            thresholds.append(0.5)
            continue
        best_f1, best_threshold = -1.0, 0.5
        for threshold in grid:
            score = float(f1_score(labels[:, index], (probs[:, index] >= threshold).astype(int), zero_division=0))
            if score > best_f1:
                best_f1, best_threshold = score, float(threshold)
        thresholds.append(best_threshold)
    return thresholds


def compute_task_metrics(y_true: Any, probabilities: Any, thresholds: Sequence[float], task_names: Sequence[str]) -> dict[str, Any]:
    """Return the existing multitask ``tasks``/``macro`` metric structure."""
    labels, probs = normalize_evaluation_inputs(y_true, probabilities)
    if len(thresholds) != labels.shape[1] or len(task_names) != labels.shape[1]:
        raise ValueError("thresholds and task_names must match the number of tasks")
    if len(set(task_names)) != len(task_names):
        raise ValueError("task_names must be unique")
    result: dict[str, Any] = {"tasks": {}, "macro": {}}
    per_task = []
    for index, task_name in enumerate(task_names):
        metrics = compute_classification_metrics(labels[:, index], probs[:, index], float(thresholds[index]))
        result["tasks"][task_name] = metrics
        per_task.append(metrics)
    for metric in MACRO_METRICS:
        result["macro"][metric] = float(np.mean([item[metric] for item in per_task]))
    return result


def summarize_multitask_folds(fold_results: Sequence[Mapping[str, Any]], task_names: Sequence[str]) -> dict[str, Any]:
    if not fold_results:
        raise ValueError("fold_results must not be empty")
    summary: dict[str, Any] = {"macro": {}, "tasks": {}}
    for metric in FOLD_METRICS:
        summary["macro"][metric] = _mean_std([fold["metrics"]["macro"][metric] for fold in fold_results])
    for task_name in task_names:
        summary["tasks"][task_name] = {
            metric: _mean_std([fold["metrics"]["tasks"][task_name][metric] for fold in fold_results])
            for metric in FOLD_METRICS
        }
    return summary


def summarize_single_folds(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    if not fold_results:
        raise ValueError("fold_results must not be empty")
    return {metric: _mean_std([fold["metrics"][metric] for fold in fold_results]) for metric in FOLD_METRICS}


def build_prediction_frame(
    predictions: Mapping[str, Any],
    task_names: Sequence[str],
    *,
    threshold_key: str = "thresholds",
    include_fold: bool = True,
) -> pd.DataFrame:
    """Build the stable metadata-first prediction CSV frame for one or more tasks."""
    metadata = {
        "patient_id": pd.Series(predictions["names"]).astype(str),
        "key": predictions["keys"],
        "event_index": predictions["event_indices"],
        "time": pd.to_datetime(predictions["times"]),
    }
    if include_fold:
        metadata = {"fold": predictions["folds"], **metadata}
    frame = pd.DataFrame(metadata)
    labels, probs = normalize_evaluation_inputs(predictions["y_true"], predictions["probs"])
    if len(frame) != labels.shape[0] or len(task_names) != labels.shape[1]:
        raise ValueError("prediction metadata, samples, and task_names must align")
    thresholds = np.asarray(predictions[threshold_key], dtype=float).reshape(-1)
    if len(thresholds) != labels.shape[1]:
        raise ValueError("prediction thresholds must match the number of tasks")
    predicted = (probs >= thresholds.reshape(1, -1)).astype(int)
    for index, task_name in enumerate(task_names):
        frame[f"y_{task_name}_true"] = labels[:, index].astype(int)
        frame[f"y_{task_name}_prob"] = probs[:, index].astype(float)
        frame[f"y_{task_name}_pred"] = predicted[:, index].astype(int)
    return frame


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}
