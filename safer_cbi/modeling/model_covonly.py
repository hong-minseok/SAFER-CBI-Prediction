"""Cov-only LightGBM HPO and frozen-fit primitives."""
from __future__ import annotations

from typing import Any, Union

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import WilcoxonPruner
from optuna.samplers import TPESampler

from .modules.contracts import LABEL_COLS, TASK_NAMES
from .modules.data import CovariateOnlyDataset
from .modules.evaluation import (
    compute_task_metrics,
    summarize_multitask_folds,
    tune_f1_thresholds,
)
from .modules.folds import inner_fold_splits
from .modules.hpo import _extract_baseline_params, _suggest_from_space
from .modules.utils import (
    THRESHOLD_GRID_MAX,
    THRESHOLD_GRID_MIN,
    THRESHOLD_GRID_POINTS,
    compute_classification_metrics,
)


Booster = Union["lgb.LGBMClassifier", float]
FLOOR_FEATURES = ("Age", "Sex", "Day_of_week", "nonwearing")


def _df(features: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(features, columns=FLOOR_FEATURES)


def _scale_pos_weight(labels: np.ndarray) -> float:
    positives = int(labels.sum())
    return float((len(labels) - positives) / positives) if positives else 1.0


def train_boosters(
    features: np.ndarray,
    labels: np.ndarray,
    params: dict[str, Any],
    n_jobs: int,
    seed: int,
) -> list[Booster]:
    """Fit one deterministic booster per horizon."""
    boosters = []
    for index in range(len(TASK_NAMES)):
        task_labels = labels[:, index]
        if len(np.unique(task_labels)) < 2:
            boosters.append(float(task_labels.mean()))
            continue
        model = lgb.LGBMClassifier(
            n_jobs=n_jobs,
            scale_pos_weight=_scale_pos_weight(task_labels),
            random_state=int(seed),
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
            objective="binary",
            subsample_freq=1,
            **params,
        )
        model.fit(_df(features), task_labels)
        boosters.append(model)
    return boosters


def _proba(booster: Booster, features: np.ndarray) -> np.ndarray:
    if isinstance(booster, float):
        return np.full(len(features), booster, dtype=np.float64)
    return booster.predict_proba(_df(features))[:, 1]


def predict_boosters(boosters: list[Booster], features: np.ndarray) -> np.ndarray:
    """Return one probability column per frozen Cov-only booster."""
    return np.column_stack([_proba(booster, features) for booster in boosters])


def run_optuna_lgbm_floor(
    train_dataset: CovariateOnlyDataset,
    fold_splits,
    search_space,
    n_trials: int,
    n_jobs: int,
    seed: int,
):
    """Return a completed five-fold macro-AUROC HPO study."""
    study = optuna.create_study(
        study_name="lgbm-floor-hpo",
        direction="maximize",
        sampler=TPESampler(seed=seed, multivariate=True, group=True),
        pruner=WilcoxonPruner(p_threshold=0.1),
    )
    baseline = _extract_baseline_params(
        search_space["parameters"], search_space.get("conditional")
    )
    study.enqueue_trial(baseline, skip_if_exists=True)
    def objective(trial):
        params = _suggest_from_space(
            trial, search_space["parameters"], search_space.get("conditional")
        )
        scores = []
        for fold_record in fold_splits:
            fold = int(fold_record["fold"])
            train_indices = np.asarray(fold_record["train_indices"], dtype=int)
            validation_indices = np.asarray(fold_record["val_indices"], dtype=int)
            boosters = train_boosters(
                train_dataset.X[train_indices],
                train_dataset.y[train_indices],
                params,
                n_jobs,
                seed + fold,
            )
            probabilities = predict_boosters(
                boosters, train_dataset.X[validation_indices]
            )
            score = float(np.mean([
                compute_classification_metrics(
                    train_dataset.y[validation_indices, index],
                    probabilities[:, index],
                    0.5,
                )["auroc"]
                for index in range(len(TASK_NAMES))
            ]))
            scores.append(score)
            trial.set_user_attr(f"fold{fold}_auroc", score)
            trial.report(score, step=fold)
            if trial.should_prune():
                return float(np.mean(scores))
        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials, catch=(ValueError,))
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise RuntimeError("No HPO trials completed successfully")
    return study


def evaluate_internal_validation_lgbm(
    train_dataset: CovariateOnlyDataset,
    fold_splits,
    params,
    n_jobs: int,
    seed: int,
):
    """Build grouped OOF predictions and nested discovery-only thresholds."""
    threshold_grid = np.linspace(
        THRESHOLD_GRID_MIN, THRESHOLD_GRID_MAX, THRESHOLD_GRID_POINTS
    )
    n_samples = len(train_dataset)
    oof_probabilities = np.full(
        (n_samples, len(TASK_NAMES)), np.nan, dtype=np.float64
    )
    oof_folds = np.full(n_samples, -1, dtype=int)
    fold_results = []
    for fold_record in fold_splits:
        fold = int(fold_record["fold"])
        train_indices = np.asarray(fold_record["train_indices"], dtype=int)
        validation_indices = np.asarray(fold_record["val_indices"], dtype=int)
        boosters = train_boosters(
            train_dataset.X[train_indices],
            train_dataset.y[train_indices],
            params,
            n_jobs,
            seed + fold,
        )
        nested_probabilities = np.zeros(
            (len(train_indices), len(TASK_NAMES)), dtype=np.float64
        )
        subset_features = train_dataset.X[train_indices]
        subset_labels = train_dataset.y[train_indices]
        subset_keys = train_dataset.keys[train_indices]
        for nested_train, nested_validation in inner_fold_splits(
            subset_keys, subset_labels[:, 0], 5, seed
        ):
            nested_boosters = train_boosters(
                subset_features[nested_train],
                subset_labels[nested_train],
                params,
                n_jobs,
                seed,
            )
            nested_probabilities[nested_validation] = predict_boosters(
                nested_boosters, subset_features[nested_validation]
            )
        thresholds = tune_f1_thresholds(
            subset_labels, nested_probabilities, threshold_grid
        )
        probabilities = predict_boosters(
            boosters, train_dataset.X[validation_indices]
        )
        labels = train_dataset.y[validation_indices]
        fold_results.append({
            "fold": fold,
            "metrics": compute_task_metrics(
                labels, probabilities, thresholds, TASK_NAMES
            ),
        })
        oof_probabilities[validation_indices] = probabilities
        oof_folds[validation_indices] = fold

    if not np.isfinite(oof_probabilities).all() or (oof_folds < 0).any():
        raise RuntimeError("Cov-only OOF predictions do not cover discovery exactly once")
    thresholds = tune_f1_thresholds(
        train_dataset.y, oof_probabilities, threshold_grid
    )
    return {
        "fold_results": fold_results,
        "fold_average": summarize_multitask_folds(fold_results, TASK_NAMES),
        "aggregate": compute_task_metrics(
            train_dataset.y, oof_probabilities, thresholds, TASK_NAMES
        ),
        "predictions": {
            "folds": oof_folds.tolist(),
            "names": train_dataset.names.tolist(),
            "keys": train_dataset.keys.tolist(),
            "event_indices": train_dataset.event_indices.tolist(),
            "times": train_dataset.times.tolist(),
            "y_true": train_dataset.y.tolist(),
            "probs": oof_probabilities.tolist(),
            "thresholds": thresholds,
        },
    }
