"""Episode-wise post-hoc calibration of explicitly supplied prediction frames.

Each episode is calibrated by a model fitted to the other episodes in the same
cohort. Calibration changes only calibration-curve and DCA inputs; ROC, PR, and
scalar performance stores remain unchanged.
"""
from __future__ import annotations

import copy
import numpy as np
from dataclasses import dataclass
from typing import Dict

from ..contracts import MODEL_HORIZONS, MODEL_ORDER
from .bootstrap_engine import (
    BootstrapStore,
    RANDOM_STATE,
    _SCALAR_METRICS,
    _fixed_calibration_bins,
    _compute_net_benefit,
    _dca_treat_all,
    EPISODE_UNIT,
    dca_nb_matrix_cluster,
)
from .data_loader import identify_episodes
from .settings import (
    CALIBRATION_METHOD,
    DCA_X_MAX_FACTOR,
    PRIMARY_HORIZON,
    SPLITS,
)


@dataclass(frozen=True)
class CalibrationSlice:
    """Probability-only beta result for one exact split/model/horizon frame."""

    split: str
    model: str
    horizon: str
    y_true: np.ndarray
    source_y_prob: np.ndarray
    y_prob: np.ndarray


def beta_calibration_domain() -> tuple[tuple[str, str, str], ...]:
    """Return the exact author-approved two-cohort learned-model beta domain."""
    return tuple(
        (split, model, horizon)
        for split in SPLITS
        for model in MODEL_ORDER
        if model != "lgbm"
        for horizon in MODEL_HORIZONS[model]
    )


def _fit_calibration(y_prob: np.ndarray, y_true: np.ndarray):
    """Fit a beta calibration model."""
    from betacal import BetaCalibration

    probs_2d = np.asarray(y_prob).ravel().reshape(-1, 1)
    y_true = np.asarray(y_true)
    cal = BetaCalibration(parameters="abm")
    cal.fit(probs_2d, y_true)
    return cal


def _apply_calibration(model, y_prob: np.ndarray) -> np.ndarray:
    """Apply a fitted beta calibration model."""
    y_prob = np.asarray(y_prob)
    probs_2d = y_prob.reshape(-1, 1) if y_prob.ndim == 1 else y_prob
    if hasattr(model, "predict_proba"):
        out = model.predict_proba(probs_2d)[:, 1]
    elif hasattr(model, "predict"):
        out = model.predict(probs_2d)
    else:
        raise TypeError(f"Unknown model type: {type(model)}")
    return np.clip(out, 0.0, 1.0).ravel()


def _loeo_beta_probabilities(frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one beta calibrator per held-out episode and return aligned arrays."""
    df = identify_episodes(frame)
    y_true = df["y_true"].to_numpy(dtype=int)
    source_y_prob = df["y_prob"].to_numpy(dtype=float)
    if not np.isfinite(source_y_prob).all():
        raise ValueError("Beta calibration inputs must be finite")
    if np.any((source_y_prob <= 0.0) | (source_y_prob >= 1.0)):
        raise ValueError("Beta calibration inputs must lie strictly inside (0, 1)")

    groups = df["episode_id"].to_numpy()
    calibrated = np.empty_like(source_y_prob)
    for group in np.unique(groups):
        train_mask = groups != group
        test_mask = groups == group
        if np.unique(y_true[train_mask]).size != 2:
            raise ValueError(
                f"Leave-one-episode-out training fold {group!r} lacks both classes"
            )
        calibrator = _fit_calibration(
            source_y_prob[train_mask], y_true[train_mask]
        )
        calibrated[test_mask] = _apply_calibration(
            calibrator, source_y_prob[test_mask]
        )
    if not np.isfinite(calibrated).all():
        raise ValueError("Beta calibration produced nonfinite probabilities")
    if np.any((calibrated < 0.0) | (calibrated > 1.0)):
        raise ValueError("Beta calibration produced probabilities outside [0, 1]")
    return y_true, source_y_prob, calibrated


def _require_quantile_bins(y_prob: np.ndarray, n_bins: int) -> None:
    """Require every deterministic quantile bin to be populated."""
    _, bin_ids = _fixed_calibration_bins(y_prob, n_bins, "quantile")
    populated = int((np.bincount(bin_ids, minlength=n_bins) > 0).sum())
    if populated != n_bins:
        raise ValueError(
            f"Beta calibration requires {n_bins} nonempty quantile bins; "
            f"observed={populated}"
        )


def build_beta_calibration_grid(
    bundle,
    *,
    method: str = CALIBRATION_METHOD,
    n_bins: int = 8,
) -> dict[str, dict[str, dict[str, CalibrationSlice]]]:
    """Build the exact 26-slice probability-only beta reliability grid."""
    if method != "beta":
        raise ValueError(
            f"Unknown calibration method: {method!r}. Only 'beta' is supported."
        )
    if n_bins != 8:
        raise ValueError("The manuscript beta-calibration contract requires 8 bins")

    grid: dict[str, dict[str, dict[str, CalibrationSlice]]] = {
        split: {} for split in SPLITS
    }
    realized = []
    for split, model, horizon in beta_calibration_domain():
        try:
            frame = bundle.frames[split][model][horizon]
        except KeyError as error:
            raise KeyError(
                f"Missing beta-calibration frame for {split}/{model}/{horizon}"
            ) from error
        y_true, source_y_prob, calibrated = _loeo_beta_probabilities(frame)
        _require_quantile_bins(calibrated, n_bins)
        grid[split].setdefault(model, {})[horizon] = CalibrationSlice(
            split=split,
            model=model,
            horizon=horizon,
            y_true=y_true,
            source_y_prob=source_y_prob,
            y_prob=calibrated,
        )
        realized.append((split, model, horizon))
    if tuple(realized) != beta_calibration_domain():
        raise RuntimeError("Realized beta-calibration domain differs from the contract")
    return grid


def aligned_slice(grid, frame, split: str, model: str, horizon: str) -> CalibrationSlice:
    """Return the grid slice whose key and source arrays match ``frame``."""
    try:
        item = grid[split][model][horizon]
    except KeyError as error:
        raise KeyError(
            f"Missing beta-calibration slice {split}/{model}/{horizon}"
        ) from error
    if (item.split, item.model, item.horizon) != (split, model, horizon):
        raise ValueError(
            f"Beta-calibration slice key drift at {split}/{model}/{horizon}"
        )
    if not (
        np.array_equal(frame["y_true"].to_numpy(dtype=int), item.y_true)
        and np.array_equal(frame["y_prob"].to_numpy(dtype=float), item.source_y_prob)
    ):
        raise ValueError(
            f"Calibration frame/store drift at {split}/beta/{model}/{horizon}"
        )
    return item


def _probability_only_store(store: BootstrapStore, calibrated: np.ndarray):
    """Copy source arrays while clearing every inherited inference result."""
    new_store = copy.copy(store)
    new_store.source_y_prob = np.asarray(store.y_prob, dtype=float)
    new_store.y_prob = np.asarray(calibrated, dtype=float)
    for field in (
        *(f"{metric}_dist" for metric in _SCALAR_METRICS),
        "auprc_lift_dist",
        "roc_tpr_matrix", "pr_prec_matrix", "dca_nb_matrix",
        "fpr_grid", "recall_grid", "dca_thresholds",
    ):
        setattr(new_store, field, np.array([]))
    for field in ("point_metrics", "roc_point", "pr_point", "dca_point", "bands"):
        setattr(new_store, field, {})
    # This store carries only the recomputed net benefit; the discrimination
    # provenance it inherited does not apply to anything left on it.
    new_store.discrimination_unit = ""
    return new_store


def _rebuild_calibrated_dca_store(
    store,
    *,
    source_frame,
    calibrated,
    plan,
    dca_thresholds,
):
    """Attach only the deliberately recomputed DCA result to calibrated scores."""
    y_true = np.asarray(store.y_true, dtype=int)
    new_store = _probability_only_store(store, calibrated)
    thresholds = np.asarray(dca_thresholds, dtype=float)
    new_store.dca_thresholds = thresholds
    prevalence = float(y_true.mean())
    new_store.dca_point = {
        "net_benefit": _compute_net_benefit(y_true, calibrated, thresholds),
        "treat_all": _dca_treat_all(prevalence, thresholds),
        "prevalence": prevalence,
    }
    new_store.dca_nb_matrix = dca_nb_matrix_cluster(
        source_frame,
        y_true,
        calibrated,
        thresholds,
        plan,
    )
    return new_store


def _threshold_grid(prevalence: float) -> np.ndarray:
    """Return one cohort's deterministic net-benefit threshold grid."""
    dca_max = prevalence * DCA_X_MAX_FACTOR
    return np.linspace(
        0.0, dca_max, max(100, int(np.ceil(dca_max / 0.005)) + 1)
    )


def primary_dca_from_calibration_grid(
    bundle,
    calibration_grid,
    *,
    n_boot: int,
    seed: int = RANDOM_STATE,
    dca_thresholds: dict[str, np.ndarray] | None = None,
) -> Dict[str, Dict[str, BootstrapStore]]:
    """Recompute the retained 24-hour beta DCA family in both cohorts.

    Net benefit belongs to the reliability-and-utility family, so every
    replicate here comes from the split's episode-cluster plan.
    """
    calibrated_stores: Dict[str, Dict[str, BootstrapStore]] = {}
    for split in SPLITS:
        plan = bundle.plan(split, EPISODE_UNIT)
        if plan.n_boot != n_boot or plan.seed != seed:
            raise ValueError(
                f"[{split}] episode plan ({plan.n_boot} draws, seed {plan.seed}) does not "
                f"match the requested beta DCA inference ({n_boot} draws, seed {seed})"
            )
        learned = [
            model
            for model in MODEL_ORDER
            if model != "lgbm"
            and PRIMARY_HORIZON
            in bundle.comparison_stores[split].get(model, {})
        ]
        stores = bundle.comparison_view(split, PRIMARY_HORIZON, models=learned)
        thresholds = (
            _threshold_grid(float(stores["multitask"].dca_point["prevalence"]))
            if dca_thresholds is None
            else np.asarray(dca_thresholds[split], dtype=float)
        )
        calibrated_stores[split] = {}
        for model, store in stores.items():
            frame = bundle.frames[split][model][PRIMARY_HORIZON]
            calibration_slice = aligned_slice(
                calibration_grid, frame, split, model, PRIMARY_HORIZON
            )
            if not (
                np.array_equal(store.y_true, calibration_slice.y_true)
                and np.array_equal(store.y_prob, calibration_slice.source_y_prob)
            ):
                raise ValueError(
                    f"Calibration grid does not align with "
                    f"{split}/{model}/{PRIMARY_HORIZON}"
                )
            calibrated_stores[split][model] = _rebuild_calibrated_dca_store(
                store,
                source_frame=frame,
                calibrated=calibration_slice.y_prob,
                plan=plan,
                dca_thresholds=thresholds,
            )
    return calibrated_stores
