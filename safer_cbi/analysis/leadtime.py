"""Lead-time sensitivity analysis with paired participant-cluster inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
)

from .settings import (
    N_BOOT, CI_ALPHA, LEADTIME_CONFIGS,
    PRIMARY_HORIZON,
)
from .data_loader import identify_episodes
from .bootstrap_engine import PARTICIPANT_UNIT


# Shared helpers

LEADTIME_MODEL = "multitask"
LEADTIME_SPLIT = "external"
DEFAULT_METRICS = ("auroc", "auprc", "f1")
# AUPRC p-value is suppressed: lead-time trimming structurally removes
# positive-class timesteps, causing a mechanical prevalence shift that
# confounds AUPRC comparison. Use AUROC instead.
_NO_PVALUE = {"auprc"}


def _compute_metric_set(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    metrics: tuple,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Metric set at a fixed threshold, unweighted or under bootstrap row weights.

    Both call paths share one scaffold. ``weights=None`` reproduces the plain
    (unweighted) estimator exactly — sklearn treats ``sample_weight=None`` as the
    unweighted call — while an array applies the hierarchical bootstrap weights.
    A degenerate (single-class) frame or draw returns all-NaN.
    """
    if weights is None:
        if len(np.unique(y)) < 2:
            return {metric: np.nan for metric in metrics}
    else:
        total = weights.sum()
        positive = weights[y == 1].sum()
        negative = weights[y == 0].sum()
        if total <= 0 or positive <= 0 or negative <= 0:
            return {metric: np.nan for metric in metrics}

    y_pred = (p >= threshold).astype(int)
    result: Dict[str, float] = {}
    if "auroc" in metrics:
        result["auroc"] = roc_auc_score(y, p, sample_weight=weights)
    if "auprc" in metrics:
        result["auprc"] = average_precision_score(y, p, sample_weight=weights)
    if "f1" in metrics:
        result["f1"] = f1_score(y, y_pred, sample_weight=weights, zero_division=0)
    return result


def _paired_pvalue(baseline: np.ndarray, condition: np.ndarray) -> float:
    """Two-sided paired bootstrap p-value with finite-sample +1 correction."""
    delta = baseline - condition
    valid = delta[~np.isnan(delta)]
    if len(valid) == 0:
        return np.nan
    denominator = len(valid) + 1
    lower = (np.count_nonzero(valid <= 0) + 1) / denominator
    upper = (np.count_nonzero(valid >= 0) + 1) / denominator
    p = 2 * min(lower, upper)
    return min(p, 1.0)


# Lead-Time Analysis — paired participant-cluster bootstrap

def run_leadtime_analysis(
    df_external: pd.DataFrame,
    threshold: float,
    *,
    plan,
    leadtime_configs: Optional[List[Tuple[str, int]]] = None,
    n_boot: int = N_BOOT,
    ci_alpha: float = CI_ALPHA,
    metrics: tuple = DEFAULT_METRICS,
    show_progress: bool = True,
    return_replicate_audit: bool = False,
):
    """
    Trim each episode tail, then evaluate every condition under the *same*
    participant-cluster draws. Drawing once on the original timeline and masking
    afterward preserves exact pairing across conditions.
    """
    from tqdm.auto import tqdm

    if leadtime_configs is None:
        leadtime_configs = LEADTIME_CONFIGS

    if plan.cluster_unit != PARTICIPANT_UNIT:
        raise ValueError(
            f"Lead-time sensitivity belongs to the performance family; got a "
            f"{plan.cluster_unit!r} plan"
        )
    if plan.n_boot != n_boot:
        raise ValueError(
            f"Participant plan carries {plan.n_boot} draws, requested {n_boot}"
        )
    df = identify_episodes(df_external).reset_index(drop=True)
    df["key"] = df["key"].astype(str)

    lo_pct = (1 - ci_alpha) / 2 * 100
    hi_pct = (1 + ci_alpha) / 2 * 100

    # Each condition is a row mask on the original frame.  This is what lets the
    # block draws remain exactly paired even though episode tails differ.
    condition_masks: Dict[str, np.ndarray] = {}
    for label, n_remove in leadtime_configs:
        mask = np.zeros(len(df), dtype=bool)
        for _, episode in df.groupby(["key", "event_index"], sort=False):
            ordered = episode.sort_values("time").index.to_numpy()
            if n_remove > 0 and len(ordered) > n_remove:
                ordered = ordered[:-n_remove]
            mask[ordered] = True
        condition_masks[label] = mask

    all_dists = {label: {m: np.zeros(n_boot) for m in metrics}
                 for label, _ in leadtime_configs}

    y = df["y_true"].to_numpy(dtype=int)
    p = df["y_prob"].to_numpy(dtype=float)
    row_code = plan.row_code_for(df)
    weights = (plan.replicate_weights(b, row_code) for b in range(n_boot))
    if show_progress:
        weights = tqdm(weights, total=n_boot, desc="Sensitivity (lead-time)", leave=False)
    for b, sampled_weights in enumerate(weights):
        for label, _ in leadtime_configs:
            mask = condition_masks[label]
            values = _compute_metric_set(
                y[mask], p[mask], threshold, metrics,
                weights=sampled_weights[mask],
            )
            for metric, value in values.items():
                all_dists[label][metric][b] = value

    # Reference = first condition (typically "Original")
    ref_label = leadtime_configs[0][0]

    # Point estimates on actual (non-bootstrapped) data
    point_estimates = {}
    for label, _ in leadtime_configs:
        mask = condition_masks[label]
        point_estimates[label] = _compute_metric_set(
            y[mask], p[mask], threshold, metrics
        )

    rows = []
    for label, _ in leadtime_configs:
        is_ref = (label == ref_label)
        dists = all_dists[label]

        for m in metrics:
            valid = dists[m][~np.isnan(dists[m])]
            point = point_estimates[label].get(m, np.nan)
            lo = np.percentile(valid, lo_pct) if len(valid) else np.nan
            hi = np.percentile(valid, hi_pct) if len(valid) else np.nan

            if is_ref or m in _NO_PVALUE:
                p_val = np.nan
            else:
                p_val = _paired_pvalue(all_dists[ref_label][m], dists[m])

            rows.append({
                "condition": label,
                "metric": m,
                "point": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "p_value": p_val,
            })

    frame = pd.DataFrame(rows)
    if not return_replicate_audit:
        return frame
    # Every condition x metric bootstrap distribution is auditable, not only the
    # summaries that reach the CSV.
    audit = {
        f"{label}/{metric}": {
            "generated": int(n_boot),
            "valid": int(np.isfinite(all_dists[label][metric]).sum()),
        }
        for label, _ in leadtime_configs
        for metric in metrics
    }
    return frame, audit


def leadtime_results(
    bundle,
    *,
    n_boot,
    seed=42,
):
    """Return the lead-time frame and its per-distribution replicate audit."""
    primary_horizon = PRIMARY_HORIZON
    threshold = bundle.thresholds[LEADTIME_MODEL][primary_horizon]
    plan = bundle.plan(LEADTIME_SPLIT, PARTICIPANT_UNIT)
    if plan.seed != seed:
        raise ValueError(
            f"Participant plan seed {plan.seed} does not match the requested {seed}"
        )
    result, replicate_audit = run_leadtime_analysis(
        bundle.frame(LEADTIME_SPLIT, LEADTIME_MODEL, primary_horizon),
        threshold=threshold,
        plan=plan,
        n_boot=n_boot,
        return_replicate_audit=True,
    )
    result.insert(0, "horizon", primary_horizon)
    result.insert(0, "model", LEADTIME_MODEL)
    result.insert(0, "split", LEADTIME_SPLIT)
    result = result.rename(columns={
        "point": "estimate",
        "ci_lo": "ci_lower",
        "ci_hi": "ci_upper",
    })
    columns = (
        "split", "model", "horizon", "condition", "metric", "estimate",
        "ci_lower", "ci_upper", "p_value",
    )
    for column in columns:
        if column not in result:
            result[column] = np.nan
    key = f"{LEADTIME_SPLIT}/{LEADTIME_MODEL}/{primary_horizon}"
    return result.loc[:, columns], {key: replicate_audit}
