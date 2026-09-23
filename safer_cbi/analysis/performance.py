"""Numeric discrimination, comparison, and episode operating metrics."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from .bootstrap_engine import extract_all_metrics_ci
from .data_loader import HORIZONS, clean_cbi_episode, identify_episodes
from .settings import (
    CI_ALPHA,
    MANUSCRIPT_DISCRIMINATION_SCHEDULE,
    model_order,
)


# Episode-level detection at an operating threshold

_TIMESTEP_HR = 0.25   # 15-min sampling (project invariant)


def episode_detection(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per-CBI-episode detection & lead-time — the episode-level counterpart to
    row-level recall at a fixed threshold.

    A CBI episode is *detected* if any alarm (y_prob >= threshold) fires inside its
    cleaned labeling window (y_true==1 rows, prior-CBI contamination trimmed). lead_time
    = hours from the first such alarm to CBI onset.

    Returns one row per CBI episode: [episode_id, window, detected, lead_time]
    (lead_time is NaN when not detected).
    """
    df = identify_episodes(df)
    cbi = df[df["episode_type"] == "cbi"]
    rows = []
    for eid in sorted(cbi["episode_id"].unique()):
        ep = cbi[cbi["episode_id"] == eid].copy()
        if "time" in ep.columns:
            ep = ep.sort_values("time")
        ep = clean_cbi_episode(ep)
        window = ep[ep["y_true"] == 1]
        if len(window) == 0:
            rows.append({"episode_id": eid, "window": 0, "detected": False, "lead_time": np.nan})
            continue
        above = window["y_prob"].values >= threshold
        if above.any():
            lead_hr = (len(window) - 1 - int(np.argmax(above))) * _TIMESTEP_HR
            rows.append({"episode_id": eid, "window": len(window), "detected": True, "lead_time": lead_hr})
        else:
            rows.append({"episode_id": eid, "window": len(window), "detected": False, "lead_time": np.nan})
    return pd.DataFrame(rows)


def episode_detection_summary(df: pd.DataFrame, threshold: float,
                              lead_floor: Optional[float] = None) -> Dict[str, float]:
    """Scalar summary of :func:`episode_detection` at one threshold.

    Returns {detection_rate, lead_time_median, n_cbi, n_detected}. When ``lead_floor``
    (hours) is supplied, also returns {timely_rate, n_timely, lead_floor}: the fraction
    of *detected* episodes whose lead-time ≥ lead_floor (denominator = detected episodes)
    — a timeliness counterpart that avoids averaging a skewed lead-time distribution.
    """
    det = episode_detection(df, threshold)
    n_cbi = len(det)
    n_detected = int(det["detected"].sum())
    lead = det.loc[det["detected"], "lead_time"].values
    out = {
        "detection_rate": n_detected / n_cbi if n_cbi else np.nan,
        "lead_time_median": float(np.median(lead)) if len(lead) else np.nan,
        "n_cbi": n_cbi,
        "n_detected": n_detected,
    }
    if lead_floor is not None:
        n_timely = int(np.sum(lead >= lead_floor))
        out["n_timely"] = n_timely
        out["timely_rate"] = n_timely / n_detected if n_detected else np.nan
        out["lead_floor"] = float(lead_floor)
    return out


def episode_false_detection(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    """Share of control episodes with at least one alarm (y_prob >= threshold).

    A control episode has no y_true==1 row at this horizon (episode_id =
    key|event_index). Any single alarm marks the whole episode falsely detected,
    so no run or contamination logic applies.

    Returns {false_detection_rate, n_alarmed, n_control}.
    """
    df = identify_episodes(df)
    ctrl = df[df["episode_type"] == "control"]
    n_control = int(ctrl["episode_id"].nunique())
    n_alarmed = int(ctrl.groupby("episode_id")["y_prob"].max().ge(threshold).sum())
    return {
        "false_detection_rate": n_alarmed / n_control if n_control else np.nan,
        "n_alarmed": n_alarmed,
        "n_control": n_control,
    }


_OPERATING_HORIZONS = HORIZONS[::-1]


def _metrics_at(y_true: np.ndarray, y_prob: np.ndarray, t: float) -> Dict[str, float]:
    """Recall / F1 / specificity at a fixed threshold (decision = y_prob >= t)."""
    yb = y_true.astype(bool)
    pos = yb.sum()
    pred = y_prob >= t
    tp = float(np.sum(pred & yb)); fp = float(np.sum(pred & ~yb))
    tn = float(np.sum(~pred & ~yb))
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / pos if pos else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {"Recall": recall, "F1": f1, "Specificity": specificity}


def _internal_operating_thresholds(store, f1_threshold: float) -> Dict[str, float]:
    """Internal-tuned operating thresholds: F1-opt (given), Youden-opt, recall>=0.5."""
    y, p = store.y_true.astype(int), store.y_prob
    fpr, tpr, roc_thr = roc_curve(y, p)
    fin = np.isfinite(roc_thr)
    youden = float(roc_thr[fin][np.argmax((tpr - fpr)[fin])])
    _, rec, pr_thr = precision_recall_curve(y, p)
    mask = rec[:-1] >= 0.5
    recall50 = float(pr_thr[mask].max()) if mask.any() else float(pr_thr.max())
    return {"F1-opt": f1_threshold, "Youden-opt": youden, "Recall≥0.5": recall50}


_OP_ORDER = ["F1-opt", "Youden-opt", "Recall≥0.5"]


def _arrays_store(bundle, split: str, model: str, horizon: str):
    """Store whose y_true/y_prob back threshold reads at (split, model, horizon).

    The primary paired store takes precedence over any stale marginal copy.
    Secondary SAFER-CBI horizons use the scheduled
    store in ``bundle.stores``.
    """
    comparisons = bundle.comparison_stores.get(split, {}).get(model, {})
    if horizon in comparisons:
        return comparisons[horizon]
    stores = bundle.stores[split].get(model, {})
    if horizon in stores:
        return stores[horizon]
    raise KeyError(f"Missing manuscript store for {split}/{model}/{horizon}")


def manuscript_discrimination_stores(bundle, split: str, models=None) -> dict:
    """Select only the scheduled manuscript discrimination cells."""
    requested = (
        {model for model, _ in MANUSCRIPT_DISCRIMINATION_SCHEDULE}
        if models is None else set(models)
    )
    selected = {}
    for model, horizon in MANUSCRIPT_DISCRIMINATION_SCHEDULE:
        if model in requested:
            selected.setdefault(model, {})[horizon] = _arrays_store(
                bundle, split, model, horizon
            )
    return selected


def _operating_point_records(bundle, model: str, horizons) -> list:
    """Return numeric operating-point records.

    One record per (split, horizon, operating point). Thresholds are tuned on
    **internal** per horizon (F1-opt / Youden-opt / recall ≥ 0.5) and the same
    threshold is read off on internal and external (honest transfer). Each record
    carries row-level recall/F1/specificity, episode-level **detection**
    (CBI episodes with ≥1 alarm in their cleaned positive window), **timely** detection
    (fraction of detected episodes alerted ≥ ½ horizon ahead; denominator = detected
    episodes), and **false detection** (share of control episodes with ≥1 alarm).
    """
    horizons = ([h for h in _OPERATING_HORIZONS
                 if h in bundle.stores["internal"].get(model, {})
                 or h in bundle.comparison_stores.get("internal", {}).get(model, {})]
                if horizons is None else list(horizons))
    thr_by_h = {h: _internal_operating_thresholds(
                    _arrays_store(bundle, "internal", model, h),
                    bundle.thresholds[model][h])
                for h in horizons}
    recs = []
    for h in horizons:
        lead_floor = 0.5 * int(h.replace("hr", ""))      # alert ≥ half the horizon ahead
        for op in _OP_ORDER:
            t = thr_by_h[h][op]
            for split in ("internal", "external"):
                st = _arrays_store(bundle, split, model, h)
                fr = bundle.frames[split][model][h]
                rl = _metrics_at(st.y_true, st.y_prob, t)
                dt = episode_detection_summary(fr, t, lead_floor=lead_floor)
                fd = episode_false_detection(fr, t)
                recs.append({
                    "model": model, "split": split, "horizon": h, "operating_point": op,
                    "threshold": t, "prevalence": float(np.mean(st.y_true)),
                    "recall": rl["Recall"], "f1": rl["F1"],
                    "episode_recall": dt["detection_rate"],
                    "timely": dt["timely_rate"],
                    "specificity": rl["Specificity"],
                    "false_detection": fd["false_detection_rate"],
                })
    return recs


def operating_point_long_format(bundle, model: str = "multitask",
                                horizons=None) -> pd.DataFrame:
    """Return one tidy row per split, horizon, and operating point.

    Columns: model, split, horizon, operating_point, threshold, prevalence, recall,
    f1, specificity (sample-level), episode_recall, timely, and false_detection
    (episode-level; false detection = share of control episodes with ≥1 alarm).
    """
    cols = ["model", "split", "horizon", "operating_point", "threshold", "prevalence",
            "recall", "f1", "episode_recall", "timely",
            "specificity", "false_detection"]
    return pd.DataFrame(_operating_point_records(bundle, model, horizons), columns=cols)


# Paired participant-cluster comparison
def _delta_dist(stores, a, b, h, metric):
    """Per-horizon paired Δ bootstrap distribution (a−b).

    Both models share each resampling draw, so ``A.{m}_dist[b] − B.{m}_dist[b]`` is the
    paired Δ*_b. Unknown metrics raise (AttributeError).
    """
    sa, sb = stores[a][h], stores[b][h]
    return getattr(sa, f"{metric}_dist") - getattr(sb, f"{metric}_dist")


def _delta_point(stores, a, b, h, metric):
    """Raw-data paired Δ point (a−b) for one horizon. Unknown metrics raise (KeyError)."""
    sa, sb = stores[a][h], stores[b][h]
    return sa.point_metrics[metric] - sb.point_metrics[metric]


def run_paired_model_comparison(
    bundle,
    splits=("internal", "external"),
    horizons=None,
    metrics=("auroc", "auprc"),
    pairs=None,
    alpha: float = CI_ALPHA,
) -> pd.DataFrame:
    """Compare models using the bundle's paired participant-cluster contrast stores."""
    from . import settings as _cfg
    from .bootstrap_engine import summarize_paired_deltas
    from statsmodels.stats.multitest import multipletests

    stores_by_split = getattr(bundle, "comparison_stores", None) or bundle.stores
    common = set(stores_by_split[splits[0]])
    for split in splits[1:]:
        common &= set(stores_by_split[split])
    models = [model for model in _cfg.model_order() if model in common]
    if pairs is None:
        pairs = [(models[i], models[j])
                 for i in range(len(models)) for j in range(i + 1, len(models))]

    rows = []
    for split in splits:
        stores = stores_by_split[split]                     # {model: {horizon: store}}
        for a, b in pairs:
            available = set(stores[a]) & set(stores[b])
            hs = ([h for h in HORIZONS if h in available]
                  if horizons is None else [h for h in horizons if h in available])
            if not hs:
                continue
            for met in metrics:
                for h in hs:
                    point = _delta_point(stores, a, b, h, met)
                    d, lo, hi, p = summarize_paired_deltas(
                        point, _delta_dist(stores, a, b, h, met), alpha)
                    rows.append({"split": split, "model_a": a, "model_b": b,
                                 "metric": met, "horizon": h, "scope": "per_horizon",
                                 "diff": d, "ci_lo": lo, "ci_hi": hi, "p_value": p})

    df = pd.DataFrame(rows)
    df["q_value"] = np.nan                                   # BH-FDR within scope (separate families)
    for _, idx in df.groupby("scope").groups.items():
        df.loc[idx, "q_value"] = multipletests(
            df.loc[idx, "p_value"].values, alpha=0.05, method="fdr_bh")[1]
    return df


def perf_long_format(
    bundle,
    models=None,
    splits=("internal", "external"),
    metrics=("auroc", "auprc", "auprc_lift", "f1", "precision",
             "sensitivity", "balanced_acc", "prevalence"),
    alpha: float = CI_ALPHA,
) -> pd.DataFrame:
    """
    Machine-friendly long-format performance dump (one tidy row per estimate).

    Columns: split, model, horizon, metric, scope, estimate, ci_lower, ci_upper.
    All rows use scope="per_horizon" and the exact manuscript schedule. Every
    interval uses stratified participant-cluster resampling. Prevalence is a
    descriptive property of the observed sample (ci_lower == ci_upper ==
    estimate) and is not resampled. Pairs with
    ``comparison_long_format`` so every reported number is reproducible from a
    parsable CSV without re-running the bootstrap.
    """
    scheduled_models = {
        model for model, _ in MANUSCRIPT_DISCRIMINATION_SCHEDULE
    }
    models = (
        [model for model in model_order() if model in scheduled_models]
        if models is None else list(models)
    )
    HOR = HORIZONS
    rows = []
    for split in splits:
        scheduled = manuscript_discrimination_stores(bundle, split, models=models)
        for m in models:
            by_horizon = scheduled.get(m, {})
            hs = [h for h in HOR if h in by_horizon]
            for h in hs:
                store = by_horizon[h]
                cis = extract_all_metrics_ci(store, alpha)
                for met in metrics:
                    pt, lo, hi = cis[met]
                    rows.append({"split": split, "model": m, "horizon": h,
                                 "metric": met, "scope": "per_horizon",
                                 "estimate": pt, "ci_lower": lo, "ci_upper": hi})
    return pd.DataFrame(rows, columns=["split", "model", "horizon", "metric",
                                       "scope", "estimate", "ci_lower", "ci_upper"])


def comparison_long_format(comp_df: pd.DataFrame) -> pd.DataFrame:
    """Re-map ``run_paired_model_comparison`` output to the shared long-format schema.

    Columns: split, model_a, model_b, metric, horizon, scope, estimate, ci_lower,
    ci_upper, p_value, q_value (estimate = paired A−B difference).
    """
    return comp_df.rename(columns={"diff": "estimate", "ci_lo": "ci_lower",
                                   "ci_hi": "ci_upper"})[
        ["split", "model_a", "model_b", "metric", "horizon", "scope",
         "estimate", "ci_lower", "ci_upper", "p_value", "q_value"]]
