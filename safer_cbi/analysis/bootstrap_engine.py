"""Stratified cluster bootstrap inference.

Two prespecified resampling families carry every reported interval. The
performance family draws participants with replacement within ever-CBI/control
strata; the reliability-and-utility family draws episodes with replacement
within CBI/control strata. Each draw carries every subordinate unit of the
selected cluster at the same multiplicity, so the number of epochs — and, in the
participant family, the number of episodes — varies across replicates.
"""
from __future__ import annotations

import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    f1_score, recall_score, precision_score, balanced_accuracy_score,
)
from . import settings as _cfg
from .settings import CI_ALPHA, GRID_POINTS, N_BOOT, RANDOM_STATE

_SCALAR_METRICS = ["auroc", "auprc", "f1", "sensitivity",
                   "specificity", "precision", "balanced_acc"]

# Cluster units, in contract order. One plan per unit per split.
PARTICIPANT_UNIT = "participant"
EPISODE_UNIT = "episode"
CLUSTER_UNITS = (PARTICIPANT_UNIT, EPISODE_UNIT)

# Replicate-audit metric families. Discrimination replicates come from the
# participant plan; utility replicates come from the episode plan.
DISCRIMINATION_AUDIT_METRICS = (
    *_SCALAR_METRICS, "auprc_lift", "roc_band", "pr_band",
)
UTILITY_AUDIT_METRICS = ("dca_band",)

# Which cluster unit each replicate family must use. Declared here so the engine,
# the checkpoint, and the promotion gate all compare against one source.
DISCRIMINATION_UNIT = PARTICIPANT_UNIT
UTILITY_UNIT = EPISODE_UNIT


def cluster_keys(frame, unit: str) -> np.ndarray:
    """Return each row's cluster key for the requested resampling unit."""
    key = frame["key"].astype(str).values
    if unit == PARTICIPANT_UNIT:
        return np.asarray(key)
    if unit == EPISODE_UNIT:
        evt = frame["event_index"].astype(str).values
        return np.asarray(np.char.add(np.char.add(key, "|"), evt))
    raise ValueError(f"Unknown cluster unit {unit!r}; expected one of {CLUSTER_UNITS}")


# Data containers
@dataclass
class BootstrapPlan:
    """Stratified cluster resampling shared across models and horizons."""
    cluster_id: np.ndarray        # (n_cluster,) sorted; protected in memory only
    cluster_label: np.ndarray     # (n_cluster,) "case"/"control"
    row_cluster_code: np.ndarray  # (n_rows,) int32 — plan frame row -> cluster
    counts: np.ndarray            # (n_boot, n_cluster) int16 draw multiplicity
    seed: int
    n_boot: int
    cluster_unit: str
    data_fingerprint: str = ""
    counts_fingerprint: str = ""  # sha1 of counts — pins the realized resampling across RNG changes

    def row_code_for(self, frame) -> np.ndarray:
        """Map an arbitrary frame's rows onto this plan's cluster codes.

        Resolution is by cluster key, never by row position, so a frame that
        shares the plan's cluster domain in a different order still resamples
        correctly. A key outside the domain is a hard failure.
        """
        keys = cluster_keys(frame, self.cluster_unit)
        domain = self.cluster_id
        code = np.searchsorted(domain, keys)
        bounded = np.minimum(code, len(domain) - 1)
        matched = (code < len(domain)) & (domain[bounded] == keys)
        if not matched.all():
            unknown = sorted({str(k) for k, ok in zip(keys, matched) if not ok})
            raise ValueError(
                f"Frame carries {len(unknown)} {self.cluster_unit} key(s) outside the "
                f"bootstrap plan domain: {unknown[:5]}"
            )
        code = code.astype(np.int32)
        # A frame missing part of the domain would silently resample a smaller
        # sample than the plan was drawn for.
        missing = len(domain) - len(np.unique(code))
        if missing:
            raise ValueError(
                f"Frame covers {len(domain) - missing} of {len(domain)} "
                f"{self.cluster_unit} clusters; the plan requires the full domain"
            )
        return code

    def replicate_indices(self, b: int, row_code: np.ndarray) -> np.ndarray:
        """Row indices for replicate ``b``, each row repeated by its cluster draw."""
        return np.repeat(np.arange(len(row_code)), self.counts[b][row_code])

    def replicate_weights(self, b: int, row_code: np.ndarray) -> np.ndarray:
        """Row multiplicities for replicate ``b`` as float weights."""
        return self.counts[b][row_code].astype(float)


@dataclass
class BootstrapStore:
    """Original data plus bootstrap distributions and curves."""
    y_true: np.ndarray
    y_prob: np.ndarray
    threshold: float

    # Scalar metric distributions — each (n_boot,)
    auroc_dist: np.ndarray = field(default_factory=lambda: np.array([]))
    auprc_dist: np.ndarray = field(default_factory=lambda: np.array([]))
    f1_dist: np.ndarray = field(default_factory=lambda: np.array([]))
    sensitivity_dist: np.ndarray = field(default_factory=lambda: np.array([]))
    specificity_dist: np.ndarray = field(default_factory=lambda: np.array([]))
    precision_dist: np.ndarray = field(default_factory=lambda: np.array([]))
    balanced_acc_dist: np.ndarray = field(default_factory=lambda: np.array([]))
    # Replicate-wise AUPRC / prevalence — prevalence varies under cluster draws.
    auprc_lift_dist: np.ndarray = field(default_factory=lambda: np.array([]))

    # Curve matrices
    roc_tpr_matrix: np.ndarray = field(default_factory=lambda: np.array([]))  # (n_boot, grid)
    pr_prec_matrix: np.ndarray = field(default_factory=lambda: np.array([]))  # (n_boot, grid)
    dca_nb_matrix: np.ndarray = field(default_factory=lambda: np.array([]))   # (n_boot, n_thr)

    # Point estimates (computed on original data, unweighted)
    point_metrics: Dict[str, float] = field(default_factory=dict)
    roc_point: Dict[str, np.ndarray] = field(default_factory=dict)
    pr_point: Dict[str, np.ndarray] = field(default_factory=dict)
    dca_point: Dict[str, np.ndarray] = field(default_factory=dict)

    # Grids
    fpr_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    recall_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    dca_thresholds: np.ndarray = field(default_factory=lambda: np.array([]))

    # Resampling provenance — which plan filled each replicate family.
    discrimination_unit: str = ""
    utility_unit: str = ""

    # Precomputed confidence-band summaries.
    bands: Dict = field(default_factory=dict)


@dataclass
class AnalysisBundle:
    """Orchestration container — nested by split/model/horizon, two plans per split."""
    frames: dict = field(default_factory=dict)      # frames[split][model][horizon] : PredictionFrame
    stores: dict = field(default_factory=dict)      # stores[split][model][horizon] : BootstrapStore
    comparison_stores: dict = field(default_factory=dict)  # primary-horizon stores for paired contrasts
    thresholds: dict = field(default_factory=dict)  # thresholds[model][horizon]    : float
    resampling: dict = field(default_factory=dict)  # resampling[split][unit]       : BootstrapPlan
    folds: dict = field(default_factory=dict)       # folds[split]                  : (n_rows,) int | None
    replicate_audit: dict = field(default_factory=dict)  # PHI-free generated/valid counts

    def plan(self, split: str, unit: str) -> BootstrapPlan:
        """Return the cluster plan for one split and resampling unit."""
        try:
            return self.resampling[split][unit]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"No {unit!r} bootstrap plan for split={split!r}") from exc

    def comparison_view(
        self, split: str, horizon: str = "24hr", models=None
    ) -> Dict[str, "BootstrapStore"]:
        """Return the paired-contrast stores for one split and horizon."""
        if split not in self.comparison_stores:
            raise KeyError(f"No comparison stores for split={split!r}")
        stores = self.comparison_stores[split]
        models = list(stores) if models is None else list(models)
        missing = [
            model for model in models
            if model not in stores or horizon not in stores[model]
        ]
        if missing:
            raise KeyError(
                f"Missing comparison stores for {split}/{horizon}: {missing}"
            )
        return {model: stores[model][horizon] for model in models}

    def frame(self, split: str, model: str, horizon: str):
        return self.frames[split][model][horizon]


# Plan construction
def build_cluster_plan(frame, *, unit: str, n_boot: int = N_BOOT,
                       seed: int = RANDOM_STATE,
                       data_fingerprint: str = "") -> BootstrapPlan:
    """Resample clusters with replacement within case/control strata.

    ``unit`` selects the cluster: ``"participant"`` groups by ``key`` and splits
    strata by whether the participant ever contributed a positive epoch;
    ``"episode"`` groups by ``key|event_index`` and splits strata by whether the
    episode is a CBI episode. Every row of a selected cluster receives the same
    multiplicity. The number of clusters drawn per stratum is fixed, so stratum
    sizes are preserved while epoch prevalence varies by replicate.
    """
    if unit not in CLUSTER_UNITS:
        raise ValueError(f"Unknown cluster unit {unit!r}; expected one of {CLUSTER_UNITS}")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    required = {"key", "event_index", "y_true"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Cluster bootstrap frame is missing {sorted(missing)}")

    keys = cluster_keys(frame, unit)
    uniq, row_code = np.unique(keys, return_inverse=True)
    y = frame["y_true"].values.astype(int)
    cluster_pos = np.zeros(len(uniq), dtype=int)
    np.maximum.at(cluster_pos, row_code, y)

    # Strata are mutually exclusive by *cluster experience*: a cluster is in the
    # case stratum if any of its epochs is positive. Every row of that cluster —
    # including any all-negative episode of a case participant — then receives
    # the same bootstrap multiplicity.
    label = np.where(cluster_pos > 0, "case", "control")
    case_idx = np.where(label == "case")[0]
    ctrl_idx = np.where(label == "control")[0]
    if not len(case_idx) or not len(ctrl_idx):
        raise ValueError(
            f"{unit} cluster bootstrap requires both case and control clusters"
        )

    rng = np.random.RandomState(seed)
    counts = np.zeros((n_boot, len(uniq)), dtype=np.int16)
    for b in range(n_boot):
        c = rng.choice(case_idx, size=len(case_idx), replace=True)
        k = rng.choice(ctrl_idx, size=len(ctrl_idx), replace=True)
        np.add.at(counts[b], c, 1)
        np.add.at(counts[b], k, 1)
    # internal invariant (not external data): a per-stratum draw cannot exceed its
    # stratum size <= n_cluster << int16 max, so this can only trip on an engine bug.
    assert counts.max() <= len(uniq), "cluster draw count exceeds n_cluster"

    counts_fp = hashlib.sha1(np.ascontiguousarray(counts)).hexdigest()[:16]
    return BootstrapPlan(
        cluster_id=uniq,
        cluster_label=label,
        row_cluster_code=row_code.astype(np.int32),
        counts=counts,
        seed=int(seed),
        n_boot=int(n_boot),
        cluster_unit=unit,
        data_fingerprint=data_fingerprint,
        counts_fingerprint=counts_fp,
    )


def build_split_plans(frame, *, n_boot: int = N_BOOT, seed: int = RANDOM_STATE,
                      data_fingerprint: str = "") -> Dict[str, BootstrapPlan]:
    """Build both contract plans for one split from its primary-horizon frame.

    Each plan seeds its own ``RandomState``; the two draw sequences run over
    different cluster domains and are never combined within a result file.
    """
    return {
        unit: build_cluster_plan(
            frame, unit=unit, n_boot=n_boot, seed=seed,
            data_fingerprint=data_fingerprint,
        )
        for unit in CLUSTER_UNITS
    }


# Helper functions
def scalar_metrics(y_true, y_prob, threshold):
    """Compute scalar metrics for one sample.

    Public because every reported point estimate — bootstrap replicate, whole
    cohort, and single cross-validation fold — must share one definition.
    """
    y_pred = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        auroc = auprc = np.nan
    else:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }


def _compute_roc_on_grid(y_true, y_prob, fpr_grid):
    """ROC curve interpolated onto a common FPR grid."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return np.interp(fpr_grid, fpr, tpr)


def _compute_pr_on_grid(y_true, y_prob, recall_grid):
    """PR curve interpolated onto a common recall grid (descending recall -> ascending)."""
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    sorted_idx = np.argsort(rec)
    return np.interp(recall_grid, rec[sorted_idx], prec[sorted_idx])


def _compute_net_benefit(y_true, y_prob, thresholds):
    """Net benefit at each threshold."""
    thresholds = np.asarray(thresholds, dtype=float)
    y_prob = np.asarray(y_prob).ravel()
    y_true = np.asarray(y_true).ravel()
    y_pred = y_prob[None, :] >= thresholds[:, None]   # (T, n) boolean
    y_true_b = (y_true == 1)
    tp = (y_pred & y_true_b).sum(axis=1)
    fp = (y_pred & ~y_true_b).sum(axis=1)
    valid = (thresholds >= 0) & (thresholds < 1)
    odds = np.full_like(thresholds, np.nan, dtype=float)
    odds[valid] = thresholds[valid] / (1 - thresholds[valid])
    nb = tp / len(y_true) - fp / len(y_true) * odds
    nb[~valid] = np.nan
    return nb


def _dca_treat_all(prevalence, thresholds):
    """Net benefit for 'Treat All' strategy."""
    return prevalence - (thresholds / (1 - thresholds)) * (1 - prevalence)


def require_contract_plans(discrimination_plan: BootstrapPlan,
                          utility_plan: BootstrapPlan) -> None:
    """Fail unless the two plans are the contract's units and agree on their draw.

    Passing the episode plan to discrimination (or the reverse) is silently
    computable, so the binding is enforced here rather than assumed by callers.
    """
    for role, plan, expected in (
        ("discrimination", discrimination_plan, DISCRIMINATION_UNIT),
        ("utility", utility_plan, UTILITY_UNIT),
    ):
        if not isinstance(plan, BootstrapPlan):
            raise TypeError(f"{role} plan must be a BootstrapPlan, got {type(plan).__name__}")
        if plan.cluster_unit != expected:
            raise ValueError(
                f"The {role} family must resample {expected} clusters; "
                f"received a {plan.cluster_unit!r} plan"
            )
    mismatched = [
        field
        for field in ("seed", "n_boot", "data_fingerprint")
        if getattr(discrimination_plan, field) != getattr(utility_plan, field)
    ]
    if mismatched:
        raise ValueError(
            f"Discrimination and utility plans disagree on {mismatched}; "
            "both families must describe the same cohort and draw count"
        )


def store_provenance(stores_by_split) -> tuple[str, str]:
    """Return the one (discrimination, utility) unit pair every store carries."""
    pairs = {
        (store.discrimination_unit, store.utility_unit)
        for by_model in stores_by_split.values()
        for by_horizon in by_model.values()
        for store in by_horizon.values()
    }
    if len(pairs) != 1:
        raise ValueError(
            f"Stores disagree on their resampling provenance: {sorted(pairs)}"
        )
    return pairs.pop()


def _progress(iterator, total, desc, show_progress):
    if not show_progress:
        return iterator
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterator
    return tqdm(iterator, total=total, desc=desc, leave=False)


# Cluster bootstrap stores
def compute_bootstrap_store_cluster(
    frame, threshold: float, *,
    discrimination_plan: BootstrapPlan,
    utility_plan: BootstrapPlan,
    grid_points: int = GRID_POINTS,
    dca_threshold_range: Optional[np.ndarray] = None,
    show_progress: bool = True,
) -> BootstrapStore:
    """Return a two-family cluster bootstrap store.

    Discrimination scalars and the ROC/PR bands are filled from
    ``discrimination_plan``; the net-benefit matrix is filled from
    ``utility_plan``. The two passes are independent by contract — no result
    file mixes them.
    """
    require_contract_plans(discrimination_plan, utility_plan)
    n_boot = discrimination_plan.n_boot
    y_true = np.asarray(frame["y_true"].values, dtype=int)
    y_prob = np.asarray(frame["y_prob"].values, dtype=float)

    fpr_grid = np.linspace(0.0, 1.0, grid_points)
    recall_grid = np.linspace(0.0, 1.0, grid_points)
    dca_thresholds = (np.linspace(0.0, 0.50, 100) if dca_threshold_range is None else dca_threshold_range)

    point_metrics = scalar_metrics(y_true, y_prob, threshold)
    fpr_orig, tpr_orig, _ = roc_curve(y_true, y_prob)
    roc_point = {"fpr": fpr_orig, "tpr": tpr_orig}
    prec_orig, rec_orig, _ = precision_recall_curve(y_true, y_prob)
    pr_point = {"precision": prec_orig, "recall": rec_orig}
    nb_orig = _compute_net_benefit(y_true, y_prob, dca_thresholds)
    prevalence = float(y_true.mean())
    dca_point = {"net_benefit": nb_orig, "treat_all": _dca_treat_all(prevalence, dca_thresholds),
                 "prevalence": prevalence}

    metric_dists = {k: np.zeros(n_boot) for k in _SCALAR_METRICS}
    auprc_lift_dist = np.full(n_boot, np.nan)
    roc_tpr_matrix = np.zeros((n_boot, grid_points))
    pr_prec_matrix = np.zeros((n_boot, grid_points))

    # Pass 1 — discrimination family.
    disc_code = discrimination_plan.row_code_for(frame)
    for b in _progress(range(n_boot), n_boot,
                       f"Bootstrap({discrimination_plan.cluster_unit})", show_progress):
        idx = discrimination_plan.replicate_indices(b, disc_code)
        yb, pb = y_true[idx], y_prob[idx]
        if len(np.unique(yb)) < 2:
            for k in _SCALAR_METRICS:
                metric_dists[k][b] = np.nan
            roc_tpr_matrix[b] = pr_prec_matrix[b] = np.nan
            continue
        m = scalar_metrics(yb, pb, threshold)
        for k in _SCALAR_METRICS:
            metric_dists[k][b] = m[k]
        # Prevalence moves with the draw, so lift is formed replicate-wise.
        auprc_lift_dist[b] = m["auprc"] / float(yb.mean())
        roc_tpr_matrix[b] = _compute_roc_on_grid(yb, pb, fpr_grid)
        pr_prec_matrix[b] = _compute_pr_on_grid(yb, pb, recall_grid)

    # Pass 2 — utility family.
    dca_nb_matrix = _net_benefit_matrix(
        frame, y_true, y_prob, dca_thresholds, utility_plan,
        show_progress=show_progress,
    )

    return BootstrapStore(
        y_true=y_true, y_prob=y_prob, threshold=threshold,
        auroc_dist=metric_dists["auroc"], auprc_dist=metric_dists["auprc"],
        f1_dist=metric_dists["f1"], sensitivity_dist=metric_dists["sensitivity"],
        specificity_dist=metric_dists["specificity"], precision_dist=metric_dists["precision"],
        balanced_acc_dist=metric_dists["balanced_acc"], auprc_lift_dist=auprc_lift_dist,
        roc_tpr_matrix=roc_tpr_matrix, pr_prec_matrix=pr_prec_matrix, dca_nb_matrix=dca_nb_matrix,
        point_metrics=point_metrics, roc_point=roc_point, pr_point=pr_point, dca_point=dca_point,
        fpr_grid=fpr_grid, recall_grid=recall_grid, dca_thresholds=dca_thresholds,
        discrimination_unit=discrimination_plan.cluster_unit,
        utility_unit=utility_plan.cluster_unit)


def _compute_store_cluster_task(split, model, horizon, frame, threshold,
                                discrimination_plan, utility_plan):
    label = f"{split}/{model}/{horizon}"
    try:
        store = compute_bootstrap_store_cluster(
            frame, threshold,
            discrimination_plan=discrimination_plan,
            utility_plan=utility_plan,
            show_progress=False)
    except Exception as e:
        n_pos = int(np.asarray(frame["y_true"].values).sum())
        raise RuntimeError(f"[{label}] cluster bootstrap failed "
                           f"(rows={len(frame)}, positives={n_pos})") from e
    return (split, model, horizon, store)


def compute_all_stores_cluster(
    bundle: AnalysisBundle,
    n_jobs: int = 1,
    show_progress: bool = True,
    *,
    splits=None,
    models=None,
    horizons=None,
) -> dict:
    """Cluster bootstrap for every split x model x horizon.

    Both plans are read from ``bundle.resampling[split]``. Models within a
    (split, horizon) share the plans and therefore the realized draws, so paired
    contrasts are exact. Returns a fresh ``{split:{model:{h:store}}}`` dict; does
    NOT touch ``bundle.stores``.
    """
    splits = list(bundle.frames) if splits is None else list(splits)
    models = (
        [model for model in _cfg.model_order() if model in bundle.frames[splits[0]]]
        if models is None else list(models)
    )
    horizons = (
        list(bundle.frames[splits[0]][models[0]])
        if horizons is None else list(horizons)
    )
    cols = ["key", "event_index", "y_true", "y_prob"]
    tasks = [
        (split, model, h, bundle.frames[split][model][h][cols],
         bundle.thresholds[model][h],
         bundle.plan(split, PARTICIPANT_UNIT), bundle.plan(split, EPISODE_UNIT))
        for split in splits for model in models for h in horizons
        if h in bundle.frames[split][model]
    ]
    if n_jobs == 1:
        iterator = tasks
        if show_progress:
            from tqdm.auto import tqdm
            iterator = tqdm(tasks, desc="Cluster stores")
        results = [_compute_store_cluster_task(*t) for t in iterator]
    else:
        import joblib
        from joblib import Parallel, delayed
        n_eff = joblib.effective_n_jobs(n_jobs)
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            gen = Parallel(n_jobs=n_jobs, pre_dispatch=n_eff, return_as="generator_unordered")(
                delayed(_compute_store_cluster_task)(*t) for t in tasks)
            if show_progress:
                from tqdm.auto import tqdm
                gen = tqdm(gen, total=len(tasks), desc="Cluster stores")
            results = list(gen)
    by_key = {(s, m, h): store for (s, m, h, store) in results}
    return {split: {model: {h: by_key[(split, model, h)] for h in horizons
                            if (split, model, h) in by_key}
                    for model in models if any((split, model, h) in by_key for h in horizons)}
            for split in splits}


# Cluster bands for the calibration figure (reliability diagram CI + recalibrated DCA)
def _fixed_calibration_bins(y_prob, n_bins: int, strategy: str):
    """Fixed calibration-bin edges and each row's bin ID.

    Bin membership depends only on a row's predicted prob, so these ids are reused across
    bootstrap replicates (a replicate's per-bin observed freq = bincount over ``binids[idx]``).
    """
    x = np.asarray(y_prob, dtype=float)
    if strategy == "quantile":
        edges = np.percentile(x, np.linspace(0, 1, n_bins + 1) * 100)
    elif strategy == "uniform_support":
        edges = np.linspace(float(x.min()), float(x.max()), n_bins + 1)
    else:  # "uniform" over [0, 1]
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    return edges, np.searchsorted(edges[1:-1], x)


def binned_calibration_cluster_ci(
    frame,
    y_true,
    y_prob,
    n_bins,
    strategy,
    plan: BootstrapPlan,
    alpha: float = CI_ALPHA,
    *,
    return_valid_counts: bool = False,
):
    """Reliability-diagram point estimate + per-bin cluster-bootstrap CI.

    Bins are fixed from the full data; each episode-cluster replicate reassigns
    its resampled rows to those fixed bins and recomputes the observed frequency,
    giving a clustering-aware CI in place of the row-independent Wilson interval.
    ``y_prob`` is passed separately so calibrated probs work; ``frame`` supplies
    the cluster keys. Returns (prob_pred, prob_true, lo, hi, counts) over
    non-empty bins (ascending predicted prob), plus per-bin valid replicate
    counts when explicitly requested.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    _, binids = _fixed_calibration_bins(y_prob, n_bins, strategy)
    cnts = np.bincount(binids, minlength=n_bins).astype(float)
    kept = np.flatnonzero(cnts > 0)                                  # ascending non-empty bins
    with np.errstate(invalid="ignore"):
        prob_pred = (np.bincount(binids, weights=y_prob, minlength=n_bins) / cnts)[kept]
        prob_true = (np.bincount(binids, weights=y_true, minlength=n_bins) / cnts)[kept]
    counts = cnts[kept].astype(int)

    row_code = plan.row_code_for(frame)
    boot = np.full((plan.n_boot, len(kept)), np.nan)
    for b in range(plan.n_boot):
        idx = plan.replicate_indices(b, row_code)
        bidx = binids[idx]
        c = np.bincount(bidx, minlength=n_bins).astype(float)
        with np.errstate(invalid="ignore"):
            obs = np.bincount(bidx, weights=y_true[idx], minlength=n_bins) / c
        boot[b] = obs[kept]
    lo = np.nanpercentile(boot, (1 - alpha) / 2 * 100, axis=0)
    hi = np.nanpercentile(boot, (1 + alpha) / 2 * 100, axis=0)
    result = (prob_pred, prob_true, lo, hi, counts)
    if return_valid_counts:
        return (*result, np.isfinite(boot).sum(axis=0).astype(int))
    return result


def _net_benefit_matrix(frame, y_true, y_prob, dca_thresholds, plan: BootstrapPlan,
                        *, show_progress: bool = False):
    """Net-benefit replicate matrix under one cluster plan."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    dca_thresholds = np.asarray(dca_thresholds, dtype=float)
    row_code = plan.row_code_for(frame)
    mat = np.full((plan.n_boot, len(dca_thresholds)), np.nan)
    for b in _progress(range(plan.n_boot), plan.n_boot,
                       f"DCA({plan.cluster_unit})", show_progress):
        idx = plan.replicate_indices(b, row_code)
        yb, pb = y_true[idx], y_prob[idx]
        if len(np.unique(yb)) >= 2:
            mat[b] = _compute_net_benefit(yb, pb, dca_thresholds)
    return mat


def dca_nb_matrix_cluster(frame, y_true, y_prob, dca_thresholds, plan: BootstrapPlan):
    """Net-benefit bootstrap matrix for recomputing DCA on calibrated probs.

    Mirrors the utility pass of ``compute_bootstrap_store_cluster``; ``y_prob``
    may differ from ``frame``'s.
    """
    return _net_benefit_matrix(frame, y_true, y_prob, dca_thresholds, plan)


# Extraction helpers
def ci_bounds(dist: np.ndarray, alpha: float = CI_ALPHA) -> Tuple[float, float]:
    """Percentile CI from a bootstrap distribution."""
    lo = (1 - alpha) / 2 * 100
    hi = (1 + alpha) / 2 * 100
    valid = dist[~np.isnan(dist)]
    if len(valid) == 0:
        return (np.nan, np.nan)
    return (np.percentile(valid, lo), np.percentile(valid, hi))


def extract_metric_ci(
    store: BootstrapStore,
    metric: str,
    alpha: float = CI_ALPHA,
) -> Tuple[float, float, float]:
    """Extract (point, ci_lo, ci_hi) percentile CI for a scalar metric."""
    point = store.point_metrics[metric]
    dist = getattr(store, f"{metric}_dist")
    lo, hi = ci_bounds(dist, alpha)
    return (point, lo, hi)


def extract_all_metrics_ci(
    store: BootstrapStore,
    alpha: float = CI_ALPHA,
) -> Dict[str, Tuple[float, float, float]]:
    """Extract CI for all scalar metrics, plus derived metrics (AUPRC Lift, Prevalence)."""
    result = {m: extract_metric_ci(store, m, alpha) for m in _SCALAR_METRICS}

    # AUPRC Lift — the interval comes from the replicate-wise AUPRC*/prevalence*
    # distribution, because cluster draws move prevalence between replicates.
    prevalence = float(store.y_true.mean())
    auprc_point = result["auprc"][0]
    lift_dist = np.asarray(getattr(store, "auprc_lift_dist", np.array([])))
    if prevalence <= 0 or lift_dist.size == 0:
        raise ValueError(
            "AUPRC lift requires a replicate-wise lift distribution and positive prevalence"
        )
    lift_lo, lift_hi = ci_bounds(lift_dist, alpha)
    result["auprc_lift"] = (auprc_point / prevalence, lift_lo, lift_hi)

    # Prevalence of the observed analysis sample (descriptive; not resampled)
    result["prevalence"] = (prevalence, prevalence, prevalence)
    return result


def summarize_paired_deltas(point: float, deltas: np.ndarray,
                             alpha: float = CI_ALPHA) -> Tuple[float, float, float, float]:
    """Percentile CI + two-sided bootstrap p from a paired delta distribution.

    p = 2*(min(#d*>=0, #d*<=0)+1)/(B+1) over the non-nan replicates, capped at 1.0.
    """
    valid = deltas[~np.isnan(deltas)]
    if len(valid) == 0:
        return (point, np.nan, np.nan, np.nan)
    lo = float(np.percentile(valid, (1 - alpha) / 2 * 100))
    hi = float(np.percentile(valid, (1 + alpha) / 2 * 100))
    B = len(valid)
    n_ge = int(np.sum(valid >= 0))
    n_le = int(np.sum(valid <= 0))
    p = 2 * (min(n_ge, n_le) + 1) / (B + 1)        # +1 bias-corrected two-sided bootstrap p
    return (point, lo, hi, min(p, 1.0))


def _precomputed_bands(store: BootstrapStore, kind: str, alpha: float) -> Dict[str, np.ndarray]:
    """Band-only fallback for stores whose bootstrap matrices were dropped."""
    bands = getattr(store, "bands", None) or {}
    if kind not in bands:
        raise ValueError(
            f"Store has neither bootstrap matrices nor precomputed {kind} bands"
        )
    if bands.get("alpha") != alpha:
        raise ValueError(
            f"Precomputed {kind} bands use alpha={bands.get('alpha')}, requested {alpha}"
        )
    return bands[kind]


def extract_roc_bands(
    store: BootstrapStore,
    alpha: float = CI_ALPHA,
) -> Dict[str, np.ndarray]:
    """Extract ROC curve bands for plotting (band-only stores fall back to ``bands``)."""
    matrix = getattr(store, "roc_tpr_matrix", None)
    if matrix is None or np.asarray(matrix).size == 0:
        return _precomputed_bands(store, "roc", alpha)
    lo_pct = (1 - alpha) / 2 * 100
    hi_pct = (1 + alpha) / 2 * 100
    valid = matrix[~np.isnan(matrix).any(axis=1)]
    auc_lo, auc_hi = ci_bounds(store.auroc_dist, alpha)
    return {
        "fpr_grid": store.fpr_grid,
        "tpr_mean": np.nanmean(valid, axis=0),
        "tpr_lo": np.nanpercentile(valid, lo_pct, axis=0),
        "tpr_hi": np.nanpercentile(valid, hi_pct, axis=0),
        "tpr_point": np.interp(store.fpr_grid, store.roc_point["fpr"], store.roc_point["tpr"]),
        "auc_point": store.point_metrics["auroc"],
        "auc_lo": auc_lo,
        "auc_hi": auc_hi,
    }


def extract_pr_bands(
    store: BootstrapStore,
    alpha: float = CI_ALPHA,
) -> Dict[str, np.ndarray]:
    """Extract PR curve bands for plotting (band-only stores fall back to ``bands``)."""
    matrix = getattr(store, "pr_prec_matrix", None)
    if matrix is None or np.asarray(matrix).size == 0:
        return _precomputed_bands(store, "pr", alpha)
    lo_pct = (1 - alpha) / 2 * 100
    hi_pct = (1 + alpha) / 2 * 100
    valid = matrix[~np.isnan(matrix).any(axis=1)]
    rec_sorted = np.argsort(store.pr_point["recall"])
    ap_lo, ap_hi = ci_bounds(store.auprc_dist, alpha)
    return {
        "recall_grid": store.recall_grid,
        "prec_mean": np.nanmean(valid, axis=0),
        "prec_lo": np.nanpercentile(valid, lo_pct, axis=0),
        "prec_hi": np.nanpercentile(valid, hi_pct, axis=0),
        "prec_point": np.interp(
            store.recall_grid,
            store.pr_point["recall"][rec_sorted],
            store.pr_point["precision"][rec_sorted],
        ),
        "ap_point": store.point_metrics["auprc"],
        "ap_lo": ap_lo,
        "ap_hi": ap_hi,
        "prevalence": store.y_true.mean(),
    }


def extract_dca_bands(
    store: BootstrapStore,
    alpha: float = CI_ALPHA,
) -> Dict[str, np.ndarray]:
    """Extract DCA net benefit bands for plotting (band-only stores fall back to ``bands``)."""
    matrix = getattr(store, "dca_nb_matrix", None)
    if matrix is None or np.asarray(matrix).size == 0:
        return _precomputed_bands(store, "dca", alpha)
    lo_pct = (1 - alpha) / 2 * 100
    hi_pct = (1 + alpha) / 2 * 100
    valid = matrix[~np.isnan(matrix).any(axis=1)]
    return {
        "thresholds": store.dca_thresholds,
        "nb_mean": np.nanmean(valid, axis=0),
        "nb_lo": np.nanpercentile(valid, lo_pct, axis=0),
        "nb_hi": np.nanpercentile(valid, hi_pct, axis=0),
        "nb_point": store.dca_point["net_benefit"],
        "treat_all": store.dca_point["treat_all"],
        "prevalence": store.dca_point["prevalence"],
    }


def attach_band_summaries(stores_by_split: dict, alpha: float = CI_ALPHA) -> None:
    """Precompute ROC/PR/DCA summaries while bootstrap matrices are available."""
    for by_model in stores_by_split.values():
        for by_horizon in by_model.values():
            for store in by_horizon.values():
                store.bands = {
                    "alpha": alpha,
                    "roc": extract_roc_bands(store, alpha),
                    "pr": extract_pr_bands(store, alpha),
                    "dca": extract_dca_bands(store, alpha),
                    "replicate_audit": store_replicate_audit(store),
                }


def _replicate_count(array) -> dict[str, int]:
    """Return generated and fully finite replicate counts for one result array."""
    values = np.asarray(array)
    if values.ndim == 0:
        raise ValueError("Bootstrap replicate arrays must have at least one dimension")
    generated = int(values.shape[0])
    finite = np.isfinite(values)
    if values.ndim > 1:
        finite = finite.reshape(generated, -1).all(axis=1)
    return {
        "generated": generated,
        "valid": int(finite.sum()),
    }


def store_replicate_audit(store: BootstrapStore) -> dict[str, dict[str, int]]:
    """Return per-metric generated/valid counts, including band-only stores."""
    cached = (getattr(store, "bands", None) or {}).get("replicate_audit")
    if cached is not None:
        return {
            metric: {
                "generated": int(counts["generated"]),
                "valid": int(counts["valid"]),
            }
            for metric, counts in cached.items()
        }
    arrays = {
        metric: getattr(store, f"{metric}_dist")
        for metric in _SCALAR_METRICS
    }
    arrays.update({
        "auprc_lift": store.auprc_lift_dist,
        "roc_band": store.roc_tpr_matrix,
        "pr_band": store.pr_prec_matrix,
        "dca_band": store.dca_nb_matrix,
    })
    audit = {
        metric: _replicate_count(array)
        for metric, array in arrays.items()
    }
    generated = {counts["generated"] for counts in audit.values()}
    if len(generated) != 1 or next(iter(generated)) < 1:
        raise ValueError("Bootstrap store arrays do not share one positive replicate count")
    return audit


def split_replicate_audit(audit: dict[str, dict[str, int]]) -> dict[str, dict]:
    """Split one store's per-metric audit into its two resampling families."""
    unknown = set(audit) - set(DISCRIMINATION_AUDIT_METRICS) - set(UTILITY_AUDIT_METRICS)
    if unknown:
        raise ValueError(f"Replicate audit carries unassigned metrics: {sorted(unknown)}")
    return {
        "discrimination": {
            metric: audit[metric]
            for metric in DISCRIMINATION_AUDIT_METRICS if metric in audit
        },
        "utility": {
            metric: audit[metric]
            for metric in UTILITY_AUDIT_METRICS if metric in audit
        },
    }
