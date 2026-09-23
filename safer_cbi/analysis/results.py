"""Validated, identifier-free aggregate result frames."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ..contracts import MODEL_HORIZONS, MODEL_ORDER, PRIMARY_HORIZON
from .settings import GRID_POINTS, MANUSCRIPT_DISCRIMINATION_SCHEDULE, SPLITS


SCHEMA_VERSION = 15
RESULT_CONTRACT_VERSION = 3
CALIBRATION_RUG_QUANTILES = 4_096
LEARNED_MODELS = tuple(model for model in MODEL_ORDER if model != "lgbm")
PERFORMANCE_METRICS = (
    "auroc", "auprc", "auprc_lift", "f1",
    "precision", "sensitivity", "balanced_acc", "prevalence",
)
# Every non-primary cell in the schedule, grouped by model. The lgbm and gru
# comparators carry the same secondary horizons as multitask; single is defined
# at the primary horizon only and therefore never appears here.
SECONDARY_CELLS = {
    model: [
        horizon
        for scheduled, horizon in MANUSCRIPT_DISCRIMINATION_SCHEDULE
        if scheduled == model and horizon != PRIMARY_HORIZON
    ]
    for model in MODEL_ORDER
}
SECONDARY_STORE_MODELS = tuple(
    model for model, horizons in SECONDARY_CELLS.items() if horizons
)
SECONDARY_HORIZONS = tuple(SECONDARY_CELLS["multitask"])
PRIMARY_MODELS = tuple(
    model
    for model, horizon in MANUSCRIPT_DISCRIMINATION_SCHEDULE
    if horizon == PRIMARY_HORIZON
)
PARTICIPANT_CLUSTER_UNIT = "participant"
EPISODE_CLUSTER_UNIT = "episode"
# Two prespecified stratified cluster resampling families. Each draws clusters
# with replacement within its strata and carries every subordinate unit of a
# drawn cluster at the same multiplicity. No result file mixes the two.
# Both plans are built once per cohort from this horizon's frame and shared by
# every model and horizon; cross-horizon identity is enforced at load.
PLAN_FRAME_HORIZON = PRIMARY_HORIZON
RESAMPLING_FAMILIES = {
    "performance": {
        "cluster_unit": PARTICIPANT_CLUSTER_UNIT,
        "cluster_key": "key",
        "strata": "participant contributed an observed CBI episode, or did not",
        "varies_across_replicates": ["episodes", "epochs", "epoch prevalence"],
        "fixed_across_replicates": ["participants drawn per stratum"],
        "result_files": [
            "performance.csv", "comparisons.csv", "curves.csv", "sensitivity.csv",
        ],
    },
    "reliability_utility": {
        "cluster_unit": EPISODE_CLUSTER_UNIT,
        "cluster_key": "key|event_index",
        "strata": "CBI episode, or control episode",
        "varies_across_replicates": [
            "unique episodes", "epochs", "epoch prevalence",
        ],
        "fixed_across_replicates": [
            "episodes drawn per stratum", "the 1:6 drawn case-control ratio",
        ],
        "result_files": ["calibration.csv", "dca.csv"],
    },
}
# Reported files that carry no cluster-bootstrap interval. Enumerated so the
# contract records why each one is unaffected by the resampling change.
UNRESAMPLED_RESULT_FILES = {
    "feature_dynamics.csv": "event-level bootstrap and cluster permutation",
    "lmm.csv": "model-based inference from episode-level random effects",
    "operating_points.csv": "no bootstrap interval",
    "calibration_rug.csv": "no bootstrap interval",
    "feature_importance.csv": "no bootstrap interval",
    "feature_concordance.csv": "no bootstrap interval",
    "fold_performance.csv": "within-fold point estimates; no bootstrap interval",
}
# Discovery cross-validation read-out. Point estimates only: the fold is the
# unit that the participant-cluster bootstrap already resamples over, so a
# within-fold interval would restate the same uncertainty on a smaller sample.
FOLD_SCOPE = {
    "split": "internal",
    "folds": 5,
    "fold_source": "patient-grouped cross-validation fold that held the row out",
    "model_horizons": {
        model: list(MODEL_HORIZONS[model]) for model in MODEL_ORDER
    },
    "metrics": list(PERFORMANCE_METRICS),
    "threshold": "sealed discovery threshold, never re-optimized within a fold",
    "expected_rows": (
        sum(len(MODEL_HORIZONS[model]) for model in MODEL_ORDER)
        * 5
        * len(PERFORMANCE_METRICS)
    ),
}
MANUSCRIPT_STORE_SCHEDULE = {
    "stores": {
        "discrimination_unit": PARTICIPANT_CLUSTER_UNIT,
        "utility_unit": EPISODE_CLUSTER_UNIT,
        "by_split": {
            split: {
                model: list(SECONDARY_CELLS[model])
                for model in SECONDARY_STORE_MODELS
            }
            for split in SPLITS
        },
    },
    "comparison_stores": {
        "discrimination_unit": PARTICIPANT_CLUSTER_UNIT,
        "utility_unit": EPISODE_CLUSTER_UNIT,
        "by_split": {
            split: {model: [PRIMARY_HORIZON] for model in PRIMARY_MODELS}
            for split in SPLITS
        },
    },
}
CALIBRATION_SCOPE = {
    "raw": {
        "splits": list(SPLITS),
        "model_horizons": {model: [PRIMARY_HORIZON] for model in LEARNED_MODELS},
        "strategy": "quantile",
        "bins_per_slice": 8,
        "expected_rows": 48,
    },
    "beta": {
        "splits": list(SPLITS),
        "model_horizons": {
            model: list(MODEL_HORIZONS[model]) for model in LEARNED_MODELS
        },
        "validation_unit": "episode",
        "strategy": "quantile",
        "bins_per_slice": 8,
        "expected_slices": 26,
        "expected_rows": 208,
    },
    "expected_rows": 256,
}
DCA_SCOPE = {
    "raw": {
        "splits": list(SPLITS), "models": list(LEARNED_MODELS),
        "horizon": PRIMARY_HORIZON,
        "points_per_slice": dict.fromkeys(SPLITS, 100),
        "expected_rows": 600,
    },
    "beta": {
        "splits": list(SPLITS), "models": list(LEARNED_MODELS),
        "horizon": PRIMARY_HORIZON,
        "points_per_slice": {"internal": 139, "external": 133},
        "expected_rows": 816,
    },
    "expected_rows": 1_416,
}
INFERENCE_CONTRACT = {
    "contract_version": RESULT_CONTRACT_VERSION,
    "resampling_families": RESAMPLING_FAMILIES,
    "plan_frame_horizon": PLAN_FRAME_HORIZON,
    # Cluster counts per stratum are fixed; epoch counts are not.
    "epoch_prevalence": "varies by replicate in both families",
    "episode_count": "varies by replicate in the performance family",
    # Secondary-horizon stores fill a net-benefit matrix that no result file
    # reads; it is retained so every store carries the same band and audit
    # surface. Only the 24-hour scope below is emitted.
    "secondary_horizon_net_benefit": "computed, audited, not emitted",
    "unresampled_result_files": UNRESAMPLED_RESULT_FILES,
    "fold_scope": FOLD_SCOPE,
    "bootstrap_replicates": {
        "generated": 5_000,
        "promotion_minimum_valid": 4_990,
        "exception_review_range": [4_990, 4_999],
    },
    "store_schedule": MANUSCRIPT_STORE_SCHEDULE,
    "calibration_scope": CALIBRATION_SCOPE,
    "dca_scope": DCA_SCOPE,
}
SENSITIVITY_SCOPE = {
    "model": "multitask",
    "horizon": "24hr",
    "splits": ["external"],
    "conditions": ["Original", "15 min", "1 hr"],
    "metrics": ["auroc", "auprc", "f1"],
}
FORBIDDEN_COLUMNS = {
    "key",
    "patient_id",
    "event_index",
    "time",
    "y_true",
    "y_prob",
}

RESULT_COLUMNS = {
    "performance.csv": (
        "split", "model", "horizon", "metric", "scope",
        "estimate", "ci_lower", "ci_upper",
    ),
    "comparisons.csv": (
        "split", "model_a", "model_b", "metric", "horizon", "scope",
        "difference", "ci_lower", "ci_upper", "p_value", "q_value",
    ),
    "operating_points.csv": (
        "model", "split", "horizon", "operating_point", "threshold",
        "prevalence", "recall", "f1", "episode_recall",
        "timely", "specificity", "false_detection",
    ),
    "curves.csv": (
        "split", "model", "horizon", "curve", "grid",
        "estimate", "bootstrap_mean", "ci_lower", "ci_upper",
    ),
    "fold_performance.csv": (
        "split", "model", "fold", "horizon", "metric", "estimate",
    ),
    "calibration.csv": (
        "split", "method", "model", "horizon", "strategy", "bin",
        "predicted", "observed", "ci_lower", "ci_upper", "count",
    ),
    "dca.csv": (
        "split", "method", "model", "horizon", "threshold",
        "net_benefit", "bootstrap_mean", "ci_lower", "ci_upper",
        "treat_all", "prevalence",
    ),
    "calibration_rug.csv": (
        "split", "model", "horizon", "quantile", "predicted",
    ),
    "feature_importance.csv": (
        "method", "split", "feature", "category", "importance", "sd",
        "rank", "epoch_weight",
    ),
    "feature_concordance.csv": (
        "comparison_axis", "method_a", "split_a", "method_b", "split_b",
        "metric", "n_features", "estimate", "p_value",
    ),
    "feature_dynamics.csv": (
        "feature", "category", "hours_to_event", "estimate", "ci_lower",
        "ci_upper", "cluster_significant",
    ),
    "lmm.csv": (
        "feature", "category", "term", "estimate", "std_error", "df",
        "statistic", "p_value", "ci_lower", "ci_upper", "q_group",
        "q_interaction", "ar1_rho", "fallback",
        "robust_fallback", "inference",
    ),
    "sensitivity.csv": (
        "split", "model", "horizon", "condition", "metric", "estimate",
        "ci_lower", "ci_upper", "p_value",
    ),
}

MACHINE_COLUMNS = {
    "split", "model", "horizon", "metric", "scope", "model_a", "model_b",
    "operating_point", "curve", "method", "strategy", "feature", "category",
    "comparison_axis", "method_a", "split_a", "method_b", "split_b", "term",
    "inference", "condition",
}
BOOLEAN_COLUMNS = {"cluster_significant", "fallback", "robust_fallback"}


def validate_result_frame(filename: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce the public aggregate schema before any file is written."""
    if filename not in RESULT_COLUMNS:
        raise ValueError(f"Unknown aggregate result: {filename}")
    forbidden = FORBIDDEN_COLUMNS & set(frame.columns)
    if forbidden:
        raise ValueError(
            f"{filename} contains row-level identifiers or predictions: {sorted(forbidden)}"
        )
    expected = list(RESULT_COLUMNS[filename])
    if list(frame.columns) != expected:
        raise ValueError(
            f"{filename} columns differ from schema: "
            f"expected={expected}, actual={list(frame.columns)}"
        )
    if not frame.empty:
        for column in frame.columns:
            if column in MACHINE_COLUMNS:
                if frame[column].map(
                    lambda value: isinstance(value, (dict, list, tuple, set))
                ).any():
                    raise TypeError(f"{filename}.{column} is not a scalar machine ID")
            elif column in BOOLEAN_COLUMNS:
                if not is_bool_dtype(frame[column]):
                    raise TypeError(f"{filename}.{column} must be boolean")
            elif not is_numeric_dtype(frame[column]):
                raise TypeError(f"{filename}.{column} must be numeric")
        if filename == "sensitivity.csv":
            expected = {
                (split, SENSITIVITY_SCOPE["model"], SENSITIVITY_SCOPE["horizon"], condition, metric)
                for split in SENSITIVITY_SCOPE["splits"]
                for condition in SENSITIVITY_SCOPE["conditions"]
                for metric in SENSITIVITY_SCOPE["metrics"]
            }
            observed = set(frame[[
                "split", "model", "horizon", "condition", "metric",
            ]].itertuples(index=False, name=None))
            if observed != expected or len(frame) != len(expected):
                raise ValueError("sensitivity.csv differs from the temporal primary-model contract")
    return frame


def _manuscript_domain_specs() -> tuple:
    """Derive the frozen (file, keys, rows, expected keys, rows per key) contracts."""
    from .calibration import beta_calibration_domain

    discrimination = {
        (split, model, horizon)
        for split in SPLITS
        for model, horizon in MANUSCRIPT_DISCRIMINATION_SCHEDULE
    }
    raw_primary = {
        (split, "raw", model, PRIMARY_HORIZON)
        for split in SPLITS
        for model in LEARNED_MODELS
    }
    return (
        (
            "performance.csv", ["split", "model", "horizon", "metric"],
            len(discrimination) * len(PERFORMANCE_METRICS),
            {(*key, metric) for key in discrimination for metric in PERFORMANCE_METRICS},
            lambda key: 1,
        ),
        (
            "curves.csv", ["split", "model", "horizon", "curve"],
            len(discrimination) * 2 * GRID_POINTS,
            {(*key, curve) for key in discrimination for curve in ("roc", "pr")},
            lambda key: GRID_POINTS,
        ),
        (
            "fold_performance.csv", ["split", "model", "fold", "horizon"],
            FOLD_SCOPE["expected_rows"],
            {
                (FOLD_SCOPE["split"], model, fold, horizon)
                for model, horizons in FOLD_SCOPE["model_horizons"].items()
                for horizon in horizons
                for fold in range(1, FOLD_SCOPE["folds"] + 1)
            },
            lambda key: len(PERFORMANCE_METRICS),
        ),
        (
            "calibration.csv", ["split", "method", "model", "horizon"], 256,
            raw_primary | {
                (split, "beta", model, horizon)
                for split, model, horizon in beta_calibration_domain()
            },
            lambda key: CALIBRATION_SCOPE[key[1]]["bins_per_slice"],
        ),
        (
            "dca.csv", ["split", "method", "model", "horizon"], 1_416,
            raw_primary | {
                (split, "beta", model, PRIMARY_HORIZON)
                for split in SPLITS
                for model in LEARNED_MODELS
            },
            lambda key: DCA_SCOPE[key[1]]["points_per_slice"][key[0]],
        ),
        (
            "calibration_rug.csv", ["split", "model", "horizon"],
            len(beta_calibration_domain()) * CALIBRATION_RUG_QUANTILES,
            set(beta_calibration_domain()),
            lambda key: CALIBRATION_RUG_QUANTILES,
        ),
    )


def validate_manuscript_domains(
    frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Validate the exact discrimination, calibration, and DCA result domains."""
    for filename, keys, rows, expected, rows_per_key in _manuscript_domain_specs():
        frame = frames[filename]
        counts = frame.groupby(keys, sort=False).size()
        if (
            len(frame) != rows
            or set(counts.index) != expected
            or counts.to_dict() != {key: rows_per_key(key) for key in expected}
        ):
            raise ValueError(
                f"{filename} differs from the exact {rows:,}-row domain"
            )
    if set(frames["calibration.csv"]["bin"]) != set(range(8)):
        raise ValueError("calibration.csv differs from the exact 256-row domain")
    return frames


def performance_frames(bundle) -> dict[str, pd.DataFrame]:
    """Build discrimination, comparison, and operating-point result frames."""
    from . import settings
    from .performance import (
        comparison_long_format,
        operating_point_long_format,
        perf_long_format,
        run_paired_model_comparison,
    )

    models = [
        model for model in settings.model_order()
        if model in bundle.comparison_stores["external"]
    ]
    comparisons = run_paired_model_comparison(
        bundle,
        splits=settings.SPLITS,
        horizons=(settings.PRIMARY_HORIZON,),
        metrics=("auroc", "auprc"),
        pairs=[("multitask", model) for model in models if model != "multitask"],
    )
    comparisons = comparison_long_format(comparisons).rename(
        columns={"estimate": "difference"}
    )
    return {
        "performance.csv": perf_long_format(bundle),
        "comparisons.csv": comparisons,
        "operating_points.csv": operating_point_long_format(
            bundle, model="multitask"
        ),
    }


def curve_frame(bundle) -> pd.DataFrame:
    """Flatten the exact manuscript ROC/PR schedule and its participant bands."""
    from . import settings
    from .bootstrap_engine import extract_pr_bands, extract_roc_bands
    from .performance import manuscript_discrimination_stores

    rows = []
    for split in settings.SPLITS:
        selected = manuscript_discrimination_stores(bundle, split)
        for model in settings.model_order():
            for horizon, store in selected.get(model, {}).items():
                for curve, extract, grid_key, value in (
                    ("roc", extract_roc_bands, "fpr_grid", "tpr"),
                    ("pr", extract_pr_bands, "recall_grid", "prec"),
                ):
                    bands = extract(store)
                    for index, grid in enumerate(bands[grid_key]):
                        rows.append({
                            "split": split, "model": model, "horizon": horizon,
                            "curve": curve, "grid": grid,
                            "estimate": bands[f"{value}_point"][index],
                            "bootstrap_mean": bands[f"{value}_mean"][index],
                            "ci_lower": bands[f"{value}_lo"][index],
                            "ci_upper": bands[f"{value}_hi"][index],
                        })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS["curves.csv"])


def fold_performance_frame(bundle) -> pd.DataFrame:
    """Point estimates inside each discovery cross-validation fold.

    The sealed discovery threshold is read off unchanged within every fold, and
    lift divides a fold's AUPRC by that same fold's positive-epoch prevalence.
    No interval is reported: participants are the bootstrap cluster, and these
    folds partition the very participants that bootstrap already resamples.
    """
    from .bootstrap_engine import scalar_metrics

    split = FOLD_SCOPE["split"]
    fold_index = bundle.folds.get(split)
    if fold_index is None:
        raise ValueError(f"{split} frames carry no cross-validation fold")
    rows = []
    for model, horizons in FOLD_SCOPE["model_horizons"].items():
        for horizon in horizons:
            frame = bundle.frames[split][model][horizon]
            if len(fold_index) != len(frame):
                raise ValueError(
                    f"[{split}/{model}/{horizon}] fold vector covers "
                    f"{len(fold_index)} rows, frame has {len(frame)}"
                )
            threshold = bundle.thresholds[model][horizon]
            y_true = frame["y_true"].to_numpy(dtype=int)
            y_prob = frame["y_prob"].to_numpy(dtype=float)
            for fold in range(1, FOLD_SCOPE["folds"] + 1):
                held = fold_index == fold - 1
                metrics = scalar_metrics(y_true[held], y_prob[held], threshold)
                metrics["prevalence"] = float(y_true[held].mean())
                metrics["auprc_lift"] = metrics["auprc"] / metrics["prevalence"]
                rows.extend(
                    {
                        "split": split, "model": model, "fold": fold,
                        "horizon": horizon, "metric": metric,
                        "estimate": metrics[metric],
                    }
                    for metric in PERFORMANCE_METRICS
                )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS["fold_performance.csv"])


def calibration_frame(
    bundle,
    *,
    n_boot: int,
    seed: int,
    calibration_grid,
) -> tuple[pd.DataFrame, dict]:
    """Flatten raw and beta reliability estimates with their per-bin audits."""
    from . import settings
    from .calibration import aligned_slice, beta_calibration_domain
    from .bootstrap_engine import EPISODE_UNIT, binned_calibration_cluster_ci

    primary = settings.PRIMARY_HORIZON
    for split in SPLITS:
        plan = bundle.plan(split, EPISODE_UNIT)
        if plan.n_boot != n_boot or plan.seed != seed:
            raise ValueError(
                f"[{split}] episode plan ({plan.n_boot} draws, seed {plan.seed}) does not "
                f"match the requested calibration inference ({n_boot} draws, seed {seed})"
            )
    learned = [
        model for model in settings.model_order()
        if model != "lgbm" and model in bundle.comparison_stores["external"]
    ]
    specifications = []
    for split in SPLITS:
        stores = bundle.comparison_view(split, primary, models=learned)
        specifications.extend(
            (split, "raw", model, primary, store)
            for model, store in stores.items()
        )
    specifications.extend(
        (
            split, "beta", model, horizon,
            aligned_slice(
                calibration_grid,
                bundle.frames[split][model][horizon],
                split, model, horizon,
            ),
        )
        for split, model, horizon in beta_calibration_domain()
    )

    rows = []
    replicate_audit = {}
    for split, method, model, horizon, store in specifications:
        frame = bundle.frames[split][model][horizon]
        if method == "raw" and not (
            np.array_equal(frame["y_true"].to_numpy(dtype=int), store.y_true)
            and np.array_equal(frame["y_prob"].to_numpy(dtype=float), store.y_prob)
        ):
            raise ValueError(
                f"Calibration frame/store drift at {split}/{method}/{model}/{horizon}"
            )
        values = binned_calibration_cluster_ci(
            frame,
            store.y_true,
            store.y_prob,
            8,
            "quantile",
            bundle.plan(split, EPISODE_UNIT),
            return_valid_counts=True,
        )
        if not all(len(part) == 8 for part in values):
            raise ValueError(
                f"Calibration slice {split}/{method}/{model}/{horizon} "
                "does not contain exactly 8 nonempty bins"
            )
        predicted, observed, low, high, counts, valid_counts = values
        replicate_audit[f"{split}/{method}/{model}/{horizon}"] = {
            f"bin_{index}": {"generated": int(n_boot), "valid": int(valid)}
            for index, valid in enumerate(valid_counts)
        }
        for index, (p_hat, p_obs, lo, hi, count) in enumerate(
            zip(predicted, observed, low, high, counts)
        ):
            rows.append({
                "split": split, "method": method, "model": model,
                "horizon": horizon, "strategy": "quantile", "bin": index,
                "predicted": p_hat, "observed": p_obs,
                "ci_lower": lo, "ci_upper": hi, "count": count,
            })
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS["calibration.csv"])
    return frame, replicate_audit


def dca_frame(bundle, calibrated_stores) -> pd.DataFrame:
    """Flatten the net-benefit point curves and bootstrap bands.

    Both families cover the learned models on both splits at 24 hours; lgbm is a
    covariate floor without a deployment curve.
    """
    from . import settings
    from .bootstrap_engine import extract_dca_bands

    primary = settings.PRIMARY_HORIZON

    def _raw_models(split):
        return [
            model for model in settings.model_order()
            if model != "lgbm" and model in bundle.comparison_stores[split]
        ]

    specifications = [
        (split, "raw", bundle.comparison_view(split, primary, models=_raw_models(split)))
        for split in settings.SPLITS
    ] + [
        (split, "beta", calibrated_stores[split]) for split in settings.SPLITS
    ]

    rows = []
    for split, method, stores in specifications:
        for model, store in stores.items():
            bands = extract_dca_bands(store)
            for index, threshold in enumerate(bands["thresholds"]):
                rows.append({
                    "split": split, "method": method, "model": model,
                    "horizon": primary, "threshold": threshold,
                    "net_benefit": bands["nb_point"][index],
                    "bootstrap_mean": bands["nb_mean"][index],
                    "ci_lower": bands["nb_lo"][index],
                    "ci_upper": bands["nb_hi"][index],
                    "treat_all": bands["treat_all"][index],
                    "prevalence": bands["prevalence"],
                })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS["dca.csv"])


def calibration_rug_frame(bundle, *, calibration_grid) -> pd.DataFrame:
    """Summarise each beta slice's calibrated scores as a fixed quantile ladder.

    Rows are order statistics of a whole slice, never tied to a single epoch.
    """
    from .calibration import aligned_slice, beta_calibration_domain

    levels = np.linspace(0.0, 1.0, CALIBRATION_RUG_QUANTILES)
    rows = []
    for split, model, horizon in beta_calibration_domain():
        item = aligned_slice(
            calibration_grid,
            bundle.frames[split][model][horizon],
            split, model, horizon,
        )
        rows.extend(
            {"split": split, "model": model, "horizon": horizon,
             "quantile": level, "predicted": predicted}
            for level, predicted in zip(levels, np.quantile(item.y_prob, levels))
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS["calibration_rug.csv"])


def build_result_frames(
    bundle,
    calibrated_stores,
    *,
    feature_importance: pd.DataFrame,
    feature_concordance: pd.DataFrame,
    feature_dynamics: pd.DataFrame,
    lmm: pd.DataFrame,
    sensitivity: pd.DataFrame,
    n_boot: int,
    seed: int,
    calibration_grid,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Assemble every aggregate numeric slice into the complete result set."""
    frames = performance_frames(bundle)
    calibration, calibration_audit = calibration_frame(
        bundle,
        n_boot=n_boot,
        seed=seed,
        calibration_grid=calibration_grid,
    )
    frames.update({
        "curves.csv": curve_frame(bundle),
        "fold_performance.csv": fold_performance_frame(bundle),
        "calibration.csv": calibration,
        "calibration_rug.csv": calibration_rug_frame(
            bundle, calibration_grid=calibration_grid
        ),
        "dca.csv": dca_frame(bundle, calibrated_stores),
        "feature_importance.csv": feature_importance,
        "feature_concordance.csv": feature_concordance,
        "feature_dynamics.csv": feature_dynamics,
        "lmm.csv": lmm,
        "sensitivity.csv": sensitivity,
    })
    return validate_manuscript_domains(frames), calibration_audit
