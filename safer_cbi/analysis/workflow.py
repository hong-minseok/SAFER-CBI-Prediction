"""Checkpoint-free numeric analysis composition."""
from __future__ import annotations

from .settings import (
    CALIBRATION_METHOD,
    MANUSCRIPT_DISCRIMINATION_SCHEDULE,
    PRIMARY_HORIZON,
    SPLITS,
)


def _validated_thresholds(inputs, thresholds) -> dict[str, dict[str, float]]:
    """Validate explicit frozen thresholds against the aligned model frames."""
    active_models = tuple(inputs["active_models"])
    if set(thresholds) != set(active_models):
        raise ValueError(
            "Frozen threshold models differ from active models: "
            f"thresholds={sorted(thresholds)}, active={sorted(active_models)}"
        )
    validated = {}
    for model in active_models:
        model_thresholds = dict(thresholds[model])
        model_horizons = set(inputs["aligned"]["internal"]["frames"][model])
        if set(model_thresholds) != model_horizons:
            raise ValueError(
                f"{model} frozen threshold set does not match prediction horizons"
            )
        validated[model] = model_thresholds
    return validated


def _store_keys(stores) -> set[tuple[str, str, str]]:
    """Return ``(split, model, horizon)`` keys from a nested store mapping."""
    return {
        (split, model, horizon)
        for split, by_model in stores.items()
        for model, by_horizon in by_model.items()
        for horizon in by_horizon
    }


def _validate_discrimination_schedule(stores, comparison_stores) -> None:
    """Require the exact 18-key manuscript discrimination store schedule."""
    marginal_keys = _store_keys(stores)
    comparison_keys = _store_keys(comparison_stores)
    overlap = marginal_keys & comparison_keys
    if overlap:
        raise RuntimeError(
            f"Discrimination store families overlap: {sorted(overlap)}"
        )
    expected = {
        (split, model, horizon)
        for split in SPLITS
        for model, horizon in MANUSCRIPT_DISCRIMINATION_SCHEDULE
    }
    actual = marginal_keys | comparison_keys
    if actual != expected:
        raise RuntimeError(
            "Manuscript discrimination schedule differs: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _replicate_audit(
    bundle,
    calibrated_stores,
    calibration_bin_ci,
    sensitivity_audit,
    *,
    n_boot: int,
) -> dict:
    """Collect PHI-free valid counts per resampling family.

    Store replicates are reported under their own family: the discrimination
    metrics come from the participant plan and the net-benefit band from the
    episode plan. Beta DCA and calibration bins belong to the episode family.
    """
    import numpy as np

    from . import results
    from .bootstrap_engine import (
        EPISODE_UNIT,
        PARTICIPANT_UNIT,
        _replicate_count,
        split_replicate_audit,
        store_replicate_audit,
    )
    from .performance import manuscript_discrimination_stores

    stores = {}
    for split in SPLITS:
        selected = manuscript_discrimination_stores(bundle, split)
        for model, by_horizon in selected.items():
            for horizon, store in by_horizon.items():
                stores[f"{split}/{model}/{horizon}"] = split_replicate_audit(
                    store_replicate_audit(store)
                )
    beta_dca = {}
    for split, by_model in calibrated_stores.items():
        for model, store in by_model.items():
            key = f"{split}/{model}/{PRIMARY_HORIZON}"
            matrix = np.asarray(store.dca_nb_matrix)
            if matrix.ndim != 2 or matrix.shape[0] != n_boot:
                raise ValueError(f"Beta DCA replicate matrix differs at {key}")
            beta_dca[key] = {"dca_band": _replicate_count(matrix)}

    expected_calibration = {
        f"{split}/raw/{model}/{PRIMARY_HORIZON}"
        for split in results.CALIBRATION_SCOPE["raw"]["splits"]
        for model in results.CALIBRATION_SCOPE["raw"]["model_horizons"]
    } | {
        f"{split}/beta/{model}/{horizon}"
        for split in results.CALIBRATION_SCOPE["beta"]["splits"]
        for model, horizons in results.CALIBRATION_SCOPE[
            "beta"
        ]["model_horizons"].items()
        for horizon in horizons
    }
    expected_sensitivity = {
        f"{split}/{results.SENSITIVITY_SCOPE['model']}/"
        f"{results.SENSITIVITY_SCOPE['horizon']}"
        for split in results.SENSITIVITY_SCOPE["splits"]
    }
    expected_sensitivity_metrics = {
        f"{condition}/{metric}"
        for condition in results.SENSITIVITY_SCOPE["conditions"]
        for metric in results.SENSITIVITY_SCOPE["metrics"]
    }
    if (
        not isinstance(sensitivity_audit, dict)
        or set(sensitivity_audit) != expected_sensitivity
        or any(
            set(metrics) != expected_sensitivity_metrics
            for metrics in sensitivity_audit.values()
        )
    ):
        raise ValueError("Lead-time sensitivity replicate audit domain changed")

    if (
        not isinstance(calibration_bin_ci, dict)
        or set(calibration_bin_ci) != expected_calibration
        or any(
            set(metrics) != {f"bin_{index}" for index in range(8)}
            for metrics in calibration_bin_ci.values()
        )
    ):
        raise ValueError("Calibration bin-CI replicate audit domain changed")

    replicates = results.INFERENCE_CONTRACT["bootstrap_replicates"]
    locked = n_boot == replicates["generated"]
    minimum_valid = replicates["promotion_minimum_valid"]
    blockers = []
    requires_exception = []

    def _check(path, metrics):
        for metric, counts in metrics.items():
            if counts["generated"] != n_boot:
                raise ValueError(
                    f"{path}/{metric} generated "
                    f"{counts['generated']} replicates, expected {n_boot}"
                )
            if locked and counts["valid"] < n_boot:
                target = (
                    blockers if counts["valid"] < minimum_valid
                    else requires_exception
                )
                target.append(f"{path}/{metric}")

    for key, by_family in stores.items():
        for family, metrics in by_family.items():
            _check(f"stores/{key}/{family}", metrics)
    for record_family, records in (
        ("beta_dca", beta_dca),
        ("calibration_bin_ci", calibration_bin_ci),
        ("sensitivity", sensitivity_audit),
    ):
        for key, metrics in records.items():
            _check(f"{record_family}/{key}", metrics)
    if blockers:
        raise RuntimeError(
            f"Promotion-blocking valid replicate counts below {minimum_valid:,}: "
            f"{blockers}"
        )
    return {
        "n_boot": int(n_boot),
        "promotion_minimum_valid": minimum_valid if locked else None,
        "resampling_families": {
            "stores/discrimination": PARTICIPANT_UNIT,
            "stores/utility": EPISODE_UNIT,
            "beta_dca": EPISODE_UNIT,
            "calibration_bin_ci": EPISODE_UNIT,
            "sensitivity": PARTICIPANT_UNIT,
        },
        "stores": stores,
        "beta_dca": beta_dca,
        "calibration_bin_ci": calibration_bin_ci,
        "sensitivity": sensitivity_audit,
        "requires_exception": requires_exception,
    }


def build_bundle(
    inputs,
    *,
    thresholds,
    n_boot,
    seed=42,
    n_jobs=-2,
    show_progress=True,
):
    """Build the exact 38-key manuscript store schedule.

    Each split gets both contract plans — participant clusters for
    discrimination, episode clusters for net benefit — built once and shared
    across every model and horizon. All data and analysis contracts are explicit
    inputs; this module performs no filesystem reads and owns no resumable state.
    """
    from . import results
    from .bootstrap_engine import (
        AnalysisBundle,
        attach_band_summaries,
        build_split_plans,
        compute_all_stores_cluster,
    )

    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    bundle = AnalysisBundle()
    for split in SPLITS:
        aligned = inputs["aligned"][split]
        bundle.frames[split] = aligned["frames"]
        bundle.folds[split] = aligned["fold_index"]
        bundle.resampling[split] = build_split_plans(
            aligned["frames"]["multitask"][PRIMARY_HORIZON],
            n_boot=n_boot,
            seed=seed,
            data_fingerprint=aligned["fingerprint"],
        )

    bundle.thresholds = _validated_thresholds(inputs, thresholds)
    bundle.stores = compute_all_stores_cluster(
        bundle,
        n_jobs=n_jobs,
        show_progress=show_progress,
        models=results.SECONDARY_STORE_MODELS,
        horizons=results.SECONDARY_HORIZONS,
    )
    bundle.comparison_stores = compute_all_stores_cluster(
        bundle,
        n_jobs=n_jobs,
        show_progress=show_progress,
        models=results.PRIMARY_MODELS,
        horizons=(PRIMARY_HORIZON,),
    )
    _validate_discrimination_schedule(bundle.stores, bundle.comparison_stores)
    attach_band_summaries(bundle.stores)
    attach_band_summaries(bundle.comparison_stores)
    return bundle


def emit_results(
    bundle,
    *,
    feature_importance_inputs,
    category_map,
    lmm_results,
    cohend_data,
    n_boot,
    seed=42,
):
    """Compute and return the complete aggregate numeric result frame set."""
    from . import calibration, feature_dynamics, feature_importance, leadtime, results

    calibration_grid = calibration.build_beta_calibration_grid(
        bundle, method=CALIBRATION_METHOD
    )
    calibrated = calibration.primary_dca_from_calibration_grid(
        bundle,
        calibration_grid,
        n_boot=n_boot,
        seed=seed,
    )
    importance = feature_importance.feature_importance_results(
        feature_importance_inputs, category_map=category_map
    )
    concordance = feature_importance.feature_concordance_results(importance)
    sensitivity, sensitivity_audit = leadtime.leadtime_results(
        bundle, n_boot=n_boot, seed=seed
    )
    frames, calibration_bin_ci = results.build_result_frames(
        bundle,
        calibrated,
        feature_importance=importance,
        feature_concordance=concordance,
        feature_dynamics=feature_dynamics.feature_dynamics_results(cohend_data),
        lmm=feature_dynamics.lmm_results_frame(lmm_results),
        sensitivity=sensitivity,
        n_boot=n_boot,
        seed=seed,
        calibration_grid=calibration_grid,
    )
    bundle.replicate_audit = _replicate_audit(
        bundle,
        calibrated,
        calibration_bin_ci,
        sensitivity_audit,
        n_boot=n_boot,
    )
    return frames


def reproduce_inference_results(
    bundle,
    *,
    n_boot: int,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Regenerate the checkpoint-backed CSVs and the full replicate audit."""
    from . import calibration, leadtime, results

    calibration_grid = calibration.build_beta_calibration_grid(
        bundle, method=CALIBRATION_METHOD
    )
    calibrated = calibration.primary_dca_from_calibration_grid(
        bundle,
        calibration_grid,
        n_boot=n_boot,
        seed=seed,
    )
    frames = results.performance_frames(bundle)
    frames["curves.csv"] = results.curve_frame(bundle)
    frames["fold_performance.csv"] = results.fold_performance_frame(bundle)
    calibration_result, calibration_bin_ci = results.calibration_frame(
        bundle,
        n_boot=n_boot,
        seed=seed,
        calibration_grid=calibration_grid,
    )
    frames["calibration.csv"] = calibration_result
    frames["calibration_rug.csv"] = results.calibration_rug_frame(
        bundle, calibration_grid=calibration_grid
    )
    frames["dca.csv"] = results.dca_frame(bundle, calibrated)
    frames["sensitivity.csv"], sensitivity_audit = leadtime.leadtime_results(
        bundle,
        n_boot=n_boot,
        seed=seed,
    )
    audit = _replicate_audit(
        bundle,
        calibrated,
        calibration_bin_ci,
        sensitivity_audit,
        n_boot=n_boot,
    )
    return frames, audit


def run_all(
    inputs,
    *,
    thresholds,
    feature_importance_inputs,
    category_map,
    lmm_results,
    cohend_data,
    n_boot,
    seed=42,
    n_jobs=-2,
    show_progress=True,
):
    """Run the checkpoint-free numeric pipeline and return bundle plus frames."""
    bundle = build_bundle(
        inputs,
        thresholds=thresholds,
        n_boot=n_boot,
        seed=seed,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )
    frames = emit_results(
        bundle,
        feature_importance_inputs=feature_importance_inputs,
        category_map=category_map,
        lmm_results=lmm_results,
        cohend_data=cohend_data,
        n_boot=n_boot,
        seed=seed,
    )
    return bundle, frames
