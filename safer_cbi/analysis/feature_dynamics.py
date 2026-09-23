"""Numeric Cohen's d trajectories and confirmatory LMM estimates."""
from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from importlib.resources import as_file, files
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from . import settings as _cfg


def _compute_cohens_d(case_vals: np.ndarray, ctrl_vals: np.ndarray) -> float:
    """Cohen's d with pooled SD."""
    n1, n2 = len(case_vals), len(ctrl_vals)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = np.nanmean(case_vals), np.nanmean(ctrl_vals)
    s1, s2 = np.nanstd(case_vals, ddof=1), np.nanstd(ctrl_vals, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (m1 - m2) / pooled


def _prepare_event_frame(
    features: List[Tuple[str, str]],
    time_range_hours: float,
    id_col: str,
    *,
    raw_frame: pd.DataFrame,
) -> Tuple[List[Tuple[str, str]], List[str], pd.DataFrame]:
    """Shared preprocessing for the Cohen's d and LMM inputs.

    Consumes the injected canonical cohort frame, masks non-wearing / zero-padded
    rows to NaN, assigns the per-episode time-to-event axis under ``id_col``, and trims
    to the requested lookback. ``id_col`` ("event_id" vs "episode_id") is the
    only genuine difference between the two callers.
    """
    top1 = features
    feature_names = [f for _, f in top1]
    print(f"Features ({len(top1)}): {top1}")

    df = raw_frame.copy()

    if "nonwearing" in df.columns:
        nw_mask = df["nonwearing"].astype(bool)
        df.loc[nw_mask, feature_names] = np.nan
    if "is_zero_padded" in df.columns:
        zp_mask = df["is_zero_padded"].astype(bool)
        df.loc[zp_mask, feature_names] = np.nan

    df[id_col] = df["key"].astype(str) + "_" + df["event_index"].astype(str)

    def _assign_time_to_event(grp):
        n = len(grp)
        grp = grp.copy()
        grp["time_to_event"] = np.arange(-(n - 1), 1) * 0.25
        return grp

    df = df.groupby(id_col, group_keys=False).apply(_assign_time_to_event)
    df = df.reset_index(drop=True)
    if id_col not in df.columns:
        df[id_col] = df["key"].astype(str) + "_" + df["event_index"].astype(str)

    df = df[df["time_to_event"] >= -time_range_hours].copy()
    return top1, feature_names, df


def compute_cohens_d_data(
    time_range_hours: float = 24.0,
    *,
    features: List[Tuple[str, str]],
    raw_frame: pd.DataFrame,
    n_boot: Optional[int] = None,
    n_permutations: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    """Compute Cohen's d, bootstrap CI, and cluster permutation results.

    Parameters
    ----------
    features : list of (category, feature_name) tuples
        Selected feature set to trace (category, feature_name).
    n_boot, n_permutations, seed : int, optional
        Runtime resampling parameters. Omitted values use the analysis defaults.

    Returns a dict with keys: top1, time_bins, and results.
    """
    import mne

    n_bootstrap = _cfg.N_BOOT if n_boot is None else int(n_boot)
    n_permutations = _cfg.N_PERM if n_permutations is None else int(n_permutations)
    ci_level = _cfg.CI_ALPHA
    random_state = _cfg.RANDOM_STATE if seed is None else int(seed)

    # ── 1-4. Shared preprocessing (feature-select → NaN mask →
    #         event id + time-to-event → time filter) ──
    print("Preparing feature-dynamics data...")
    top1, _, df = _prepare_event_frame(
        features, time_range_hours, "event_id", raw_frame=raw_frame
    )

    # ── 5. Time bins (15-min) ──
    bin_width = 0.25
    df["time_bin"] = (df["time_to_event"] / bin_width).round().astype(int) * bin_width
    time_bins = np.sort(df["time_bin"].unique())

    # ── 6. Pre-pivot: (n_events, n_time_bins) arrays per feature ──
    #   This avoids repeated DataFrame filtering in bootstrap loops.
    print("Pre-pivoting data into event × time arrays...")
    is_case = df["is_case"].astype(bool)
    case_events = df.loc[is_case, "event_id"].unique()
    ctrl_events = df.loc[~is_case, "event_id"].unique()
    n_case, n_ctrl = len(case_events), len(ctrl_events)
    n_tb = len(time_bins)

    # Map event_id -> integer index for fast indexing
    case_id_to_idx = {eid: i for i, eid in enumerate(case_events)}
    ctrl_id_to_idx = {eid: i for i, eid in enumerate(ctrl_events)}
    tb_to_idx = {tb: j for j, tb in enumerate(time_bins)}

    # Pre-allocate NaN arrays: (n_events, n_time_bins) per feature
    pivoted = {}  # feat -> {"case": array, "ctrl": array}
    for _, feat in top1:
        pivoted[feat] = {
            "case": np.full((n_case, n_tb), np.nan),
            "ctrl": np.full((n_ctrl, n_tb), np.nan),
        }

    # Single pass over DataFrame to fill arrays
    event_ids = df["event_id"].values
    time_bin_vals = df["time_bin"].values
    is_case_vals = is_case.values
    for _, feat in top1:
        feat_vals = df[feat].values
        case_arr = pivoted[feat]["case"]
        ctrl_arr = pivoted[feat]["ctrl"]
        for row_idx in range(len(df)):
            val = feat_vals[row_idx]
            if np.isnan(val):
                continue
            eid = event_ids[row_idx]
            tb = time_bin_vals[row_idx]
            j = tb_to_idx.get(tb)
            if j is None:
                continue
            if is_case_vals[row_idx]:
                i = case_id_to_idx.get(eid)
                if i is not None:
                    case_arr[i, j] = val
            else:
                i = ctrl_id_to_idx.get(eid)
                if i is not None:
                    ctrl_arr[i, j] = val

    print(f"  Pivoted: {n_case} case events, {n_ctrl} control events, "
          f"{n_tb} time bins, {len(top1)} features")

    # ── 7. Point estimate Cohen's d (vectorized) ──
    print("Computing Cohen's d...")
    results = {}
    for cat, feat in top1:
        case_arr = pivoted[feat]["case"]  # (n_case, n_tb)
        ctrl_arr = pivoted[feat]["ctrl"]  # (n_ctrl, n_tb)
        d_vals = np.empty(n_tb)
        for j in range(n_tb):
            cv = case_arr[:, j]
            ctv = ctrl_arr[:, j]
            cv = cv[~np.isnan(cv)]
            ctv = ctv[~np.isnan(ctv)]
            d_vals[j] = _compute_cohens_d(cv, ctv)
        results[feat] = {"category": cat, "d": d_vals}

    # ── 8. Bootstrap CI (event-level resampling, vectorized) ──
    print(f"Bootstrap CI (n_boot={n_bootstrap})...")
    rng = np.random.RandomState(random_state)
    alpha_lo = (1 - ci_level) / 2
    alpha_hi = 1 - alpha_lo

    # Pre-generate all bootstrap indices (shared across features)
    case_boot_idx = rng.randint(0, n_case, size=(n_bootstrap, n_case))
    ctrl_boot_idx = rng.randint(0, n_ctrl, size=(n_bootstrap, n_ctrl))

    for cat, feat in top1:
        case_arr = pivoted[feat]["case"]  # (n_case, n_tb)
        ctrl_arr = pivoted[feat]["ctrl"]  # (n_ctrl, n_tb)
        boot_d = np.empty((n_bootstrap, n_tb))

        for b in range(n_bootstrap):
            # Fancy-index rows → (n_case, n_tb), (n_ctrl, n_tb)
            c_sample = case_arr[case_boot_idx[b]]
            t_sample = ctrl_arr[ctrl_boot_idx[b]]

            # Vectorized Cohen's d per time bin (ignoring NaN)
            n1 = np.sum(~np.isnan(c_sample), axis=0).astype(float)
            n2 = np.sum(~np.isnan(t_sample), axis=0).astype(float)
            m1 = np.nanmean(c_sample, axis=0)
            m2 = np.nanmean(t_sample, axis=0)
            s1 = np.nanstd(c_sample, axis=0, ddof=1)
            s2 = np.nanstd(t_sample, axis=0, ddof=1)
            pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            d = (m1 - m2) / pooled
            d[pooled == 0] = 0.0
            d[(n1 < 2) | (n2 < 2)] = np.nan
            boot_d[b] = d

        results[feat]["ci_lo"] = np.nanpercentile(boot_d, alpha_lo * 100, axis=0)
        results[feat]["ci_hi"] = np.nanpercentile(boot_d, alpha_hi * 100, axis=0)
        print(f"  {feat} done")

    # ── 9. Cluster permutation test (Cohen's d as test statistic) ──
    #   Use Cohen's d directly as the test statistic so that cluster formation
    #   depends on effect magnitude (N-invariant), not statistical power.
    #   Cluster-forming threshold: |d| >= 0.2 (Cohen's "small" effect, 1988).
    #   FWER controlled via event-level label permutation (Maris & Oostenveld 2007).
    CLUSTER_D_THRESH = 0.2  # Cohen's small effect benchmark

    def _cohens_d_stat(x, y):
        """Custom stat_fun for MNE: returns Cohen's d (N-invariant)."""
        n1, n2 = x.shape[0], y.shape[0]
        m1, m2 = x.mean(axis=0), y.mean(axis=0)
        s1, s2 = x.var(axis=0, ddof=1), y.var(axis=0, ddof=1)
        pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        d = np.divide(m1 - m2, pooled, out=np.zeros_like(m1), where=pooled > 0)
        return d

    print(f"Cluster permutation test (n_perm={n_permutations}, |d|>={CLUSTER_D_THRESH})...")
    for cat, feat in top1:
        # Reuse pre-pivoted arrays; interpolate NaN per event, then fill remainder
        case_2d = pivoted[feat]["case"].copy()
        ctrl_2d = pivoted[feat]["ctrl"].copy()

        # Per-event linear interpolation (row-wise)
        for arr in (case_2d, ctrl_2d):
            for i in range(arr.shape[0]):
                row = pd.Series(arr[i])
                arr[i] = row.interpolate(method="linear", limit_direction="both").values

        # Drop fully-NaN events, fill remaining NaN with 0
        case_2d = case_2d[~np.isnan(case_2d).all(axis=1)]
        ctrl_2d = ctrl_2d[~np.isnan(ctrl_2d).all(axis=1)]
        case_2d = np.nan_to_num(case_2d, nan=0.0)
        ctrl_2d = np.nan_to_num(ctrl_2d, nan=0.0)

        print(f"  {feat}: N_case={case_2d.shape[0]}, N_ctrl={ctrl_2d.shape[0]}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, clusters, cluster_pv, _ = mne.stats.permutation_cluster_test(
                    [case_2d, ctrl_2d],
                    stat_fun=_cohens_d_stat,
                    threshold=CLUSTER_D_THRESH,
                    n_permutations=n_permutations,
                    tail=0,
                    n_jobs=1,
                    seed=random_state,
                    out_type="indices",
                    verbose=False,
                )
                sig_mask = np.zeros(len(time_bins), dtype=bool)
                for cl, pv in zip(clusters, cluster_pv):
                    if pv < 0.05:
                        sig_mask[cl[0]] = True
                n_sig = sig_mask.sum()
                results[feat]["sig_cluster"] = sig_mask
                print(f"  {feat}: {len(clusters)} clusters, "
                      f"{sum(1 for pv in cluster_pv if pv < 0.05)} sig, "
                      f"{n_sig}/{len(time_bins)} time points shaded")
            except Exception as e:
                raise RuntimeError(
                    f"Cluster permutation test failed for feature {feat!r}"
                ) from e

    print("Computation complete.")
    return {"top1": top1, "time_bins": time_bins, "results": results}


AR1_BOUNDARY = 0.999
LMM_INFERENCE_CONTRACT = {
    "primary": "random-slope AR(1) LMM",
    "random_effect_fallback": "random-intercept AR(1) LMM",
    "ar1_boundary": AR1_BOUNDARY,
    "boundary_fallback": "no-AR LMM with participant-clustered CR2/Satterthwaite inference",
}

LMM_RAW_COLUMNS = (
    "feature", "term", "estimate", "ci_lower", "ci_upper", "p_value",
    "n_obs", "n_groups", "converged",
    "ar1_rho", "fallback", "ar1_boundary", "robust_fallback", "robust_df",
    "inference",
)
LMM_POST_COLUMNS = (
    *LMM_RAW_COLUMNS,
    "q_interaction",
    "q_group",
    "category",
)


def prepare_lmm_data(
    time_range_hours: float = 24.0,
    *,
    features: List[Tuple[str, str]],
    raw_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare long-format DataFrame for LMM, identical data to Cohen's d.

    Parameters
    ----------
    features : list of (category, feature_name) tuples
        Selected feature set to model (category, feature_name).
    """

    # 1-6. Shared preprocessing (identical transforms to Cohen's d; the id
    #      column is "episode_id" here vs "event_id" in compute_cohens_d_data).
    _, feature_names, df = _prepare_event_frame(
        features, time_range_hours, "episode_id", raw_frame=raw_frame
    )

    required = {"age", "sex", "site"}
    missing = required - set(df)
    if missing:
        raise ValueError(
            f"LMM frame is missing caller-supplied demographics: {sorted(missing)}"
        )

    # 8. Melt to long format
    id_cols = ["episode_id", "key", "is_case", "time_to_event", "age", "sex", "site"]
    long = df[id_cols + feature_names].melt(
        id_vars=id_cols,
        value_vars=feature_names,
        var_name="feature_name",
        value_name="feature_value",
    )

    # 9. Drop NaN feature values
    long = long.dropna(subset=["feature_value", "age", "sex"]).reset_index(drop=True)

    n_episodes = long["episode_id"].nunique()
    n_features = long["feature_name"].nunique()
    print(f"LMM data ready: {len(long):,} rows, {n_episodes} episodes, {n_features} features")

    return long


def _attach_categories(
    results: pd.DataFrame,
    category_map: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Attach complete, unambiguous feature categories to LMM output."""
    feature_category = {}
    for category, features in category_map.items():
        for feature in features:
            previous = feature_category.setdefault(feature, category)
            if previous != category:
                raise ValueError(
                    f"Feature {feature!r} belongs to both {previous!r} and {category!r}"
                )
    missing = sorted(set(results["feature"]) - set(feature_category))
    if missing:
        raise ValueError(f"LMM features missing from category map: {missing}")
    output = results.copy()
    output["category"] = output["feature"].map(feature_category)
    return output


def run_lmm(
    df: pd.DataFrame,
    *,
    category_map: Mapping[str, Sequence[str]],
    rscript,
) -> pd.DataFrame:
    """Run the bundled LMM script with an explicit R executable."""
    rscript_bin = Path(rscript)
    if not rscript_bin.is_file():
        raise EnvironmentError(f"Rscript does not exist: {rscript_bin}")

    # Save to temp CSV and call R
    resource = files("safer_cbi.analysis").joinpath("lmm_fit.R")
    with as_file(resource) as r_script:
        with tempfile.TemporaryDirectory(prefix="lmm_") as tmp_dir:
            input_csv = Path(tmp_dir) / "lmm_input.csv"
            output_csv = Path(tmp_dir) / "lmm_output.csv"

            df.to_csv(input_csv, index=False)
            print(f"Input CSV saved: {len(df):,} rows")

            result = subprocess.run(
                [str(rscript_bin), str(r_script), str(input_csv), str(output_csv)],
                capture_output=True, text=True, timeout=600,
                env={**os.environ, "SAFER_AR1_BOUNDARY": str(AR1_BOUNDARY)},
            )

            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    print(f"  [R] {line}")

            if result.returncode != 0:
                raise RuntimeError(
                    f"R script failed (code {result.returncode}):\n{result.stderr}"
                )
            if not output_csv.exists():
                raise RuntimeError(f"R script did not create output: {output_csv}")

            results = pd.read_csv(output_csv)
            print(f"R output: {len(results)} rows")

    if tuple(results.columns) != LMM_RAW_COLUMNS:
        raise ValueError(
            "R LMM output schema differs: "
            f"expected={list(LMM_RAW_COLUMNS)}, actual={list(results.columns)}"
        )

    # BH-FDR correction
    results = _apply_fdr(results)
    results = _attach_categories(results, category_map)
    if tuple(results.columns) != LMM_POST_COLUMNS:
        raise ValueError("Postprocessed LMM schema differs from the 18-column contract")
    return results


def _apply_fdr(results: pd.DataFrame) -> pd.DataFrame:
    """Apply BH-FDR correction for interaction and group terms across features."""
    from statsmodels.stats.multitest import multipletests

    results = results.copy()

    def _fdr_for_term(term_name: str, q_col: str) -> None:
        results[q_col] = np.nan
        mask = results["term"] == term_name
        if mask.sum() > 0:
            pvals = results.loc[mask, "p_value"].values
            valid = ~np.isnan(pvals)
            if valid.sum() > 0:
                _, qvals, _, _ = multipletests(pvals[valid], method="fdr_bh")
                q_full = np.full(len(pvals), np.nan)
                q_full[valid] = qvals
                results.loc[mask, q_col] = q_full

    # FDR across features for the interaction and the group term.
    _fdr_for_term("is_caseTRUE:time_to_event", "q_interaction")
    _fdr_for_term("is_caseTRUE", "q_group")

    return results


def feature_dynamics_results(data: dict) -> pd.DataFrame:
    """Flatten Cohen's d trajectories into an aggregate numeric frame."""
    rows = []
    time_bins = np.asarray(data["time_bins"], dtype=float)
    for category, feature in data["top1"]:
        values = data["results"][feature]
        significant = np.asarray(
            values.get("sig_cluster", np.zeros(len(time_bins), dtype=bool)),
            dtype=bool,
        )
        for index, hours_to_event in enumerate(time_bins):
            rows.append({
                "feature": feature,
                "category": category,
                "hours_to_event": hours_to_event,
                "estimate": values["d"][index],
                "ci_lower": values["ci_lo"][index],
                "ci_upper": values["ci_hi"][index],
                "cluster_significant": bool(significant[index]),
            })
    return pd.DataFrame(
        rows,
        columns=(
            "feature", "category", "hours_to_event", "estimate", "ci_lower",
            "ci_upper", "cluster_significant",
        ),
    )


# Terms retained in lmm.csv: the case main effect, the time slope, and their
# interaction. Nuisance covariate rows (intercept/age/sex/site) are fit but not
# output — only these three terms belong to the result contract.
LMM_REPORTED_TERMS = ("is_caseTRUE", "time_to_event", "is_caseTRUE:time_to_event")


def lmm_results_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Select the three output LMM terms with explicit unavailable estimates.

    The fit and its covariates are unchanged; the filter is applied at frame
    assembly, keeping only the case, slope, and interaction terms.
    """
    if tuple(results.columns) != LMM_POST_COLUMNS:
        raise ValueError("LMM results must follow the strict 18-column contract")
    source = results.copy()

    fallback = source["fallback"].map(
        lambda value: str(value).lower() == "true" if not pd.isna(value) else False
    )
    robust = source["robust_fallback"].map(
        lambda value: str(value).lower() == "true" if not pd.isna(value) else False
    )
    frame = pd.DataFrame({
        "feature": source["feature"],
        "category": source["category"],
        "term": source["term"],
        "estimate": source["estimate"],
        "std_error": np.full(len(source), np.nan),
        "df": np.full(len(source), np.nan),
        "statistic": np.full(len(source), np.nan),
        "p_value": source["p_value"],
        "ci_lower": source["ci_lower"],
        "ci_upper": source["ci_upper"],
        "q_group": source["q_group"],
        "q_interaction": source["q_interaction"],
        "ar1_rho": source["ar1_rho"],
        "fallback": fallback,
        "robust_fallback": robust,
        "inference": source["inference"],
    })
    frame = frame[frame["term"].isin(LMM_REPORTED_TERMS)].reset_index(drop=True)
    return frame
