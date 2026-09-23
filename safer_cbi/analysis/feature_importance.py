"""Numeric feature-importance estimates, concordance, and feature selection."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureImportanceInputs:
    """The four in-memory attribution payloads used by numeric analysis."""

    internal_ig: Mapping[str, Any]
    external_ig: Mapping[str, Any]
    internal_gradshap: Mapping[str, Any]
    external_gradshap: Mapping[str, Any]

    def payload(self, split: str, method: str) -> Mapping[str, Any]:
        try:
            return getattr(self, f"{split}_{method}")
        except AttributeError as exc:
            raise ValueError(
                f"Unknown feature-importance payload: split={split!r}, method={method!r}"
            ) from exc


def _category_index(
    category_map: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Invert a category map, rejecting ambiguous feature ownership."""
    feature_category: dict[str, str] = {}
    for category, features in category_map.items():
        for feature in features:
            previous = feature_category.setdefault(feature, category)
            if previous != category:
                raise ValueError(
                    f"Feature {feature!r} belongs to both {previous!r} and {category!r}"
                )
    return feature_category


def _require_categories(features, feature_category: Mapping[str, str]) -> None:
    missing = sorted(set(features) - set(feature_category))
    if missing:
        raise ValueError(f"Features missing from category map: {missing}")


def _rank_by_importance(shares: dict) -> dict:
    """1-indexed dense rank by descending importance share."""
    return {
        feature: rank
        for rank, feature in enumerate(
            sorted(shares, key=lambda name: -shares[name]), start=1
        )
    }


def feature_importance_results(
    inputs: FeatureImportanceInputs,
    *,
    category_map: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Return tidy feature-level IG/GradientSHAP estimates for both cohorts.

    Emits per-cohort rows (split in {internal, external}) plus, for method="ig", an
    epoch-weighted split="overall" family. Each overall share is
    ``(internal_share * w_int + external_share * w_ext) / (w_int + w_ext)``
    over the already-normalized cohort shares. Overall rows carry
    ``epoch_weight`` = the combined scored-epoch total and ``sd`` = NaN
    (no cross-cohort spread is defined).
    """
    feature_category = _category_index(category_map)
    rows = []
    for method in ("ig", "gradshap"):
        shares_by_split, epochs_by_split = {}, {}
        for split in ("internal", "external"):
            data = inputs.payload(split, method)
            if split == "internal":
                records = [
                    (
                        item["feature"],
                        float(item["importance_mean"]),
                        float(item["importance_std"]),
                    )
                    for item in data["per_feature"]
                ]
                epoch_weight = sum(
                    int(fold["n_heldout"])
                    for fold in data.get("meta", {}).get("folds", [])
                )
            else:
                records = [
                    (item["feature"], float(item["importance"]), np.nan)
                    for item in data["macro"]
                ]
                epoch_weight = int(data.get("n_scored_windows", 0))
            if epoch_weight <= 0:
                raise ValueError(
                    f"{method}/{split} feature importance has no scored-epoch count"
                )
            total = sum(value for _, value, _ in records)
            if total <= 0:
                raise ValueError(f"{method}/{split} feature importance sums to zero")
            _require_categories((feature for feature, _, _ in records), feature_category)
            normalized = [
                (
                    feature,
                    value / total,
                    sd / total if not np.isnan(sd) else np.nan,
                )
                for feature, value, sd in records
            ]
            ranks = _rank_by_importance({f: imp for f, imp, _ in normalized})
            shares_by_split[split] = {f: imp for f, imp, _ in normalized}
            epochs_by_split[split] = epoch_weight
            for feature, importance, sd in normalized:
                rows.append({
                    "method": method,
                    "split": split,
                    "feature": feature,
                    "category": feature_category[feature],
                    "importance": importance,
                    "sd": sd,
                    "rank": ranks[feature],
                    "epoch_weight": epoch_weight,
                })
        if method == "ig":
            internal, external = shares_by_split["internal"], shares_by_split["external"]
            if set(internal) != set(external):
                raise ValueError("Overall IG shares require aligned cohort feature sets")
            w_int, w_ext = epochs_by_split["internal"], epochs_by_split["external"]
            total_epochs = w_int + w_ext
            overall = {
                feature: (internal[feature] * w_int + external[feature] * w_ext)
                / total_epochs
                for feature in internal
            }
            overall_ranks = _rank_by_importance(overall)
            for feature in internal:
                rows.append({
                    "method": method,
                    "split": "overall",
                    "feature": feature,
                    "category": feature_category[feature],
                    "importance": overall[feature],
                    "sd": np.nan,
                    "rank": overall_ranks[feature],
                    "epoch_weight": total_epochs,
                })
    return pd.DataFrame(rows, columns=(
        "method", "split", "feature", "category", "importance", "sd",
        "rank", "epoch_weight",
    ))


def feature_concordance_results(
    importance: pd.DataFrame,
) -> pd.DataFrame:
    """Return numeric method/cohort concordance without display formatting."""
    from scipy.stats import kendalltau, spearmanr

    importance = importance.copy()

    def vector(method, split):
        subset = importance[
            (importance["method"] == method) & (importance["split"] == split)
        ]
        return subset.set_index("feature")["importance"].sort_index()

    pairs = (
        ("method", "ig", "external", "gradshap", "external"),
        ("method", "ig", "internal", "gradshap", "internal"),
        ("cohort", "ig", "external", "ig", "internal"),
        ("cohort", "gradshap", "external", "gradshap", "internal"),
    )
    rows = []
    for axis, method_a, split_a, method_b, split_b in pairs:
        a, b = vector(method_a, split_a), vector(method_b, split_b)
        if not a.index.equals(b.index):
            raise ValueError("Feature sets differ across concordance vectors")
        for metric, result in (
            ("spearman", spearmanr(a, b)),
            ("kendall", kendalltau(a, b)),
        ):
            rows.append({
                "comparison_axis": axis,
                "method_a": method_a,
                "split_a": split_a,
                "method_b": method_b,
                "split_b": split_b,
                "metric": metric,
                "n_features": len(a),
                "estimate": result.statistic,
                "p_value": result.pvalue,
            })
    return pd.DataFrame(rows, columns=(
        "comparison_axis", "method_a", "split_a", "method_b", "split_b",
        "metric", "n_features", "estimate", "p_value",
    ))


def _fi_vector(
    inputs: FeatureImportanceInputs,
    split: str,
    method: str = "ig",
) -> dict:
    data = inputs.payload(split, method)
    if split == "external":
        return {row["feature"]: row["importance"] for row in data["macro"]}
    return {row["feature"]: row["importance_mean"] for row in data["per_feature"]}


def overall_shares(
    inputs: FeatureImportanceInputs,
    *,
    category_map: Mapping[str, Sequence[str]],
    method: str = "ig",
) -> dict:
    """Return the epoch-weighted overall share per feature, keyed by feature."""
    frame = feature_importance_results(inputs, category_map=category_map)
    overall = frame[(frame["method"] == method) & (frame["split"] == "overall")]
    if overall.empty:
        raise ValueError(f"{method} feature importance has no overall rows")
    return dict(zip(overall["feature"], overall["importance"]))


def select_topn_intext(
    inputs: FeatureImportanceInputs,
    *,
    category_map: Mapping[str, Sequence[str]],
    top_n: int = 12,
    method: str = "ig",
):
    """Select features by overall importance and return cohort ranks.

    Selection reads the same epoch-weighted overall share Figure 3 plots, so one
    importance axis orders both figures. The pool is the features both cohorts
    score, excluding "Basics"; the feature name breaks exact ties.
    """
    feature_category = _category_index(category_map)
    internal = _fi_vector(inputs, "internal", method)
    external = _fi_vector(inputs, "external", method)
    _require_categories(set(internal) | set(external), feature_category)
    overall = overall_shares(inputs, category_map=category_map, method=method)
    pool = [
        feature for feature in external
        if feature in internal
        and feature_category.get(feature) != "Basics"
    ]
    internal_rank = {
        feature: rank
        for rank, feature in enumerate(
            sorted(pool, key=lambda name: internal[name], reverse=True), start=1
        )
    }
    external_rank = {
        feature: rank
        for rank, feature in enumerate(
            sorted(pool, key=lambda name: external[name], reverse=True), start=1
        )
    }
    ordered = sorted(
        pool, key=lambda feature: (-overall[feature], feature)
    )[:top_n]
    return (
        [(feature_category[feature], feature) for feature in ordered],
        {
            feature: (internal_rank[feature], external_rank[feature])
            for feature in ordered
        },
    )
