"""Multi-ATT per-feature attribution and fold-checkpoint utilities."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Mapping, Sequence
import warnings

from captum.attr import GradientShap, IntegratedGradients
import numpy as np
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .artifacts import CheckpointMetadata, rebuild_model_from_checkpoint
from .contracts import LABEL_COLS
from .data import MultiHorizonDataset, SubsetDataset, iter_patient_windows
from .utils import FI_SEED, SEQ_LEN, cleanup_gpu


def compute_ig_attributions(
    model,
    dataset,
    task_index,
    device,
    baseline,
    batch_size=64,
    n_steps=50,
    internal_batch_size=10,
    is_multitask=False,
    return_convergence_delta=False,
):
    """Return Integrated Gradients with shape ``(N, T, F)``."""
    model.eval()

    def forward(features):
        logits = model(features)
        return logits[:, task_index, 1] if is_multitask else logits[:, 1]

    method = IntegratedGradients(forward)
    baseline = torch.as_tensor(baseline, dtype=torch.float32, device=device).flatten()
    attributions, deltas = [], []
    for features, _ in tqdm(
        DataLoader(dataset, batch_size=batch_size, shuffle=False),
        desc="IG batches",
        leave=False,
    ):
        features = features.to(device).requires_grad_(True)
        if baseline.numel() != features.shape[-1]:
            raise ValueError("Attribution baseline does not match the feature contract")
        expanded = baseline.view(1, 1, -1).expand_as(features)
        recurrent = (
            torch.backends.cudnn.flags(enabled=False)
            if features.is_cuda and hasattr(model, "gru") else nullcontext()
        )
        with warnings.catch_warnings(), recurrent:
            warnings.filterwarnings("ignore", message="Internal batch size", module="captum")
            output = method.attribute(
                features,
                baselines=expanded,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=return_convergence_delta,
            )
        if return_convergence_delta:
            attribution, delta = output
            deltas.append(delta.detach().cpu().numpy())
        else:
            attribution = output
        attributions.append(attribution.detach().cpu().numpy())
    values = np.concatenate(attributions, axis=0)
    return (values, np.concatenate(deltas)) if return_convergence_delta else values


def compute_gradshap_attributions(
    model,
    dataset,
    task_index,
    device,
    baselines,
    batch_size=8,
    n_samples=50,
    stdevs=0.0,
    is_multitask=False,
    seed=FI_SEED,
    return_convergence_delta=False,
):
    """Return Gradient SHAP attributions with shape ``(N, T, F)``."""
    model.eval()

    def forward(features):
        logits = model(features)
        return logits[:, task_index, 1] if is_multitask else logits[:, 1]

    method = GradientShap(forward)
    baselines = torch.as_tensor(baselines, dtype=torch.float32, device=device)
    attributions, deltas = [], []
    for features, _ in tqdm(
        DataLoader(dataset, batch_size=batch_size, shuffle=False),
        desc="GradientShap batches",
        leave=False,
    ):
        features = features.to(device)
        np.random.seed(seed)
        torch.manual_seed(seed)
        recurrent = (
            torch.backends.cudnn.flags(enabled=False)
            if features.is_cuda and hasattr(model, "gru") else nullcontext()
        )
        with recurrent:
            output = method.attribute(
                features,
                baselines=baselines,
                n_samples=n_samples,
                stdevs=stdevs,
                return_convergence_delta=return_convergence_delta,
            )
        if return_convergence_delta:
            attribution, delta = output
            deltas.append(delta.detach().cpu().numpy())
        else:
            attribution = output
        attributions.append(attribution.detach().cpu().numpy())
    values = np.concatenate(attributions, axis=0)
    return (values, np.concatenate(deltas)) if return_convergence_delta else values


def aggregate_attributions(attrs, feature_cols):
    """Return the retained per-feature mean absolute attribution."""
    values = np.abs(attrs).mean(axis=(0, 1))
    return {"per_feature": sorted(
        [
            {"feature": feature, "importance": round(float(value), 6)}
            for feature, value in zip(feature_cols, values)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )}


def _wearing_window_mask(dataset):
    if not hasattr(dataset, "df") or "nonwearing" not in dataset.df:
        raise ValueError("Attribution dataset lacks the nonwearing window contract")
    masks = []
    for group, start in iter_patient_windows(dataset.df):
        masks.append(
            group["nonwearing"].astype(float).to_numpy()[start:start + SEQ_LEN] == 0
        )
    if len(masks) != len(dataset):
        raise ValueError("Wearing-mask windows do not align with the dataset")
    mask = np.stack(masks)
    if not mask.any():
        raise ValueError("Attribution baseline has no wearing timepoints")
    return mask


def _wearing_baseline(dataset, feature_cols, *, mask=None, features=None, k=200):
    mask = _wearing_window_mask(dataset) if mask is None else mask
    features = np.asarray(dataset.X) if features is None else np.asarray(features)
    fractions = mask.mean(axis=1)
    fully_wearing = int(np.count_nonzero(fractions == 1.0))
    majority_wearing = int(np.count_nonzero(fractions >= 0.9))
    indices = np.flatnonzero(fractions == 1.0)
    fallback = "none"
    if not len(indices):
        indices = np.flatnonzero(fractions >= 0.9)
        fallback = "majority_wearing"
    if len(indices):
        if len(indices) > k:
            indices = np.sort(np.random.default_rng(FI_SEED).choice(indices, k, replace=False))
        distribution = features[indices].astype(np.float32)
    else:
        mean = features[mask].mean(axis=0).astype(np.float32)
        distribution = np.broadcast_to(
            mean, (1, SEQ_LEN, len(feature_cols))
        ).copy()
        fallback = "timepoint_mean_broadcast"
    point = distribution.mean(axis=(0, 1)).astype(np.float32)
    metadata = {
        "baseline_id": "wearing_window_distribution_standardized",
        "space": "model-standardized feature space",
        "selection": "fully wearing windows; >=90% wearing fallback; timepoint-mean broadcast final fallback",
        "k_requested": int(k),
        "k_effective": int(len(distribution)),
        "fallback": fallback,
        "wearing_fraction": round(float(mask.mean()), 6),
        "source_windows": int(len(features)),
        "fully_wearing_windows": fully_wearing,
        "majority_wearing_windows": majority_wearing,
        "seed": FI_SEED,
    }
    return distribution, point, metadata


def _per_feature(attrs, feature_cols):
    return aggregate_attributions(attrs, feature_cols)["per_feature"]


def compute_feature_importance(
    model,
    dataset,
    feature_cols,
    device,
    task_names=None,
    method="both",
    batch_size=256,
    n_steps=50,
    shap_batch_size=8,
    n_samples=None,
    return_convergence_delta=False,
    baseline_dataset=None,
):
    """Return the two retained temporal per-feature payloads."""
    if method != "both" or not task_names:
        raise ValueError("Feature importance requires both methods and all multitask heads")
    n_samples = n_steps if n_samples is None else n_samples
    baseline_source = dataset if baseline_dataset is None else baseline_dataset
    distribution, point, baseline_meta = _wearing_baseline(
        baseline_source, feature_cols
    )
    artifacts = {}
    specifications = (
        ("ig", "integrated_gradients", point),
        ("gradshap", "gradient_shap", distribution),
    )
    for token, method_name, attribution_baseline in specifications:
        task_values = []
        for task_index in range(len(task_names)):
            if token == "ig":
                values = compute_ig_attributions(
                    model, dataset, task_index, device, attribution_baseline,
                    batch_size=batch_size, n_steps=n_steps, is_multitask=True,
                    return_convergence_delta=return_convergence_delta,
                )
            else:
                values = compute_gradshap_attributions(
                    model, dataset, task_index, device, attribution_baseline,
                    batch_size=shap_batch_size, n_samples=n_samples, is_multitask=True,
                    return_convergence_delta=return_convergence_delta,
                )
            task_values.append(values[0] if return_convergence_delta else values)
        macro = np.mean(task_values, axis=0)
        artifacts[token] = {
            "method": method_name,
            "resolution": "per_feature",
            "n_scored_windows": int(len(dataset)),
            "macro": _per_feature(macro, feature_cols),
            "tasks": {
                task: _per_feature(values, feature_cols)
                for task, values in zip(task_names, task_values)
            },
        }
        cleanup_gpu()
    provenance = {
        "scope": "temporal",
        "n_scored_windows": int(len(dataset)),
        "methods": {
            "integrated_gradients": {"n_steps": int(n_steps)},
            "gradient_shap": {
                "n_samples": int(n_samples),
                "stdevs": 0.0,
                "seed": int(FI_SEED),
            },
        },
        "baseline": {"source": "full_discovery", **baseline_meta},
    }
    return artifacts, provenance


def _fold_importance(
    model,
    dataset,
    device,
    method,
    point,
    distribution,
    task_names,
    *,
    batch_size,
    n_steps,
    shap_batch_size,
    n_samples,
):
    task_values = []
    for task_index in range(len(task_names)):
        if method == "ig":
            values = compute_ig_attributions(
                model, dataset, task_index, device, point,
                batch_size=batch_size, n_steps=n_steps, is_multitask=True,
            )
        else:
            values = compute_gradshap_attributions(
                model, dataset, task_index, device, distribution,
                batch_size=shap_batch_size, n_samples=n_samples, is_multitask=True,
            )
        task_values.append(values)
    return np.abs(np.mean(task_values, axis=0)).mean(axis=(0, 1))


def compute_oof_feature_importance(
    checkpoints: Sequence[Mapping],
    train_frame,
    device,
    method="both",
    batch_size=256,
    n_steps=50,
    shap_batch_size=8,
    n_samples=None,
):
    """Return the retained OOF vectors and stability payload."""
    if method != "both":
        raise ValueError("OOF feature importance requires both methods")
    if not checkpoints:
        raise ValueError("OOF feature importance requires fold checkpoints")
    ordered = sorted(checkpoints, key=lambda checkpoint: int(checkpoint["fold"]))
    metadata = [CheckpointMetadata.from_payload(checkpoint) for checkpoint in ordered]
    if any(item.model_type != "multitask" for item in metadata):
        raise ValueError("OOF feature importance is retained only for Multi-ATT")
    n_samples = n_steps if n_samples is None else n_samples
    feature_cols = list(metadata[0].features)
    task_names = list(metadata[0].task_names)
    dataset = MultiHorizonDataset(train_frame, feature_cols, list(LABEL_COLS))
    wearing = _wearing_window_mask(dataset)
    raw_features = np.asarray(dataset.X)
    all_indices = np.arange(len(dataset))
    per_fold = {"ig": [], "gradshap": []}
    fold_meta = []
    for checkpoint, checkpoint_metadata in zip(ordered, metadata):
        if (
            checkpoint_metadata.features != metadata[0].features
            or checkpoint_metadata.task_names != metadata[0].task_names
        ):
            raise ValueError("OOF fold checkpoint contracts do not match")
        selection = checkpoint.get("epoch_selection", {})
        if checkpoint.get("training_protocol") != "fold_specific_early_stopping":
            raise RuntimeError("OOF attribution checkpoint is not early-stopped")
        if checkpoint.get("training_epochs") != selection.get("best_epoch"):
            raise RuntimeError("OOF attribution checkpoint epoch provenance differs")
        validation_indices = np.asarray(checkpoint["val_indices"], dtype=int)
        train_indices = np.setdiff1d(all_indices, validation_indices)
        mean = np.asarray(checkpoint["standardize_mu"], dtype=np.float32)
        std = np.asarray(checkpoint["standardize_sd"], dtype=np.float32)
        validation = SubsetDataset(
            ((raw_features[validation_indices] - mean) / std).astype(np.float32),
            dataset.y[validation_indices],
        )
        train_features = ((raw_features[train_indices] - mean) / std).astype(np.float32)
        distribution, point, baseline_meta = _wearing_baseline(
            None,
            feature_cols,
            mask=wearing[train_indices],
            features=train_features,
        )
        model = rebuild_model_from_checkpoint(checkpoint, len(feature_cols), device)
        for method_name in per_fold:
            per_fold[method_name].append(_fold_importance(
                model,
                validation,
                device,
                method_name,
                point,
                distribution,
                task_names,
                batch_size=batch_size,
                n_steps=n_steps,
                shap_batch_size=shap_batch_size,
                n_samples=n_samples,
            ))
        fold_meta.append({
            "fold": int(checkpoint["fold"]),
            "n_heldout": int(len(validation_indices)),
            "best_epoch": int(selection["best_epoch"]),
            "baseline": {"source": "fold_training", **baseline_meta},
        })
        del model
        cleanup_gpu()

    artifacts = {}
    stability = {}
    for token, vectors in per_fold.items():
        values = np.stack(vectors)
        normalized = values / np.where(values.sum(axis=1, keepdims=True) == 0, 1.0,
                                       values.sum(axis=1, keepdims=True))
        mean = normalized.mean(axis=0)
        std = normalized.std(axis=0)
        rows = sorted(
            [
                {
                    "feature": feature,
                    "importance_mean": round(float(estimate), 6),
                    "importance_std": round(float(sd), 6),
                }
                for feature, estimate, sd in zip(feature_cols, mean, std)
            ],
            key=lambda row: row["importance_mean"],
            reverse=True,
        )
        artifacts[token] = {
            "meta": {
                "method": "integrated_gradients" if token == "ig" else "gradient_shap",
                "resolution": "per_feature",
                "aggregation": "per-fold sum-normalized proportion, then fold mean +/- std",
                "source": "oof_heldout_windows",
                "folds": [
                    {"fold": row["fold"], "n_heldout": row["n_heldout"]}
                    for row in fold_meta
                ],
            },
            "per_feature": rows,
        }
        correlations = [
            round(float(spearmanr(values[left], values[right]).statistic), 4)
            for left in range(len(values))
            for right in range(left + 1, len(values))
        ]
        stability[token] = {
            "mean_pairwise_spearman": (
                round(float(np.mean(correlations)), 4) if correlations else None
            ),
            "pairwise_spearman": correlations,
        }
    artifacts["stability"] = stability
    provenance = {
        "scope": "discovery_oof",
        "n_folds": int(len(fold_meta)),
        "methods": {
            "integrated_gradients": {"n_steps": int(n_steps)},
            "gradient_shap": {
                "n_samples": int(n_samples),
                "stdevs": 0.0,
                "seed": int(FI_SEED),
            },
        },
        "folds": fold_meta,
    }
    return artifacts, provenance
