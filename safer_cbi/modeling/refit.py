"""Repository-independent refit and evaluation calculations."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.utils.data import DataLoader

from . import model_covonly
from .modules import neural_training
from .modules.artifacts import CheckpointMetadata, rebuild_model_from_checkpoint
from .modules.beta_calibration import (
    BetaCalibrationModel,
    crossfit_beta_calibration,
    fit_beta_calibration,
)
from .modules.data import (
    CovariateOnlyDataset,
    MultiHorizonDataset,
    SingleHorizonDataset,
    SubsetDataset,
)
from .modules.evaluation import (
    build_prediction_frame,
    compute_task_metrics,
    summarize_multitask_folds,
    summarize_single_folds,
    tune_f1_thresholds,
)
from .modules.models import build_gru_model, build_neural_model
from .modules.utils import (
    cleanup_gpu,
    compute_classification_metrics,
    create_criterion,
    seed_everything,
)


@dataclass(frozen=True)
class NeuralSpec:
    """Architecture and task semantics needed by the neural refit core."""

    name: str
    model_type: str
    task_names: tuple[str, ...]
    label_cols: tuple[str, ...]

    @property
    def is_single(self) -> bool:
        return self.model_type == "singletask"


@dataclass(frozen=True)
class LoaderConfig:
    workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int = 2

    @classmethod
    def for_host(cls) -> "LoaderConfig":
        workers = min(2, os.cpu_count() or 1)
        return cls(
            workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )

    def kwargs(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "num_workers": self.workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if self.workers > 0:
            values["prefetch_factor"] = self.prefetch_factor
        return values


@dataclass(frozen=True)
class TorchRuntimePolicy:
    """Process-wide Torch controls shared by every supported entrypoint."""

    deterministic_algorithms: bool = True
    deterministic_warn_only: bool = True
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = True
    matmul_allow_tf32: bool = True
    cudnn_allow_tf32: bool = True
    float32_matmul_precision: str = "high"
    cuda_allocator_config: str = "expandable_segments:True"


LOCKED_TORCH_RUNTIME_POLICY = TorchRuntimePolicy()


def apply_torch_runtime(
    policy: TorchRuntimePolicy = LOCKED_TORCH_RUNTIME_POLICY,
) -> None:
    """Apply the declared process-wide Torch runtime policy."""
    torch.backends.cudnn.benchmark = policy.cudnn_benchmark
    torch.backends.cudnn.deterministic = policy.cudnn_deterministic
    torch.backends.cuda.matmul.allow_tf32 = policy.matmul_allow_tf32
    torch.backends.cudnn.allow_tf32 = policy.cudnn_allow_tf32
    torch.set_float32_matmul_precision(policy.float32_matmul_precision)
    torch.use_deterministic_algorithms(
        policy.deterministic_algorithms,
        warn_only=policy.deterministic_warn_only,
    )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", policy.cuda_allocator_config)


@dataclass(frozen=True)
class RefitRuntime:
    """Explicit execution settings; no repository or lineage state."""

    device: str
    use_amp: bool
    mono_lambda: float
    grad_clip_norm: float
    min_delta: float
    patience: int
    max_epochs: int
    loader: LoaderConfig


@dataclass(frozen=True)
class EpochSelection:
    best_epoch: int
    stopped_epoch: int
    best_auroc: float
    best_auprc: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "stopped_epoch": self.stopped_epoch,
            "best_auroc": self.best_auroc,
            "best_auprc": self.best_auprc,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class FoldFit:
    fold: int
    held_out_indices: np.ndarray
    raw_train_features: np.ndarray
    model: Any
    selection: EpochSelection


@dataclass(frozen=True)
class NeuralDiscoveryFit:
    population: dict[str, int]
    prediction: pd.DataFrame
    internal_metrics: dict[str, Any]
    epoch_records: tuple[dict[str, Any], ...]
    selected_epoch: int
    thresholds: tuple[float, ...]
    calibration: dict[str, dict[str, Any]]
    final_model: Any
    standardize_mean: np.ndarray
    standardize_std: np.ndarray
    final_seed: int


@dataclass(frozen=True)
class CovariateDiscoveryFit:
    population: dict[str, int]
    prediction: pd.DataFrame
    internal_metrics: dict[str, Any]
    thresholds: tuple[float, ...]
    boosters: tuple[Any, ...]
    scale_pos_weight: dict[str, float]


@dataclass(frozen=True)
class EvaluationResult:
    population: dict[str, int]
    prediction: pd.DataFrame
    metrics: dict[str, Any]


def derive_seed(master_seed: int, model: str, phase: str, fold: int | str) -> int:
    token = f"{master_seed}|{model}|{phase}|{fold}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def neural_runtime(spec: NeuralSpec, runtime: RefitRuntime) -> neural_training.NeuralRuntime:
    return neural_training.NeuralRuntime(
        device=runtime.device,
        use_amp=runtime.use_amp,
        task_names=spec.task_names,
        mono_lambda=0.0 if spec.is_single else runtime.mono_lambda,
        grad_clip_norm=runtime.grad_clip_norm,
        min_delta=runtime.min_delta,
        patience=runtime.patience,
        single_output=spec.is_single,
    )


def make_dataset(spec: NeuralSpec, frame: pd.DataFrame, feature_cols: list[str]):
    if spec.is_single:
        return SingleHorizonDataset(frame, feature_cols, list(spec.label_cols))
    return MultiHorizonDataset(frame, feature_cols, list(spec.label_cols))


def build_model(
    spec: NeuralSpec,
    params: Mapping[str, Any],
    input_size: int,
    runtime: RefitRuntime,
):
    if spec.model_type == "gru":
        model = build_gru_model(
            input_size=input_size,
            hidden_size=int(params["hidden_size"]),
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
            num_tasks=len(spec.task_names),
        )
    else:
        model_kind = "singletask" if spec.is_single else "multitask"
        model = build_neural_model(
            model_kind,
            input_size=input_size,
            d_model=int(params["d_model"]),
            nhead=int(params["nhead"]),
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
            num_tasks=len(spec.task_names),
        )
    return model.cuda() if runtime.device == "cuda" else model.to(runtime.device)


def focal_alpha(
    spec: NeuralSpec,
    labels: np.ndarray,
    runtime: RefitRuntime,
) -> torch.Tensor:
    if spec.is_single:
        counts = np.bincount(labels.astype(int), minlength=2)
        if (counts == 0).any():
            raise ValueError("Discovery single-task labels must contain both classes")
        values = counts.sum() / (2.0 * counts)
    else:
        values = np.zeros((labels.shape[1], 2), dtype=np.float32)
        for index in range(labels.shape[1]):
            counts = np.bincount(labels[:, index].astype(int), minlength=2)
            if (counts == 0).any():
                raise ValueError(
                    f"Discovery task {spec.task_names[index]} lacks an outcome class"
                )
            values[index] = counts.sum() / (2.0 * counts)
    return torch.tensor(values, dtype=torch.float32, device=runtime.device)


def standardize_fold(
    dataset,
    train_indices: np.ndarray,
    held_out_indices: np.ndarray,
):
    train_x = dataset.X[train_indices]
    held_out_x = dataset.X[held_out_indices]
    flat = train_x.reshape(-1, train_x.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = (flat.std(axis=0) + 1e-6).astype(np.float32)
    train = SubsetDataset(
        ((train_x - mean) / std).astype(np.float32), dataset.y[train_indices]
    )
    held_out = SubsetDataset(
        ((held_out_x - mean) / std).astype(np.float32),
        dataset.y[held_out_indices],
    )
    return train, held_out, mean, std


def make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    runtime: RefitRuntime,
    seed: int | None = None,
):
    generator = None
    if shuffle:
        if seed is None:
            raise ValueError("A deterministic loader seed is required when shuffle=True")
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        **runtime.loader.kwargs(),
    )


def _train_epoch(
    spec: NeuralSpec,
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    runtime: RefitRuntime,
) -> None:
    neural_training.train_epoch(
        model,
        loader,
        optimizer,
        criterion,
        neural_runtime(spec, runtime),
        scaler=scaler,
    )


def _validation_metrics(spec: NeuralSpec, model, loader, runtime: RefitRuntime):
    return neural_training.validation_metrics(
        model, loader, neural_runtime(spec, runtime)
    )


def fit_fold_best(
    spec: NeuralSpec,
    train_dataset,
    validation_dataset,
    params: Mapping[str, Any],
    class_weights: torch.Tensor,
    seed: int,
    runtime: RefitRuntime,
) -> tuple[Any, EpochSelection]:
    """Fit one fold once and restore its actual early-stopped best state."""
    seed_everything(seed)
    model = build_model(spec, params, train_dataset.X.shape[-1], runtime)
    batch_size = int(params["batch_size"])
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        runtime=runtime,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        runtime=runtime,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]))
    criterion = create_criterion(float(params["focal_gamma"]), class_weights)
    scaler = torch.amp.GradScaler("cuda") if runtime.use_amp else None
    best_auroc = -np.inf
    best_auprc = np.nan
    best_epoch = 0
    best_state = None
    wait = 0
    stopped_epoch = runtime.max_epochs
    for epoch_index in range(runtime.max_epochs):
        _train_epoch(
            spec, model, train_loader, optimizer, criterion, scaler, runtime
        )
        metric_values = _validation_metrics(
            spec, model, validation_loader, runtime
        )
        score = float(metric_values["macro_auroc"])
        if score > best_auroc + runtime.min_delta:
            best_auroc = score
            best_auprc = float(metric_values["macro_auprc"])
            best_epoch = epoch_index + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= runtime.patience:
                stopped_epoch = epoch_index + 1
                break
    if best_epoch < 1 or best_state is None:
        raise RuntimeError(f"{spec.name} did not select a valid epoch")
    model.load_state_dict(best_state)
    del optimizer, criterion, scaler, best_state
    return model, EpochSelection(
        best_epoch=int(best_epoch),
        stopped_epoch=int(stopped_epoch),
        best_auroc=float(best_auroc),
        best_auprc=float(best_auprc),
        seed=int(seed),
    )


def fit_fixed_epochs(
    spec: NeuralSpec,
    train_dataset,
    params: Mapping[str, Any],
    class_weights: torch.Tensor,
    epochs: int,
    seed: int,
    runtime: RefitRuntime,
):
    """Fit the full dataset for a fixed epoch count, without validation input."""
    if epochs < 1:
        raise ValueError("Fixed training epochs must be positive")
    seed_everything(seed)
    model = build_model(spec, params, train_dataset.X.shape[-1], runtime)
    loader = make_loader(
        train_dataset,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        seed=seed,
        runtime=runtime,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]))
    criterion = create_criterion(float(params["focal_gamma"]), class_weights)
    scaler = torch.amp.GradScaler("cuda") if runtime.use_amp else None
    for _ in range(int(epochs)):
        _train_epoch(spec, model, loader, optimizer, criterion, scaler, runtime)
    del optimizer, criterion, scaler
    return model


@torch.no_grad()
def predict(
    spec: NeuralSpec,
    model,
    dataset,
    batch_size: int,
    runtime: RefitRuntime,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(
        dataset, batch_size=batch_size, shuffle=False, runtime=runtime
    )
    margins, labels = neural_training.collect_margins(
        model, loader, neural_runtime(spec, runtime)
    )
    probabilities = expit(margins)
    if spec.is_single:
        return probabilities.reshape(-1, 1), labels.reshape(-1, 1)
    return probabilities, labels


def compute_metrics(
    spec: NeuralSpec,
    labels,
    probabilities,
    thresholds,
):
    if spec.is_single:
        return compute_classification_metrics(
            np.asarray(labels).reshape(-1),
            np.asarray(probabilities).reshape(-1),
            float(thresholds[0]),
        )
    return compute_task_metrics(
        labels, probabilities, thresholds, spec.task_names
    )


def population_counts(dataset) -> dict[str, int]:
    labels = dataset.y if np.ndim(dataset.y) == 1 else dataset.y[:, -1]
    frame = pd.DataFrame(
        {
            "key": np.asarray(dataset.keys).astype(str),
            "event_index": np.asarray(dataset.event_indices).astype(str),
            "label": np.asarray(labels).astype(int),
        }
    )
    episode = frame.groupby(["key", "event_index"], sort=False)["label"].max()
    participant = episode.groupby(level=0).agg(["max", "sum", "count"])
    return {
        "participants": int(len(participant)),
        "positive_participants": int(participant["max"].sum()),
        "episodes": int(len(episode)),
        "positive_episodes": int(episode.sum()),
        "prediction_epochs": int(len(dataset)),
    }


def fit_calibrators(
    spec: NeuralSpec,
    labels,
    probabilities,
    folds,
) -> tuple[dict[str, dict[str, Any]], np.ndarray]:
    deployment: dict[str, dict[str, Any]] = {}
    crossfitted = np.zeros_like(probabilities, dtype=float)
    for index, task in enumerate(spec.task_names):
        model = fit_beta_calibration(probabilities[:, index], labels[:, index])
        deployment[task] = model.to_dict()
        crossfitted[:, index] = crossfit_beta_calibration(
            probabilities[:, index], labels[:, index], folds
        )
    return deployment, crossfitted


def prediction_frame(
    spec: NeuralSpec,
    dataset,
    folds,
    labels,
    probabilities,
    thresholds,
    calibrated=None,
) -> pd.DataFrame:
    frame = build_prediction_frame(
        {
            "folds": folds,
            "names": dataset.names,
            "keys": dataset.keys,
            "event_indices": dataset.event_indices,
            "times": dataset.times,
            "y_true": labels,
            "probs": probabilities,
            "thresholds": thresholds,
        },
        spec.task_names,
        include_fold=folds is not None,
    )
    if calibrated is not None:
        for index, task in enumerate(spec.task_names):
            frame[f"y_{task}_prob_calibrated"] = calibrated[:, index]
    return frame


def _internal_metrics(
    spec: NeuralSpec,
    fold_metrics,
    aggregate,
    thresholds,
) -> dict[str, Any]:
    clean = [
        {"fold": fold, "metrics": metric_values}
        for fold, metric_values in fold_metrics
    ]
    if spec.is_single:
        return {
            "fold_average": summarize_single_folds(clean),
            "fold_results": clean,
            "aggregate_pooled": aggregate,
            "deployment_params": {"threshold": float(thresholds[0])},
            "reporting_role": "post-tuning grouped 5-fold OOF development estimate",
            "discovery_oof_protocol": "fold_specific_early_stopping",
        }
    return {
        "fold_average": summarize_multitask_folds(clean, spec.task_names),
        "fold_results": clean,
        "aggregate_pooled": aggregate,
        "deployment_params": {
            "thresholds": {
                task: float(thresholds[index])
                for index, task in enumerate(spec.task_names)
            }
        },
        "reporting_role": "post-tuning grouped 5-fold OOF development estimate",
        "discovery_oof_protocol": "fold_specific_early_stopping",
    }


def fit_neural_discovery(
    spec: NeuralSpec,
    discovery_frame: pd.DataFrame,
    feature_cols: list[str],
    folds: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    runtime: RefitRuntime,
    master_seed: int,
    *,
    on_fold: Callable[[FoldFit], None] | None = None,
) -> NeuralDiscoveryFit:
    """Fit fold-specific OOF models and one median-epoch full model."""
    dataset = make_dataset(spec, discovery_frame.copy(), feature_cols)
    population = population_counts(dataset)
    n_samples, n_tasks = len(dataset), len(spec.task_names)
    oof_probabilities = np.full((n_samples, n_tasks), np.nan, dtype=np.float32)
    oof_labels = np.zeros((n_samples, n_tasks), dtype=np.int64)
    oof_folds = np.full(n_samples, -1, dtype=int)
    epoch_records: list[dict[str, Any]] = []
    for fold_record in folds:
        fold = int(fold_record["fold"])
        train_indices = np.asarray(fold_record["train_indices"], dtype=int)
        held_out_indices = np.asarray(fold_record["val_indices"], dtype=int)
        train, held_out, _, _ = standardize_fold(
            dataset, train_indices, held_out_indices
        )
        seed = derive_seed(master_seed, spec.name, "epoch_selection", fold)
        weights = focal_alpha(spec, train.y, runtime)
        model, selection = fit_fold_best(
            spec, train, held_out, params, weights, seed, runtime
        )
        epoch_records.append({"fold": fold, **selection.to_dict()})
        probabilities, labels = predict(
            spec, model, held_out, int(params["batch_size"]), runtime
        )
        oof_probabilities[held_out_indices] = probabilities
        oof_labels[held_out_indices] = labels
        oof_folds[held_out_indices] = fold
        if on_fold is not None:
            on_fold(
                FoldFit(
                    fold=fold,
                    held_out_indices=held_out_indices,
                    raw_train_features=dataset.X[train_indices],
                    model=model,
                    selection=selection,
                )
            )
        del model, train, held_out, weights
        cleanup_gpu()
    if not np.isfinite(oof_probabilities).all() or (oof_folds < 0).any():
        raise RuntimeError(
            f"{spec.name} OOF predictions do not cover discovery exactly once"
        )

    selected_epoch = int(
        np.median([record["best_epoch"] for record in epoch_records])
    )
    thresholds = tuple(tune_f1_thresholds(oof_labels, oof_probabilities))
    calibration, crossfitted = fit_calibrators(
        spec, oof_labels, oof_probabilities, oof_folds
    )
    prediction = prediction_frame(
        spec,
        dataset,
        oof_folds.tolist(),
        oof_labels,
        oof_probabilities,
        thresholds,
        crossfitted,
    )
    fold_metrics = []
    for fold in np.unique(oof_folds):
        mask = oof_folds == fold
        fold_metrics.append(
            (
                int(fold),
                compute_metrics(
                    spec,
                    oof_labels[mask],
                    oof_probabilities[mask],
                    thresholds,
                ),
            )
        )
    aggregate = compute_metrics(
        spec, oof_labels, oof_probabilities, thresholds
    )
    internal_metrics = _internal_metrics(
        spec, fold_metrics, aggregate, thresholds
    )

    flat = dataset.X.reshape(-1, dataset.X.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = (flat.std(axis=0) + 1e-6).astype(np.float32)
    standardized = SubsetDataset(
        ((dataset.X - mean) / std).astype(np.float32), dataset.y
    )
    final_seed = derive_seed(
        master_seed, spec.name, "final_full_discovery", "all"
    )
    final_weights = focal_alpha(spec, dataset.y, runtime)
    final_model = fit_fixed_epochs(
        spec,
        standardized,
        params,
        final_weights,
        selected_epoch,
        final_seed,
        runtime,
    )
    del standardized, dataset, final_weights
    cleanup_gpu()
    return NeuralDiscoveryFit(
        population=population,
        prediction=prediction,
        internal_metrics=internal_metrics,
        epoch_records=tuple(epoch_records),
        selected_epoch=selected_epoch,
        thresholds=thresholds,
        calibration=calibration,
        final_model=final_model,
        standardize_mean=mean,
        standardize_std=std,
        final_seed=final_seed,
    )


def _scale_pos_weights(
    labels: np.ndarray, task_names: Sequence[str]
) -> dict[str, float]:
    weights = {}
    for index, task in enumerate(task_names):
        task_labels = labels[:, index]
        positives = int(task_labels.sum())
        weights[task] = (
            float((len(task_labels) - positives) / positives)
            if positives
            else 1.0
        )
    return weights


def fit_covariate_discovery(
    discovery_frame: pd.DataFrame,
    folds,
    params: Mapping[str, Any],
    n_jobs: int,
    feature_cols: Sequence[str],
    label_cols: Sequence[str],
    task_names: Sequence[str],
    seed: int,
) -> CovariateDiscoveryFit:
    dataset = CovariateOnlyDataset(
        discovery_frame.copy(), list(feature_cols), list(label_cols)
    )
    population = population_counts(dataset)
    evaluation = model_covonly.evaluate_internal_validation_lgbm(
        dataset, folds, params, n_jobs, seed
    )
    prediction = build_prediction_frame(
        evaluation["predictions"], task_names, include_fold=True
    )
    thresholds = tuple(
        float(value) for value in evaluation["predictions"]["thresholds"]
    )
    boosters = tuple(
        model_covonly.train_boosters(dataset.X, dataset.y, params, n_jobs, seed)
    )
    weights = _scale_pos_weights(dataset.y, task_names)
    internal_metrics = {
        "fold_average": evaluation["fold_average"],
        "fold_results": evaluation["fold_results"],
        "aggregate_pooled": evaluation["aggregate"],
        "deployment_params": {
            "thresholds": {
                task: thresholds[index]
                for index, task in enumerate(task_names)
            }
        },
        "reporting_role": "post-tuning grouped 5-fold OOF development estimate",
    }
    return CovariateDiscoveryFit(
        population=population,
        prediction=prediction,
        internal_metrics=internal_metrics,
        thresholds=thresholds,
        boosters=boosters,
        scale_pos_weight=weights,
    )


def evaluate_neural_checkpoint(
    spec: NeuralSpec,
    checkpoint: Mapping[str, Any],
    temporal_frame: pd.DataFrame,
    feature_cols: list[str],
    calibrators: Sequence[BetaCalibrationModel],
    runtime: RefitRuntime,
) -> EvaluationResult:
    dataset = make_dataset(spec, temporal_frame.copy(), feature_cols)
    population = population_counts(dataset)
    mean = np.asarray(checkpoint["standardize_mu"], dtype=np.float32)
    std = np.asarray(checkpoint["standardize_sd"], dtype=np.float32)
    dataset.X = ((dataset.X - mean) / std).astype(np.float32)
    model = rebuild_model_from_checkpoint(
        checkpoint, len(feature_cols), runtime.device
    )
    params = checkpoint["best_hyperparams"]
    probabilities, labels = predict(
        spec, model, dataset, int(params["batch_size"]), runtime
    )
    thresholds = (
        [float(checkpoint["threshold"])]
        if spec.is_single
        else [
            float(checkpoint["thresholds"][task])
            for task in spec.task_names
        ]
    )
    calibrated = np.column_stack(
        [
            calibrator.predict(probabilities[:, index])
            for index, calibrator in enumerate(calibrators)
        ]
    )
    prediction = prediction_frame(
        spec,
        dataset,
        None,
        labels,
        probabilities,
        thresholds,
        calibrated,
    )
    metric_values = compute_metrics(spec, labels, probabilities, thresholds)
    del model, dataset
    cleanup_gpu()
    return EvaluationResult(population, prediction, metric_values)


def evaluate_covariate_artifact(
    artifact: Mapping[str, Any],
    metadata: CheckpointMetadata,
    temporal_frame: pd.DataFrame,
) -> EvaluationResult:
    dataset = CovariateOnlyDataset(
        temporal_frame.copy(), list(metadata.features), list(metadata.label_cols)
    )
    population = population_counts(dataset)
    probabilities = model_covonly.predict_boosters(
        artifact["boosters"], dataset.X
    )
    thresholds = [
        float(artifact["thresholds"][task]) for task in metadata.task_names
    ]
    prediction = build_prediction_frame(
        {
            "names": dataset.names,
            "keys": dataset.keys,
            "event_indices": dataset.event_indices,
            "times": dataset.times,
            "y_true": dataset.y,
            "probs": probabilities,
            "thresholds": thresholds,
        },
        metadata.task_names,
        include_fold=False,
    )
    metric_values = compute_task_metrics(
        dataset.y, probabilities, thresholds, metadata.task_names
    )
    return EvaluationResult(population, prediction, metric_values)
