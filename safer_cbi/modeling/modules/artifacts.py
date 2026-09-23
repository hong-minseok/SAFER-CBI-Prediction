"""Versioned metadata validation for modeling checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

from .contracts import ModelKind, horizon_for_label, normalize_horizon
from .models import build_gru_model, build_neural_model
from .utils import validate_attention_dimensions


CHECKPOINT_SCHEMA_VERSION = 1


def detect_model_kind(payload: Mapping[str, Any]) -> ModelKind:
    """Return the explicit current artifact kind."""
    explicit = payload.get("model_type")
    allowed = {"multitask", "gru", "singletask", "lgbm"}
    if explicit not in allowed:
        raise ValueError(f"Unsupported checkpoint model_type: {explicit!r}")
    kind = explicit

    if kind in {"multitask", "gru"} and "label_col" in payload:
        raise ValueError("Multi-horizon checkpoint cannot contain label_col")
    if kind == "singletask" and ("task_names" in payload or "label_cols" in payload):
        raise ValueError("Singletask checkpoint cannot contain task_names or label_cols")
    return kind  # type: ignore[return-value]


def _string_list(payload: Mapping[str, Any], key: str, *, required: bool = True) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None and not required:
        return ()
    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"Checkpoint {key} must be a non-empty string list")
    if len(set(value)) != len(value):
        raise ValueError(f"Checkpoint {key} contains duplicates")
    return tuple(value)


def _numeric_vector(payload: Mapping[str, Any], key: str, expected: int) -> tuple[float, ...]:
    value = payload.get(key)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        raise ValueError(f"Checkpoint {key} must have one value per feature ({expected})")
    if not all(isinstance(item, Real) for item in value):
        raise ValueError(f"Checkpoint {key} must contain only numbers")
    numbers = tuple(float(item) for item in value)
    if key == "standardize_sd" and any(item <= 0 for item in numbers):
        raise ValueError("Checkpoint standardize_sd values must be positive")
    return numbers


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: int
    model_type: ModelKind
    features: tuple[str, ...]
    task_names: tuple[str, ...]
    label_cols: tuple[str, ...]
    standardize_mu: tuple[float, ...]
    standardize_sd: tuple[float, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CheckpointMetadata":
        schema_version = payload.get("schema_version")
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported checkpoint schema_version: {schema_version!r}")

        model_type = detect_model_kind(payload)
        features = _string_list(payload, "features")
        if model_type == "lgbm":
            mu = ()
            sd = ()
        else:
            mu = _numeric_vector(payload, "standardize_mu", len(features))
            sd = _numeric_vector(payload, "standardize_sd", len(features))

        if model_type in {"multitask", "gru"}:
            task_names = _string_list(payload, "task_names")
            label_cols = _string_list(payload, "label_cols")
            if len(task_names) != len(label_cols):
                raise ValueError("Checkpoint task_names and label_cols must have equal length")
            expected_labels = tuple(f"y_{normalize_horizon(task)}_true" for task in task_names)
            if label_cols != expected_labels:
                raise ValueError("Checkpoint task_names and label_cols do not map positionally")
            thresholds = payload.get("thresholds")
            if thresholds is not None:
                if not isinstance(thresholds, Mapping) or tuple(thresholds) != task_names:
                    raise ValueError("Checkpoint thresholds must follow task_names order")
                if not all(isinstance(value, Real) for value in thresholds.values()):
                    raise ValueError("Checkpoint thresholds must be numeric")
        elif model_type == "singletask":
            label_col = payload.get("label_col")
            horizon_for_label(label_col)
            task_names = (horizon_for_label(label_col),)
            label_cols = (label_col,)
            threshold = payload.get("threshold")
            if threshold is not None and not isinstance(threshold, Real):
                raise ValueError("Checkpoint threshold must be numeric")
        else:
            task_names = _string_list(payload, "task_names")
            label_cols = _string_list(payload, "label_cols")
            if len(task_names) != len(label_cols):
                raise ValueError("LGBM artifact task_names and label_cols must have equal length")
            expected_labels = tuple(f"y_{normalize_horizon(task)}_true" for task in task_names)
            if label_cols != expected_labels:
                raise ValueError("LGBM artifact task_names and label_cols do not map positionally")
            thresholds = payload.get("thresholds")
            if not isinstance(thresholds, Mapping) or tuple(thresholds) != task_names:
                raise ValueError("LGBM artifact thresholds must follow task_names order")
            if not all(isinstance(value, Real) for value in thresholds.values()):
                raise ValueError("LGBM artifact thresholds must be numeric")
            boosters = payload.get("boosters")
            if not isinstance(boosters, (list, tuple)) or len(boosters) != len(task_names):
                raise ValueError("LGBM artifact must contain one booster per task")

        hyperparameters = payload.get("best_hyperparams")
        if not isinstance(hyperparameters, Mapping):
            raise ValueError("Checkpoint best_hyperparams must be a mapping")
        if model_type != "lgbm" and "focal_gamma" not in hyperparameters:
            raise ValueError("Neural checkpoint best_hyperparams is missing focal_gamma")
        if model_type in {"multitask", "singletask"}:
            d_model, nhead = validate_attention_dimensions(
                hyperparameters.get("d_model"), hyperparameters.get("nhead")
            )
            if d_model != hyperparameters["d_model"] or nhead != hyperparameters["nhead"]:
                raise ValueError("Attention dimensions must be explicit integers")
        if model_type == "gru":
            for key in ("hidden_size", "num_layers", "dropout"):
                if key not in hyperparameters:
                    raise ValueError(f"GRU checkpoint is missing hyperparameter {key}")

        return cls(schema_version, model_type, features, task_names, label_cols, mu, sd)


def rebuild_model_from_checkpoint(checkpoint, input_size: int, device: str):
    """Rebuild a neural model from a validated current payload."""
    metadata = CheckpointMetadata.from_payload(checkpoint)
    params = checkpoint["best_hyperparams"]
    kind = metadata.model_type
    if kind == "gru":
        model = build_gru_model(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            num_tasks=len(checkpoint["task_names"]),
        )
    elif kind in {"multitask", "singletask"}:
        d_model, nhead = validate_attention_dimensions(
            params["d_model"], params["nhead"]
        )
        model = build_neural_model(
            kind,
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            num_tasks=len(checkpoint.get("task_names", ())),
        )
    else:
        raise ValueError(f"Unsupported neural checkpoint type: {kind!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()
