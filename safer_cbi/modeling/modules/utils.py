# -*- coding: utf-8 -*-
"""Shared constants, configuration, losses, and metrics."""

import gc
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

# Configuration Constants
DEFAULT_GRAD_CLIP_NORM = 1.0
SEED = 2024
FI_SEED = 241112
THRESHOLD_GRID_MIN = 0.01
THRESHOLD_GRID_MAX = 0.95
THRESHOLD_GRID_POINTS = 191
DEFAULT_ES_MIN_DELTA = 1e-4
ES_PATIENCE = 8
SEQ_LEN = 96
STEP_MINUTES = 15


@dataclass(frozen=True)
class Cfg:
    """HPO epoch budget."""
    epochs: int = 30


# Reproducibility
def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs. Call explicitly at entrypoint start."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Focal Loss
class FocalLoss(nn.Module):
    """Focal Loss for class imbalance. Reference: Lin et al. ICCV 2017.

    Supports both single-task and multi-task configurations.
    For multi-task, use alpha with shape (num_tasks, 2) and provide task_indices.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.per_task_alpha = (alpha is not None and alpha.dim() == 2)

    def forward(self, inputs, targets, task_indices=None):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_term = (1 - p_t) ** self.gamma
        focal_loss = focal_term * ce_loss

        if self.alpha is not None:
            if self.per_task_alpha:
                alpha_t = self.alpha[task_indices, targets]
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def create_criterion(focal_gamma: float, focal_alpha: Optional[torch.Tensor] = None) -> nn.Module:
    """Create loss criterion (FocalLoss or CrossEntropyLoss)."""
    if focal_gamma > 0:
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
    return nn.CrossEntropyLoss()


def compute_multitask_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    *,
    mono_lambda: float,
    num_tasks: int,
) -> torch.Tensor:
    """Apply the shared multitask classification and monotonicity loss."""
    batch_size = outputs.shape[0]
    flat_outputs = outputs.view(-1, 2)
    flat_targets = targets.view(-1)
    if getattr(criterion, "per_task_alpha", False):
        task_indices = torch.arange(num_tasks, device=outputs.device).repeat(batch_size)
        base_loss = criterion(flat_outputs, flat_targets, task_indices=task_indices)
    else:
        base_loss = criterion(flat_outputs, flat_targets)
    if mono_lambda <= 0:
        return base_loss
    probabilities = torch.sigmoid(outputs[:, :, 1] - outputs[:, :, 0])
    violations = F.relu(probabilities[:, :-1] - probabilities[:, 1:])
    return base_loss + mono_lambda * violations.mean()


def cleanup_gpu() -> None:
    """Release GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Positional Encoding
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + d_model % 2])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)].unsqueeze(0)


def validate_attention_dimensions(d_model: int, nhead: int) -> tuple[int, int]:
    """Return explicit compatible attention dimensions or fail."""
    if (
        isinstance(d_model, bool)
        or isinstance(nhead, bool)
        or not isinstance(d_model, (int, np.integer))
        or not isinstance(nhead, (int, np.integer))
    ):
        raise ValueError("d_model and nhead must be positive integers")
    d_model, nhead = int(d_model), int(nhead)
    if d_model <= 0 or nhead <= 0:
        raise ValueError("d_model and nhead must be positive integers")
    if d_model % nhead:
        raise ValueError("d_model must be divisible by nhead")
    return d_model, nhead


# Metrics
def compute_classification_metrics(
    y_true: np.ndarray, probs: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Compute classification metrics for a single binary task."""
    preds = (probs >= threshold).astype(int)
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, probs))
        auprc = float(average_precision_score(y_true, probs))
    else:
        auroc, auprc = 0.5, 0.5
    return {
        'auroc': auroc,
        'auprc': auprc,
        'f1': float(f1_score(y_true, preds, zero_division=0)),
        'precision': float(precision_score(y_true, preds, zero_division=0)),
        'recall': float(recall_score(y_true, preds, zero_division=0)),
        'accuracy': float(accuracy_score(y_true, preds)),
        'positive_class_proportion': float(np.mean(y_true)),
    }
