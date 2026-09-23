"""Shared training mechanics for the two multi-horizon neural encoders."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from .utils import compute_multitask_loss


@dataclass(frozen=True)
class NeuralRuntime:
    """Immutable execution settings for shared multi-horizon training."""

    device: str
    use_amp: bool
    task_names: tuple[str, ...]
    mono_lambda: float
    grad_clip_norm: float
    min_delta: float
    patience: int
    single_output: bool = False


def train_epoch(model, loader, optimizer, criterion, runtime: NeuralRuntime, scaler=None) -> None:
    """Train one multi-horizon epoch."""
    model.train()
    if scaler is None and runtime.use_amp:
        scaler = torch.amp.GradScaler("cuda")
    for features, labels in loader:
        features = features.to(runtime.device)
        labels = labels.to(runtime.device)
        with torch.amp.autocast("cuda", enabled=runtime.use_amp):
            outputs = model(features)
            loss = (
                criterion(outputs, labels.view(-1))
                if runtime.single_output
                else compute_multitask_loss(
                    outputs,
                    labels,
                    criterion,
                    mono_lambda=runtime.mono_lambda,
                    num_tasks=len(runtime.task_names),
                )
            )
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), runtime.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), runtime.grad_clip_norm)
            optimizer.step()
        optimizer.zero_grad()


@torch.no_grad()
def collect_margins(model, loader, runtime: NeuralRuntime) -> tuple[np.ndarray, np.ndarray]:
    """Collect logit margins and labels in loader order."""
    model.eval()
    margins, labels = [], []
    for features, batch_labels in loader:
        outputs = model(features.to(runtime.device))
        if runtime.single_output:
            margins.append((outputs[:, 1] - outputs[:, 0]).cpu().numpy())
            labels.append(batch_labels.numpy().reshape(-1))
        else:
            margins.append((outputs[:, :, 1] - outputs[:, :, 0]).cpu().numpy())
            labels.append(batch_labels.numpy())
    if runtime.single_output:
        return (
            np.concatenate(margins) if margins else np.zeros(0, dtype=np.float32),
            np.concatenate(labels) if labels else np.zeros(0, dtype=np.int64),
        )
    width = len(runtime.task_names)
    return (
        np.vstack(margins) if margins else np.zeros((0, width), dtype=np.float32),
        np.vstack(labels) if labels else np.zeros((0, width), dtype=np.int64),
    )


def validation_metrics(model, loader, runtime: NeuralRuntime) -> dict[str, float]:
    """Compute macro AUROC and AUPRC over the configured horizons."""
    margins, labels = collect_margins(model, loader, runtime)
    probabilities = expit(margins)
    if runtime.single_output:
        if len(np.unique(labels)) < 2:
            return {"macro_auroc": 0.5, "macro_auprc": 0.5}
        return {
            "macro_auroc": float(roc_auc_score(labels, probabilities)),
            "macro_auprc": float(average_precision_score(labels, probabilities)),
        }
    aurocs, auprcs = [], []
    for index in range(probabilities.shape[1]):
        task_labels = labels[:, index]
        if len(np.unique(task_labels)) > 1:
            aurocs.append(float(roc_auc_score(task_labels, probabilities[:, index])))
            auprcs.append(
                float(average_precision_score(task_labels, probabilities[:, index]))
            )
        else:
            aurocs.append(0.5)
            auprcs.append(0.5)
    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else 0.5,
        "macro_auprc": float(np.mean(auprcs)) if auprcs else 0.5,
    }


def train_with_early_stopping(
    model,
    train_loader,
    validation_loader,
    optimizer,
    criterion,
    runtime: NeuralRuntime,
    *,
    max_epochs: int,
) -> tuple[float, float, int, int, int]:
    """Train and restore the best macro-AUROC epoch."""
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    best_auroc = -float("inf")
    best_auprc = 0.5
    best_state = None
    best_epoch = 0
    wait = 0
    stopped_epoch = max_epochs
    scaler = torch.amp.GradScaler("cuda") if runtime.use_amp else None

    for epoch_index in range(max_epochs):
        train_epoch(model, train_loader, optimizer, criterion, runtime, scaler=scaler)
        metrics = validation_metrics(model, validation_loader, runtime)
        if metrics["macro_auroc"] > best_auroc + runtime.min_delta:
            best_auroc = metrics["macro_auroc"]
            best_auprc = metrics["macro_auprc"]
            best_epoch = epoch_index + 1
            wait = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            wait += 1
            if wait >= runtime.patience:
                stopped_epoch = epoch_index + 1
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auroc, best_auprc, best_epoch, stopped_epoch, max_epochs
