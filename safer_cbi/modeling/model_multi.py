"""Multi-ATT hyperparameter optimization calculations."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import optuna
import torch
from optuna.pruners import WilcoxonPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Dataset

from .modules import neural_training
from .modules.data import standardize_fold
from .modules.hpo import _extract_baseline_params, _suggest_from_space
from .modules.models import build_neural_model
from .modules.utils import Cfg, cleanup_gpu, create_criterion


def _build_and_train(
    train_subset,
    validation_subset,
    params,
    feature_cols,
    task_names,
    *,
    epochs,
    focal_alpha,
    runtime: neural_training.NeuralRuntime,
    loader_kwargs: Mapping | None = None,
    loader_seed: int,
):
    model = build_neural_model(
        "multitask",
        input_size=len(feature_cols),
        d_model=int(params["d_model"]),
        nhead=int(params["nhead"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"]),
        num_tasks=len(task_names),
    ).to(runtime.device)
    generator = torch.Generator().manual_seed(int(loader_seed))
    kwargs = dict(loader_kwargs or {})
    train_loader = DataLoader(
        train_subset,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        generator=generator,
        **kwargs,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=int(params["batch_size"]),
        shuffle=False,
        **kwargs,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]))
    criterion = create_criterion(float(params["focal_gamma"]), focal_alpha)
    best_auroc, best_auprc, *_ = neural_training.train_with_early_stopping(
        model,
        train_loader,
        validation_loader,
        optimizer,
        criterion,
        runtime,
        max_epochs=epochs,
    )
    return model, best_auroc, best_auprc


def run_optuna_multitask_standard(
    train_dataset: Dataset,
    cfg: Cfg,
    *,
    fold_splits,
    focal_alpha,
    search_space,
    feature_cols,
    task_names,
    runtime: neural_training.NeuralRuntime,
    seed: int,
    n_trials=50,
    loader_kwargs: Mapping | None = None,
) -> optuna.Study:
    """Return a completed five-fold macro-AUROC HPO study."""
    if runtime.single_output:
        raise ValueError("Multi-ATT requires multi-output runtime semantics")
    study = optuna.create_study(
        study_name="multi-attention-hpo",
        direction="maximize",
        sampler=TPESampler(seed=seed, multivariate=True, group=True),
        pruner=WilcoxonPruner(p_threshold=0.1),
    )
    study.enqueue_trial(
        _extract_baseline_params(
            search_space["parameters"], search_space.get("conditional")
        ),
        skip_if_exists=True,
    )

    def objective(trial):
        params = _suggest_from_space(
            trial, search_space["parameters"], search_space.get("conditional")
        )
        scores = []
        for fold_record in fold_splits:
            fold = int(fold_record["fold"])
            train_indices = np.asarray(fold_record["train_indices"], dtype=int)
            validation_indices = np.asarray(fold_record["val_indices"], dtype=int)
            train_subset, validation_subset = standardize_fold(
                train_dataset, train_indices, validation_indices
            )
            _, auroc, auprc = _build_and_train(
                train_subset,
                validation_subset,
                params,
                feature_cols,
                task_names,
                epochs=cfg.epochs,
                focal_alpha=focal_alpha,
                runtime=runtime,
                loader_kwargs=loader_kwargs,
                loader_seed=seed + fold,
            )
            scores.append(float(auroc))
            trial.set_user_attr(f"fold{fold}_auroc", float(auroc))
            trial.set_user_attr(f"fold{fold}_auprc", float(auprc))
            trial.report(float(auroc), step=fold)
            cleanup_gpu()
            if trial.should_prune():
                return float(np.mean(scores))
        return float(np.mean(scores))

    study.optimize(
        objective,
        n_trials=n_trials,
        gc_after_trial=True,
        catch=(RuntimeError, ValueError, torch.cuda.OutOfMemoryError),
    )
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise RuntimeError("No HPO trials completed successfully")
    return study
