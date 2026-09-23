# -*- coding: utf-8 -*-
"""In-memory preprocessing and window construction shared by all models."""

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .contracts import LABEL_COLS
from .utils import SEQ_LEN, STEP_MINUTES

def preprocess(df: pd.DataFrame, feature_cols: List[str], label_cols: List[str]) -> pd.DataFrame:
    """Normalize a raw frame for windowing: synthesize patient_id, cast types,
    sort by (patient_id, time), and fill features/labels. Returns a new frame."""
    df = df.copy()
    if 'patient_id' not in df.columns and 'key' in df.columns and 'event_index' in df.columns:
        df['patient_id'] = df['key'].astype(str) + '_' + df['event_index'].astype(str)
    if 'nonwearing' in df.columns:
        df['nonwearing'] = df['nonwearing'].astype(float)
    df['time'] = pd.to_datetime(df['time'])
    df['patient_id'] = df['patient_id'].astype('category')
    df = df.sort_values(['patient_id', 'time']).reset_index(drop=True)
    req_cols = {"patient_id", "time"} | set(feature_cols) | set(label_cols)
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df[feature_cols] = df[feature_cols].fillna(0.0).astype(np.float32)
    df[label_cols] = df[label_cols].fillna(0).astype(np.int64)
    return df


# Sliding-window construction (shared by all datasets)
def _valid_ranges(valid_mask, total_len: int, seq_len: int = SEQ_LEN) -> List[Tuple[int, int]]:
    """Return contiguous row ranges separated by invalid cadence transitions."""
    if len(valid_mask) != max(0, total_len - 1):
        raise ValueError("valid_mask length must equal total_len - 1")
    boundaries = [index + 1 for index, valid in enumerate(valid_mask) if not valid]
    starts = [0, *boundaries]
    ends = [*boundaries, total_len]
    return [
        (start, end)
        for start, end in zip(starts, ends)
        if end - start >= seq_len
    ]


def iter_patient_windows(df, seq_len: int = SEQ_LEN, step_minutes: int = STEP_MINUTES):
    """Yield (group, i) for every valid sliding window across patients.

    Per patient (grouped by 'patient_id'), rows are sorted by 'time' and split into
    ranges of regular `step_minutes` cadence. Within each range, yields the window-start
    index i; the window rows are group.iloc[i:i+seq_len] and the label row is
    group.iloc[i+seq_len].
    """
    for _, group in df.groupby('patient_id', observed=True):
        group = group.sort_values('time').reset_index(drop=True)
        time_diffs = group['time'].diff().dt.total_seconds() / 60
        valid_mask = (np.abs(time_diffs[1:] - step_minutes) < 1).values
        ranges = [(0, len(group))] if valid_mask.all() else _valid_ranges(valid_mask, len(group), seq_len)
        for start, end in ranges:
            g = group.iloc[start:end]
            for i in range(len(g) - seq_len):
                yield g, i


def build_samples(df: pd.DataFrame, feature_cols: List[str], label_cols: List[str]):
    """Construct sliding-window samples from a preprocessed frame.

    Returns (X, y, names, keys, event_indices, times):
      X     (N, SEQ_LEN, F) float32   — window features
      y     (N, len(label_cols)) int64 — labels at the step after each window
      names (N,)                       — patient_id (episode id) per window
      keys  (N,) | None                — patient key per window (None if absent)
      event_indices (N,) | None        — episode index per window (None if absent)
      times (N,) datetime64[ns]        — timestamp of the last window step
    """
    X_list, y_list, name_list, time_list = [], [], [], []
    key_list, event_index_list = [], []
    for g, i in iter_patient_windows(df):
        X_list.append(g.iloc[i:i + SEQ_LEN][feature_cols].values)
        y_list.append(g.iloc[i + SEQ_LEN][label_cols].values)
        name_list.append(g['patient_id'].iloc[0])
        time_list.append(g['time'].iloc[i + SEQ_LEN - 1])
        key_list.append(g['key'].iloc[0] if 'key' in g.columns else None)
        event_index_list.append(g['event_index'].iloc[0] if 'event_index' in g.columns else None)
    if not X_list:
        raise RuntimeError('No valid samples generated.')
    keys = np.array(key_list) if key_list and key_list[0] is not None else None
    event_indices = np.array(event_index_list) if event_index_list and event_index_list[0] is not None else None
    return (np.stack(X_list).astype(np.float32),
            np.stack(y_list).astype(np.int64),
            np.array(name_list),
            keys,
            event_indices,
            np.array(time_list, dtype='datetime64[ns]'))


def collapse_to_floor(X: np.ndarray, nw_idx: int) -> np.ndarray:
    """Collapse (N, SEQ_LEN, F) windows to (N, F): last-step value per feature, with
    nonwearing replaced by the per-window mean. The per-window loop preserves the exact
    float32 summation order of the original window-wise computation."""
    out = X[:, -1, :].copy()
    for n in range(X.shape[0]):
        out[n, nw_idx] = X[n, :, nw_idx].mean()
    return out.astype(np.float32)


# Dataset wrappers and standardization
class MultiHorizonDataset(Dataset):
    """Sliding-window dataset with one label per prediction horizon."""

    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 label_cols: List[str] = list(LABEL_COLS)):
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.df = preprocess(df, feature_cols, label_cols)
        self.X, self.y, self.names, self.keys, self.event_indices, self.times = build_samples(
            self.df, feature_cols, label_cols,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.y[idx]).long()


class SingleHorizonDataset(Dataset):
    """Sliding-window dataset for one prediction horizon."""

    def __init__(self, df: pd.DataFrame, feature_cols: List[str], label_cols: List[str]):
        if len(label_cols) != 1:
            raise ValueError("Single-horizon datasets require exactly one label column")
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.df = preprocess(df, feature_cols, label_cols)
        self.X, labels, self.names, self.keys, self.event_indices, self.times = build_samples(
            self.df, feature_cols, label_cols,
        )
        self.y = labels[:, 0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)


class CovariateOnlyDataset:
    """Covariate-only windows collapsed to their LightGBM input representation."""

    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 label_cols: List[str] = list(LABEL_COLS)):
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.df = preprocess(df, feature_cols, label_cols)
        windows, self.y, self.names, self.keys, self.event_indices, self.times = build_samples(
            self.df, feature_cols, label_cols,
        )
        self.X = collapse_to_floor(windows, feature_cols.index("nonwearing"))

    def __len__(self):
        return len(self.y)


class SubsetDataset(Dataset):
    """Lightweight Dataset wrapping pre-sliced numpy arrays."""
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


def standardize_datasets(train_ds, val_ds, device: str = 'cpu', chunk_size: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize train/val datasets in-place. Returns 1D mu, sd arrays of shape (F,)."""
    if device != 'cpu' and torch.cuda.is_available():
        Xtr_gpu = torch.tensor(train_ds.X.reshape(-1, train_ds.X.shape[-1]), dtype=torch.float32, device=device)
        mu = Xtr_gpu.mean(dim=0, keepdim=True)
        sd = Xtr_gpu.std(dim=0, keepdim=True) + 1e-6
        for ds in (train_ds, val_ds):
            for i in range(0, len(ds.X), chunk_size):
                end = min(i + chunk_size, len(ds.X))
                chunk = torch.tensor(ds.X[i:end], dtype=torch.float32, device=device)
                ds.X[i:end] = ((chunk - mu) / sd).cpu().numpy().astype(np.float32)
        return mu.squeeze(0).cpu().numpy(), sd.squeeze(0).cpu().numpy()
    else:
        Xtr = train_ds.X.reshape(-1, train_ds.X.shape[-1])
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-6
        train_ds.X = ((train_ds.X - mu) / sd).astype(np.float32)
        val_ds.X = ((val_ds.X - mu) / sd).astype(np.float32)
        return mu.astype(np.float32), sd.astype(np.float32)


def standardize_fold(full_ds, train_idx, val_idx) -> Tuple[SubsetDataset, SubsetDataset]:
    """Fit mu/sd on raw full_ds.X[train_idx] only and apply to both fold splits.

    Returns (train_fold_ds, val_fold_ds). Fitting strictly on the fold's training rows
    avoids the validation-into-train leakage of a single global standardization. Mirrors
    the CPU numerics of standardize_datasets; must receive raw (un-standardized) X.
    """
    Xtr, Xva = full_ds.X[train_idx], full_ds.X[val_idx]
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(axis=0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(axis=0) + 1e-6
    train_X = ((Xtr - mu) / sd).astype(np.float32)
    val_X = ((Xva - mu) / sd).astype(np.float32)
    return SubsetDataset(train_X, full_ds.y[train_idx]), SubsetDataset(val_X, full_ds.y[val_idx])
