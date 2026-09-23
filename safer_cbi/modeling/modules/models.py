"""Shared attention-based temporal model definitions."""

from __future__ import annotations

from typing import Literal, overload

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import ModelKind, TASK_NAMES
from .utils import SEQ_LEN, SinusoidalPositionalEncoding, validate_attention_dimensions


class MultiHorizonModel(nn.Module):
    """Attention-based temporal model with one head per prediction horizon."""

    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1,
                 num_tasks: int = len(TASK_NAMES)):
        super().__init__()
        d_model, nhead = validate_attention_dimensions(d_model, nhead)
        self.num_tasks = num_tasks
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=2048)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout,
                batch_first=True, norm_first=True, activation=F.gelu,
            ) for _ in range(num_layers)
        ])
        self.task_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 2),
            ) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for layer in self.encoder_layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        h = x[:, -1, :]
        return torch.stack([classifier(h) for classifier in self.task_classifiers], dim=1)


class SingleHorizonModel(nn.Module):
    """Attention-based temporal model for one prediction horizon."""

    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, max_len: int = SEQ_LEN):
        super().__init__()
        d_model, nhead = validate_attention_dimensions(d_model, nhead)
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout,
                batch_first=True, activation=F.gelu, norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for layer in self.encoder_layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.classifier(x[:, -1, :])


class MultiHorizonGRUModel(nn.Module):
    """Multi-horizon encoder control with one head per prediction horizon.

    Training and prediction stay on the fused cuDNN path rather than using the
    attention model's activation-checkpoint wrapper. Attribution temporarily
    uses the native recurrent backward because cuDNN RNN backward requires
    training mode.
    """

    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.1,
                 num_tasks: int = len(TASK_NAMES)):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.task_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 2),
            ) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial_state = x.new_zeros(self.num_layers, x.shape[0], self.hidden_size)
        encoded, _ = self.gru(x, initial_state)
        h = encoded[:, -1, :]
        return torch.stack([classifier(h) for classifier in self.task_classifiers], dim=1)


def build_gru_model(*, input_size: int, hidden_size: int = 256,
                    num_layers: int = 2, dropout: float = 0.1,
                    num_tasks: int = len(TASK_NAMES)) -> MultiHorizonGRUModel:
    """Build the multi-horizon GRU encoder control."""
    return MultiHorizonGRUModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_tasks=num_tasks,
    )


@overload
def build_neural_model(model_kind: Literal["multitask"], *, input_size: int, d_model: int = 64,
                      nhead: int = 4, num_layers: int = 2, dropout: float = 0.1,
                      num_tasks: int = len(TASK_NAMES), max_len: int = SEQ_LEN) -> MultiHorizonModel: ...


@overload
def build_neural_model(model_kind: Literal["singletask"], *, input_size: int, d_model: int = 64,
                      nhead: int = 4, num_layers: int = 2, dropout: float = 0.1,
                      num_tasks: int = len(TASK_NAMES), max_len: int = SEQ_LEN) -> SingleHorizonModel: ...


def build_neural_model(model_kind: ModelKind, *, input_size: int, d_model: int = 64,
                      nhead: int = 4, num_layers: int = 2, dropout: float = 0.1,
                      num_tasks: int = len(TASK_NAMES), max_len: int = SEQ_LEN) -> nn.Module:
    """Build a supported attention-based temporal model."""
    if model_kind == "multitask":
        return MultiHorizonModel(
            input_size=input_size, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout, num_tasks=num_tasks,
        )
    if model_kind == "singletask":
        return SingleHorizonModel(
            input_size=input_size, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout, max_len=max_len,
        )
    raise ValueError(f"build_neural_model does not support model kind {model_kind!r}")
