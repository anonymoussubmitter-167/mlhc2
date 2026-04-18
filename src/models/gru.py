# MIT License — Anonymous Authors, 2026
# Part of ATLAS
"""GRU-based mortality prediction model for RSB quantification."""

import torch
import torch.nn as nn
import numpy as np


class MortalityGRU(nn.Module):
    """Simple GRU for ICU mortality prediction from tabular features.

    Uses a single-step GRU (features treated as a sequence of 1 step
    with feature dimension = input_dim). For temporal data, expand to
    multi-step sequences.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        h = self.input_proj(x)         # (batch, hidden)
        h = h.unsqueeze(1)             # (batch, 1, hidden)
        out, _ = self.gru(h)           # (batch, 1, hidden)
        logit = self.classifier(out[:, -1, :])  # (batch, 1)
        return logit.squeeze(-1)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
