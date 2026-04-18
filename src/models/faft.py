# MIT License — Anonymous Authors, 2026
"""Fairness-Aware Feature Transformer (FAFT) for ICU mortality prediction.

Architecture:
  1. Feature Tokenizer: each scalar feature → learned d-dim token
  2. [CLS] token prepended
  3. Transformer encoder (pre-norm, multi-head self-attention)
  4. Mortality head on [CLS] representation
  5. Gradient Reversal + adversarial demographic heads (age, race)
     → encoder learns representations that predict mortality but
       resist demographic prediction

Reference for FT-Transformer backbone:
  Gorishniy et al. "Revisiting Deep Learning Models for Tabular Data." NeurIPS 2021.
Reference for gradient reversal:
  Ganin et al. "Domain-Adversarial Training of Neural Networks." JMLR 2016.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Gradient Reversal ─────────────────────────────────────────────────────────

class _GRL(torch.autograd.Function):
    """Gradient Reversal Layer with tunable alpha."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x, alpha=1.0):
    return _GRL.apply(x, alpha)


# ── Feature Tokenizer ─────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """Maps n_features scalar values → n_features × d_model tokens.

    Each feature i: token_i = W_i * x_i + b_i  (W_i, b_i ∈ R^d)
    """
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        # Per-feature weight and bias
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias   = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        # weight: (n_features, d_model)
        # → (batch, n_features, d_model)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


# ── Transformer Block (Pre-Norm) ──────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop(h)
        # Pre-norm feed-forward
        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.drop(h)
        return x


# ── FAFT ─────────────────────────────────────────────────────────────────────

class FAFT(nn.Module):
    """Fairness-Aware Feature Transformer.

    Args:
        n_features:   number of input scalar features
        n_age_groups: number of age-group classes for adversarial head
        n_race_groups: number of race classes for adversarial head
        d_model:      token embedding dimension (default 64)
        n_heads:      attention heads (default 4)
        n_layers:     transformer depth (default 2)
        d_ff:         FFN hidden size (default 256)
        dropout:      dropout rate (default 0.1)
    """
    def __init__(self,
                 n_features: int,
                 n_age_groups: int,
                 n_race_groups: int,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 d_ff: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.encoder = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_ff, dropout)
              for _ in range(n_layers)]
        )

        # Mortality prediction head (on [CLS])
        self.mortality_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Adversarial demographic heads (on [CLS] after gradient reversal)
        self.age_head  = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, n_age_groups),
        )
        self.race_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, n_race_groups),
        )

    def forward(self, x: torch.Tensor, alpha: float = 0.0):
        """
        Args:
            x:     (batch, n_features) — scaled clinical features
            alpha: gradient reversal strength (0 = no reversal)

        Returns:
            mortality_logit: (batch,)
            age_logit:       (batch, n_age_groups)   or None if alpha=0
            race_logit:      (batch, n_race_groups)  or None if alpha=0
        """
        batch = x.size(0)

        # Tokenize features → (batch, n_features, d_model)
        tokens = self.tokenizer(x)

        # Prepend [CLS]
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)   # (batch, n_feat+1, d_model)

        # Encode
        tokens = self.encoder(tokens)

        # [CLS] representation
        cls_repr = tokens[:, 0, :]   # (batch, d_model)

        # Mortality logit
        mort_logit = self.mortality_head(cls_repr).squeeze(-1)

        # Adversarial heads (only meaningful during training when alpha > 0)
        if alpha > 0:
            rev = gradient_reversal(cls_repr, alpha)
            age_logit  = self.age_head(rev)
            race_logit = self.race_head(rev)
        else:
            age_logit  = None
            race_logit = None

        return mort_logit, age_logit, race_logit

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
