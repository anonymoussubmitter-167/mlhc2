# MIT License — Anonymous Authors, 2026
"""Group-Adaptive Fairness-Aware Feature Transformer (GA-FAFT).

Key novelty: per-demographic-group pairwise AUROC ranking loss that directly
targets worst-case group AUROC, combined with the FAFT feature tokenizer and
transformer backbone.

Architecture overview:
  1. Feature Tokenizer: each scalar feature → d-dim token (per-feature affine)
  2. [CLS] token + Transformer encoder (pre-norm, multi-head self-attention)
  3. Mortality prediction head on [CLS]
  4. Per-group pairwise AUROC ranking loss (novel — see GroupAUROCLoss)
  5. Gradient Reversal adversarial heads for age and race (carry-over from FAFT)
  6. Per-group temperature scaling (learned T_g per demographic group, post-hoc)

Novel contribution vs. FAFT:
  FAFT uses GRL to push representations away from demographic decodability.
  GA-FAFT ADDITIONALLY targets the AUROC metric directly per group via a
  differentiable pairwise ranking loss. No prior work has combined
  Feature Tokenizer + Transformer + per-group pairwise AUROC training
  for tabular ICU data (verified via literature search).

Loss function:
  L = L_CE + λ_rank * L_rank + λ_adv * L_adv

  L_rank = max_g[-AUC_g]  (worst-case group AUROC, approximated via pairwise sigmoid)
         = max_g { (1/|P_g||N_g|) * Σ_{i∈P_g, j∈N_g} σ(margin - (s_i - s_j)) }

  where P_g = positives in group g, N_g = negatives in group g, σ = sigmoid,
  margin ≥ 0 is a slack hyperparameter that upweights hard pairs.

  L_adv = GRL adversarial loss on age/race classification (same as FAFT)

References:
  - Feature Tokenizer + Transformer: Gorishniy et al., NeurIPS 2021
  - Gradient Reversal: Ganin et al., JMLR 2016
  - Pairwise AUC optimization: Yan et al., ICML 2003; Calders & Jaroszewicz, 2007
  - Group AUC fairness theory: arXiv:1902.05826
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Gradient Reversal ─────────────────────────────────────────────────────────

class _GRL(torch.autograd.Function):
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
    """Per-feature affine projection: x_i → W_i * x_i + b_i ∈ R^d."""
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias   = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) → (B, n_features, d_model)
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
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop(h)
        h = self.norm2(x)
        h = self.ff(h)
        return x + self.drop(h)


# ── Per-Group Pairwise AUROC Loss (core novelty) ──────────────────────────────

class GroupAUROCLoss(nn.Module):
    """Differentiable per-group AUROC ranking loss.

    For each demographic group g, approximates -AUC_g as a sum over
    (positive, negative) pairs using a sigmoid surrogate:

        L_rank(g) = (1 / |P_g| * |N_g|) * Σ_{i∈P_g, j∈N_g}
                    sigmoid(margin - (logit_i - logit_j))

    Total loss = max_g L_rank(g)  [worst-case fairness]
              OR mean_g L_rank(g) [average fairness]

    The max variant targets worst-case group AUROC (minimax fairness).
    The mean variant smoothly encourages all groups toward high AUROC.

    Implementation note: we use a mini-batch pair approximation.
    With B samples per group in the mini-batch:
      - Pairs formed as outer product of positives × negatives
      - Complexity: O(B^2) per group — manageable with small batch size

    Args:
        margin:     surrogate margin (default 1.0). Larger → harder pairs weighted more.
        max_pairs:  max positive/negative pairs to sample per group per batch
                    (for memory efficiency)
        mode:       "max" (minimax) or "mean" (average group AUROC)
    """
    def __init__(self, margin: float = 1.0, max_pairs: int = 512, mode: str = "max"):
        super().__init__()
        self.margin = margin
        self.max_pairs = max_pairs
        self.mode = mode

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                group_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:    (B,) — raw mortality scores (not sigmoided)
            labels:    (B,) — binary mortality labels
            group_ids: (B,) — integer group indices (e.g., age_group or race_group)

        Returns:
            scalar loss
        """
        unique_groups = group_ids.unique()
        group_losses = []

        for g in unique_groups:
            g_mask = (group_ids == g)
            g_logits = logits[g_mask]
            g_labels = labels[g_mask]

            pos_mask = (g_labels == 1)
            neg_mask = (g_labels == 0)

            n_pos = pos_mask.sum().item()
            n_neg = neg_mask.sum().item()

            if n_pos < 1 or n_neg < 1:
                continue

            pos_logits = g_logits[pos_mask]  # (n_pos,)
            neg_logits = g_logits[neg_mask]  # (n_neg,)

            # Sub-sample pairs if too many
            if n_pos * n_neg > self.max_pairs:
                n_sample = int(self.max_pairs ** 0.5) + 1
                if n_pos > n_sample:
                    idx = torch.randperm(n_pos, device=logits.device)[:n_sample]
                    pos_logits = pos_logits[idx]
                if n_neg > n_sample:
                    idx = torch.randperm(n_neg, device=logits.device)[:n_sample]
                    neg_logits = neg_logits[idx]

            # Pairwise differences: (n_pos, n_neg)
            # diff[i, j] = logit_pos_i - logit_neg_j  (should be > 0 for correct ranking)
            diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)

            # Sigmoid surrogate loss: σ(margin - diff) for each pair
            # = 0 when diff >> margin (correctly ranked with margin)
            # = 0.5 when diff = margin
            # → 1 when diff << margin (wrong ranking)
            pair_loss = torch.sigmoid(self.margin - diff)  # (n_pos, n_neg)
            group_loss = pair_loss.mean()
            group_losses.append(group_loss)

        if not group_losses:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        stacked = torch.stack(group_losses)
        if self.mode == "max":
            return stacked.max()
        else:
            return stacked.mean()


# ── GA-FAFT ───────────────────────────────────────────────────────────────────

class GAFAFT(nn.Module):
    """Group-Adaptive Fairness-Aware Feature Transformer.

    Extends FAFT with per-group pairwise AUROC ranking loss and
    per-group learned temperature scaling for calibration fairness.

    Args:
        n_features:     number of input scalar features
        n_age_groups:   number of age-group classes
        n_race_groups:  number of race classes
        d_model:        token embedding dimension (default 64)
        n_heads:        attention heads (default 4)
        n_layers:       transformer depth (default 2)
        d_ff:           FFN hidden dimension (default 256)
        dropout:        dropout rate (default 0.1)
        rank_margin:    pairwise AUROC loss margin (default 1.0)
        rank_max_pairs: max pairs per group per batch (default 512)
        rank_mode:      "max" (minimax) or "mean" (default "max")
    """
    def __init__(self,
                 n_features: int,
                 n_age_groups: int,
                 n_race_groups: int,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 rank_margin: float = 1.0,
                 rank_max_pairs: int = 512,
                 rank_mode: str = "max"):
        super().__init__()

        # Backbone (same as FAFT)
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.encoder   = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_ff, dropout)
              for _ in range(n_layers)]
        )

        # Mortality head
        self.mortality_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Adversarial demographic heads (GRL)
        self.age_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Linear(32, n_age_groups),
        )
        self.race_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Linear(32, n_race_groups),
        )

        # Per-group temperature scaling (one log-temperature per age group + race group)
        # log(T_g) initialized to 0 → T_g = 1 (identity)
        self.age_log_temp  = nn.Parameter(torch.zeros(n_age_groups))
        self.race_log_temp = nn.Parameter(torch.zeros(n_race_groups))

        # Group AUROC ranking loss module
        self.group_auroc_loss = GroupAUROCLoss(
            margin=rank_margin,
            max_pairs=rank_max_pairs,
            mode=rank_mode,
        )

        # Store dims for external use
        self.n_age_groups  = n_age_groups
        self.n_race_groups = n_race_groups

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder: feature tokens → [CLS] representation."""
        B = x.size(0)
        tokens = self.tokenizer(x)                   # (B, n_feat, d)
        cls    = self.cls_token.expand(B, -1, -1)    # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)     # (B, n_feat+1, d)
        tokens = self.encoder(tokens)
        return tokens[:, 0, :]                       # [CLS] repr

    def forward(self, x: torch.Tensor, alpha: float = 0.0,
                age_ids: torch.Tensor = None,
                race_ids: torch.Tensor = None):
        """
        Args:
            x:        (B, n_features)
            alpha:    GRL strength (0 during eval/warm-up)
            age_ids:  (B,) integer age group indices (for temp scaling)
            race_ids: (B,) integer race group indices (for temp scaling)

        Returns:
            mort_logit:  (B,) raw mortality logit (before temperature scaling)
            age_logit:   (B, n_age_groups) or None
            race_logit:  (B, n_race_groups) or None
        """
        cls_repr  = self.encode(x)
        mort_logit = self.mortality_head(cls_repr).squeeze(-1)

        if alpha > 0:
            rev = gradient_reversal(cls_repr, alpha)
            age_logit  = self.age_head(rev)
            race_logit = self.race_head(rev)
        else:
            age_logit  = None
            race_logit = None

        return mort_logit, age_logit, race_logit

    def scaled_logit(self, mort_logit: torch.Tensor,
                     age_ids: torch.Tensor = None,
                     race_ids: torch.Tensor = None) -> torch.Tensor:
        """Apply per-group temperature scaling to logits.

        Temperature T_g = exp(log_T_g). Scaled logit = logit / T_g.
        During training, this is NOT used in the loss (to avoid confounding).
        Used for calibrated inference.
        """
        scaled = mort_logit.clone()
        if age_ids is not None:
            T_g = torch.exp(self.age_log_temp[age_ids])
            scaled = scaled / T_g.clamp(min=0.5, max=4.0)
        return scaled

    def compute_loss(self,
                     mort_logit: torch.Tensor,
                     labels: torch.Tensor,
                     age_logit: torch.Tensor = None,
                     race_logit: torch.Tensor = None,
                     age_ids: torch.Tensor = None,
                     race_ids: torch.Tensor = None,
                     lambda_rank: float = 0.5,
                     lambda_adv: float = 0.1) -> dict:
        """Compute total loss and component losses.

        Args:
            mort_logit:   (B,) raw mortality logit
            labels:       (B,) binary mortality labels
            age_logit:    (B, n_age_groups) or None
            race_logit:   (B, n_race_groups) or None
            age_ids:      (B,) integer age group IDs for ranking loss
            race_ids:     (B,) integer race group IDs for ranking loss
            lambda_rank:  weight on group AUROC ranking loss
            lambda_adv:   weight on adversarial demographic loss

        Returns:
            dict with keys: total, ce, rank_age, rank_race, adv_age, adv_race
        """
        # Primary: binary cross-entropy
        L_ce = F.binary_cross_entropy_with_logits(mort_logit, labels.float())

        # Per-group AUROC ranking loss (age groups)
        L_rank_age = torch.tensor(0.0, device=mort_logit.device)
        if age_ids is not None:
            L_rank_age = self.group_auroc_loss(mort_logit, labels, age_ids)

        # Per-group AUROC ranking loss (race groups)
        L_rank_race = torch.tensor(0.0, device=mort_logit.device)
        if race_ids is not None:
            L_rank_race = self.group_auroc_loss(mort_logit, labels, race_ids)

        L_rank = (L_rank_age + L_rank_race) / 2.0

        # Adversarial demographic classification loss (GRL)
        L_adv_age  = torch.tensor(0.0, device=mort_logit.device)
        L_adv_race = torch.tensor(0.0, device=mort_logit.device)
        if age_logit is not None and age_ids is not None:
            L_adv_age = F.cross_entropy(age_logit, age_ids)
        if race_logit is not None and race_ids is not None:
            L_adv_race = F.cross_entropy(race_logit, race_ids)
        L_adv = (L_adv_age + L_adv_race) / 2.0

        total = L_ce + lambda_rank * L_rank + lambda_adv * L_adv

        return {
            "total": total,
            "ce": L_ce,
            "rank_age": L_rank_age,
            "rank_race": L_rank_race,
            "adv_age": L_adv_age,
            "adv_race": L_adv_race,
        }

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
