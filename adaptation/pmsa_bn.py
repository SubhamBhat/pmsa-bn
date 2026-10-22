"""
adaptation/pmsa_bn.py — Core PMSA-BN Module
=============================================
Prior-Matched Shift-Adaptive BatchNorm (PMSA-BN)

This module wraps a standard nn.BatchNorm2d layer and replaces
its normalization at test time using the 8-step algorithm.

ALGORITHM STEPS (implemented in forward()):
  Step 3: Class-conditional target statistics (per channel)
  Step 4: James-Stein / Bayesian shrinkage (stabilises batch size 1)
  Step 5: Prior-matched recombination (removes label-proportion confound)
  Step 6: Scale-free Wasserstein distance per channel
  Step 7: Channel mixing weight α_c = 1 - exp(-D_c / τ)
  Step 8: Mix statistics and normalize
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PMSABatchNorm2d(nn.Module):
    """
    Drop-in replacement for nn.BatchNorm2d that performs PMSA-BN adaptation.

    Usage:
        1. Wrap the model using `replace_bn_with_pmsa(model)`
        2. Call `adaptor.set_pseudo_mask(mask)` before each test image
        3. Forward pass produces adapted output

    Attributes:
        bn             : original BatchNorm2d layer (weights/bias preserved)
        num_classes    : number of segmentation classes (default 3)
        src_class_mean : [K, C] per-class, per-channel source mean
        src_class_var  : [K, C] per-class, per-channel source variance
        src_prior      : [K] source class proportions π_k^src
        adapt_mode     : bool — True enables adaptation; False = standard BN
        pseudo_mask    : [1, K, H, W] soft probability map (set externally)
        tau            : temperature τ for α_c computation
        kappa_mu       : shrinkage pseudo-count for mean
        kappa_sigma    : shrinkage pseudo-count for variance
    """

    def __init__(self, bn_layer: nn.BatchNorm2d, num_classes: int = 3,
                 tau: float = 1.0, kappa_mu: float = 8.0, kappa_sigma: float = 16.0):
        super().__init__()

        # ── Preserve original BN parameters ──────────────────────────────
        self.bn = bn_layer         # keeps weight (γ), bias (β), eps, momentum
        C = bn_layer.num_features

        # ── Source class statistics (set in Phase 0) ─────────────────────
        self.register_buffer("src_class_mean", torch.zeros(num_classes, C))
        self.register_buffer("src_class_var",  torch.ones(num_classes, C))
        self.register_buffer("src_prior",      torch.ones(num_classes) / num_classes)

        # ── Adaptation state ─────────────────────────────────────────────
        self.num_classes = num_classes
        self.tau         = tau
        self.kappa_mu    = kappa_mu
        self.kappa_sigma = kappa_sigma

        self.adapt_mode  = False
        self.pseudo_mask: Optional[torch.Tensor] = None

        # ── Logged diagnostics (useful for generating paper figures) ──────
        self.last_D_c   : Optional[torch.Tensor] = None   # [C] shift distances
        self.last_alpha_c: Optional[torch.Tensor] = None  # [C] mixing weights

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def set_source_stats(self, class_mean: torch.Tensor,
                         class_var: torch.Tensor,
                         prior: torch.Tensor):
        """
        Store Phase-0 source statistics.

        Args:
            class_mean : [K, C] per-class per-channel mean
            class_var  : [K, C] per-class per-channel variance
            prior      : [K]   source pixel proportion per class
        """
        device = self.bn.weight.device
        self.src_class_mean = class_mean.to(device)
        self.src_class_var  = class_var.to(device)
        self.src_prior      = prior.to(device)

    def set_pseudo_mask(self, mask: torch.Tensor):
        """
        Set the current pseudo-mask (Step 2 output).

        Args:
            mask : [K, H, W] or [1, K, H, W] soft probability map
                   Values in [0, 1]; high-confidence binary mask recommended.
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)           # → [1, K, H, W]
        self.pseudo_mask = mask

    # ─────────────────────────────────────────────────────────────────────
    # Internal computation steps
    # ─────────────────────────────────────────────────────────────────────

    def _step3_class_conditional_stats(
        self,
        x: torch.Tensor,           # [B, C, H, W]
        mask: torch.Tensor         # [B, K, H, W]  — already at layer resolution
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Step 3: Compute μ̂_{k,c}^tgt and σ̂²_{k,c}^tgt for each class k and channel c.

        Returns:
            tgt_mean : [K, C]
            tgt_var  : [K, C]
            Nk       : [K]    effective pixel count per class
        """
        B, C, H, W = x.shape
        K = self.num_classes

        tgt_mean = torch.zeros(K, C, device=x.device)
        tgt_var  = torch.zeros(K, C, device=x.device)
        Nk       = torch.zeros(K, device=x.device)

        for k in range(K):
            mask_k = mask[:, k:k+1, :, :]     # [B, 1, H, W]
            Nk[k]  = mask_k.sum()

            if Nk[k] > 1e-6:
                # Weighted mean: Σ mask_k * x  / Σ mask_k
                weighted_sum = (mask_k * x).sum(dim=[0, 2, 3])   # [C]
                mu_k = weighted_sum / Nk[k]                       # [C]

                # Weighted variance
                residual = x - mu_k.view(1, C, 1, 1)             # [B, C, H, W]
                var_k = (mask_k * residual.pow(2)).sum(dim=[0, 2, 3]) / Nk[k]

                tgt_mean[k] = mu_k
                tgt_var[k]  = var_k
            else:
                # No pixels → use source prior as fallback (will be corrected in Step 4)
                tgt_mean[k] = self.src_class_mean[k]
                tgt_var[k]  = self.src_class_var[k]

        return tgt_mean, tgt_var, Nk

    def _step4_bayesian_shrinkage(
        self,
        tgt_mean: torch.Tensor,   # [K, C]
        tgt_var:  torch.Tensor,   # [K, C]
        Nk:       torch.Tensor    # [K]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 4: James-Stein shrinkage toward source statistics.

        γ_{k,c} = N_k / (N_k + κ_μ)           ← weight on target mean
        η_{k,c} = max(0, N_k−1) / (max(0, N_k−1) + κ_σ)  ← weight on target var

        When N_k → 0 (tiny structure, deep layer): smoothly reverts to source.
        When N_k >> κ  (large structure, shallow): uses 100% target.
        """
        shrunk_mean = torch.zeros_like(tgt_mean)
        shrunk_var  = torch.zeros_like(tgt_var)

        for k in range(self.num_classes):
            nk = Nk[k].item()

            gamma_mu    = nk / (nk + self.kappa_mu)
            gamma_sigma = max(0.0, nk - 1.0) / (max(0.0, nk - 1.0) + self.kappa_sigma)

            shrunk_mean[k] = gamma_mu    * tgt_mean[k] + (1.0 - gamma_mu)    * self.src_class_mean[k]
            shrunk_var[k]  = gamma_sigma * tgt_var[k]  + (1.0 - gamma_sigma) * self.src_class_var[k]

        return shrunk_mean, shrunk_var

    def _step5_prior_matched_recombination(
        self,
        class_mean: torch.Tensor,   # [K, C]
        class_var:  torch.Tensor    # [K, C]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 5: Recombine class stats using SOURCE proportions (π_k^src).

        Law of Total Variance:
          μ_PM  = Σ_k π_k^src · μ_k
          σ²_PM = Σ_k π_k^src · σ_k²          (within-class)
                + Σ_k π_k^src · (μ_k − μ_PM)²  (between-class)

        Key: using π_k^src instead of target's π_k^tgt removes the
        label-proportion confound. Any remaining shift is pure appearance shift.
        """
        pi = self.src_prior   # [K]

        # μ_PM: [C]
        pm_mean = (pi.unsqueeze(1) * class_mean).sum(dim=0)

        # Within-class variance: [C]
        within_var = (pi.unsqueeze(1) * class_var).sum(dim=0)

        # Between-class variance: [C]
        diff        = class_mean - pm_mean.unsqueeze(0)          # [K, C]
        between_var = (pi.unsqueeze(1) * diff.pow(2)).sum(dim=0) # [C]

        pm_var = within_var + between_var    # [C]

        return pm_mean, pm_var

    def _step6_wasserstein_distance(
        self,
        pm_mean: torch.Tensor,   # [C]  prior-matched target mean
        pm_var:  torch.Tensor,   # [C]  prior-matched target variance
        src_mean: torch.Tensor,  # [C]  source BN running mean
        src_var:  torch.Tensor,  # [C]  source BN running variance
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Step 6: Scale-free normalised 2-Wasserstein distance per channel.

        D_c = [(μ_PM - μ_src)² + (σ_PM - σ_src)²] / (σ_src² + ε)

        Why Wasserstein (not KL)?
          KL has log(σ_src/σ_tgt) → explodes when σ_tgt → 0.
          W₂ is symmetric, bounded below by 0, and dividing by σ_src²
          makes it scale-free across all channels.

        Returns:
            D_c : [C]  non-negative shift distance per channel
        """
        sigma_pm  = pm_var.clamp(min=0.0).sqrt()
        sigma_src = src_var.clamp(min=0.0).sqrt()

        numerator   = (pm_mean - src_mean).pow(2) + (sigma_pm - sigma_src).pow(2)
        denominator = src_var + eps

        return numerator / denominator   # [C]

    def _step7_mixing_weights(self, D_c: torch.Tensor) -> torch.Tensor:
        """
        Step 7: Channel mixing weight.

            α_c = 1 − exp(−D_c / τ)  ∈ [0, 1)

        D_c ≈ 0 → α_c ≈ 0  (no shift → keep source)
        D_c >> τ → α_c → 1  (large shift → use target)
        """
        return 1.0 - torch.exp(-D_c / self.tau)   # [C]

    # ─────────────────────────────────────────────────────────────────────
    # Forward Pass
    # ─────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        If adapt_mode=False  → standard BatchNorm forward (training-time behaviour)
        If adapt_mode=True   → PMSA-BN adaptation (Steps 3–8)
        """
        # ── Standard mode (source training) ──────────────────────────────
        if not self.adapt_mode or self.pseudo_mask is None:
            return self.bn(x)

        B, C, H, W = x.shape

        # Downsample pseudo-mask to layer resolution (Step 3 prerequisite)
        mask = self.pseudo_mask.to(x.device)        # [1, K, H_orig, W_orig]
        if mask.shape[2] != H or mask.shape[3] != W:
            mask = F.interpolate(
                mask, size=(H, W),
                mode="bilinear", align_corners=False
            )
        mask = mask.expand(B, -1, -1, -1)           # [B, K, H, W]

        # ── Step 3: Class-conditional target statistics ───────────────────
        with torch.no_grad():
            tgt_mean, tgt_var, Nk = self._step3_class_conditional_stats(x, mask)

        # ── Step 4: Bayesian shrinkage ────────────────────────────────────
        shrunk_mean, shrunk_var = self._step4_bayesian_shrinkage(tgt_mean, tgt_var, Nk)

        # ── Step 5: Prior-matched recombination ───────────────────────────
        pm_mean, pm_var = self._step5_prior_matched_recombination(shrunk_mean, shrunk_var)

        # ── Step 6: Wasserstein distance per channel ──────────────────────
        src_mean = self.bn.running_mean.detach()   # [C]
        src_var  = self.bn.running_var.detach()    # [C]

        D_c = self._step6_wasserstein_distance(pm_mean, pm_var, src_mean, src_var)

        # ── Step 7: Compute α_c ───────────────────────────────────────────
        alpha_c = self._step7_mixing_weights(D_c)   # [C]

        # Store for diagnostics / paper figures
        self.last_D_c    = D_c.detach().cpu()
        self.last_alpha_c = alpha_c.detach().cpu()

        # ── Step 8: Mix statistics and normalize ──────────────────────────
        hat_mean = alpha_c * pm_mean + (1.0 - alpha_c) * src_mean   # [C]
        hat_var  = alpha_c * pm_var  + (1.0 - alpha_c) * src_var    # [C]

        hat_mean_4d = hat_mean.view(1, C, 1, 1)
        hat_std_4d  = (hat_var + self.bn.eps).clamp(min=1e-12).sqrt().view(1, C, 1, 1)

        x_norm = (x - hat_mean_4d) / hat_std_4d

        # Apply learned affine (γ, β) — NEVER modified
        gamma = self.bn.weight.view(1, C, 1, 1)
        beta  = self.bn.bias.view(1, C, 1, 1)

        return gamma * x_norm + beta


# ─────────────────────────────────────────────────────────────────────────
# Model Conversion Utility
# ─────────────────────────────────────────────────────────────────────────

def replace_bn_with_pmsa(model: nn.Module, num_classes: int = 3,
                          tau: float = 1.0, kappa_mu: float = 8.0,
                          kappa_sigma: float = 16.0) -> nn.Module:
    """
    Recursively replace every nn.BatchNorm2d in `model` with PMSABatchNorm2d.
    Preserves all learned weights (γ, β) and running statistics (μ, σ²).

    Call this AFTER loading the source-trained checkpoint.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            pmsa = PMSABatchNorm2d(
                bn_layer=module,
                num_classes=num_classes,
                tau=tau,
                kappa_mu=kappa_mu,
                kappa_sigma=kappa_sigma
            )
            setattr(model, name, pmsa)
        else:
            replace_bn_with_pmsa(module, num_classes, tau, kappa_mu, kappa_sigma)
    return model


def set_adapt_mode(model: nn.Module, adapt: bool):
    """Toggle adaptation mode for all PMSA layers."""
    for module in model.modules():
        if isinstance(module, PMSABatchNorm2d):
            module.adapt_mode = adapt


def set_pseudo_mask_all_layers(model: nn.Module, mask: torch.Tensor):
    """Set pseudo-mask for all PMSA layers simultaneously."""
    for module in model.modules():
        if isinstance(module, PMSABatchNorm2d):
            module.set_pseudo_mask(mask)


def get_all_diagnostics(model: nn.Module):
    """
    Collect D_c and α_c from all PMSA layers.
    Returns dict: {layer_idx: {"D_c": tensor, "alpha_c": tensor}}
    Used to generate the per-layer per-channel heatmap figure.
    """
    diagnostics = {}
    idx = 0
    for module in model.modules():
        if isinstance(module, PMSABatchNorm2d):
            if module.last_D_c is not None:
                diagnostics[idx] = {
                    "D_c":     module.last_D_c,
                    "alpha_c": module.last_alpha_c
                }
            idx += 1
    return diagnostics
