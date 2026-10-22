"""
adaptation/adaptor.py — High-Level Test-Time Adaptation Orchestrator
=====================================================================
This file wires together:
  - Pass 1: initial forward pass with source statistics
  - Pseudo-mask generation with anatomical gating (Step 2)
  - Pass 2: recalibrated PMSA-BN forward pass (Steps 3–8)
  - Optional EM iterations (Step 9)

Also implements:
  - AdaBN baseline
  - α-BN (global single mixing weight) baseline
for ablation table generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from adaptation.pmsa_bn import (
    PMSABatchNorm2d, set_adapt_mode,
    set_pseudo_mask_all_layers, get_all_diagnostics
)


# ─────────────────────────────────────────────────────────────────────────
# Pseudo-Mask Generation (Step 2)
# ─────────────────────────────────────────────────────────────────────────

def generate_pseudo_mask(
    probs: torch.Tensor,          # [1, K, H, W]  softmax probabilities
    tau_hi: float = config.TAU_HI,
    tau_lo: float = config.TAU_LO,
    enforce_anatomy: bool = True
) -> torch.Tensor:
    """
    Convert raw probability map into a high-confidence pseudo-mask.

    Step 2 rules:
      - Pixel assigned to class k only if P_k > tau_hi
      - Pixels with max confidence < tau_hi → set to 0 (ignored)
      - Anatomical constraint: cup ∈ disc  (class 2 ⊆ class 1 ∪ class 2)

    Args:
        probs          : [1, K, H, W] softmax output from Pass 1
        tau_hi         : confidence threshold for foreground (default 0.9)
        tau_lo         : confidence below which pixel is background (default 0.1)
        enforce_anatomy: whether to enforce cup ⊆ disc containment

    Returns:
        mask : [1, K, H, W]  soft mask (binary with uncertain=0)
    """
    K = probs.shape[1]
    mask = torch.zeros_like(probs)   # [1, K, H, W]

    max_prob, argmax_class = probs.max(dim=1, keepdim=True)   # [1,1,H,W]

    for k in range(K):
        # Keep pixel for class k only if it is the argmax AND confidence > tau_hi
        confident = (argmax_class == k) & (max_prob > tau_hi)   # [1,1,H,W]
        mask[:, k:k+1, :, :] = confident.float()

    # Anatomical constraint: cup (class 2) cannot be outside disc (class 1 or 2)
    if enforce_anatomy and K >= 3:
        disc_region = (mask[:, 1:2, :, :] + mask[:, 2:3, :, :]).clamp(0, 1)  # disc + cup
        mask[:, 2:3, :, :] = mask[:, 2:3, :, :] * disc_region                 # cup ⊆ disc region

    # Safety check: if cup-to-disc ratio > 0.85 (anatomically impossible),
    # zero out cup mask to trigger shrinkage
    if enforce_anatomy and K >= 3:
        n_cup  = mask[:, 2, :, :].sum().item()
        n_disc = (mask[:, 1, :, :] + mask[:, 2, :, :]).sum().item() + 1e-6
        cdr    = n_cup / n_disc
        if cdr > 0.85 or n_cup < 5:
            mask[:, 2:3, :, :] = 0.0   # too noisy → fall back to shrinkage in Step 4

    return mask


# ─────────────────────────────────────────────────────────────────────────
# PMSA-BN Adaptor — Main Class
# ─────────────────────────────────────────────────────────────────────────

class PMSAAdaptor:
    """
    High-level wrapper for PMSA-BN test-time adaptation.

    Usage:
        adaptor = PMSAAdaptor(model, em_iterations=2)
        pred, diag = adaptor.adapt(image)   # image: [1, 3, H, W]
    """

    def __init__(
        self,
        model: nn.Module,
        em_iterations: int = config.EM_ITERATIONS,
        enforce_anatomy: bool = True,
        device: str = config.DEVICE
    ):
        self.model           = model.to(device)
        self.em_iterations   = em_iterations
        self.enforce_anatomy = enforce_anatomy
        self.device          = device

        # Initially use source stats (adapt_mode=False)
        set_adapt_mode(self.model, adapt=False)

    @torch.no_grad()
    def adapt(self, image: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Full PMSA-BN adaptation pipeline for a single test image.

        Pass 1: Forward pass with source statistics → rough prediction P(0)
        Repeat for em_iterations:
            Generate pseudo-mask from P(t)
            Set mask in all PMSA layers
            Forward pass with adapted statistics → P(t+1)

        Args:
            image : [1, 3, H, W]  normalized test image tensor

        Returns:
            final_pred : [1, K, H, W]  final class probability map
            diagnostics: dict with per-layer D_c and α_c (for figures)
        """
        x = image.to(self.device)

        # ── Pass 1: Source statistics (adapt_mode off) ────────────────────
        set_adapt_mode(self.model, adapt=False)
        self.model.eval()

        logits = self.model(x)                          # [1, K, H, W]
        probs  = F.softmax(logits, dim=1)               # P(0)

        # ── EM Iterations ─────────────────────────────────────────────────
        set_adapt_mode(self.model, adapt=True)

        for iteration in range(self.em_iterations):
            # Step 2: Generate pseudo-mask from current probs
            pseudo_mask = generate_pseudo_mask(
                probs, enforce_anatomy=self.enforce_anatomy
            )                                           # [1, K, H, W]

            # Set mask in all PMSA-BN layers
            set_pseudo_mask_all_layers(self.model, pseudo_mask)

            # Pass 2: Forward with PMSA-BN adaptation (Steps 3–8)
            logits = self.model(x)
            probs  = F.softmax(logits, dim=1)           # P(t+1)

        # ── Collect diagnostics ───────────────────────────────────────────
        diagnostics = get_all_diagnostics(self.model)

        # Reset adapt mode for next image (episodic — no persistent state drift)
        set_adapt_mode(self.model, adapt=False)

        return probs, diagnostics

    def predict_source_only(self, image: torch.Tensor) -> torch.Tensor:
        """Baseline: run with source statistics only (no adaptation)."""
        set_adapt_mode(self.model, adapt=False)
        self.model.eval()
        x = image.to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        return F.softmax(logits, dim=1)


# ─────────────────────────────────────────────────────────────────────────
# Baselines (for Ablation Table)
# ─────────────────────────────────────────────────────────────────────────

class AdaBNAdaptor:
    """
    Baseline: AdaBN — update ALL BN stats with target domain statistics.
    No class splitting, no prior matching, no per-channel weights.

    Implementation: simply set BN to eval mode + feed target images
    to update running stats with momentum=1.0 (full replacement).
    """

    def __init__(self, model: nn.Module, device: str = config.DEVICE):
        self.model  = model.to(device)
        self.device = device

    @torch.no_grad()
    def adapt(self, image: torch.Tensor) -> torch.Tensor:
        x = image.to(self.device)

        # Temporarily set BN to training mode to accumulate target stats
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
                module.momentum = 1.0   # full replacement

        _ = self.model(x)   # updates running mean/var with target stats

        # Switch back to eval for final prediction
        self.model.eval()
        logits = self.model(x)
        return F.softmax(logits, dim=1)


class AlphaBNAdaptor:
    """
    Baseline: α-BN — single global mixing weight α for all channels.

    Intermediate ablation between source-only and full PMSA-BN.
    """

    def __init__(self, model: nn.Module, alpha: float = 0.5,
                 device: str = config.DEVICE):
        self.model  = model.to(device)
        self.alpha  = alpha
        self.device = device

        # Store original running stats
        self._src_stats = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self._src_stats[name] = {
                    "mean": module.running_mean.clone(),
                    "var":  module.running_var.clone()
                }

    @torch.no_grad()
    def adapt(self, image: torch.Tensor) -> torch.Tensor:
        x = image.to(self.device)

        # Collect target statistics
        tgt_stats = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
                module.momentum = 1.0

        _ = self.model(x)

        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                if name in self._src_stats:
                    src_m = self._src_stats[name]["mean"].to(self.device)
                    src_v = self._src_stats[name]["var"].to(self.device)
                    # Global α mix (same weight for all channels)
                    module.running_mean = self.alpha * module.running_mean + (1 - self.alpha) * src_m
                    module.running_var  = self.alpha * module.running_var  + (1 - self.alpha) * src_v

        logits = self.model(x)
        return F.softmax(logits, dim=1)
