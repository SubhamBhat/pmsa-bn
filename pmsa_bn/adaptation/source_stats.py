"""
adaptation/source_stats.py — Phase 0: Collect Source Statistics
================================================================
Runs the source training set through the trained model once, collecting
per-class, per-channel mean and variance at every BatchNorm layer.

These stats are stored on disk and loaded at test time.

HOW TO RUN:
    python -c "from adaptation.source_stats import collect_and_save; collect_and_save()"
  or via:
    python collect_stats.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple

import sys
from pmsa_bn import config
from pmsa_bn.adaptation.pmsa_bn import PMSABatchNorm2d


# ─────────────────────────────────────────────────────────────────────────
# Hook-Based Statistics Collector
# ─────────────────────────────────────────────────────────────────────────

class SourceStatCollector:
    """
    Attaches forward hooks to every BN layer in the model.
    For each BN layer, accumulates per-class feature statistics across
    the entire source training set using Welford's online algorithm
    (numerically stable running mean + variance).

    Usage:
        collector = SourceStatCollector(model, num_classes=3)
        for images, labels in dataloader:
            probs = softmax(model(images))    ← class probabilities
            collector.update(probs, labels)   ← register current batch
        stats = collector.finalize()
    """

    def __init__(self, model: nn.Module, num_classes: int = 3):
        self.model       = model
        self.num_classes = num_classes
        self.hooks       = []

        # Containers: list of dicts, one per BN layer
        # Each dict: {"sum_k": [K,C], "sum_sq_k": [K,C], "count_k": [K]}
        self._layer_accum = []
        self._bn_modules  = []
        self._current_labels: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self):
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, PMSABatchNorm2d)):
                # Get channel count
                bn = module.bn if isinstance(module, PMSABatchNorm2d) else module
                C  = bn.num_features

                # Initialize accumulators
                accum = {
                    "sum_k":    torch.zeros(self.num_classes, C),
                    "sum_sq_k": torch.zeros(self.num_classes, C),
                    "count_k":  torch.zeros(self.num_classes),
                }
                self._layer_accum.append(accum)
                self._bn_modules.append(module)

                idx = len(self._layer_accum) - 1

                def make_hook(layer_idx):
                    def hook_fn(mod, inp, out):
                        # inp[0]: [B, C, H, W] feature map before BN
                        x = inp[0].detach().cpu()
                        if self._current_labels is None:
                            return

                        B, C, H, W = x.shape

                        # Downsample labels to feature map resolution
                        lbl = self._current_labels.float().unsqueeze(1)  # [B,1,H0,W0]
                        lbl_down = F.interpolate(lbl, size=(H, W),
                                                 mode="nearest").squeeze(1).long()  # [B,H,W]

                        for k in range(self.num_classes):
                            # Build binary mask for class k
                            mask_k = (lbl_down == k).float()   # [B, H, W]
                            count  = mask_k.sum()
                            if count < 1:
                                continue

                            # Weighted per-channel sum: [C]
                            # x: [B, C, H, W], mask_k: [B, H, W] → [B,1,H,W]
                            mask_k_4d   = mask_k.unsqueeze(1)
                            channel_sum = (x * mask_k_4d).sum(dim=[0, 2, 3])   # [C]
                            channel_sq  = (x.pow(2) * mask_k_4d).sum(dim=[0, 2, 3])

                            acc = self._layer_accum[layer_idx]
                            acc["sum_k"][k]    += channel_sum
                            acc["sum_sq_k"][k] += channel_sq
                            acc["count_k"][k]  += count

                    return hook_fn

                h = module.register_forward_hook(make_hook(idx))
                self.hooks.append(h)

    def update(self, labels: torch.Tensor):
        """
        Set the ground-truth labels for the upcoming forward pass.
        Call this BEFORE model(images).

        Args:
            labels: [B, H, W]  integer class labels (0, 1, ..., K-1)
        """
        self._current_labels = labels.cpu()

    def finalize(self) -> Dict:
        """
        Compute final per-class per-channel mean and variance from accumulators.

        Returns dict with keys:
            "class_mean"  : list of [K, C] tensors, one per BN layer
            "class_var"   : list of [K, C] tensors
            "prior"       : [K] global class pixel proportions
        """
        class_means = []
        class_vars  = []

        for acc in self._layer_accum:
            K, C = acc["sum_k"].shape
            means  = torch.zeros(K, C)
            vars_  = torch.ones(K, C)

            for k in range(K):
                n = acc["count_k"][k].item()
                if n > 0:
                    mu       = acc["sum_k"][k] / n               # [C]
                    e_x2     = acc["sum_sq_k"][k] / n            # E[x²]
                    var      = (e_x2 - mu.pow(2)).clamp(min=1e-6)
                    means[k] = mu
                    vars_[k] = var

            class_means.append(means)
            class_vars.append(vars_)

        # Global prior: average over all layers
        global_prior = torch.zeros(self.num_classes)
        for acc in self._layer_accum:
            total = acc["count_k"].sum().clamp(min=1e-6)
            global_prior += acc["count_k"] / total
        global_prior /= len(self._layer_accum)
        global_prior /= global_prior.sum()

        return {
            "class_mean": class_means,
            "class_var":  class_vars,
            "prior":      global_prior
        }

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────

def save_source_stats(stats: Dict, save_dir: str):
    """Save collected source statistics to disk."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(stats, os.path.join(save_dir, "source_stats.pth"))
    print(f"[Phase 0] Source statistics saved to {save_dir}/source_stats.pth")


def load_source_stats(save_dir: str) -> Dict:
    """Load source statistics from disk."""
    path = os.path.join(save_dir, "source_stats.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Source stats not found at {path}. Run collect_stats.py first."
        )
    stats = torch.load(path, map_location="cpu")
    print(f"[Phase 0] Loaded source statistics from {path}")
    return stats


# ─────────────────────────────────────────────────────────────────────────
# Inject Loaded Stats into PMSA Model
# ─────────────────────────────────────────────────────────────────────────

def inject_source_stats(model: nn.Module, stats: Dict):
    """
    Push collected source stats into each PMSABatchNorm2d layer.
    Must be called after replace_bn_with_pmsa(model) and
    after load_source_stats().
    """
    pmsa_layers = [m for m in model.modules() if isinstance(m, PMSABatchNorm2d)]
    class_means = stats["class_mean"]
    class_vars  = stats["class_var"]
    prior       = stats["prior"]

    assert len(pmsa_layers) == len(class_means), (
        f"Mismatch: {len(pmsa_layers)} PMSA layers vs "
        f"{len(class_means)} saved stats entries."
    )

    for layer, mu, var in zip(pmsa_layers, class_means, class_vars):
        layer.set_source_stats(mu, var, prior)

    print(f"[Phase 0] Injected source stats into {len(pmsa_layers)} PMSA-BN layers.")


# ─────────────────────────────────────────────────────────────────────────
# Full Phase-0 Pipeline
# ─────────────────────────────────────────────────────────────────────────

def collect_and_save(model: nn.Module, dataloader, device: str = "cuda",
                     save_dir: str = config.STATS_DIR):
    """
    Complete Phase-0 pipeline:
      1. Attach hooks
      2. Run source training set through model
      3. Finalize statistics
      4. Save to disk

    Args:
        model      : source-trained model (with standard BN or PMSA-BN)
        dataloader : source training DataLoader (with ground-truth labels)
        device     : "cuda" or "cpu"
        save_dir   : where to save source_stats.pth
    """
    model.eval()
    model.to(device)

    collector = SourceStatCollector(model, num_classes=config.NUM_CLASSES)

    print("[Phase 0] Collecting source statistics...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Phase 0"):
            images, labels, _ = batch
            images = images.to(device)

            collector.update(labels)          # register labels for hook
            _ = model(images)                 # triggers hooks

    stats = collector.finalize()
    collector.remove_hooks()

    save_source_stats(stats, save_dir)
    return stats
