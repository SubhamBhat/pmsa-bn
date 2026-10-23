"""
utils/visualization.py — Paper Figure Generation
=================================================
Generates the 4 key figures required for the ISBI 4-page paper.

Figure 2: Per-layer, per-channel D_c heatmap (α_c values)
Figure 3: Dice vs. Batch Size robustness curve
Figure 4: Predicted vs. True Cup-to-Disc Ratio scatter plot
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────────
# Figure 2: Per-Layer Per-Channel α_c Heatmap
# ─────────────────────────────────────────────────────────────────────────

def plot_alpha_heatmap(
    all_diagnostics: List[Dict],    # list of diagnostics dicts over test images
    save_path: str = "results/fig2_alpha_heatmap.pdf",
    max_layers: int = 20,
    max_channels: int = 64
):
    """
    Plot average α_c (channel mixing weights) as a heatmap:
      - Rows: layer index (encoder → decoder)
      - Columns: channel index
      - Color: α_c ∈ [0, 1]  (0=keeps source, 1=full target)

    Shows that early layers adapt more (appearance shift)
    and deep layers adapt less (semantic preservation).
    """
    # Average α_c across all test images
    layer_keys = sorted(all_diagnostics[0].keys())[:max_layers]
    num_layers  = len(layer_keys)

    # Build matrix [num_layers, max_channels]
    alpha_matrix = np.zeros((num_layers, max_channels))

    for diag in all_diagnostics:
        for i, lk in enumerate(layer_keys):
            if lk in diag:
                alpha_c = diag[lk]["alpha_c"].numpy()
                C = min(len(alpha_c), max_channels)
                alpha_matrix[i, :C] += alpha_c[:C]

    alpha_matrix /= len(all_diagnostics)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        alpha_matrix,
        ax=ax,
        cmap="YlOrRd",
        vmin=0.0, vmax=1.0,
        xticklabels=8,
        yticklabels=True,
        cbar_kws={"label": r"$\alpha_c$ (Mixing Weight)"}
    )
    ax.set_xlabel("Channel Index $c$", fontsize=12)
    ax.set_ylabel("Layer Index (Encoder → Decoder)", fontsize=12)
    ax.set_title(r"Per-Layer, Per-Channel Mixing Weights $\alpha_c$" + "\n"
                 r"(0 = keep source, 1 = full target)", fontsize=13)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Figure 2] Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────
# Figure 3: Dice vs. Batch Size Curve
# ─────────────────────────────────────────────────────────────────────────

def plot_batch_size_curve(
    results_by_method: Dict[str, Dict[int, float]],
    # e.g. {"AdaBN": {1: 0.45, 2: 0.60, 4: 0.71, 8: 0.78},
    #        "PMSA-BN": {1: 0.79, 2: 0.80, 4: 0.81, 8: 0.81}}
    metric: str = "Cup Dice",
    save_path: str = "results/fig3_batch_size_curve.pdf"
):
    """
    Line plot of Cup Dice as a function of batch size.
    Shows PMSA-BN is flat (stable) while baselines degrade at B=1.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    styles = {
        "Source Only": dict(color="gray",   linestyle="--",  marker="x"),
        "AdaBN":       dict(color="tomato", linestyle="-.",  marker="s"),
        "α-BN":        dict(color="orange", linestyle=":",   marker="^"),
        "PMSA-BN (Ours)": dict(color="steelblue", linestyle="-", marker="o", linewidth=2.5),
    }

    for method, bs_dict in results_by_method.items():
        batch_sizes = sorted(bs_dict.keys())
        scores      = [bs_dict[b] for b in batch_sizes]
        style = styles.get(method, dict(marker="D"))
        ax.plot(batch_sizes, scores, label=method, **style)

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1\n(Clinical)", "2", "4", "8", "16"])
    ax.set_xlabel("Batch Size $B$", fontsize=12)
    ax.set_ylabel(f"{metric}", fontsize=12)
    ax.set_title(f"{metric} vs. Batch Size\n(Optic Cup, REFUGE→Drishti)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.3, 0.95])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Figure 3] Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────
# Figure 4: Predicted vs True CDR Scatter
# ─────────────────────────────────────────────────────────────────────────

def plot_cdr_scatter(
    true_cdrs: List[float],
    pred_cdrs_source: List[float],
    pred_cdrs_adabn: List[float],
    pred_cdrs_pmsa: List[float],
    save_path: str = "results/fig4_cdr_scatter.pdf"
):
    """
    Scatter plot of predicted CDR vs. true CDR.
    A perfect model lies on the y=x diagonal.
    Domain shift causes shrinkage (underprediction of cup size).
    PMSA-BN should stay close to diagonal.

    CDR = Cup-to-Disc Ratio = cup area / disc area
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)

    pairs = [
        ("Source Only", pred_cdrs_source, "gray"),
        ("AdaBN",       pred_cdrs_adabn,  "tomato"),
        ("PMSA-BN (Ours)", pred_cdrs_pmsa, "steelblue"),
    ]

    for ax, (method, preds, color) in zip(axes, pairs):
        ax.scatter(true_cdrs, preds, alpha=0.6, s=25, color=color, edgecolors="none")
        # y = x diagonal
        lims = [0.0, 1.0]
        ax.plot(lims, lims, "k--", linewidth=1.2, label="y = x (ideal)")

        # Regression line
        coeffs = np.polyfit(true_cdrs, preds, 1)
        x_fit  = np.linspace(min(true_cdrs), max(true_cdrs), 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "-", color=color,
                linewidth=1.5, alpha=0.8)

        # Compute correlation
        corr = np.corrcoef(true_cdrs, preds)[0, 1]

        ax.set_title(f"{method}\nr = {corr:.3f}", fontsize=11)
        ax.set_xlabel("True CDR", fontsize=11)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Predicted CDR", fontsize=11)
    fig.suptitle("Predicted vs. True Cup-to-Disc Ratio (REFUGE→Drishti)",
                 fontsize=13, y=1.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Figure 4] Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────
# Segmentation Overlay Visualization
# ─────────────────────────────────────────────────────────────────────────

def visualize_predictions(
    image: np.ndarray,     # [H, W, 3]  original RGB image
    gt: np.ndarray,        # [H, W]     ground truth labels
    pred: np.ndarray,      # [H, W]     predicted labels
    save_path: str = None,
    title: str = ""
):
    """
    Side-by-side: original image | ground truth overlay | prediction overlay
    Color coding: Green=Rim, Red=Cup, Black=Background
    """
    CLASS_COLORS = {
        0: np.array([0,   0,   0]),    # background → black
        1: np.array([0,   255, 0]),    # rim        → green
        2: np.array([255, 0,   0]),    # cup        → red
    }

    def overlay(img, mask, alpha=0.45):
        overlay_img = img.copy().astype(float)
        for cls, color in CLASS_COLORS.items():
            if cls == 0:
                continue
            region = (mask == cls)
            for c in range(3):
                overlay_img[:, :, c][region] = (
                    alpha * color[c] + (1 - alpha) * overlay_img[:, :, c][region]
                )
        return overlay_img.astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(image);           axes[0].set_title("Image");            axes[0].axis("off")
    axes[1].imshow(overlay(image, gt));   axes[1].set_title("Ground Truth"); axes[1].axis("off")
    axes[2].imshow(overlay(image, pred)); axes[2].set_title(f"Prediction {title}"); axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
