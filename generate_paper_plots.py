"""Generate ultra-compact publication figures for the 4-page PMSA-BN manuscript."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Colour-blind safe academic palette (Okabe-Ito inspired)
C_SRC   = "#666666"
C_ADA   = "#D55E00"  # rust red / vermilion
C_ALPHA = "#E69F00"  # amber / orange
C_PMSA  = "#0072B2"  # deep royal blue
C_ORAC  = "#009E73"  # Bluish green


def make_fig1_compact_3panel():
    """
    Single horizontal 1x3 figure (height ~2.3 in) designed to fit in a 4-page paper
    while replacing the separate ablation table:
      (a) Step-by-step component ablation (PH2 -> ISIC18, B=1)
      (b) Dice by ground-truth lesion-area quartile Q1-Q4 (proving label-proportion shift)
      (c) Dice vs. test batch size B in {1, 2, 4, 8, 16}
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 2.35),
                             gridspec_kw={"width_ratios": [1.22, 1.05, 1.0], "wspace": 0.32})

    # ── Panel (a): Component Ablation ──
    ax = axes[0]
    labels = [
        "1. Source Only",
        "2. AdaBN ($B=1$)",
        r"3. + Class-split (tgt $\pi$)",
        r"4. + Prior match (src $\pi^{\mathrm{src}}$)",
        "5. + Bayes shrinkage",
        r"6. + Per-ch. $\alpha_c$ (PMSA)",
        "7. + Pass 2 refine",
    ]
    vals = [0.8312, 0.6804, 0.6658, 0.7882, 0.8397, 0.8344, 0.8346]
    colors = [C_SRC, C_ADA, "#CC79A7", "#56B4E9", "#009E73", C_PMSA, C_PMSA]

    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, vals, height=0.62, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.axvline(0.8312, color=C_SRC, linestyle="--", linewidth=0.9)
    ax.axvline(0.8401, color=C_ORAC, linestyle=":", linewidth=1.0)

    for bar, v in zip(bars, vals):
        ax.text(v + 0.004, bar.get_y() + bar.get_height() / 2.0,
                f"{v:.3f}", va="center", ha="left", fontsize=7.3,
                fontweight="bold" if v >= 0.834 else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.4)
    ax.set_xlim(0.58, 0.885)
    ax.set_xlabel("Target Dice (PH$^2 \\to$ ISIC18, $B=1$)")
    ax.set_title("(a) Component Ablation", fontweight="bold", pad=5)
    ax.grid(axis="x", linestyle=":", alpha=0.45)

    # ── Panel (b): Dice by GT Lesion Area Quartile (Q1 = smallest lesions <7.5%) ──
    ax = axes[1]
    quartiles = ["Q1\n(<7.5%)", "Q2\n(7.5–18%)", "Q3\n(18–35%)", "Q4\n(>35%)"]
    x = np.arange(len(quartiles))
    w = 0.20

    q_src   = [0.774, 0.831, 0.867, 0.853]
    q_ada   = [0.242, 0.732, 0.896, 0.852]
    q_alpha = [0.733, 0.842, 0.888, 0.843]
    q_pmsa  = [0.775, 0.834, 0.870, 0.858]

    ax.bar(x - 1.5*w, q_src,   width=w, color=C_SRC,   edgecolor="black", linewidth=0.4, label="Source Only")
    ax.bar(x - 0.5*w, q_ada,   width=w, color=C_ADA,   edgecolor="black", linewidth=0.4, label="AdaBN ($B=1$)")
    ax.bar(x + 0.5*w, q_alpha, width=w, color=C_ALPHA, edgecolor="black", linewidth=0.4, label=r"$\alpha$-BN (0.5)")
    ax.bar(x + 1.5*w, q_pmsa,  width=w, color=C_PMSA,  edgecolor="black", linewidth=0.4, label="PMSA-BN")

    # Clean vertical callout right above Q1 AdaBN bar
    ax.text(0 - 0.5*w, 0.255, ".242", ha="center", va="bottom",
            fontsize=6.8, color=C_ADA, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(quartiles, fontsize=7.0)
    ax.set_ylim(0.15, 0.96)
    ax.set_ylabel("Mean Dice")
    ax.set_xlabel("Target Foreground Area Quartile")
    ax.set_title("(b) Robustness to Small Objects", fontweight="bold", pad=5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, handlelength=1.0, borderpad=0.28, fontsize=6.9)
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    # ── Panel (c): Dice vs Batch Size B in {1, 2, 4, 8, 16} ──
    ax = axes[2]
    bs = [1, 2, 4, 8, 16]
    bx = np.arange(len(bs))

    b_src   = [0.8312] * 5
    b_ada   = [0.6804, 0.7525, 0.7905, 0.7998, 0.8069]
    b_alpha = [0.8263, 0.8354, 0.8351, 0.8351, 0.8351]
    b_pmsa  = [0.8344, 0.8328, 0.8320, 0.8320, 0.8317]

    ax.plot(bx, b_src,   linestyle="--", color=C_SRC,   linewidth=1.2, label="Source Only")
    ax.plot(bx, b_ada,   marker="s", markersize=4.2, color=C_ADA,   linewidth=1.4, label="AdaBN")
    ax.plot(bx, b_alpha, marker="^", markersize=4.2, color=C_ALPHA, linewidth=1.2, label=r"$\alpha$-BN (0.5)")
    ax.plot(bx, b_pmsa,  marker="o", markersize=4.5, color=C_PMSA,  linewidth=1.6, label="PMSA-BN")

    ax.set_xticks(bx)
    ax.set_xticklabels([str(b) for b in bs])
    ax.set_ylim(0.65, 0.86)
    ax.set_xlabel("Test Batch Size $B$")
    ax.set_ylabel("Mean Dice")
    ax.set_title("(c) Batch-Size Invariance", fontweight="bold", pad=5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, handlelength=1.3, borderpad=0.28, fontsize=6.9)
    ax.grid(True, linestyle=":", alpha=0.45)

    pdf_path = os.path.join(OUT_DIR, "fig_paper_compact.pdf")
    png_path = os.path.join(OUT_DIR, "fig_paper_compact.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path} and {png_path}")


def make_fig2_modality_and_tau():
    """
    Optional second compact 1x2 figure:
      (a) Representative high-shift and collapse-prone cohorts across all 4 modalities
      (b) Source-validation temperature selection curve (clean vs shifted source val)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 2.30),
                             gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.28})

    # ── Panel (a): Cross-Modality Benchmark Highlights ──
    ax = axes[0]
    cohorts = [
        "Fundus\nPAPILA",
        "Fundus\nG1020 Cup",
        "Polyp\nK$\\to$ETIS",
        "Polyp\nK$\\to$Clinic",
        "CXR\nCOVID-QU",
        "Derm.\nISIC$\\to$HAM",
        "Derm.\nPH$^2\\to$ISIC",
    ]
    x = np.arange(len(cohorts))
    w = 0.20

    v_src   = [0.690, 0.612, 0.624, 0.826, 0.930, 0.919, 0.831]
    v_ada   = [0.881, 0.563, 0.377, 0.716, 0.923, 0.822, 0.680]
    v_alpha = [0.782, 0.610, 0.603, 0.816, 0.930, 0.910, 0.826]
    v_pmsa  = [0.891, 0.613, 0.651, 0.834, 0.940, 0.923, 0.834]

    ax.bar(x - 1.5*w, v_src,   width=w, color=C_SRC,   edgecolor="black", linewidth=0.4, label="Source Only")
    ax.bar(x - 0.5*w, v_ada,   width=w, color=C_ADA,   edgecolor="black", linewidth=0.4, label="AdaBN ($B=1$)")
    ax.bar(x + 0.5*w, v_alpha, width=w, color=C_ALPHA, edgecolor="black", linewidth=0.4, label=r"$\alpha$-BN (0.5)")
    ax.bar(x + 1.5*w, v_pmsa,  width=w, color=C_PMSA,  edgecolor="black", linewidth=0.4, label="PMSA-BN")

    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, fontsize=7.1)
    ax.set_ylim(0.30, 1.09)
    ax.set_yticks([0.3, 0.5, 0.7, 0.9])
    ax.set_ylabel("Dice Score ($B=1$)")
    ax.set_title("(a) Cross-Modality Adaptation & Collapse Prevention", fontweight="bold", pad=5)
    ax.legend(ncol=4, loc="upper center", frameon=True, framealpha=0.92, fontsize=6.9, handlelength=1.1)
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    # ── Panel (b): Source-Validation Temperature Selection Protocol ──
    ax = axes[1]
    taus = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    tx = np.arange(len(taus))
    clean_dice   = [0.9407, 0.9434, 0.9462, 0.9512, 0.9540, 0.9548]
    shifted_dice = [0.9417, 0.9435, 0.9445, 0.9476, 0.9505, 0.9509]

    ax.axhline(0.9552, color=C_SRC, linestyle="--", linewidth=1.0, label="Clean Source Only (0.9552)")
    ax.axhline(0.9552 - 0.003, color=C_ADA, linestyle=":", linewidth=0.9, label="Max allowed drop (-0.003)")
    ax.plot(tx, clean_dice,   marker="o", markersize=4.2, color=C_PMSA,  linewidth=1.5, label="PMSA-BN Clean Val")
    ax.plot(tx, shifted_dice, marker="s", markersize=4.0, color=C_ORAC,  linewidth=1.3, label="PMSA-BN Shifted Val")

    ax.scatter([5], [0.9548], s=55, facecolors="none", edgecolors="black", linewidths=1.3, zorder=5)
    ax.annotate("Selected $\\tau=10$", xy=(5, 0.9548), xytext=(2.2, 0.9532),
                fontsize=7.3, fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=0.8))

    ax.set_xticks(tx)
    ax.set_xticklabels([str(t) for t in taus])
    ax.set_ylim(0.938, 0.9575)
    ax.set_xlabel("Temperature $\\tau$ (log-spaced grid)")
    ax.set_ylabel("Source Validation Dice")
    ax.set_title("(b) Target-Free Selection of $\\tau$", fontweight="bold", pad=5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=6.7, handlelength=1.2)
    ax.grid(True, linestyle=":", alpha=0.45)

    pdf_path = os.path.join(OUT_DIR, "fig_modality_gains.pdf")
    png_path = os.path.join(OUT_DIR, "fig_modality_gains.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path} and {png_path}")


if __name__ == "__main__":
    make_fig1_compact_3panel()
    make_fig2_modality_and_tau()
