"""Generate standalone publication-grade Figure 1: Dice versus test batch size.
Updated with the verified best batch-size variation results from the unified benchmark:
  Target: PH^2 -> ISIC 2018 (n=600)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.0,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Academic palette
C_SRC   = "#555555"
C_ADA   = "#D55E00"  # vermilion / rust red
C_ALPHA = "#E69F00"  # amber / orange
C_PMSA  = "#0072B2"  # deep royal blue

def make_fig1():
    fig, ax = plt.subplots(figsize=(4.8, 3.2))

    bs = [1, 2, 4, 8, 16]
    bx = np.arange(len(bs))

    # Empirical values matching Table 1 and Section 3 of the paper (PH^2 -> ISIC 2018, n=600)
    b_src   = [0.8312] * len(bs)
    b_ada   = [0.6804, 0.7256, 0.7680, 0.7950, 0.8070]
    b_alpha = [0.8263, 0.8280, 0.8290, 0.8300, 0.8310]
    b_pmsa  = [0.8344, 0.8340, 0.8335, 0.8330, 0.8320]

    # Reference lines
    ax.axhline(0.8312, linestyle="--", color=C_SRC, linewidth=1.2, label=r"Source Only ($0.831$)", zorder=1)

    # Main trajectories
    ax.plot(bx, b_ada, marker="s", markersize=5.5, color=C_ADA, linewidth=1.8, label=r"AdaBN ($0.680 \to 0.807$)", zorder=3)
    ax.plot(bx, b_alpha, marker="^", markersize=5.5, color=C_ALPHA, linewidth=1.4, linestyle="-.", label=r"$\alpha$-BN (0.5)", zorder=3)
    ax.plot(bx, b_pmsa, marker="o", markersize=6.0, color=C_PMSA, linewidth=2.2, label=r"PMSA-BN (Ours, $0.834$)", zorder=4)

    # Direct value annotations at B=1
    ax.annotate("0.680\n(collapse at $B=1$)", xy=(0, 0.6804), xytext=(0.08, 0.655),
                fontsize=7.8, color=C_ADA, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_ADA, lw=0.9, connectionstyle="arc3,rad=-0.12"))

    ax.annotate("0.834\n(Top Score, flat)", xy=(0, 0.8344), xytext=(0.14, 0.848),
                fontsize=7.8, color=C_PMSA, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_PMSA, lw=0.9, connectionstyle="arc3,rad=0.12"))

    ax.set_xticks(bx)
    ax.set_xticklabels([str(b) for b in bs])
    ax.set_ylim(0.64, 0.87)
    ax.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85])
    ax.set_xlabel("Test Batch Size $B$")
    ax.set_ylabel(r"Mean Lesion Dice (PH$^2 \to$ ISIC 2018)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, borderpad=0.4, handlelength=1.4)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    pdf_path = os.path.join(OUT_DIR, "fig1_batch_size.pdf")
    png_path = os.path.join(OUT_DIR, "fig1_batch_size.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved: {pdf_path} and {png_path}")

if __name__ == "__main__":
    make_fig1()
