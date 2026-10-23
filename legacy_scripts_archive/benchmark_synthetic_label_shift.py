import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Academic styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.0,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def simulate_proposition_1():
    """
    Direct empirical verification of Proposition 1:
    Pure label-proportion shift with ZERO appearance shift.
    
    Generates synthetic medical images where foreground (lesion/cup)
    appearance p(z|c=1) ~ N(mu_1, sigma_1^2) and background
    appearance p(z|c=0) ~ N(mu_0, sigma_0^2) are strictly invariant.
    Only the spatial foreground area fraction pi_tgt varies from 0.01 to 0.40.
    
    Source prior: pi_src = 0.03 (e.g. tiny optic cup or small polyp).
    """
    np.random.seed(42)
    torch.manual_seed(42)

    pi_src = 0.03
    mu_0, sigma_0 = 0.0, 1.0     # background appearance
    mu_1, sigma_1 = 3.0, 1.2     # foreground appearance (contrast = 3.0)

    # In source domain, BN calibrated running stats:
    # mu_src = (1-pi_src)*mu_0 + pi_src*mu_1
    # var_src = (1-pi_src)*sigma_0^2 + pi_src*sigma_1^2 + pi_src*(1-pi_src)*(mu_1-mu_0)^2
    mu_src = (1.0 - pi_src) * mu_0 + pi_src * mu_1
    var_src = (1.0 - pi_src) * (sigma_0**2) + pi_src * (sigma_1**2) + pi_src * (1.0 - pi_src) * ((mu_1 - mu_0)**2)

    # Target foreground area fractions to evaluate
    pi_targets = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45]
    H, W = 128, 128
    N_pixels = H * W
    n_images = 100

    results = {"pi": pi_targets, "source": [], "adabn": [], "pmsa": []}

    # Optimal Bayes threshold under source statistics
    # z_norm = (z - mu) / sqrt(var)
    # A linear classifier w*z_norm + b > 0 classifies foreground
    # Under source domain, optimal threshold in z-space is ~ 1.5
    w_clf = 2.0
    b_clf = - (1.5 - mu_src) / np.sqrt(var_src) * w_clf

    for pi_tgt in pi_targets:
        d_src_list, d_ada_list, d_pmsa_list = [], [], []

        for _ in range(n_images):
            # Ground truth mask with fraction pi_tgt
            mask = np.zeros(N_pixels, dtype=bool)
            n_fg = int(round(pi_tgt * N_pixels))
            fg_indices = np.random.choice(N_pixels, n_fg, replace=False)
            mask[fg_indices] = True

            # Generate pixels with ZERO appearance change
            z = np.zeros(N_pixels, dtype=np.float32)
            z[~mask] = np.random.normal(mu_0, sigma_0, size=(~mask).sum())
            z[mask]  = np.random.normal(mu_1, sigma_1, size=mask.sum())

            # ── 1. Source Only Normalization ──
            z_src = (z - mu_src) / np.sqrt(var_src + 1e-5)
            pred_src = (w_clf * z_src + b_clf) > 0

            # ── 2. AdaBN Normalization (single-image sample mean and sample variance) ──
            mu_tgt = np.mean(z)
            var_tgt = np.var(z)
            z_ada = (z - mu_tgt) / np.sqrt(var_tgt + 1e-5)
            pred_ada = (w_clf * z_ada + b_clf) > 0

            # ── 3. PMSA-BN Normalization ──
            # Pass 1 pseudo-mask (using Source model)
            pseudo_fg = pred_src
            pseudo_bg = ~pred_src

            # Class-conditional target statistics (with shrinkage)
            n_pseudo_fg = pseudo_fg.sum()
            n_pseudo_bg = pseudo_bg.sum()

            mu_1_tgt = np.mean(z[pseudo_fg]) if n_pseudo_fg > 0 else mu_1
            var_1_tgt = np.var(z[pseudo_fg]) if n_pseudo_fg > 1 else sigma_1**2

            mu_0_tgt = np.mean(z[pseudo_bg]) if n_pseudo_bg > 0 else mu_0
            var_0_tgt = np.var(z[pseudo_bg]) if n_pseudo_bg > 1 else sigma_0**2

            # Empirical Bayes shrinkage towards source class stats
            kappa_mu, kappa_sigma = 8.0, 16.0
            shrunk_mu_1 = (n_pseudo_fg * mu_1_tgt + kappa_mu * mu_1) / (n_pseudo_fg + kappa_mu)
            shrunk_var_1 = (n_pseudo_fg * var_1_tgt + kappa_sigma * (sigma_1**2)) / (n_pseudo_fg + kappa_sigma)
            shrunk_mu_0 = (n_pseudo_bg * mu_0_tgt + kappa_mu * mu_0) / (n_pseudo_bg + kappa_mu)
            shrunk_var_0 = (n_pseudo_bg * var_0_tgt + kappa_sigma * (sigma_0**2)) / (n_pseudo_bg + kappa_sigma)

            # Prior-Matched Recombination (using SOURCE PRIOR pi_src!)
            mu_pm = (1.0 - pi_src) * shrunk_mu_0 + pi_src * shrunk_mu_1
            var_pm = (1.0 - pi_src) * shrunk_var_0 + pi_src * shrunk_var_1 + \
                     pi_src * (1.0 - pi_src) * ((shrunk_mu_1 - shrunk_mu_0)**2)

            z_pmsa = (z - mu_pm) / np.sqrt(var_pm + 1e-5)
            pred_pmsa = (w_clf * z_pmsa + b_clf) > 0

            # Compute Dice scores
            def dice(p, g):
                inter = (p & g).sum()
                return (2.0 * inter + 1e-6) / (p.sum() + g.sum() + 1e-6)

            d_src_list.append(dice(pred_src, mask))
            d_ada_list.append(dice(pred_ada, mask))
            d_pmsa_list.append(dice(pred_pmsa, mask))

        results["source"].append(np.mean(d_src_list))
        results["adabn"].append(np.mean(d_ada_list))
        results["pmsa"].append(np.mean(d_pmsa_list))

    print("="*65)
    print("PROPOSITION 1 EMPIRICAL VERIFICATION: PURE LABEL-PROPORTION SHIFT")
    print("="*65)
    print("Target Area Fraction pi_tgt       Source Only    AdaBN (B=1)    PMSA-BN (Ours)")
    print("-" * 68)
    for i, pi in enumerate(pi_targets):
        print(f"pi_tgt = {pi:4.2f} ({pi*100:4.1f}%)        {results['source'][i]:.4f}         {results['adabn'][i]:.4f}         {results['pmsa'][i]:.4f}")
    print("="*65)

    # Plot figure
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.axvline(pi_src, color="#888888", linestyle=":", linewidth=1.2, label=f"Source Prior $\\pi^{{\\mathrm{{src}}}} = {pi_src}$")
    ax.plot(pi_targets, results["source"], label="Source Only (Fixed $\\mu^{\\mathrm{src}}, \\sigma^{\\mathrm{src}}$)", color="#555555", linestyle="--", linewidth=1.5)
    ax.plot(pi_targets, results["adabn"], label="AdaBN (Collapses under shift)", color="#D55E00", marker="s", markersize=5, linewidth=1.8)
    ax.plot(pi_targets, results["pmsa"], label="PMSA-BN (Invariant via Prior Matching)", color="#0072B2", marker="o", markersize=6, linewidth=2.2)

    ax.set_xlabel(r"Target Foreground Area Fraction $\pi^{\mathrm{tgt}}$ (Zero Appearance Change)")
    ax.set_ylabel("Segmentation Dice")
    ax.set_title("Empirical Verification of Proposition 1", fontweight="bold", pad=8)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=8.0)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(0.40, 1.00)

    plt.tight_layout()
    out_png = "c:/Users/Subham Bhat/Downloads/tds/pmsa_bn_poc/fig_prop1_synthetic_label_shift.png"
    out_pdf = "c:/Users/Subham Bhat/Downloads/tds/pmsa_bn_poc/fig_prop1_synthetic_label_shift.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    simulate_proposition_1()
