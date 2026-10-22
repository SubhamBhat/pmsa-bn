"""
test_adapt.py — Test-Time Adaptation & Evaluation
===================================================
Evaluates PMSA-BN and all baselines on the target domain.
Also generates paper figures 2, 3, 4.

RUN:
    python test_adapt.py
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy

import config
from train import build_model
from data.fundus_dataset import get_dataloaders
from adaptation.pmsa_bn import replace_bn_with_pmsa
from adaptation.source_stats import load_source_stats, inject_source_stats
from adaptation.adaptor import PMSAAdaptor, AdaBNAdaptor, AlphaBNAdaptor
from utils.metrics import evaluate_batch, aggregate_results, print_results_table
from utils.visualization import (
    plot_alpha_heatmap, plot_batch_size_curve, plot_cdr_scatter
)

os.makedirs(config.RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────

def load_source_model(device):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    model = build_model()
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Test] Source model loaded from {ckpt_path}")
    return model


def load_pmsa_model(device):
    """Load model + replace BN with PMSA-BN + inject source stats."""
    source_model = load_source_model(device)

    # Replace all BN layers with PMSA-BN
    pmsa_model = replace_bn_with_pmsa(
        copy.deepcopy(source_model),
        num_classes=config.NUM_CLASSES,
        tau=config.ALPHA_TAU,
        kappa_mu=config.KAPPA_MU,
        kappa_sigma=config.KAPPA_SIGMA
    )

    # Inject source class statistics
    stats = load_source_stats(config.STATS_DIR)
    inject_source_stats(pmsa_model, stats)

    return pmsa_model, source_model


# ─────────────────────────────────────────────────────────────────────────
# CDR Helper
# ─────────────────────────────────────────────────────────────────────────

def compute_cdr(pred_or_label: np.ndarray) -> float:
    """Compute Cup-to-Disc Ratio from label map."""
    disc = (pred_or_label >= 1).sum()
    cup  = (pred_or_label == 2).sum()
    return float(cup) / max(float(disc), 1.0)


# ─────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────────────────────────────────

def run_evaluation():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Test] Device: {device}")

    pmsa_model, source_model = load_pmsa_model(device)

    _, _, target_loader = get_dataloaders(
        source="REFUGE",
        target="Drishti",
        batch_size=config.EVAL_BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        num_workers=config.NUM_WORKERS
    )

    # ── Instantiate all methods ───────────────────────────────────────────
    methods = {
        "Source Only":    None,                     # handled separately
        "AdaBN":          AdaBNAdaptor(copy.deepcopy(source_model), device=str(device)),
        "α-BN (α=0.5)":  AlphaBNAdaptor(copy.deepcopy(source_model), alpha=0.5, device=str(device)),
        "PMSA-BN (Ours)": PMSAAdaptor(pmsa_model, em_iterations=config.EM_ITERATIONS, device=str(device)),
    }

    all_results   = {m: [] for m in methods}
    all_diag      = []      # for Figure 2
    true_cdrs     = []      # for Figure 4
    pred_cdrs     = {m: [] for m in methods}

    source_model.eval().to(device)

    print("\n[Test] Running evaluation on target domain (Drishti)...")

    for images, labels, stems in tqdm(target_loader, desc="Evaluating"):
        images = images.to(device)
        gt_np  = labels.squeeze().numpy()

        true_cdrs.append(compute_cdr(gt_np))

        # ── Source Only ──────────────────────────────────────────────────
        with torch.no_grad():
            src_probs = F.softmax(source_model(images), dim=1)
        all_results["Source Only"].append(evaluate_batch(src_probs.cpu(), labels))
        pred_cdrs["Source Only"].append(compute_cdr(src_probs.argmax(1).squeeze().cpu().numpy()))

        # ── AdaBN ────────────────────────────────────────────────────────
        # Re-load fresh model copy to avoid state accumulation
        adabn_model = copy.deepcopy(source_model)
        adabn = AdaBNAdaptor(adabn_model, device=str(device))
        adabn_probs = adabn.adapt(images)
        all_results["AdaBN"].append(evaluate_batch(adabn_probs.cpu(), labels))
        pred_cdrs["AdaBN"].append(compute_cdr(adabn_probs.argmax(1).squeeze().cpu().numpy()))

        # ── α-BN ─────────────────────────────────────────────────────────
        abn_model = copy.deepcopy(source_model)
        abn = AlphaBNAdaptor(abn_model, alpha=0.5, device=str(device))
        abn_probs = abn.adapt(images)
        all_results["α-BN (α=0.5)"].append(evaluate_batch(abn_probs.cpu(), labels))
        pred_cdrs["α-BN (α=0.5)"].append(compute_cdr(abn_probs.argmax(1).squeeze().cpu().numpy()))

        # ── PMSA-BN ──────────────────────────────────────────────────────
        adaptor = methods["PMSA-BN (Ours)"]
        pmsa_probs, diag = adaptor.adapt(images)
        all_results["PMSA-BN (Ours)"].append(evaluate_batch(pmsa_probs.cpu(), labels))
        pred_cdrs["PMSA-BN (Ours)"].append(compute_cdr(pmsa_probs.argmax(1).squeeze().cpu().numpy()))
        all_diag.append(diag)

    # ── Aggregate and print results ───────────────────────────────────────
    aggregated = {m: aggregate_results(r) for m, r in all_results.items()}
    print_results_table(aggregated)

    # Save numeric results
    import json
    with open(os.path.join(config.RESULTS_DIR, "results.json"), "w") as f:
        json.dump(aggregated, f, indent=2)

    # ── Generate Figures ──────────────────────────────────────────────────

    # Figure 2: α_c heatmap
    if all_diag and any(d for d in all_diag):
        plot_alpha_heatmap(
            all_diag,
            save_path=os.path.join(config.RESULTS_DIR, "fig2_alpha_heatmap.pdf")
        )

    # Figure 4: CDR scatter
    plot_cdr_scatter(
        true_cdrs         = true_cdrs,
        pred_cdrs_source  = pred_cdrs["Source Only"],
        pred_cdrs_adabn   = pred_cdrs["AdaBN"],
        pred_cdrs_pmsa    = pred_cdrs["PMSA-BN (Ours)"],
        save_path=os.path.join(config.RESULTS_DIR, "fig4_cdr_scatter.pdf")
    )

    print(f"\n[Test] All results and figures saved to: {config.RESULTS_DIR}/")
    return aggregated


# ─────────────────────────────────────────────────────────────────────────
# Batch-Size Robustness Experiment (Figure 3)
# ─────────────────────────────────────────────────────────────────────────

def run_batch_size_experiment():
    """
    Evaluate all methods at batch sizes B ∈ {1, 2, 4, 8}.
    Generates Figure 3.
    """
    print("\n[Batch Size Exp] Running...")
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    batch_sizes = [1, 2, 4, 8]

    pmsa_model, source_model = load_pmsa_model(device)

    bs_results = {
        "Source Only":      {},
        "AdaBN":            {},
        "PMSA-BN (Ours)":   {}
    }

    for bs in batch_sizes:
        print(f"  Batch size B = {bs}")
        _, _, loader = get_dataloaders(
            source="REFUGE", target="Drishti",
            batch_size=bs, image_size=config.IMAGE_SIZE,
            num_workers=0   # avoid multiprocessing issues
        )

        src_only_res = []
        adabn_res    = []
        pmsa_res     = []

        source_model.eval()

        for images, labels, _ in loader:
            images = images.to(device)

            # Source only
            with torch.no_grad():
                sp = F.softmax(source_model(images), dim=1)
            src_only_res.append(evaluate_batch(sp.cpu(), labels))

            # AdaBN (fresh copy each batch)
            adabn = AdaBNAdaptor(copy.deepcopy(source_model), device=str(device))
            ap = adabn.adapt(images)
            adabn_res.append(evaluate_batch(ap.cpu(), labels))

            # PMSA-BN
            adaptor = PMSAAdaptor(copy.deepcopy(pmsa_model), em_iterations=1, device=str(device))
            pp, _   = adaptor.adapt(images[0:1])   # always B=1 per image for PMSA
            pmsa_res.append(evaluate_batch(pp.cpu(), labels[0:1]))

        bs_results["Source Only"][bs]    = aggregate_results(src_only_res)["mean_dice_cup"]
        bs_results["AdaBN"][bs]          = aggregate_results(adabn_res)["mean_dice_cup"]
        bs_results["PMSA-BN (Ours)"][bs] = aggregate_results(pmsa_res)["mean_dice_cup"]

    plot_batch_size_curve(
        bs_results,
        metric="Cup Dice",
        save_path=os.path.join(config.RESULTS_DIR, "fig3_batch_size_curve.pdf")
    )

    print(f"[Batch Size Exp] Figure 3 saved.")
    return bs_results


# ─────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PMSA-BN — Proof of Concept Evaluation")
    print("  Target: REFUGE → Drishti-GS (domain shift)")
    print("=" * 60)

    run_evaluation()
    run_batch_size_experiment()
