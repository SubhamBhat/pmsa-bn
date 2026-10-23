"""
run_ablation.py — Full Ablation Table Generator
================================================
Generates the ablation table for the ISBI paper:

  Method                              | Disc Dice | Cup Dice | HD95 Cup
  ─────────────────────────────────── | ───────── | ──────── | ────────
  Source Only                         |  ...      |  ...     |  ...
  AdaBN (full target stats)           |  ...      |  ...     |  ...
  + Class-Split Statistics (Step 3)   |  ...      |  ...     |  ...
  + Prior Matching (Step 5)           |  ...      |  ...     |  ...
  + Bayesian Shrinkage (Step 4)       |  ...      |  ...     |  ...
  + Per-Channel α_c (Steps 6-7)      |  ...      |  ...     |  ...
  + EM Iteration × 2 (Full PMSA-BN)  |  ...      |  ...     |  ...

RUN:
    python run_ablation.py
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
from adaptation.pmsa_bn import (
    PMSABatchNorm2d, replace_bn_with_pmsa,
    set_adapt_mode, set_pseudo_mask_all_layers
)
from adaptation.source_stats import load_source_stats, inject_source_stats
from adaptation.adaptor import PMSAAdaptor, generate_pseudo_mask
from utils.metrics import evaluate_batch, aggregate_results, print_results_table


def load_pmsa_model(device):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    model = build_model()
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pmsa_model = replace_bn_with_pmsa(
        copy.deepcopy(model),
        num_classes=config.NUM_CLASSES,
        tau=config.ALPHA_TAU,
        kappa_mu=config.KAPPA_MU,
        kappa_sigma=config.KAPPA_SIGMA
    )
    stats = load_source_stats(config.STATS_DIR)
    inject_source_stats(pmsa_model, stats)
    return pmsa_model, model


# ─────────────────────────────────────────────────────────────────────────
# Ablation Variants
# ─────────────────────────────────────────────────────────────────────────

class AblationPMSA(PMSAAdaptor):
    """
    Configurable ablation version of PMSA-BN.
    Allows toggling each step on/off to isolate contributions.
    """

    def __init__(self, model, device,
                 use_class_split=True,    # Step 3: separate class stats
                 use_prior_match=True,    # Step 5: prior-matched recombination
                 use_shrinkage=True,      # Step 4: Bayesian shrinkage
                 use_per_channel=True,    # Steps 6-7: per-channel α_c
                 em_iters=1):
        super().__init__(model, em_iterations=em_iters, device=device)
        self.use_class_split = use_class_split
        self.use_prior_match = use_prior_match
        self.use_shrinkage   = use_shrinkage
        self.use_per_channel = use_per_channel

    @torch.no_grad()
    def adapt(self, image):
        x = image.to(self.device)

        # Pass 1: source stats
        set_adapt_mode(self.model, adapt=False)
        self.model.eval()
        logits = self.model(x)
        probs  = F.softmax(logits, dim=1)

        set_adapt_mode(self.model, adapt=True)

        for _ in range(self.em_iterations):
            pseudo_mask = generate_pseudo_mask(probs)
            set_pseudo_mask_all_layers(self.model, pseudo_mask)

            # Override PMSA layer behavior based on ablation flags
            for module in self.model.modules():
                if isinstance(module, PMSABatchNorm2d):
                    module._ablation_flags = {
                        "use_class_split": self.use_class_split,
                        "use_prior_match": self.use_prior_match,
                        "use_shrinkage":   self.use_shrinkage,
                        "use_per_channel": self.use_per_channel,
                    }

            logits = self.model(x)
            probs  = F.softmax(logits, dim=1)

        set_adapt_mode(self.model, adapt=False)
        return probs, {}


def run_ablation():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    pmsa_model, source_model = load_pmsa_model(device)

    _, _, loader = get_dataloaders(
        source="REFUGE", target="Drishti",
        batch_size=1, image_size=config.IMAGE_SIZE,
        num_workers=config.NUM_WORKERS
    )

    source_model.eval().to(device)

    # Define ablation methods (additive — each adds one component)
    ablation_configs = [
        ("1. Source Only",                    dict(use_class_split=False, use_prior_match=False, use_shrinkage=False, use_per_channel=False, em_iters=0)),
        ("2. AdaBN (full target)",            None),    # handled separately
        ("3. + Class-Split Stats",            dict(use_class_split=True,  use_prior_match=False, use_shrinkage=False, use_per_channel=False, em_iters=1)),
        ("4. + Prior Matching",               dict(use_class_split=True,  use_prior_match=True,  use_shrinkage=False, use_per_channel=False, em_iters=1)),
        ("5. + Bayesian Shrinkage",           dict(use_class_split=True,  use_prior_match=True,  use_shrinkage=True,  use_per_channel=False, em_iters=1)),
        ("6. + Per-Channel α_c",              dict(use_class_split=True,  use_prior_match=True,  use_shrinkage=True,  use_per_channel=True,  em_iters=1)),
        ("7. + EM×2 (Full PMSA-BN)",          dict(use_class_split=True,  use_prior_match=True,  use_shrinkage=True,  use_per_channel=True,  em_iters=2)),
    ]

    all_agg = {}

    for name, cfg in ablation_configs:
        print(f"\n→ Evaluating: {name}")
        results = []

        for images, labels, _ in tqdm(loader, leave=False, desc=name):
            images = images.to(device)

            if name.startswith("1."):   # Source only
                with torch.no_grad():
                    probs = F.softmax(source_model(images), dim=1)

            elif name.startswith("2."):   # AdaBN
                from adaptation.adaptor import AdaBNAdaptor
                adabn = AdaBNAdaptor(copy.deepcopy(source_model), device=str(device))
                probs = adabn.adapt(images)

            else:
                ablator = AblationPMSA(
                    copy.deepcopy(pmsa_model), device=str(device), **cfg
                )
                probs, _ = ablator.adapt(images)

            results.append(evaluate_batch(probs.cpu(), labels))

        agg = aggregate_results(results)
        all_agg[name] = agg

    print_results_table(all_agg)

    # Save
    import json
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "ablation.json"), "w") as f:
        json.dump(all_agg, f, indent=2)

    print(f"\n[Ablation] Results saved to {config.RESULTS_DIR}/ablation.json")


if __name__ == "__main__":
    print("=" * 60)
    print("  PMSA-BN — Ablation Study")
    print("=" * 60)
    run_ablation()
