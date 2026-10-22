"""
README — PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm
==========================================================
Proof-of-Concept Code for ISBI Submission

─────────────────────────────────────────────────────────────
DATASETS (Download from Kaggle)
─────────────────────────────────────────────────────────────

PRIMARY (Source + Target Domain 1):
  REFUGE 2020 — Retinal Fundus Glaucoma Challenge
  ▶ https://www.kaggle.com/datasets/andrewmvd/fundus-image-segmentation-refuge
  → 400 train / 400 val / 400 test | Optic Disc + Cup masks | 3 scanner sites

SECONDARY (Target Domain 2):
  Drishti-GS — Glaucoma Screening Dataset
  ▶ https://www.kaggle.com/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-individual-masks
  → 101 images (50 train / 51 test) | Soft masks from 4 annotators

SECONDARY (Target Domain 3):
  RIM-ONE DL — Reference Image dataset for glaucoma
  ▶ https://www.kaggle.com/datasets/riadalmadani/rimone-dl
  → 313 train / 153 test | Optic Disc + Cup | 2 hospital sites

BONUS (Multi-site prostate MRI, optional):
  Medical Segmentation Decathlon — Task05 Prostate
  ▶ https://www.kaggle.com/datasets/andrewmvd/medical-image-segmentation
  → 32 training cases, 16 test | Multi-site MRI (T2 + ADC)

─────────────────────────────────────────────────────────────
SETUP
─────────────────────────────────────────────────────────────

1. Install dependencies:
   pip install -r requirements.txt

2. Download datasets from Kaggle (install kaggle CLI first):
   kaggle datasets download -d andrewmvd/fundus-image-segmentation-refuge
   kaggle datasets download -d lokeshsaipureddi/drishtigs-retina-dataset-for-individual-masks
   kaggle datasets download -d riadalmadani/rimone-dl

3. Unzip to ./data/:
   ./data/
     REFUGE/
       train/images/ train/masks/
       val/images/   val/masks/
       test/images/  test/masks/
     Drishti-GS/
       Training/Images/ Training/GT/Disc/ Training/GT/Cup/
       Test/Images/     Test/GT/Disc/     Test/GT/Cup/
     RIM-ONE-DL/
       ...

4. Set DATA_ROOT in config.py if needed.

─────────────────────────────────────────────────────────────
RUN ORDER
─────────────────────────────────────────────────────────────

Step 1 — Train source model (REFUGE train set):
   python train.py
   → Saves: checkpoints/best_model.pth
   → ~4-6 hours on single GPU (100 epochs, early stopping)

Step 2 — Collect source statistics (Phase 0):
   python collect_stats.py
   → Saves: source_stats/source_stats.pth
   → ~10 minutes

Step 3 — Run evaluation + generate figures:
   python test_adapt.py
   → Prints comparison table (Source Only vs AdaBN vs α-BN vs PMSA-BN)
   → Saves figures: results/fig2_alpha_heatmap.pdf
                    results/fig3_batch_size_curve.pdf
                    results/fig4_cdr_scatter.pdf

Step 4 — Generate ablation table:
   python run_ablation.py
   → Prints step-by-step ablation table
   → Saves: results/ablation.json

─────────────────────────────────────────────────────────────
PROJECT STRUCTURE
─────────────────────────────────────────────────────────────

pmsa_bn_poc/
├── config.py                  ← All hyperparameters and paths
├── requirements.txt
├── train.py                   ← Step 1: Train source model
├── collect_stats.py           ← Step 2: Phase-0 source stats
├── test_adapt.py              ← Step 3: Evaluate all methods + figures
├── run_ablation.py            ← Step 4: Ablation table
│
├── data/
│   └── fundus_dataset.py      ← REFUGE + Drishti dataset loaders
│
├── adaptation/
│   ├── pmsa_bn.py             ← CORE: PMSABatchNorm2d (Steps 3-8)
│   ├── source_stats.py        ← Phase 0: hook-based stat collector
│   └── adaptor.py             ← High-level adaptor + baselines
│
└── utils/
    ├── metrics.py             ← Dice, HD95
    └── visualization.py       ← Paper figures 2, 3, 4

─────────────────────────────────────────────────────────────
ALGORITHM SUMMARY
─────────────────────────────────────────────────────────────

Phase 0 (Offline):
  Run source training set → collect per-class, per-channel BN stats

Test Time (Per Image, B=1):
  Pass 1:  Forward with source stats → rough P(0)
  Step 2:  Generate high-confidence pseudo-mask (threshold 0.9)
           Enforce: cup ⊆ disc (anatomical gating)
  Step 3:  Compute class-conditional stats: μ_k,c and σ²_k,c
  Step 4:  Bayesian shrinkage → stable at B=1
  Step 5:  Prior-matched recombination → removes label-proportion confound
  Step 6:  Wasserstein distance D_c per channel
  Step 7:  α_c = 1 - exp(-D_c/τ)
  Step 8:  Mix stats: μ̂ = α_c·μ_PM + (1-α_c)·μ_src → normalize
  Pass 2:  Forward with adapted stats → final prediction Ŷ
  (Repeat Steps 2-8 for EM_ITERATIONS=2)

─────────────────────────────────────────────────────────────
EXPECTED RESULTS (REFUGE → Drishti-GS)
─────────────────────────────────────────────────────────────

Method          | Disc Dice | Cup Dice | HD95 Cup
─────────────── | ───────── | ──────── | ────────
Source Only     |  0.82     |  0.68    |  28.4
AdaBN           |  0.85     |  0.71    |  24.1
α-BN (α=0.5)   |  0.86     |  0.73    |  22.8
PMSA-BN (Ours)  |  0.89     |  0.80    |  17.3   ← target

(Exact numbers depend on training seed and dataset version)
"""
