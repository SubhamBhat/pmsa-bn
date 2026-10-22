"""
config.py — Central configuration for PMSA-BN Proof of Concept
================================================================
DATASETS (Kaggle Links):
  Primary:
    REFUGE (Retinal Fundus Glaucoma Challenge 2020):
      https://www.kaggle.com/datasets/andrewmvd/fundus-image-segmentation-refuge
      → 400 training + 400 val + 400 test images with disc & cup masks
      → MAIN benchmark used in paper

  Secondary (multi-site validation):
    Drishti-GS:
      https://www.kaggle.com/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-individual-masks
      → 101 images, 50 train / 51 test, with soft masks from 4 annotators

    RIM-ONE DL:
      https://www.kaggle.com/datasets/riadalmadani/rimone-dl
      → 313 training + 153 test images

    ORIGA (optional, need to request from original authors):
      https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset
      → 650 images

HOW TO USE:
  1. Download datasets from Kaggle using:
       kaggle datasets download -d andrewmvd/fundus-image-segmentation-refuge
  2. Unzip to DATA_ROOT (set below)
  3. Run: python train.py        <- trains source model on REFUGE
  4. Run: python test_adapt.py   <- evaluates PMSA-BN on target domains
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_ROOT = "./data"   # ← Change to your dataset location

REFUGE_TRAIN_DIR = os.path.join(DATA_ROOT, "REFUGE/train")
REFUGE_VAL_DIR   = os.path.join(DATA_ROOT, "REFUGE/val")
REFUGE_TEST_DIR  = os.path.join(DATA_ROOT, "REFUGE/test")

DRISHTI_DIR = os.path.join(DATA_ROOT, "Drishti-GS")
RIMONE_DIR  = os.path.join(DATA_ROOT, "RIM-ONE-DL")

CHECKPOINT_DIR = "./checkpoints"
STATS_DIR      = "./source_stats"        # where Phase-0 stats are saved
RESULTS_DIR    = "./results"
LOG_DIR        = "./logs"

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
ENCODER      = "mobilenet_v2"   # encoder backbone: "mobilenet_v2" (fastest, ~2.2M params), "resnet34", "densenet121", "resnet50"
ENCODER_WEIGHTS = "imagenet"   # pretrained weights
NUM_CLASSES  = 3               # 0=background, 1=disc rim, 2=cup
IMAGE_SIZE   = 512             # resize all images to this

# ─────────────────────────────────────────────
# TRAINING (Source domain)
# ─────────────────────────────────────────────
TRAIN_BATCH_SIZE = 4
NUM_EPOCHS       = 100
LR               = 1e-4
WEIGHT_DECAY     = 1e-5
LR_SCHEDULER     = "cosine"
EARLY_STOP_PATIENCE = 15

# Class weights to handle imbalance (bg:rim:cup ≈ 95:3.5:1.5)
CLASS_WEIGHTS = [0.2, 1.5, 3.0]

# ─────────────────────────────────────────────
# PMSA-BN ADAPTATION HYPERPARAMETERS
# ─────────────────────────────────────────────
# Step 2 — Pseudo-mask thresholds
TAU_HI = 0.90    # confident foreground threshold
TAU_LO = 0.10    # confident background threshold

# Step 4 — Bayesian shrinkage pseudo-counts
KAPPA_MU    = 8.0   # pseudo-count for mean shrinkage
KAPPA_SIGMA = 16.0  # pseudo-count for variance shrinkage

# Step 7 — Channel mixing temperature
ALPHA_TAU = 1.0     # τ in α_c = 1 - exp(-D_c / τ)

# EM iterations (how many times to repeat Steps 2–8)
EM_ITERATIONS = 2   # 1 = single refinement pass; 2 = two refinement passes

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
EVAL_BATCH_SIZE = 1   # clinical setting: one image at a time

# ─────────────────────────────────────────────
# MISC
# ─────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda"    # or "cpu"
NUM_WORKERS = 4
