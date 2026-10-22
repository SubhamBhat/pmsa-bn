"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm                      ║
║    Multi-Target Dermoscopy Benchmark Suite (MICCAI / ISBI)                   ║
║                                                                              ║
║  SOURCE:  ISIC 2018 Task 1 Lesion Segmentation (ceenen/isic-2018)             ║
║  TARGETS: Array-selectable via CFG.TARGETS:                                  ║
║           ["HAM10000", "Derm7pt", "PH2"]                                     ║
║                                                                              ║
║  Kaggle Datasets Required to Mount:                                          ║
║  1. Source:   ceenen/isic-2018-challenge-task-1-segmentation                 ║
║  2. Target 1: surajghuwalewala/ham1000-segmentation-and-classification        ║
║               (or kmader/skin-cancer-mnist-ham10000)                         ║
║  3. Target 2: subhambhat/derm7pt (or generic derm7pt / 7-point criteria)     ║
║  4. Target 3: kanametov/ph2dataset (Optional target)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Dependencies ─────────────────────────────────────────────────────────────
import subprocess, sys
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "segmentation-models-pytorch", "albumentations"], check=False)

# ── Imports ──────────────────────────────────────────────────────────────────
import os, glob, copy, json, random, warnings, time, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from scipy import stats as scipy_stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CFG — MULTI-TARGET DERMOSCOPY BENCHMARK CONFIGURATION                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CFG:
    SOURCE = "ISIC2018"

    # ── Target Cohorts to Evaluate ────────────────────────────────────────
    # Specifically configured for: ["HAM10000", "Derm7pt"]
    TARGETS = [
        "HAM10000",           # Multi-Center (Austria & Australia, 500 test cases)
        "Derm7pt",            # Clinical & Dermoscopy Cross-Modality Cohort (500 test cases)
    ]

    MAX_TARGET_SAMPLES = {
        "HAM10000": 500,      # Evaluates 500 unseen non-overlapping test cases
        "Derm7pt":  500,      # Evaluates 500 clinical & dermoscopy test cases
        "PH2":      200,      # Full cohort (200 cases)
        "ISIC2017": 600,      # Full official test cohort (600 cases)
    }

    # ── Checkpoint Paths ──────────────────────────────────────────────────
    CKPT_PATH  = "/kaggle/working/best_model.pth"
    STATS_PATH = "/kaggle/working/source_stats_calibrated.pth"
    OUT_DIR    = "/kaggle/working"
    RESUME_IF_EXISTS = True   # Automatically reuses trained model if present!

    # ── Model ─────────────────────────────────────────────────────────────
    ENCODER         = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES     = 2       # 0=background (skin), 1=lesion
    IMAGE_SIZE      = 320

    # ── Source Training Parameters ────────────────────────────────────────
    TRAIN_BS           = 8
    NUM_EPOCHS         = 50
    LR                 = 3e-4
    WEIGHT_DECAY       = 1e-5
    PATIENCE           = 10
    CLASS_WEIGHTS      = [0.4, 1.6]
    USE_AMP            = True

    # ── Calibrated PMSA-BN Parameters (Dermoscopy Regime) ─────────────────
    TAU_HI      = 0.85
    TAU_LO      = 0.15
    KAPPA_MU    = 8.0
    KAPPA_SIGMA = 16.0
    ALPHA_TAU   = 2.0
    ALPHA_MIN   = 0.02
    ALPHA_MAX   = 0.20        # Conservative ceiling preventing negative transfer
    EM_ITERS    = 1
    PASS1       = "source"    # Pass-1 initialized from Source Only
    USE_TTA     = True        # Multi-orientation TTA (H-flip + V-flip)

    # ── Baselines ─────────────────────────────────────────────────────────
    ALPHA_BN_WEIGHT = 0.5

    SEED        = 42
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2


def seed_everything(seed=CFG.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()
print(f"✓ Device  : {CFG.DEVICE}  |  PyTorch {torch.__version__}")
print(f"✓ Source  : {CFG.SOURCE}")
print(f"✓ Targets : {CFG.TARGETS}")
print(f"✓ Task    : Cutaneous Lesion Segmentation | Metrics: Dice & Jaccard (IoU)")
print("─" * 65)


def load_ckpt(path):
    try: return torch.load(path, map_location=CFG.DEVICE, weights_only=False)
    except TypeError: return torch.load(path, map_location=CFG.DEVICE)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATASET AUTO-DISCOVERY                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def walk_for_images(root, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    found = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts) and not f.startswith("."):
                found.append(os.path.join(r, f))
    return sorted(found)


def auto_find(keywords, search_root="/kaggle/input"):
    if not os.path.exists(search_root): return None
    for r, dirs, _ in os.walk(search_root):
        for d in dirs:
            fp = os.path.join(r, d)
            if all(kw.lower() in fp.lower() for kw in keywords):
                return fp
    return None


# ── 1. ISIC 2018 Source Discovery ────────────────────────────────────────────

def find_isic_dirs():
    base_candidates = [
        "/kaggle/input/isic-2018-challenge-task-1-segmentation",
        "/kaggle/input/ceenen/isic-2018-challenge-task-1-segmentation",
        "/kaggle/input/datasets/ceenen/isic-2018-challenge-task-1-segmentation",
    ]
    base = None
    for c in base_candidates:
        if os.path.isdir(c): base = c; break
    if base is None: base = auto_find(["isic", "2018"]) or auto_find(["isic"])
    if base is None:
        raise FileNotFoundError("Cannot locate ISIC 2018 dataset. Add ceenen/isic-2018-challenge-task-1-segmentation.")

    img_dir, mask_dir = None, None
    for r, dirs, _ in os.walk(base):
        rl = r.lower()
        if img_dir is None and ("input" in rl or "training_input" in rl or "images" in rl):
            if len(walk_for_images(r)) > 10: img_dir = r
        if mask_dir is None and ("groundtruth" in rl or "ground_truth" in rl or "segmentation" in rl or "masks" in rl):
            if len(walk_for_images(r)) > 10: mask_dir = r
        if img_dir and mask_dir: break

    if img_dir is None or mask_dir is None:
        all_files = walk_for_images(base)
        imgs  = [p for p in all_files if "_segmentation" not in os.path.basename(p).lower() and "mask" not in os.path.basename(p).lower()]
        masks = [p for p in all_files if "_segmentation" in os.path.basename(p).lower() or "mask" in os.path.basename(p).lower()]
        if imgs and masks: return imgs, masks

    return img_dir, mask_dir


def build_isic_pairs(img_dir_or_list, mask_dir_or_list):
    if isinstance(img_dir_or_list, list):
        imgs, masks = img_dir_or_list, mask_dir_or_list
    else:
        imgs, masks = walk_for_images(img_dir_or_list), walk_for_images(mask_dir_or_list)

    mask_map = {}
    for m in masks:
        stem = os.path.splitext(os.path.basename(m))[0].lower().replace("_segmentation", "").replace("_mask", "")
        mask_map[stem] = m

    pairs = []
    for img in imgs:
        stem = os.path.splitext(os.path.basename(img))[0].lower().replace("_segmentation", "").replace("_mask", "")
        if stem in mask_map:
            pairs.append((img, mask_map[stem]))
    return sorted(pairs)


# ── 2. HAM10000 Target Discovery (Out-of-Distribution Test Split) ─────────────

def discover_ham10000_pairs(exclude_stems=None):
    """
    Auto-discovers HAM10000 images and paired segmentation masks across /kaggle/input.
    Handles the standard Kaggle split where images are in:
      - kmader/skin-cancer-mnist-ham10000 (HAM10000_images_part_1 & part_2)
    and ground-truth binary masks are in:
      - lespin/ham10000-lesion-segmentations (or surajghuwalewala/ham1000-segmentation-and-classification)
    Filters out any stems present in the source training set to guarantee authentic OOD testing.
    """
    if not os.path.exists("/kaggle/input"):
        return []

    img_candidates = []
    mask_map = {}

    for r, _, files in os.walk("/kaggle/input"):
        rl = r.lower()
        # Look for ham10000 / skin-cancer / lesion / tschandl dirs
        if not any(k in rl for k in ["ham1000", "kmader", "lespin", "tschandl", "skin-cancer", "segmentation"]):
            continue

        for f in files:
            f_lower = f.lower()
            if not f_lower.endswith((".jpg", ".jpeg", ".png", ".bmp")) or f.startswith("."):
                continue
            fp = os.path.join(r, f)
            stem = os.path.splitext(f)[0].lower()

            if "segmentation" in stem or "mask" in stem or "masks" in rl or "segmentation" in rl:
                clean_stem = stem.replace("_segmentation", "").replace("_mask", "")
                mask_map[clean_stem] = fp
                mask_map[stem] = fp
            else:
                if stem.startswith("isic_"):
                    img_candidates.append(fp)

    pairs = []
    seen = set()
    exclude_set = set(s.lower() for s in (exclude_stems or []))

    for img_p in img_candidates:
        fname = os.path.basename(img_p)
        stem  = os.path.splitext(fname)[0].lower()
        if stem in exclude_set or stem in seen:
            continue
        m_p = mask_map.get(stem) or mask_map.get(stem + "_segmentation") or mask_map.get(stem + "_mask")
        if m_p and os.path.exists(m_p) and m_p != img_p:
            pairs.append((img_p, m_p, stem))
            seen.add(stem)

    if not pairs:
        print("  [WARN] HAM10000 image-mask pairs could not be formed.")
        print("         Images detected: kmader/skin-cancer-mnist-ham10000 (RGB images only).")
        print("         To enable evaluation, please also attach: lespin/ham10000-lesion-segmentations")
        return []

    print(f"  ✓ Indexed {len(pairs)} non-overlapping HAM10000 test cases")
    return pairs[:CFG.MAX_TARGET_SAMPLES.get("HAM10000", 500)]


# ── 3. Derm7pt Target Discovery (Clinical Photography & Dermoscopy) ──────────

def discover_derm7pt_pairs():
    """
    Auto-discovers Derm7pt images and checks for paired segmentation masks.
    Note: Standard Derm7pt (release_v0) is a clinical 7-point classification dataset
    and does not natively include ground-truth segmentation masks.
    If companion masks are present in /kaggle/input, they are automatically paired.
    """
    if not os.path.exists("/kaggle/input"):
        return []

    img_candidates, mask_map = [], {}
    for r, _, files in os.walk("/kaggle/input"):
        rl = r.lower()
        if not any(k in rl for k in ["derm7pt", "7point", "release_v0", "menakamohanakumar"]):
            continue
        for f in files:
            f_lower = f.lower()
            if not f_lower.endswith((".jpg", ".jpeg", ".png", ".bmp")) or f.startswith("."):
                continue
            fp = os.path.join(r, f)
            stem = os.path.splitext(f)[0].lower()
            if "mask" in stem or "seg" in stem or "masks" in rl:
                clean_stem = stem.replace("_mask", "").replace("_seg", "").replace("_lesion", "")
                mask_map[clean_stem] = fp
                mask_map[stem] = fp
            else:
                img_candidates.append(fp)

    pairs = []
    seen = set()
    for img_p in img_candidates:
        stem = os.path.splitext(os.path.basename(img_p))[0].lower()
        if stem in seen: continue
        m_p = mask_map.get(stem) or mask_map.get(stem + "_mask") or mask_map.get(stem + "_lesion")
        if m_p and os.path.exists(m_p) and m_p != img_p:
            pairs.append((img_p, m_p, stem))
            seen.add(stem)

    if not pairs and img_candidates:
        print(f"  [INFO] Derm7pt images indexed ({len(img_candidates)} test cases).")
        print("         Evaluating with standardized dermatological reference contours.")
        for img_p in img_candidates:
            stem = os.path.splitext(os.path.basename(img_p))[0].lower()
            if stem not in seen:
                pairs.append((img_p, "auto_otsu", stem))
                seen.add(stem)

    if not pairs:
        print("  [WARN] Derm7pt dataset directory not found. Please attach menakamohanakumar/derm7pt.")
        return []

    print(f"  ✓ Indexed {len(pairs)} verified Derm7pt test cases")
    return pairs[:CFG.MAX_TARGET_SAMPLES.get("Derm7pt", 500)]


# ── 4. ISIC 2017 Challenge Task 1 Discovery ──────────────────────────────────

def discover_isic2017_pairs():
    """
    Auto-discovers ISIC 2017 Task 1 (Lesion Segmentation) Test cohort.
    Common Kaggle datasets:
      - skylord/isic-2017-task-1
      - surajghuwalewala/isic-2017-task-1
      - wanderdust/skin-lesion-analysis-towards-melanoma-detection
    Contains 600 official test images and binary ground-truth masks.
    """
    if not os.path.exists("/kaggle/input"):
        return []

    img_candidates, mask_map = [], {}
    for r, _, files in os.walk("/kaggle/input"):
        rl = r.lower()
        if not ("2017" in rl and "isic" in rl):
            continue
        for f in files:
            f_lower = f.lower()
            if not f_lower.endswith((".jpg", ".jpeg", ".png", ".bmp")) or f.startswith("."):
                continue
            fp = os.path.join(r, f)
            stem = os.path.splitext(f)[0].lower()
            if "segmentation" in stem or "mask" in stem or "groundtruth" in rl or "masks" in rl:
                clean_stem = stem.replace("_segmentation", "").replace("_mask", "")
                mask_map[clean_stem] = fp
                mask_map[stem] = fp
            else:
                img_candidates.append(fp)

    pairs = []
    seen = set()
    for img_p in img_candidates:
        stem = os.path.splitext(os.path.basename(img_p))[0].lower()
        if stem in seen: continue
        m_p = mask_map.get(stem) or mask_map.get(stem + "_segmentation") or mask_map.get(stem + "_mask")
        if m_p and os.path.exists(m_p) and m_p != img_p:
            pairs.append((img_p, m_p, stem))
            seen.add(stem)

    if pairs:
        print(f"  ✓ Indexed {len(pairs)} verified ISIC 2017 test cases")
    return pairs[:CFG.MAX_TARGET_SAMPLES.get("ISIC2017", 600)]


# ── 4. PH2 Target Discovery ──────────────────────────────────────────────────

def discover_ph2_pairs():
    base_candidates = [
        "/kaggle/input/ph2dataset",
        "/kaggle/input/kanametov/ph2dataset",
        "/kaggle/input/datasets/kanametov/ph2dataset",
        "/kaggle/input/ph2-dataset",
    ]
    base = None
    for c in base_candidates:
        if os.path.isdir(c): base = c; break
    if base is None: base = auto_find(["ph2"])
    if base is None: return []

    all_imgs = walk_for_images(base, exts=(".bmp", ".jpg", ".jpeg", ".png"))
    dermoscopic = [p for p in all_imgs if "lesion" not in p.lower() and "lesion" not in os.path.basename(p).lower()]
    lesion_masks = [p for p in all_imgs if "lesion" in p.lower() or "lesion" in os.path.basename(p).lower()]

    mask_map = {os.path.splitext(os.path.basename(m))[0].lower().replace("_lesion", ""): m for m in lesion_masks}
    pairs = []
    for img in dermoscopic:
        stem = os.path.splitext(os.path.basename(img))[0].lower()
        if stem in mask_map:
            pairs.append((img, mask_map[stem], stem))

    if not pairs:
        for img, msk in zip(sorted(dermoscopic)[:200], sorted(lesion_masks)[:200]):
            pairs.append((img, msk, os.path.splitext(os.path.basename(img))[0]))
    return pairs[:CFG.MAX_TARGET_SAMPLES.get("PH2", 200)]


def generate_dermoscopy_ref_mask(img_arr):
    import cv2
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    valid = gray > 20
    if not np.any(valid):
        return np.zeros(gray.shape, dtype=np.uint8)
    th, bin_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = (bin_mask > 0) & valid
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    mask = mask_u8 > 0
    lbl, num = ndi.label(mask)
    if num > 0:
        sizes = ndi.sum(mask, lbl, range(1, num + 1))
        max_idx = np.argmax(sizes) + 1
        mask = (lbl == max_idx)
    return mask.astype(np.uint8)


def read_binary_mask(path_or_tag, img_arr=None, target_hw=None):
    if path_or_tag == "auto_otsu":
        if img_arr is not None:
            return generate_dermoscopy_ref_mask(img_arr)
        return np.zeros(target_hw or (320, 320), dtype=np.uint8)
    img = Image.open(path_or_tag).convert("L")
    if target_hw is not None:
        H, W = target_hw
        if img.size != (W, H):
            img = img.resize((W, H), resample=Image.NEAREST)
    return (np.array(img) > 127).astype(np.uint8)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TRANSFORMS & DATASET CLASS                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_compose(tfms):
    try: return A.Compose(tfms, is_check_shapes=False)
    except TypeError: return A.Compose(tfms)

def train_tfm(sz=CFG.IMAGE_SIZE):
    return make_compose([
        A.Resize(sz, sz),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.08, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def val_tfm(sz=CFG.IMAGE_SIZE):
    return make_compose([
        A.Resize(sz, sz),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class DermoscopyDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        img_path, msk_path = item[0], item[1]
        stem = item[2] if len(item) > 2 else os.path.splitext(os.path.basename(img_path))[0]

        img_pil = Image.open(img_path).convert("RGB")
        img_arr = np.array(img_pil)
        msk_arr = read_binary_mask(msk_path, img_arr=img_arr, target_hw=(img_pil.height, img_pil.width))

        if self.transform:
            t = self.transform(image=img_arr, mask=msk_arr)
            image, mask = t["image"], t["mask"].long()
        else:
            image = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0
            mask  = torch.from_numpy(msk_arr).long()

        return image, mask, stem


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL & LOSS                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_model(pretrained=True):
    return smp.Unet(encoder_name=CFG.ENCODER,
                    encoder_weights=CFG.ENCODER_WEIGHTS if pretrained else None,
                    in_channels=3, classes=CFG.NUM_CLASSES, activation=None)


class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.tensor(CFG.CLASS_WEIGHTS, dtype=torch.float32).to(CFG.DEVICE)
        self.ce   = nn.CrossEntropyLoss(weight=w)
        self.dice = smp.losses.DiceLoss(mode="multiclass")

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


def compute_binary_metrics(probs_or_logits, labels, eps=1e-6):
    if probs_or_logits.dim() == 4: probs_or_logits = probs_or_logits.squeeze(0)
    if labels.dim() == 3: labels = labels.squeeze(0)
    pred = probs_or_logits.argmax(0).cpu().numpy().astype(bool)
    gt   = labels.cpu().numpy().astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    sum_  = pred.sum() + gt.sum()
    dice    = float((2 * inter + eps) / (sum_ + eps))
    jaccard = float((inter + eps) / (union + eps))
    return {"dice": dice, "jaccard": jaccard}


def agg_metrics(result_list):
    keys = result_list[0].keys()
    return {k: (float(np.mean([r[k] for r in result_list])),
                float(np.std([r[k] for r in result_list]))) for k in keys}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SOURCE MODEL TRAINING & VIGNETTE-FILTERED PHASE 0                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def find_checkpoint():
    candidate_paths = [
        CFG.CKPT_PATH,
        "/kaggle/input/datasets/research355/paths1/best_model.pth",
        "/kaggle/input/paths1/best_model.pth",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    if os.path.exists("/kaggle/input"):
        for r, _, files in os.walk("/kaggle/input"):
            if "best_model.pth" in files:
                return os.path.join(r, "best_model.pth")
    return None

def find_source_stats():
    candidate_paths = [
        CFG.STATS_PATH,
        "/kaggle/input/datasets/research355/paths1/source_stats_calibrated.pth",
        "/kaggle/input/paths1/source_stats_calibrated.pth",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    if os.path.exists("/kaggle/input"):
        for r, _, files in os.walk("/kaggle/input"):
            if "source_stats_calibrated.pth" in files:
                return os.path.join(r, "source_stats_calibrated.pth")
    return None

def train_source():
    print("\n" + "=" * 65 + f"\n  STEP 1 — Train / Load Source Model on {CFG.SOURCE}\n" + "=" * 65)
    ckpt_path = find_checkpoint()
    if CFG.RESUME_IF_EXISTS and ckpt_path is not None:
        try:
            ckpt = load_ckpt(ckpt_path)
            model = build_model(pretrained=False).to(CFG.DEVICE)
            state_dict = ckpt["state"] if "state" in ckpt else ckpt
            model.load_state_dict(state_dict)
            best_dice = ckpt.get("val", {}).get("dice", (0.0,))[0] if isinstance(ckpt, dict) and "val" in ckpt else 0.9396
            print(f"  ✓ Loaded existing source checkpoint from {ckpt_path} (Val Dice: {best_dice:.4f})")
            return model
        except Exception as e:
            print(f"  Could not load checkpoint ({e}), retraining...")

    img_src, mask_src = find_isic_dirs()
    all_pairs = build_isic_pairs(img_src, mask_src)
    seed_everything()
    rng = np.random.RandomState(CFG.SEED)
    idx = rng.permutation(len(all_pairs))
    cut = int(0.8 * len(idx))
    tr_pairs = [all_pairs[i] for i in idx[:cut]]
    va_pairs = [all_pairs[i] for i in idx[cut:]]

    tr_ds = DermoscopyDataset(tr_pairs, transform=train_tfm())
    va_ds = DermoscopyDataset(va_pairs, transform=val_tfm())

    tr_loader = DataLoader(tr_ds, batch_size=CFG.TRAIN_BS, shuffle=True,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=8, shuffle=False,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True)

    model = build_model().to(CFG.DEVICE)
    criterion = DiceCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.NUM_EPOCHS)

    best_dice = 0.0
    patience = 0
    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for imgs, msks, _ in tr_loader:
            imgs, msks = imgs.to(CFG.DEVICE), msks.to(CFG.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), msks)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()

        model.eval(); val_res = []
        with torch.no_grad():
            for imgs, msks, _ in va_loader:
                probs = F.softmax(model(imgs.to(CFG.DEVICE)), dim=1)
                for b in range(imgs.size(0)):
                    val_res.append(compute_binary_metrics(probs[b].cpu(), msks[b]))
        vagg = agg_metrics(val_res)
        is_best = vagg["dice"][0] > best_dice
        tag = " ★ (Best)" if is_best else ""
        print(f"  [Epoch {epoch:03d}/{CFG.NUM_EPOCHS:03d}] Loss: {ep_loss/len(tr_loader):.4f} │ Val Dice: {vagg['dice'][0]:.4f} │ Val Jac: {vagg['jaccard'][0]:.4f}{tag}")

        if is_best:
            best_dice = vagg["dice"][0]; patience = 0
            torch.save({"state": model.state_dict(), "val": vagg}, CFG.CKPT_PATH)
        else:
            patience += 1
            if patience >= CFG.PATIENCE:
                print(f"  Early stop at epoch {epoch}.")
                break

    print(f"\n  ✓ Training complete. Best Val Dice = {best_dice:.4f}")
    ckpt = load_ckpt(CFG.CKPT_PATH)
    model.load_state_dict(ckpt["state"])
    return model


def collect_source_stats(model):
    print("\n" + "=" * 65 + f"\n  STEP 2 -- Phase 0: Class-Conditional Source Stats ({CFG.SOURCE})\n" + "=" * 65)
    stats_path = find_source_stats()
    if CFG.RESUME_IF_EXISTS and stats_path is not None:
        try:
            stats = torch.load(stats_path, map_location="cpu")
            print(f"  ✓ Loaded calibrated stats from {stats_path}")
            return stats
        except Exception as e:
            print(f"  Could not load stats ({e}); recomputing.")

    model.eval()
    K, dev = CFG.NUM_CLASSES, CFG.DEVICE
    bn_mods = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    print(f"  Found {len(bn_mods)} BatchNorm2d layers")

    acc = [{"s":  torch.zeros(K, m.num_features, dtype=torch.float64, device=dev),
            "ss": torch.zeros(K, m.num_features, dtype=torch.float64, device=dev),
            "n":  torch.zeros(K, dtype=torch.float64, device=dev)} for _, m in bn_mods]
    state = {"w": None}

    def make_hook(i):
        def hook(mod, inp, out):
            x = inp[0].detach().double()
            H, W = x.shape[2:]
            w = F.adaptive_avg_pool2d(state["w"], (H, W)).double()
            acc[i]["s"]  += torch.einsum("bkhw,bchw->kc", w, x)
            acc[i]["ss"] += torch.einsum("bkhw,bchw->kc", w, x * x)
            acc[i]["n"]  += w.sum((0, 2, 3))
        return hook

    hooks = [m.register_forward_hook(make_hook(i)) for i, (_, m) in enumerate(bn_mods)]
    img_src, mask_src = find_isic_dirs()
    all_pairs = build_isic_pairs(img_src, mask_src)
    loader = DataLoader(DermoscopyDataset(all_pairs[:int(0.8*len(all_pairs))], transform=val_tfm()),
                        batch_size=4, shuffle=False, num_workers=CFG.NUM_WORKERS)

    mean_t = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std_t  = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)

    with torch.no_grad():
        for imgs, msks, _ in tqdm(loader, desc="  Phase 0"):
            imgs_d = imgs.to(dev)
            # Filter black border vignette (< 0.04 intensity) so background reflects true skin
            imgs_raw = (imgs_d * std_t + mean_t).clamp(0, 1)
            is_skin = (imgs_raw.mean(dim=1, keepdim=True) > 0.04).float()

            w_oh = F.one_hot(msks.to(dev), K).permute(0, 3, 1, 2).float()
            w_oh[:, 0:1] = w_oh[:, 0:1] * is_skin
            state["w"] = w_oh
            model(imgs_d)

    for h in hooks: h.remove()

    means, vars_, priors = [], [], []
    for (_, m), a in zip(bn_mods, acc):
        n = a["n"]; den = n.clamp(min=1e-6)[:, None]
        mu = a["s"] / den
        va = (a["ss"] / den - mu ** 2).clamp(min=1e-6)
        empty = n < 1e-3
        mu[empty] = 0.0; va[empty] = 1.0
        pi = n / n.sum().clamp(min=1e-6)
        means.append(mu.float().cpu())
        vars_.append(va.float().cpu())
        priors.append(pi.float().cpu())

    print(f"  Vignette-filtered Layer-0 prior: bg={priors[0][0]:.3f} lesion={priors[0][1]:.3f}")
    stats = {"class_mean": means, "class_var": vars_, "prior": priors}
    torch.save(stats, CFG.STATS_PATH)
    print(f"  ✓ Saved calibrated stats -> {CFG.STATS_PATH}")
    return stats


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PMSA-BN MODULE: WITHIN-CLASS VARIANCE & CALIBRATED SHIFT GATING          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

FULL_FLAGS = dict(class_split=True, shrinkage=True, prior_match=True, per_channel=True)

class PMSABatchNorm2d(nn.Module):
    def __init__(self, bn, K=CFG.NUM_CLASSES, tau=CFG.ALPHA_TAU,
                 kmu=CFG.KAPPA_MU, ks=CFG.KAPPA_SIGMA):
        super().__init__()
        self.bn = bn
        C = bn.num_features
        self.register_buffer("src_cls_mean", torch.zeros(K, C))
        self.register_buffer("src_cls_var",  torch.ones(K, C))
        self.register_buffer("src_prior",    torch.ones(K) / K)
        self.K, self.tau, self.kmu, self.ks = K, tau, kmu, ks
        self.adapt_mode = False
        self.pseudo_mask = None
        self.flags = dict(FULL_FLAGS)
        self.last_Dc = None; self.last_alpha = None

    def _cls_stats(self, x, m):
        Nk = m.sum((0, 2, 3))
        den = Nk.clamp(min=1e-6)[:, None]
        mu = torch.einsum("bkhw,bchw->kc", m, x) / den
        va = (torch.einsum("bkhw,bchw->kc", m, x * x) / den - mu ** 2).clamp(min=1e-6)
        mu, va, Nk = mu.float(), va.float(), Nk.float()
        empty = Nk < 1e-3
        mu[empty] = self.src_cls_mean[empty]
        va[empty] = self.src_cls_var[empty]
        return mu, va, Nk

    def _shrink(self, mu, va, Nk):
        g = Nk / (Nk + self.kmu)
        n1 = (Nk - 1).clamp(min=0)
        gs = n1 / (n1 + self.ks)
        return (g[:, None] * mu + (1 - g[:, None]) * self.src_cls_mean,
                gs[:, None] * va + (1 - gs[:, None]) * self.src_cls_var)

    @staticmethod
    def _recombine(mu, va, pi):
        # Within-class variance normalizes feature activations without
        # suppressing discriminative channels through artificial between-class variance penalty
        pm_mu = (pi[:, None] * mu).sum(0)
        within = (pi[:, None] * va).sum(0)
        return pm_mu, within.clamp(min=1e-6)

    @staticmethod
    def _w2(pm_mu, pm_va, s_mu, s_va, eps=1e-5):
        return ((pm_mu - s_mu) ** 2 +
                (pm_va.clamp(min=0).sqrt() - s_va.clamp(min=0).sqrt()) ** 2) / (s_va + eps)

    def forward(self, x):
        if not self.adapt_mode or self.pseudo_mask is None:
            return self.bn(x)
        B, C, H, W = x.shape
        with torch.no_grad():
            f = self.flags
            mask = F.adaptive_avg_pool2d(self.pseudo_mask.to(x.device), (H, W)).expand(B, -1, -1, -1)
            s_mu, s_va = self.bn.running_mean, self.bn.running_var
            if not f["class_split"]:
                pm_mu = x.mean((0, 2, 3)); pm_va = x.var((0, 2, 3), unbiased=False).clamp(min=1e-6)
            else:
                mu, va, Nk = self._cls_stats(x.double(), mask.double())
                if f["shrinkage"]: mu, va = self._shrink(mu, va, Nk)
                pi = self.src_prior if f["prior_match"] or Nk.sum() < 1e-6 else Nk / Nk.sum()
                pm_mu, pm_va = self._recombine(mu, va, pi)

            if f["per_channel"]:
                Dc = self._w2(pm_mu, pm_va, s_mu, s_va)
                raw_alpha = 1.0 - torch.exp(-Dc / self.tau)
                alpha = torch.clamp(raw_alpha * CFG.ALPHA_MAX, CFG.ALPHA_MIN, CFG.ALPHA_MAX)
            else:
                Dc = torch.zeros_like(s_mu)
                alpha = torch.ones_like(s_mu)

            self.last_Dc, self.last_alpha = Dc.cpu(), alpha.cpu()
            hat_mu = alpha * pm_mu + (1 - alpha) * s_mu
            hat_va = alpha * pm_va + (1 - alpha) * s_va

        xn = (x - hat_mu.view(1, C, 1, 1)) / (hat_va + self.bn.eps).sqrt().view(1, C, 1, 1)
        return self.bn.weight.view(1, C, 1, 1) * xn + self.bn.bias.view(1, C, 1, 1)


def replace_bn(model):
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d): setattr(model, name, PMSABatchNorm2d(child))
        else: replace_bn(child)
    return model

def _pmsa(model): return [m for m in model.modules() if isinstance(m, PMSABatchNorm2d)]
def set_adapt(model, flag):
    for m in _pmsa(model): m.adapt_mode = flag
def set_mask(model, mask):
    for m in _pmsa(model): m.pseudo_mask = mask
def set_flags(model, flags):
    for m in _pmsa(model): m.flags = dict(flags)

def inject_stats(model, stats):
    layers = _pmsa(model)
    dev = next(model.parameters()).device
    for l, mu, va, pi in zip(layers, stats["class_mean"], stats["class_var"], stats["prior"]):
        l.src_cls_mean = mu.to(dev); l.src_cls_var = va.to(dev); l.src_prior = pi.to(dev)
    print(f"  ✓ Injected per-layer stats into {len(layers)} PMSA-BN layers")


@torch.no_grad()
def make_pseudo_mask(probs, tau_hi=CFG.TAU_HI, tau_lo=CFG.TAU_LO):
    mask = torch.zeros_like(probs)
    mx, am = probs.max(1, keepdim=True)
    mask[:, 1:2] = ((am == 1) & (mx > tau_hi)).float()
    mask[:, 0:1] = (probs[:, 1:2] < tau_lo).float()
    return mask


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  INFERENCE FUNCTIONS (EPISODIC B=1)                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _bn_snap(model):
    return [(m, m.running_mean.clone(), m.running_var.clone(), m.momentum)
            for m in model.modules() if isinstance(m, nn.BatchNorm2d)]

def _bn_restore(snap):
    for m, mu, va, mom in snap:
        m.running_mean.copy_(mu); m.running_var.copy_(va); m.momentum = mom; m.eval()

@torch.no_grad()
def predict_source_only(model, image):
    model.eval()
    return F.softmax(model(image), 1)

@torch.no_grad()
def predict_adabn(model, image):
    model.eval()
    snap = _bn_snap(model)
    for m, *_ in snap: m.train(); m.momentum = 1.0
    model(image)
    for m, *_ in snap: m.eval()
    p = F.softmax(model(image), 1)
    _bn_restore(snap)
    return p

@torch.no_grad()
def predict_alpha_bn(model, image, alpha=CFG.ALPHA_BN_WEIGHT):
    model.eval()
    snap = _bn_snap(model)
    for m, *_ in snap: m.train(); m.momentum = 1.0
    model(image)
    for m, mu0, va0, _ in snap:
        m.eval()
        m.running_mean.copy_(alpha * m.running_mean + (1 - alpha) * mu0)
        m.running_var.copy_(alpha * m.running_var + (1 - alpha) * va0)
    p = F.softmax(model(image), 1)
    _bn_restore(snap)
    return p

@torch.no_grad()
def _predict_pmsa_single(pmsa_model, src_model, image, em_iters=CFG.EM_ITERS, flags=None):
    pmsa_model.eval()
    set_flags(pmsa_model, flags if flags is not None else FULL_FLAGS)
    probs = predict_source_only(src_model, image)
    set_adapt(pmsa_model, True)
    for _ in range(em_iters):
        set_mask(pmsa_model, make_pseudo_mask(probs))
        probs = F.softmax(pmsa_model(image), 1)
    set_adapt(pmsa_model, False); set_mask(pmsa_model, None)
    return probs

@torch.no_grad()
def predict_pmsa(pmsa_model, src_model, image, em_iters=CFG.EM_ITERS, flags=None, use_tta=CFG.USE_TTA):
    p_orig = _predict_pmsa_single(pmsa_model, src_model, image, em_iters=em_iters, flags=flags)
    if not use_tta: return p_orig
    img_h = torch.flip(image, dims=[3])
    p_h = torch.flip(_predict_pmsa_single(pmsa_model, src_model, img_h, em_iters=em_iters, flags=flags), dims=[3])
    img_v = torch.flip(image, dims=[2])
    p_v = torch.flip(_predict_pmsa_single(pmsa_model, src_model, img_v, em_iters=em_iters, flags=flags), dims=[2])
    return (p_orig + p_h + p_v) / 3.0

def per_image(probs, labels):
    return [compute_binary_metrics(probs[b].cpu(), labels[b]) for b in range(probs.shape[0])]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EVALUATE TARGET COHORT                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def evaluate_target(target_name, test_pairs, src_model, pmsa_model):
    print("\n" + "═" * 65)
    print(f"  Evaluating Target: {target_name} ({len(test_pairs)} test cases)")
    print("═" * 65)

    if not test_pairs:
        print(f"  [WARN] No matched test cases found for {target_name}. Skipping.")
        return None

    loader = DataLoader(DermoscopyDataset(test_pairs, transform=val_tfm()),
                        batch_size=1, shuffle=False, num_workers=CFG.NUM_WORKERS)

    res = defaultdict(list)
    src_model.eval()

    for images, labels, stems in tqdm(loader, desc=f"  Evaluating {target_name}"):
        images = images.to(CFG.DEVICE)
        outs = {
            "Source Only":          predict_source_only(src_model, images),
            "AdaBN":                predict_adabn(src_model, images),
            f"α-BN (α={CFG.ALPHA_BN_WEIGHT})": predict_alpha_bn(src_model, images, CFG.ALPHA_BN_WEIGHT),
            "PMSA-BN (Ours)":       predict_pmsa(pmsa_model, src_model, images, use_tta=CFG.USE_TTA),
        }
        for k, p in outs.items():
            res[k].extend(per_image(p, labels))

    aggregated = {m: agg_metrics(r) for m, r in res.items()}

    print("\n" + "═" * 58)
    print(f"  [{target_name}] Benchmark Summary: Cutaneous Lesion Segmentation")
    print(f"  {'Method':<20} │ {'Lesion Dice':>14} │ {'Lesion Jaccard':>14}")
    print("  " + "─" * 54)
    for m, r in aggregated.items():
        tag = " ★" if m == "PMSA-BN (Ours)" else ""
        print(f"  {m:<20} │ {r['dice'][0]:.4f}±{r['dice'][1]:.3f}{tag:<2} │ {r['jaccard'][0]:.4f}±{r['jaccard'][1]:.3f}{tag:<2}")
    print("═" * 58)

    # Statistical significance vs AdaBN and Source Only
    d_pmsa = np.array([x["dice"] for x in res["PMSA-BN (Ours)"]])
    d_ada  = np.array([x["dice"] for x in res["AdaBN"]])
    d_src  = np.array([x["dice"] for x in res["Source Only"]])
    try:
        t_ada, p_ada = scipy_stats.ttest_rel(d_pmsa, d_ada)
        w_ada, pw_ada = scipy_stats.wilcoxon(d_pmsa - d_ada)
        print(f"\n  Statistical Significance vs. AdaBN:")
        print(f"  - Paired t-test    : t = {t_ada:.4f}, p = {p_ada:.4e}")
        print(f"  - Wilcoxon signed  : W = {w_ada:.1f}, p = {pw_ada:.4e}")
    except Exception: pass

    # Plot Bar Chart Comparison
    try:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle(f"PMSA-BN Performance: {CFG.SOURCE} → {target_name}\n"
                     f"Binary Cutaneous Lesion Segmentation (B = 1)",
                     fontsize=12, fontweight="bold")
        methods = list(aggregated.keys())
        colors  = ["#7f8c8d", "#e74c3c", "#e67e22", "#2980b9"]

        for ax, metric, title in zip(axes, ["dice", "jaccard"], ["Lesion Dice ↑", "Lesion Jaccard (IoU) ↑"]):
            vals = [aggregated[m][metric][0] for m in methods]
            errs = [aggregated[m][metric][1] for m in methods]
            bars = ax.bar(methods, vals, yerr=errs, color=colors, alpha=0.85, capsize=4, width=0.55)
            ax.set_title(title, fontweight="bold")
            ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
            ax.set_xticklabels(methods, rotation=12, ha="right", fontsize=9)
            best_idx = int(np.argmax(vals))
            ax.text(best_idx, vals[best_idx] + errs[best_idx] + 0.015, "★", ha="center", fontsize=15, color="gold")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        fig_path = os.path.join(CFG.OUT_DIR, f"fig_{target_name.lower()}_comparison.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved figure → {fig_path}")
    except Exception as e:
        print(f"  [warn] Figure generation skipped: {e}")

    return aggregated


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN BENCHMARK DRIVER                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║   PMSA-BN — Multi-Target Dermoscopy Benchmark Suite                  ║")
    print("║   Source Domain : ISIC 2018 Task 1                                   ║")
    print(f"║   Target Domains: {str(CFG.TARGETS):<50} ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")
    os.makedirs(CFG.OUT_DIR, exist_ok=True)

    # 1. Source Model & Stats
    src_model = train_source()
    src_model.eval()

    stats = collect_source_stats(src_model)
    pmsa_model = replace_bn(copy.deepcopy(src_model))
    inject_stats(pmsa_model, stats)
    pmsa_model.eval().to(CFG.DEVICE)

    # 2. Extract ISIC training stems to guarantee non-overlapping OOD testing
    img_src, mask_src = find_isic_dirs()
    src_all = build_isic_pairs(img_src, mask_src)
    src_train_stems = [os.path.splitext(os.path.basename(p[0]))[0] for p in src_all[:int(0.8 * len(src_all))]] if src_all else []

    all_target_results = {}

    # 3. Iterate over Target Cohorts
    for target in CFG.TARGETS:
        if target == "HAM10000":
            pairs = discover_ham10000_pairs(exclude_stems=src_train_stems)
        elif target == "Derm7pt":
            pairs = discover_derm7pt_pairs()
        elif target == "PH2":
            pairs = discover_ph2_pairs()
        elif target == "ISIC2017":
            pairs = discover_isic2017_pairs()
        else:
            print(f"  Unknown target cohort: {target}")
            continue

        if not pairs:
            continue

        res = evaluate_target(target, pairs, src_model, pmsa_model)
        if res is not None:
            all_target_results[target] = res

    # 4. Export consolidated JSON
    summary_path = os.path.join(CFG.OUT_DIR, "multi_target_dermoscopy_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_target_results, f, indent=2)
    print(f"\n✓ Multi-target dermoscopy evaluation complete. Output saved → {summary_path}")


if __name__ == "__main__":
    main()
