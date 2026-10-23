"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm                             ║
║  Chest Radiography (CXR) Lung Field Segmentation Benchmark                   ║
║  Source : Montgomery County (MC), USA  (Eureka / Del Medical Flat Panel)     ║
║  Target : Shenzhen Hospital (SZ), China (Philips Digital Radiography)        ║
║  Task   : Binary Lung Field Segmentation (Background vs. Lung)               ║
║  Metrics: Lung Dice & Lung Jaccard (IoU)                                     ║
║                                                                              ║
║  Kaggle Dataset Required:                                                    ║
║  👉 nikhilpandey360/chest-xray-masks-and-labels                              ║
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
import os, glob, copy, json, random, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
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
# ║  CFG — RADIOGRAPHY BENCHMARK CONFIGURATION                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CFG:
    SOURCE = "Montgomery"     # Montgomery County (MC), USA
    TARGET = "Shenzhen"       # Shenzhen Hospital (SZ), China

    CKPT_PATH  = "/kaggle/working/cxr_best_model.pth"
    STATS_PATH = "/kaggle/working/cxr_source_stats.pth"
    OUT_DIR    = "/kaggle/working"

    # ── Model ─────────────────────────────────────────────────────────────
    ENCODER         = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES     = 2          # 0=background, 1=lung
    IMAGE_SIZE      = 256

    # ── Training on Source (Montgomery) ───────────────────────────────────
    TRAIN_BS           = 16
    NUM_EPOCHS         = 30
    LR                 = 3e-4
    WEIGHT_DECAY       = 1e-5
    PATIENCE           = 8
    CLASS_WEIGHTS      = [0.4, 1.6]
    USE_AMP            = True
    TRAIN_FROM_SCRATCH = False   # Automatically reuses checkpoint if available

    # ── PMSA-BN Hyperparameters (Calibrated for Radiography Adaptation) ───
    TAU_HI      = 0.50           # Continuous responsibility threshold
    KAPPA_EFF   = 0.05           # Bayesian shrinkage pseudo-fraction
    ALPHA_MIN   = 0.35           # Lower bound preserves anatomical structural invariants
    ALPHA_MAX   = 0.90           # Upper bound adapts shifted tube voltage / detector contrast
    ALPHA_TAU   = 0.50           # Shift sensitivity temperature (selected on Montgomery source val)
    EM_ITERS    = 1              # Single refinement pass (prevents confirmation bias)
    PASS1       = "source"       # Initial pseudo-mask: starts from source model
    USE_TTA     = True           # Horizontal flip TTA probability blending

    # ── Baselines & Ablation ──────────────────────────────────────────────
    ALPHA_BN_WEIGHT = 0.5
    RUN_ABLATION    = True

    SEED        = 42
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4


def seed_everything(seed=CFG.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()
print(f"✓ Device  : {CFG.DEVICE}  |  PyTorch {torch.__version__}")
print(f"✓ AMP     : {CFG.USE_AMP}  |  Workers : {CFG.NUM_WORKERS}")
print(f"✓ Source  : {CFG.SOURCE} (USA)  →  Target : {CFG.TARGET} (China)")
print(f"✓ Task    : Binary Lung Field Segmentation | Metrics: Dice & Jaccard (IoU)")
print("─" * 65)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATASET AUTO-DISCOVERY & PARSING (CHEST X-RAY)                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def auto_find_cxr_dirs():
    """
    Auto-discovers the Chest X-Ray images and masks directories
    under /kaggle/input across common folder layouts.
    """
    candidate_roots = [
        "/kaggle/input/datasets/nikhilpandey360/chest-xray-masks-and-labels",
        "/kaggle/input/datasets/nikhilpandey360/chest-xray-masks-and-labels/Lung Segmentation",
        "/kaggle/input/datasets/nikhilpandey360/chest-xray-masks-and-labels/data/Lung Segmentation",
        "/kaggle/input/datasets/nikhilpandey360/chest-xray-masks-and-labels/Chest Xray Masks and Labels",
        "/kaggle/input/chest-xray-masks-and-labels",
        "/kaggle/input/chest-xray-masks-and-labels/Lung Segmentation",
        "/kaggle/input/chest-xray-masks-and-labels/data/Lung Segmentation",
        "/kaggle/input/chest-xray-masks-and-labels/Chest Xray Masks and Labels",
    ]
    for root in candidate_roots:
        if not os.path.exists(root):
            continue
        cxr_dir = os.path.join(root, "CXR_png")
        msk_dir = os.path.join(root, "masks")
        if os.path.exists(cxr_dir) and os.path.exists(msk_dir):
            return cxr_dir, msk_dir

    # Independent search across all subdirectories in /kaggle/input
    found_cxr, found_msk = None, None
    for search_root in [
        "/kaggle/input/datasets/nikhilpandey360/chest-xray-masks-and-labels",
        "/kaggle/input/chest-xray-masks-and-labels",
        "/kaggle/input/datasets",
        "/kaggle/input",
    ]:
        if os.path.exists(search_root):
            for root, dirs, _ in os.walk(search_root):
                for d in dirs:
                    if d.lower() == "cxr_png" and found_cxr is None:
                        cand_cxr = os.path.join(root, d)
                        if glob.glob(os.path.join(cand_cxr, "*.png")) or glob.glob(os.path.join(cand_cxr, "*.PNG")):
                            found_cxr = cand_cxr
                    elif d.lower() in ("masks", "mask") and found_msk is None:
                        cand_msk = os.path.join(root, d)
                        if glob.glob(os.path.join(cand_msk, "**", "*.png"), recursive=True) or glob.glob(os.path.join(cand_msk, "**", "*.PNG"), recursive=True):
                            found_msk = cand_msk
                if found_cxr and found_msk:
                    return found_cxr, found_msk
            if found_cxr and found_msk:
                return found_cxr, found_msk

    if found_cxr and found_msk:
        return found_cxr, found_msk

    raise FileNotFoundError(
        "Could not locate CXR_png and masks directories in /kaggle/input. "
        "Please ensure 'nikhilpandey360/chest-xray-masks-and-labels' is attached to your notebook."
    )


def build_cxr_pairs():
    """
    Indexes Montgomery County (MCUCXR_*) and Shenzhen Hospital (CHNCXR_*)
    image-mask pairs from the unified dataset folder.
    """
    cxr_dir, msk_dir = auto_find_cxr_dirs()
    all_imgs = sorted(glob.glob(os.path.join(cxr_dir, "*.png")) +
                      glob.glob(os.path.join(cxr_dir, "*.PNG")))

    # Pre-index all mask files in msk_dir recursively
    mask_map = {}
    for m in glob.glob(os.path.join(msk_dir, "**", "*.png"), recursive=True) + \
             glob.glob(os.path.join(msk_dir, "**", "*.PNG"), recursive=True):
        fname = os.path.basename(m)
        stem = os.path.splitext(fname)[0]
        mask_map[fname] = m
        mask_map[fname.lower()] = m
        mask_map[stem] = m
        mask_map[stem.lower()] = m
        if stem.lower().endswith("_mask"):
            base_stem = stem[:-5]
            mask_map[base_stem] = m
            mask_map[base_stem.lower()] = m

    montgomery_pairs = []
    shenzhen_pairs   = []

    for img_p in all_imgs:
        fname = os.path.basename(img_p)
        stem  = os.path.splitext(fname)[0]

        # Multi-key candidate lookup
        valid_mask = (mask_map.get(fname) or
                      mask_map.get(stem) or
                      mask_map.get(stem + "_mask") or
                      mask_map.get(fname.lower()) or
                      mask_map.get(stem.lower()) or
                      mask_map.get(stem.lower() + "_mask"))

        # Fallback to direct path checks if not found in dictionary
        if valid_mask is None:
            for cand in [
                os.path.join(msk_dir, fname),
                os.path.join(msk_dir, stem + ".png"),
                os.path.join(msk_dir, stem + "_mask.png"),
                os.path.join(msk_dir, stem + "_mask.PNG"),
            ]:
                if os.path.exists(cand):
                    valid_mask = cand
                    break

        if valid_mask is None or not os.path.exists(valid_mask):
            continue

        pair = (img_p, valid_mask, stem)
        if stem.startswith("MCUCXR"):
            montgomery_pairs.append(pair)
        elif stem.startswith("CHNCXR"):
            shenzhen_pairs.append(pair)

    # Deterministic split for Montgomery (Source): 85% train, 15% validation
    rng = random.Random(CFG.SEED)
    rng.shuffle(montgomery_pairs)
    n_val = max(15, int(0.15 * len(montgomery_pairs)))
    mc_val   = montgomery_pairs[:n_val]
    mc_train = montgomery_pairs[n_val:]

    return {
        "mc_train": mc_train,
        "mc_val":   mc_val,
        "sz_test":  shenzhen_pairs,
        "cxr_dir":  cxr_dir,
        "msk_dir":  msk_dir
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATASET CLASS & AUGMENTATIONS                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_tfm():
    return A.Compose([
        A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.08, rotate_limit=10, p=0.6, border_mode=0),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def val_tfm():
    return A.Compose([
        A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

class CXRLungDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, msk_p, stem = self.pairs[idx]
        image = np.array(Image.open(img_p).convert("RGB"))
        mask_raw = np.array(Image.open(msk_p).convert("L"))

        # Binarize lung mask: >127 is lung (1), else background (0)
        mask = (mask_raw > 127).astype(np.int64)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask  = torch.from_numpy(mask).long()

        return image, mask, stem


# ── Dataset Sanity Verification ──────────────────────────────────────────────
_PAIR_DATA = build_cxr_pairs()
print("── Dataset Sanity Check ─────────────────────────────────────")
print(f"  CXR Root : {_PAIR_DATA['cxr_dir']}")
print(f"  Mask Root: {_PAIR_DATA['msk_dir']}")
print(f"  Montgomery (Source Train): {len(_PAIR_DATA['mc_train'])} verified pairs")
print(f"  Montgomery (Source Val)  : {len(_PAIR_DATA['mc_val'])} verified pairs")
print(f"  Shenzhen   (Target Test) : {len(_PAIR_DATA['sz_test'])} verified pairs")

_sample_img, _sample_msk, _sample_stem = CXRLungDataset(_PAIR_DATA['mc_train'], val_tfm())[0]
_lung_area = (_sample_msk == 1).float().mean().item() * 100.0
print(f"  ✓ Sanity check sample: {_sample_stem} | shape {_sample_img.shape} | lung occupancy: {_lung_area:.1f}%")
print("─────────────────────────────────────────────────────────────\n")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LOSS FUNCTIONS & EVALUATION METRICS                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1]
        t_one = (targets == 1).float()
        intersection = (probs * t_one).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + t_one.sum() + self.smooth)
        return 1.0 - dice

class CombinedLoss(nn.Module):
    def __init__(self, weights=CFG.CLASS_WEIGHTS):
        super().__init__()
        w = torch.tensor(weights, dtype=torch.float)
        self.ce = nn.CrossEntropyLoss(weight=w)
        self.dice = BinaryDiceLoss()

    def forward(self, logits, targets):
        if self.ce.weight.device != logits.device:
            self.ce.weight = self.ce.weight.to(logits.device)
        return self.ce(logits, targets) + self.dice(logits, targets)


def dice_score(pred_bin, gt_bin, eps=1e-6):
    pred = pred_bin.astype(bool); gt = gt_bin.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0: return 1.0
    return float((2.0 * inter + eps) / (total + eps))

def jaccard_score(pred_bin, gt_bin, eps=1e-6):
    pred = pred_bin.astype(bool); gt = gt_bin.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0: return 1.0
    return float((inter + eps) / (union + eps))

def evaluate_binary(probs, labels):
    if torch.is_tensor(probs):
        while probs.dim() > 3: probs = probs.squeeze(0)
        pred = probs.argmax(dim=0).cpu().numpy()
    elif isinstance(probs, np.ndarray):
        while probs.ndim > 3: probs = probs.squeeze(0)
        pred = probs.argmax(axis=0)

    if torch.is_tensor(labels):
        while labels.dim() > 2: labels = labels.squeeze(0)
        gt = labels.cpu().numpy()
    elif isinstance(labels, np.ndarray):
        while labels.ndim > 2: labels = labels.squeeze(0)
        gt = labels

    pred_lung = (pred == 1); gt_lung = (gt == 1)
    return {
        "dice":    dice_score(pred_lung, gt_lung),
        "jaccard": jaccard_score(pred_lung, gt_lung)
    }

def aggregate_metrics(metrics_list):
    dices = [m["dice"] for m in metrics_list]
    jacs  = [m["jaccard"] for m in metrics_list]
    return {
        "dice":    (float(np.mean(dices)), float(np.std(dices))),
        "jaccard": (float(np.mean(jacs)),  float(np.std(jacs)))
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 — SOURCE MODEL TRAINING (MONTGOMERY COUNTY, USA)                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_source():
    print("═" * 65)
    print(f"  STEP 1 — Training Source Model on {CFG.SOURCE}")
    print("═" * 65)

    model = smp.Unet(
        encoder_name=CFG.ENCODER,
        encoder_weights=CFG.ENCODER_WEIGHTS,
        classes=CFG.NUM_CLASSES,
        in_channels=3
    ).to(CFG.DEVICE)

    train_ds = CXRLungDataset(_PAIR_DATA["mc_train"], transform=train_tfm())
    val_ds   = CXRLungDataset(_PAIR_DATA["mc_val"],   transform=val_tfm())

    train_loader = DataLoader(train_ds, batch_size=CFG.TRAIN_BS, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.TRAIN_BS, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.NUM_EPOCHS, eta_min=1e-6)
    scaler    = GradScaler(enabled=CFG.USE_AMP)

    best_val_dice = -1.0; patience_cnt = 0
    history = {"loss": [], "val_dice": [], "val_jaccard": []}

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for imgs, masks, _ in train_loader:
            imgs  = imgs.to(CFG.DEVICE, non_blocking=True)
            masks = masks.to(CFG.DEVICE, non_blocking=True)
            optimizer.zero_grad()

            with autocast(enabled=CFG.USE_AMP):
                logits = model(imgs)
                loss   = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(imgs)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_res = []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(CFG.DEVICE, non_blocking=True)
                with autocast(enabled=CFG.USE_AMP):
                    probs = F.softmax(model(imgs), dim=1)
                for p, m in zip(probs.cpu(), masks.cpu()):
                    val_res.append(evaluate_binary(p, m))

        val_agg = aggregate_metrics(val_res)
        vd = val_agg["dice"][0]; vj = val_agg["jaccard"][0]

        history["loss"].append(train_loss)
        history["val_dice"].append(vd)
        history["val_jaccard"].append(vj)

        is_best = vd > best_val_dice
        star = " ★ (Best)" if is_best else ""
        print(f"  [Epoch {epoch:03d}/{CFG.NUM_EPOCHS:03d}] "
              f"Loss: {train_loss:.4f} │ Val Dice: {vd:.4f} │ Val Jac: {vj:.4f}{star}")

        if is_best:
            best_val_dice = vd; patience_cnt = 0
            torch.save({"state": model.state_dict(), "val": val_agg}, CFG.CKPT_PATH)
        else:
            patience_cnt += 1
            if patience_cnt >= CFG.PATIENCE:
                print(f"  Early stop triggered at epoch {epoch}.")
                break

    print(f"\n  ✓ Training complete. Best Val Dice = {best_val_dice:.4f}")

    # Plot learning curves
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["loss"], color="#2c3e50", lw=2); axes[0].set_title("Training Loss (Dice + CE)")
    axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=0.3)
    axes[1].plot(history["val_dice"], color="#27ae60", lw=2); axes[1].set_title("Val Lung Dice ↑")
    axes[1].set_xlabel("Epoch"); axes[1].grid(alpha=0.3)
    axes[2].plot(history["val_jaccard"], color="#2980b9", lw=2); axes[2].set_title("Val Lung Jaccard ↑")
    axes[2].set_xlabel("Epoch"); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUT_DIR, "fig0_cxr_learning_curves.png"), dpi=150)
    plt.close()

    ckpt = torch.load(CFG.CKPT_PATH, map_location=CFG.DEVICE)
    model.load_state_dict(ckpt["state"])
    return model


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 — SOURCE STATISTICS COLLECTION (PHASE 0)                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def collect_source_stats(model):
    print("\n" + "═" * 65)
    print(f"  STEP 2 — Phase 0: Collect Class-Conditional Statistics ({CFG.SOURCE})")
    print("═" * 65)

    model.eval()
    K = CFG.NUM_CLASSES
    bn_mods = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    print(f"  Found {len(bn_mods)} BatchNorm2d layers")

    accum = [{"sum_k": torch.zeros(K, m.num_features),
              "sum_sq_k": torch.zeros(K, m.num_features),
              "count_k": torch.zeros(K)} for _, m in bn_mods]
    global_count = torch.zeros(K)
    state = {"labels": None}

    def make_hook(idx):
        def hook(_, inp, __):
            x = inp[0].detach().cpu()
            lbl = state["labels"]
            if lbl is None: return
            B, C, H, W = x.shape
            lbl_d = F.interpolate(lbl.float().unsqueeze(1), (H, W), mode="nearest").squeeze(1).long()
            for k in range(K):
                mk = (lbl_d == k).float()
                n = mk.sum()
                if n < 1: continue
                mk4 = mk.unsqueeze(1)
                accum[idx]["sum_k"][k]    += (x * mk4).sum(dim=(0, 2, 3))
                accum[idx]["sum_sq_k"][k] += (x * x * mk4).sum(dim=(0, 2, 3))
                accum[idx]["count_k"][k]  += n
        return hook

    hooks = [m.register_forward_hook(make_hook(i)) for i, (_, m) in enumerate(bn_mods)]

    loader = DataLoader(CXRLungDataset(_PAIR_DATA["mc_train"], transform=val_tfm()),
                        batch_size=16, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    with torch.no_grad():
        for imgs, msks, _ in tqdm(loader, desc="  Collecting Source Stats"):
            state["labels"] = msks
            with autocast(enabled=CFG.USE_AMP):
                _ = model(imgs.to(CFG.DEVICE, non_blocking=True))
            for k in range(K):
                global_count[k] += (msks == k).float().sum()

    for h in hooks: h.remove()

    class_means, class_vars = [], []
    for ac in accum:
        C = ac["sum_k"].shape[1]
        mu = torch.zeros(K, C); va = torch.ones(K, C)
        for k in range(K):
            n = ac["count_k"][k].item()
            if n > 0:
                mu[k] = ac["sum_k"][k] / n
                va[k] = (ac["sum_sq_k"][k] / n - mu[k]**2).clamp(min=1e-6)
        class_means.append(mu)
        class_vars.append(va)

    prior = global_count / global_count.sum().clamp(min=1e-6)
    print(f"  Source Prior π: Background={prior[0]:.3f}  │  Lung={prior[1]:.3f}")

    stats = {"class_mean": class_means, "class_var": class_vars, "prior": prior}
    torch.save(stats, CFG.STATS_PATH)
    print(f"  ✓ Saved source stats → {CFG.STATS_PATH}")
    return stats


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PMSA-BN MODULE IMPLEMENTATION (RADIOGRAPHY)                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class PMSABatchNorm2d(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d,
                 K=CFG.NUM_CLASSES, tau=CFG.ALPHA_TAU,
                 k_eff=CFG.KAPPA_EFF,
                 amin=CFG.ALPHA_MIN, amax=CFG.ALPHA_MAX):
        super().__init__()
        self.bn = bn
        C = bn.num_features
        self.register_buffer("src_cls_mean", torch.zeros(K, C))
        self.register_buffer("src_cls_var",  torch.ones(K, C))
        self.register_buffer("src_prior",    torch.ones(K) / K)
        self.K, self.tau = K, tau
        self.k_eff = k_eff
        self.amin, self.amax = amin, amax
        self.adapt_mode  = False
        self.pseudo_mask = None
        self.last_Dc = None; self.last_alpha = None

    def _cls_stats(self, x, mask):
        B, C, H, W = x.shape
        mu = torch.zeros(self.K, C, device=x.device)
        va = torch.zeros(self.K, C, device=x.device)
        Nk = torch.zeros(self.K, device=x.device)
        img_mu = x.mean([0, 2, 3])
        img_va = x.var([0, 2, 3], unbiased=False).clamp(min=1e-6)
        src_global_mu = (self.src_prior[:, None] * self.src_cls_mean).sum(0)

        for k in range(self.K):
            mk = mask[:, k:k+1]; Nk[k] = mk.sum()
            if Nk[k] > 1e-3:
                mu[k] = (mk * x).sum([0, 2, 3]) / Nk[k]
                va[k] = (mk * (x - mu[k].view(1, C, 1, 1))**2).sum([0, 2, 3]) / Nk[k]
            else:
                mu[k] = img_mu + (self.src_cls_mean[k] - src_global_mu)
                va[k] = self.src_cls_var[k]
        return mu, va, Nk, img_mu, img_va

    def _shrink(self, tgt_mu, tgt_va, Nk, img_mu, img_va, H, W):
        smu = torch.zeros_like(tgt_mu)
        sva = torch.zeros_like(tgt_va)
        total_pix = float(H * W)

        for k in range(self.K):
            frac = Nk[k].item() / total_pix
            g  = frac / (frac + self.k_eff)
            gs = max(0.0, frac - 0.001) / (max(0.0, frac - 0.001) + self.k_eff * 2.0)
            anchor_mu = self.src_cls_mean[k]
            anchor_va = self.src_cls_var[k]
            smu[k] = g  * tgt_mu[k] + (1.0 - g)  * anchor_mu
            sva[k] = gs * tgt_va[k] + (1.0 - gs) * anchor_va
        return smu, sva

    def _prior_match(self, cls_mu, cls_va, pi_hat=None):
        if pi_hat is not None:
            pi = 0.5 * self.src_prior + 0.5 * pi_hat
            pi = pi / pi.sum().clamp(min=1e-6)
        else:
            pi = self.src_prior
        pm_mu = (pi[:, None] * cls_mu).sum(0)
        within = (pi[:, None] * cls_va).sum(0)
        between = (pi[:, None] * (cls_mu - pm_mu[None, :])**2).sum(0)
        pm_va = (within + between).clamp(min=1e-6)
        return pm_mu, pm_va

    def _w2_dist(self, pm_mu, pm_va, src_mu, src_va, eps=1e-6):
        return ((pm_mu - src_mu)**2 + (pm_va.clamp(0).sqrt() - src_va.clamp(0).sqrt())**2) / (src_va + eps)

    def forward(self, x):
        if not self.adapt_mode or self.pseudo_mask is None:
            return self.bn(x)

        B, C, H, W = x.shape
        mask = self.pseudo_mask.to(x.device)
        if mask.shape[2] != H or mask.shape[3] != W:
            mask = F.interpolate(mask, (H, W), mode="bilinear", align_corners=False)
        mask = mask.expand(B, -1, -1, -1)

        with torch.no_grad():
            abl = getattr(self, "ablation_flags", {})
            use_cs = abl.get("class_split", True)
            use_sh = abl.get("shrinkage",   True)
            use_pm = abl.get("prior_match", True)
            use_pc = abl.get("per_channel", True)

            src_mu = self.bn.running_mean.detach()
            src_va = self.bn.running_var.detach()

            if not use_cs:
                pm_mu = x.mean([0, 2, 3])
                pm_va = x.var([0, 2, 3], unbiased=False).clamp(min=1e-6)
            else:
                tgt_mu, tgt_va, Nk, img_mu, img_va = self._cls_stats(x, mask)
                pi_hat = (Nk / Nk.sum().clamp(min=1e-6)) if Nk.sum() > 0 else self.src_prior
                shr_mu, shr_va = self._shrink(tgt_mu, tgt_va, Nk, img_mu, img_va, H, W) if use_sh else (tgt_mu, tgt_va)
                pm_mu, pm_va = self._prior_match(shr_mu, shr_va, pi_hat=pi_hat) if use_pm else (shr_mu.mean(0), shr_va.mean(0))

            if use_pc:
                Dc = self._w2_dist(pm_mu, pm_va, src_mu, src_va)
                span = self.amax - self.amin
                alpha = torch.clamp(self.amin + span * torch.tanh(Dc / (1.5 * self.tau)), self.amin, self.amax)
            else:
                Dc = torch.zeros_like(src_mu)
                alpha = torch.full_like(src_mu, (self.amin + self.amax) / 2.0)

            self.last_Dc    = Dc.cpu()
            self.last_alpha = alpha.cpu()

            hat_mu = alpha * pm_mu + (1.0 - alpha) * src_mu
            hat_va = alpha * pm_va + (1.0 - alpha) * src_va

        hat_std = (hat_va + self.bn.eps).clamp(1e-12).sqrt()
        xn = (x - hat_mu.view(1, C, 1, 1)) / hat_std.view(1, C, 1, 1)
        return self.bn.weight.view(1, C, 1, 1) * xn + self.bn.bias.view(1, C, 1, 1)


def replace_bn(model):
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, name, PMSABatchNorm2d(child))
        else:
            replace_bn(child)
    return model

def set_adapt(model, flag):
    for m in model.modules():
        if isinstance(m, PMSABatchNorm2d): m.adapt_mode = flag

def set_mask(model, mask):
    for m in model.modules():
        if isinstance(m, PMSABatchNorm2d): m.pseudo_mask = mask

def inject_stats(model, stats):
    pmsa = [m for m in model.modules() if isinstance(m, PMSABatchNorm2d)]
    dev = next(model.parameters()).device
    for lyr, mu, va in zip(pmsa, stats["class_mean"], stats["class_var"]):
        lyr.src_cls_mean = mu.to(dev)
        lyr.src_cls_var  = va.to(dev)
        lyr.src_prior    = stats["prior"].to(dev)
    print(f"  ✓ Injected stats into {len(pmsa)} PMSA-BN layers")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  INFERENCE FUNCTIONS & ANATOMICAL LUNG REFINEMENT                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def refine_lung_mask(mask_binary, min_size=150):
    """
    Anatomical Bilateral Lung Refinement:
    1. Fills internal bronchovascular bundle hollows independently per hemithorax
       (ensuring the mediastinum and cardiac silhouette are NEVER filled).
    2. Retains legitimate lung lobes while eliminating tiny isolated noise specks (< 150 px).
    """
    mask_np = mask_binary.squeeze().cpu().numpy().astype(bool)
    if not mask_np.any():
        return mask_binary

    H, W = mask_np.shape
    mid_x = W // 2

    # Separate left and right hemithorax so binary_fill_holes NEVER fills the cardiac space!
    left_filled  = ndi.binary_fill_holes(mask_np[:, :mid_x])
    right_filled = ndi.binary_fill_holes(mask_np[:, mid_x:])
    filled = np.hstack([left_filled, right_filled])

    labeled, num_features = ndi.label(filled)
    if num_features > 2:
        sizes = ndi.sum(filled, labeled, range(1, num_features + 1))
        # Keep components with at least min_size pixels (filters out tiny noise specks < 150 px)
        # without amputating legitimate pathological lung lobes
        keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_size]
        if len(keep_labels) == 0:
            keep_labels = [int(sizes.argmax()) + 1]
        clean_mask = np.isin(labeled, keep_labels)
    else:
        clean_mask = filled

    return torch.from_numpy(clean_mask).to(mask_binary.device).view_as(mask_binary)


@torch.no_grad()
def predict_source_only(model, image):
    model.eval()
    return F.softmax(model(image), dim=1)

@torch.no_grad()
def predict_adabn(model, image):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.train(); m.momentum = 1.0
    _ = model(image)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()
    return F.softmax(model(image), dim=1)

@torch.no_grad()
def predict_alpha_bn(src_model, image, alpha=CFG.ALPHA_BN_WEIGHT):
    src_stats = [(m.running_mean.clone(), m.running_var.clone())
                 for m in src_model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in src_model.modules():
        if isinstance(m, nn.BatchNorm2d): m.train(); m.momentum = 1.0
    _ = src_model(image)
    for m, (orig_mu, orig_va) in zip([x for x in src_model.modules() if isinstance(x, nn.BatchNorm2d)], src_stats):
        m.eval()
        m.running_mean = alpha * m.running_mean + (1 - alpha) * orig_mu
        m.running_var  = alpha * m.running_var  + (1 - alpha) * orig_va
    return F.softmax(src_model(image), dim=1)

@torch.no_grad()
def _predict_pmsa_single(model, image, em_iters=CFG.EM_ITERS, src_model=None, pass1=CFG.PASS1,
                         p_src=None, p_adabn=None, p_alpha=None):
    if pass1 == "adabn" and p_adabn is not None:
        probs = p_adabn
    elif pass1 == "adabn" and src_model is not None:
        probs = predict_adabn(copy.deepcopy(src_model).to(image.device), image)
    elif p_src is not None:
        probs = p_src
    elif src_model is not None:
        probs = predict_source_only(copy.deepcopy(src_model).to(image.device), image)
    else:
        probs = predict_source_only(model, image)

    set_adapt(model, True)
    for _ in range(em_iters):
        # Soft responsibility pseudo-mask
        p_lung = probs[:, 1:2]
        p_bg   = probs[:, 0:1]
        soft_m = torch.cat([p_bg, p_lung], dim=1)
        set_mask(model, soft_m)
        probs = F.softmax(model(image), dim=1)
    set_adapt(model, False); set_mask(model, None)
    return probs

@torch.no_grad()
def predict_pmsa(model, image, em_iters=CFG.EM_ITERS, src_model=None, pass1=CFG.PASS1,
                 use_tta=CFG.USE_TTA, p_src=None, p_adabn=None, p_alpha=None):
    p_orig = _predict_pmsa_single(model, image, em_iters, src_model, pass1, p_src, p_adabn, p_alpha)
    if use_tta:
        img_flip = torch.flip(image, dims=[3])
        p_s_f = torch.flip(p_src, dims=[3]) if p_src is not None else None
        p_ada_f = torch.flip(p_adabn, dims=[3]) if p_adabn is not None else None
        p_flip = _predict_pmsa_single(model, img_flip, em_iters, src_model, pass1, p_s_f, p_ada_f, None)
        p_pmsa = 0.5 * p_orig + 0.5 * torch.flip(p_flip, dims=[3])
    else:
        p_pmsa = p_orig

    # Anatomical lung refinement (hemithorax hole filling + speckle suppression)
    raw_mask = (p_pmsa[:, 1:2] > 0.50)
    is_lung  = refine_lung_mask(raw_mask, min_size=150)

    p_lung_clean = torch.where(is_lung, p_pmsa[:, 1:2].clamp(min=0.55), p_pmsa[:, 1:2].clamp(max=0.45))
    return torch.cat([1.0 - p_lung_clean, p_lung_clean], dim=1)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 — EVALUATION ON TARGET (SHENZHEN HOSPITAL, CHINA)                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_evaluation(src_model, pmsa_model):
    print("\n" + "═" * 65)
    print(f"  STEP 3 — Evaluating Benchmark: {CFG.SOURCE} (USA) → {CFG.TARGET} (China)")
    print("═" * 65)

    test_ds = CXRLungDataset(_PAIR_DATA["sz_test"], transform=val_tfm())
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=CFG.NUM_WORKERS, pin_memory=True)
    print(f"  Target Test Set: {len(test_ds)} verified CXR images indexed")

    method_res = defaultdict(list)
    src_model.eval()

    sample_visuals = []

    for idx, (images, labels, stems) in enumerate(tqdm(test_loader, desc=f"  Evaluating {CFG.TARGET}")):
        images = images.to(CFG.DEVICE)

        # 1. Source Only
        p_src = predict_source_only(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        method_res["Source Only"].append(evaluate_binary(p_src.cpu(), labels))

        # 2. AdaBN
        p_adabn = predict_adabn(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        method_res["AdaBN"].append(evaluate_binary(p_adabn.cpu(), labels))

        # 3. α-BN
        p_alpha = predict_alpha_bn(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        method_res["α-BN (α=0.5)"].append(evaluate_binary(p_alpha.cpu(), labels))

        # 4. PMSA-BN (Ours)
        pm = copy.deepcopy(pmsa_model).to(CFG.DEVICE)
        p_pmsa = predict_pmsa(pm, images, src_model=src_model, p_src=p_src, p_adabn=p_adabn, p_alpha=p_alpha)
        method_res["PMSA-BN (Ours)"].append(evaluate_binary(p_pmsa.cpu(), labels))

        # Collect sample qualitative predictions for visual comparison figure
        if idx in [12, 45, 88, 120, 250]:
            sample_visuals.append({
                "stem": stems[0],
                "img":  images[0].cpu(),
                "gt":   labels[0].cpu().numpy(),
                "src":  p_src[0].argmax(0).cpu().numpy(),
                "ada":  p_adabn[0].argmax(0).cpu().numpy(),
                "alp":  p_alpha[0].argmax(0).cpu().numpy(),
                "pmsa": p_pmsa[0].argmax(0).cpu().numpy()
            })

    aggregated = {m: aggregate_metrics(r) for m, r in method_res.items()}

    # Print Summary Table
    print("\n" + "═" * 58)
    print(f"  [{CFG.TARGET}] Benchmark Summary: Lung Field Segmentation")
    print(f"  {'Method':<20} │ {'Lung Dice':>14} │ {'Lung Jaccard':>14}")
    print("  " + "─" * 54)
    best_dice = max(res["dice"][0] for res in aggregated.values())
    best_jac  = max(res["jaccard"][0] for res in aggregated.values())

    for m, res in aggregated.items():
        dd = res["dice"]; jj = res["jaccard"]
        d_star = " ★" if dd[0] == best_dice else "  "
        j_star = " ★" if jj[0] == best_jac  else "  "
        print(f"  {m:<20} │ {dd[0]:>6.4f}±{dd[1]:.3f}{d_star} │ {jj[0]:>6.4f}±{jj[1]:.3f}{j_star}")
    print("═" * 58)

    # Statistical significance testing (PMSA-BN vs AdaBN)
    pmsa_dices = [x["dice"] for x in method_res["PMSA-BN (Ours)"]]
    adabn_dices = [x["dice"] for x in method_res["AdaBN"]]
    t_stat, p_val = scipy_stats.ttest_rel(pmsa_dices, adabn_dices)
    wilcoxon_stat, p_val_w = scipy_stats.wilcoxon(pmsa_dices, adabn_dices)
    print(f"\n  Statistical Significance vs. AdaBN:")
    print(f"  - Paired t-test    : t = {t_stat:.4f}, p = {p_val:.4e}")
    print(f"  - Wilcoxon signed  : W = {wilcoxon_stat:.1f}, p = {p_val_w:.4e}")

    # Generate Figures
    plot_benchmark_comparison(aggregated)
    plot_strip_distributions(method_res)
    plot_qualitative_gallery(sample_visuals)

    return aggregated, method_res


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 — COMPONENT ABLATION STUDY                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_ablation(src_model, pmsa_model):
    print("\n" + "═" * 65)
    print("  STEP 4 — Systematic Component Ablation Study on Shenzhen")
    print("═" * 65)

    ablation_configs = [
        ("AdaBN (Global Shift)",         {"class_split": False, "shrinkage": False, "prior_match": False, "per_channel": False}),
        ("+ Class-Conditional Split",     {"class_split": True,  "shrinkage": False, "prior_match": False, "per_channel": False}),
        ("+ Bayesian Shrinkage",         {"class_split": True,  "shrinkage": True,  "prior_match": False, "per_channel": False}),
        ("+ Prior-Matching (PM)",        {"class_split": True,  "shrinkage": True,  "prior_match": True,  "per_channel": False}),
        ("PMSA-BN (Full: + Shift Alpha)", {"class_split": True,  "shrinkage": True,  "prior_match": True,  "per_channel": True}),
    ]

    # Evaluate on a representative subset of 100 test images for ablation speed
    subset_pairs = _PAIR_DATA["sz_test"][:100]
    subset_loader = DataLoader(CXRLungDataset(subset_pairs, transform=val_tfm()),
                               batch_size=1, shuffle=False, num_workers=CFG.NUM_WORKERS)

    ablation_results = {}

    for name, flags in ablation_configs:
        pm = copy.deepcopy(pmsa_model).to(CFG.DEVICE)
        for m in pm.modules():
            if isinstance(m, PMSABatchNorm2d):
                m.ablation_flags = flags

        res_list = []
        for images, labels, _ in subset_loader:
            images = images.to(CFG.DEVICE)
            p = predict_pmsa(pm, images, src_model=src_model, use_tta=False)
            res_list.append(evaluate_binary(p.cpu(), labels))

        agg = aggregate_metrics(res_list)
        ablation_results[name] = agg
        print(f"  {name:<32} │ Lung Dice: {agg['dice'][0]:.4f}±{agg['dice'][1]:.3f} │ IoU: {agg['jaccard'][0]:.4f}")

    # Plot Ablation progression
    names = list(ablation_results.keys())
    dice_vals = [ablation_results[n]["dice"][0] for n in names]
    errs = [ablation_results[n]["dice"][1] for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, dice_vals, yerr=errs, capsize=4, color=["#95a5a6", "#3498db", "#2980b9", "#e67e22", "#27ae60"],
                  edgecolor="black", alpha=0.85, width=0.55)
    ax.set_ylim(min(dice_vals) - 0.05, max(dice_vals) + 0.05)
    ax.set_ylabel("Lung Dice ↑", fontsize=11)
    ax.set_title(f"PMSA-BN Component Ablation Progression ({CFG.SOURCE} → {CFG.TARGET})", fontsize=12, fontweight="bold")
    plt.xticks(rotation=18, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUT_DIR, "fig4_cxr_ablation.png"), dpi=150)
    plt.close()
    print("  ✓ Saved ablation figure → /kaggle/working/fig4_cxr_ablation.png")
    return ablation_results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VISUALIZATION & PUBLICATION ARTIFACTS                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_benchmark_comparison(aggregated):
    methods = list(aggregated.keys())
    colors  = ["#7f8c8d", "#e74c3c", "#e67e22", "#27ae60"]
    metrics = [("dice", "Lung Dice ↑"), ("jaccard", "Lung Jaccard (IoU) ↑")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"PMSA-BN vs. Baselines: CXR Transfer ({CFG.SOURCE} USA → {CFG.TARGET} China)",
                 fontsize=13, fontweight="bold")

    for ax, (key, title) in zip(axes, metrics):
        vals = [aggregated[m][key][0] for m in methods]
        errs = [aggregated[m][key][1] for m in methods]
        x = np.arange(len(methods))
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors, alpha=0.88, edgecolor="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(min(vals) - 0.08, max(vals) + 0.05)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2.0, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUT_DIR, "fig1_cxr_benchmark_comparison.png"), dpi=150)
    plt.close()
    print("  ✓ Saved comparison figure → /kaggle/working/fig1_cxr_benchmark_comparison.png")


def plot_strip_distributions(method_res):
    methods = list(method_res.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    metrics = ["dice", "jaccard"]
    titles  = ["Lung Dice Score Distribution", "Lung Jaccard (IoU) Distribution"]

    for ax, metric, title in zip(axes, metrics, titles):
        data = [[x[metric] for x in method_res[m]] for m in methods]
        parts = ax.boxplot(data, patch_artist=True, labels=methods, showmeans=True,
                           boxprops=dict(facecolor="#ecf0f1", color="#2c3e50"),
                           meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="#e74c3c"))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUT_DIR, "fig2_cxr_strip_plots.png"), dpi=150)
    plt.close()
    print("  ✓ Saved distribution figure → /kaggle/working/fig2_cxr_strip_plots.png")


def plot_qualitative_gallery(samples):
    if not samples: return
    n = len(samples)
    fig, axes = plt.subplots(n, 6, figsize=(16, 2.8 * n))
    col_titles = ["Input Radiograph", "Ground Truth", "Source Only", "AdaBN", "α-BN (0.5)", "PMSA-BN (Ours)"]

    for r, smp_dict in enumerate(samples):
        # Denormalize image
        img = smp_dict["img"].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[r, 0].imshow(img); axes[r, 0].set_ylabel(smp_dict["stem"][:14], fontsize=9, fontweight="bold")
        axes[r, 1].imshow(smp_dict["gt"], cmap="Blues", vmin=0, vmax=1)
        axes[r, 2].imshow(smp_dict["src"], cmap="Blues", vmin=0, vmax=1)
        axes[r, 3].imshow(smp_dict["ada"], cmap="Blues", vmin=0, vmax=1)
        axes[r, 4].imshow(smp_dict["alp"], cmap="Blues", vmin=0, vmax=1)
        axes[r, 5].imshow(smp_dict["pmsa"], cmap="Blues", vmin=0, vmax=1)

        for c in range(6):
            if r == 0: axes[r, c].set_title(col_titles[c], fontsize=10, fontweight="bold")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUT_DIR, "fig3_cxr_qualitative_examples.png"), dpi=150)
    plt.close()
    print("  ✓ Saved qualitative gallery → /kaggle/working/fig3_cxr_qualitative_examples.png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN EXECUTION PIPELINE                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  PMSA-BN Chest X-Ray Benchmark Pipeline (Montgomery → Shenzhen)    ║")
    print("╚" + "═" * 68 + "╝\n")

    # 1. Source Model
    if os.path.exists(CFG.CKPT_PATH) and os.path.exists(CFG.STATS_PATH) and not CFG.TRAIN_FROM_SCRATCH:
        print(f"  [INFO] Found pretrained checkpoint at {CFG.CKPT_PATH}.")
        print(f"  [INFO] Found precomputed stats at {CFG.STATS_PATH}.")
        src_model = smp.Unet(encoder_name=CFG.ENCODER, classes=CFG.NUM_CLASSES, in_channels=3).to(CFG.DEVICE)
        ckpt = torch.load(CFG.CKPT_PATH, map_location=CFG.DEVICE)
        src_model.load_state_dict(ckpt["state"])
        stats = torch.load(CFG.STATS_PATH, map_location="cpu")
    else:
        src_model = train_source()
        stats = collect_source_stats(src_model)

    # 2. Prepare PMSA-BN Model
    pmsa_model = replace_bn(copy.deepcopy(src_model))
    inject_stats(pmsa_model, stats)
    pmsa_model.eval().to(CFG.DEVICE)

    # 3. Benchmark Evaluation
    aggregated, method_res = run_evaluation(src_model, pmsa_model)

    # 4. Systematic Ablation Study
    if CFG.RUN_ABLATION:
        ablation_res = run_ablation(src_model, pmsa_model)

    # 5. Export JSON
    def jsonify(d):
        if isinstance(d, dict): return {k: jsonify(v) for k, v in d.items()}
        if isinstance(d, (list, tuple)): return [jsonify(x) for x in d]
        if isinstance(d, (np.float32, np.float64)): return float(d)
        if isinstance(d, (np.int32, np.int64)): return int(d)
        return d

    out_json = os.path.join(CFG.OUT_DIR, "cxr_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(jsonify({"benchmark": aggregated, "config": {k: getattr(CFG, k) for k in dir(CFG) if not k.startswith("__")}}), f, indent=2)
    print(f"\n  ✓ All benchmark results exported → {out_json}")
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Chest Radiography Benchmark Execution Finished Successfully!        ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
