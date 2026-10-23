"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm                      ║
║    Multi-Target Chest Radiography Benchmark Suite (MICCAI / ISBI)            ║
║                                                                              ║
║  SOURCE:  Montgomery County (MC, USA) or Shenzhen Hospital (SZ, China)       ║
║  TARGETS: Array-selectable via CFG.TARGETS:                                  ║
║           ["JSRT", "COVID-QU-Ex"]                                            ║
║                                                                              ║
║  Kaggle Datasets Required to Mount:                                          ║
║  1. Source:   nikhilpandey360/chest-xray-masks-and-labels                    ║
║  2. Target 1: raddar/nodules-in-chest-xrays (JSRT)                            ║
║  3. Target 2: tawsifurrahman/covid19-radiography-database (COVID-QU-Ex)      ║
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
# ║  CFG — MULTI-TARGET RADIOGRAPHY BENCHMARK CONFIGURATION                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CFG:
    SOURCE = "Montgomery"     # "Montgomery" (USA) or "Shenzhen" (China)

    # ── Target Cohorts to Evaluate ────────────────────────────────────────
    TARGETS = [
        "JSRT",               # Japanese Society of Radiological Technology (247 CXRs)
        "COVID-QU-Ex",        # COVID-QU-Ex Multi-Center Radiography Database
    ]

    # Evaluates up to MAX_SAMPLES per target for fast benchmark execution
    MAX_TARGET_SAMPLES = {
        "JSRT":        247,   # Full cohort
        "COVID-QU-Ex": 400,   # 100 cases balanced per class (COVID, Normal, Opacity, Pneumonia)
    }

    # ── Checkpoint Paths ──────────────────────────────────────────────────
    CKPT_PATH  = "/kaggle/working/cxr_best_model.pth"
    STATS_PATH = "/kaggle/working/cxr_source_stats.pth"
    OUT_DIR    = "/kaggle/working"

    # ── Model ─────────────────────────────────────────────────────────────
    ENCODER         = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES     = 2          # 0=background, 1=lung
    IMAGE_SIZE      = 256

    # ── Training on Source (if checkpoint not present) ────────────────────
    TRAIN_BS           = 16
    NUM_EPOCHS         = 30
    LR                 = 3e-4
    WEIGHT_DECAY       = 1e-5
    PATIENCE           = 8
    CLASS_WEIGHTS      = [0.4, 1.6]
    USE_AMP            = True
    TRAIN_FROM_SCRATCH = False   # Reuses /kaggle/working checkpoint automatically

    # ── PMSA-BN Hyperparameters ───────────────────────────────────────────
    TAU_HI      = 0.50
    KAPPA_EFF   = 0.05
    ALPHA_MIN   = 0.20           # Calibrated lower bound
    ALPHA_MAX   = 0.70           # Calibrated upper bound
    ALPHA_TAU   = 0.50           # Shift sensitivity temperature
    EM_ITERS    = 1              # Single refinement pass
    PASS1       = "alpha"        # Start pseudo-mask from alpha-BN
    USE_TTA     = True           # Horizontal flip TTA

    # ── Baselines ─────────────────────────────────────────────────────────
    ALPHA_BN_WEIGHT = 0.5

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
print(f"✓ Source  : {CFG.SOURCE}")
print(f"✓ Targets : {CFG.TARGETS}")
print(f"✓ Task    : Bilateral Lung Field Segmentation | Metrics: Dice & Jaccard (IoU)")
print("─" * 65)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATASET AUTO-DISCOVERY & PARSING                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def auto_find_cxr_source_dirs():
    candidate_roots = [
        "/kaggle/input/datasets/nikhilpandey360/chest-xray-masks-and-labels",
        "/kaggle/input/datasets/nikhilpandey360/chest-xray-masks-and-labels/Lung Segmentation",
        "/kaggle/input/chest-xray-masks-and-labels",
        "/kaggle/input/chest-xray-masks-and-labels/Lung Segmentation",
    ]
    for root in candidate_roots:
        if not os.path.exists(root): continue
        c = os.path.join(root, "CXR_png")
        m = os.path.join(root, "masks")
        if os.path.exists(c) and os.path.exists(m):
            return c, m

    # Generic search
    for r in ["/kaggle/input", "/kaggle/input/datasets"]:
        if os.path.exists(r):
            for root, dirs, _ in os.walk(r):
                if "CXR_png" in dirs and "masks" in dirs:
                    return os.path.join(root, "CXR_png"), os.path.join(root, "masks")
    return None, None


def build_source_pairs(source_name=CFG.SOURCE):
    cxr_dir, msk_dir = auto_find_cxr_source_dirs()
    if cxr_dir is None:
        return None, None

    all_imgs = sorted(glob.glob(os.path.join(cxr_dir, "*.png")) + glob.glob(os.path.join(cxr_dir, "*.PNG")))
    mask_map = {}
    for m in glob.glob(os.path.join(msk_dir, "**", "*.png"), recursive=True) + glob.glob(os.path.join(msk_dir, "**", "*.PNG"), recursive=True):
        fname = os.path.basename(m)
        stem = os.path.splitext(fname)[0]
        mask_map[fname] = m
        mask_map[stem] = m
        if stem.endswith("_mask"):
            mask_map[stem[:-5]] = m

    prefix = "MCUCXR" if source_name.lower().startswith("mont") else "CHNCXR"
    pairs = []
    for img_p in all_imgs:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        if stem.startswith(prefix):
            m_p = mask_map.get(stem) or mask_map.get(stem + "_mask")
            if m_p and os.path.exists(m_p):
                pairs.append((img_p, m_p, stem))

    rng = random.Random(CFG.SEED)
    rng.shuffle(pairs)
    n_val = max(15, int(0.15 * len(pairs)))
    return pairs[n_val:], pairs[:n_val]


def discover_jsrt_pairs():
    """
    Auto-discovers JSRT images and paired bilateral lung masks.
    Supported Kaggle Handles:
    - abduzzami/jsrt-247-image-lung-segmentation-mask-dataset (content/jsrt/cxr and content/jsrt/masks)
    - raddar/nodules-in-chest-xrays-jsrt
    - kmader/pulmonary-chest-xray-abnormalities
    """
    search_dirs = [
        "/kaggle/input/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt",
        "/kaggle/input/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset",
        "/kaggle/input/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt",
        "/kaggle/input/jsrt-247-image-lung-segmentation-mask-dataset",
        "/kaggle/input/datasets/raddar/nodules-in-chest-xrays-jsrt",
        "/kaggle/input/nodules-in-chest-xrays-jsrt",
        "/kaggle/input/pulmonary-chest-xray-abnormalities",
        "/kaggle/input/datasets/kmader/pulmonary-chest-xray-abnormalities",
        "/kaggle/input/jsrt",
    ]
    found_root = None
    for d in search_dirs:
        if os.path.exists(d):
            found_root = d; break

    if found_root is None:
        for r, dirs, _ in os.walk("/kaggle/input"):
            if any("jsrt" in d.lower() or "nodule" in d.lower() for d in dirs):
                found_root = r; break

    if found_root is None:
        return []

    all_files = sorted(glob.glob(os.path.join(found_root, "**", "*.png"), recursive=True) +
                       glob.glob(os.path.join(found_root, "**", "*.PNG"), recursive=True) +
                       glob.glob(os.path.join(found_root, "**", "*.jpg"), recursive=True))

    mask_dict = {}
    img_candidates = []
    for p in all_files:
        p_lower = p.lower()
        fname = os.path.basename(p)
        stem  = os.path.splitext(fname)[0].lower()
        if "/masks/" in p_lower or "\\masks\\" in p_lower or stem.endswith("_mask"):
            base_stem = stem[:-5] if stem.endswith("_mask") else stem
            mask_dict[base_stem] = p
            mask_dict[stem] = p
        elif "/cxr/" in p_lower or "\\cxr\\" in p_lower or "/images/" in p_lower or "\\images\\" in p_lower:
            img_candidates.append(p)
        elif stem.startswith("jpcln") or stem.startswith("jpcnn"):
            if not ("/masks/" in p_lower or "\\masks\\" in p_lower):
                img_candidates.append(p)

    pairs = []
    seen = set()
    for img_p in img_candidates:
        fname = os.path.basename(img_p)
        stem  = os.path.splitext(fname)[0].lower()
        if stem in seen:
            continue
        m_p = mask_dict.get(stem) or mask_dict.get(stem + "_mask")
        if m_p and os.path.exists(m_p) and m_p != img_p:
            pairs.append((img_p, m_p, os.path.splitext(fname)[0]))
            seen.add(stem)

    return pairs[:CFG.MAX_TARGET_SAMPLES["JSRT"]]


def discover_covid_qu_ex_pairs():
    """
    Auto-discovers COVID-QU-Ex / COVID-19 Radiography Database.
    Kaggle Handle: tawsifurrahman/covid19-radiography-database
    """
    search_dirs = [
        "/kaggle/input/datasets/tawsifurrahman/covid19-radiography-database/COVID-19_Radiography_Dataset",
        "/kaggle/input/datasets/tawsifurrahman/covid19-radiography-database",
        "/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset",
        "/kaggle/input/covid19-radiography-database",
        "/kaggle/input/covid19-radiography-dataset",
        "/kaggle/input/COVID-19_Radiography_Dataset",
    ]
    found_root = None
    for d in search_dirs:
        if os.path.exists(d):
            found_root = d; break

    if found_root is None:
        for r, dirs, _ in os.walk("/kaggle/input"):
            if any("covid19" in d.lower() or "covid-19" in d.lower() for d in dirs):
                found_root = r; break

    if found_root is None:
        return []

    classes = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]
    pairs = []
    max_per_class = CFG.MAX_TARGET_SAMPLES["COVID-QU-Ex"] // len(classes)

    for cls_name in classes:
        cls_imgs = sorted(glob.glob(os.path.join(found_root, "**", cls_name, "images", "*.png"), recursive=True))
        cls_pairs = []
        for img_p in cls_imgs:
            fname = os.path.basename(img_p)
            stem  = os.path.splitext(fname)[0]
            p_dir = os.path.dirname(img_p)
            cands = [
                os.path.join(os.path.dirname(p_dir), "masks", fname),
                os.path.join(os.path.dirname(p_dir), "masks", stem + ".png"),
                img_p.replace("images", "masks"),
            ]
            msk_p = None
            for c in cands:
                if os.path.exists(c):
                    msk_p = c; break
            if msk_p is not None:
                cls_pairs.append((img_p, msk_p, stem))
        pairs.extend(cls_pairs[:max_per_class])

    return pairs


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATASET CLASS & AUGMENTATIONS                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_tfm():
    try:
        return A.Compose([
            A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.08, rotate_limit=10, p=0.5, border_mode=0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], is_check_shapes=False)
    except TypeError:
        return A.Compose([
            A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.08, rotate_limit=10, p=0.5, border_mode=0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def val_tfm():
    try:
        return A.Compose([
            A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], is_check_shapes=False)
    except TypeError:
        return A.Compose([
            A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        image_pil = Image.open(img_p).convert("RGB")
        mask_pil  = Image.open(msk_p).convert("L")

        # Synchronize spatial dimensions if dataset packages slight resolution mismatches
        if image_pil.size != mask_pil.size:
            mask_pil = mask_pil.resize(image_pil.size, resample=Image.NEAREST)

        image = np.array(image_pil, dtype=np.uint8)
        mask_raw = np.array(mask_pil, dtype=np.uint8)
        mask = (mask_raw > 127).astype(np.int64)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask  = torch.from_numpy(mask).long()

        return image, mask, stem


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LOSSES & METRICS                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1:2]
        tgts  = (targets == 1).float().unsqueeze(1)
        inter = (probs * tgts).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + tgts.sum(dim=(2, 3))
        dice  = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.tensor(CFG.CLASS_WEIGHTS, device=CFG.DEVICE).float()
        self.ce   = nn.CrossEntropyLoss(weight=w)
        self.dice = SoftDiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)


def dice_score(pred_bin, gt_bin, eps=1e-6):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum()
    return float((2.0 * inter + eps) / (union + eps))

def jaccard_score(pred_bin, gt_bin, eps=1e-6):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return float((inter + eps) / (union + eps))

def evaluate_binary(probs, labels):
    if torch.is_tensor(probs):
        while probs.dim() > 3: probs = probs.squeeze(0)
        pred = probs.argmax(dim=0).cpu().numpy()
    else:
        while probs.ndim > 3: probs = probs.squeeze(0)
        pred = probs.argmax(axis=0)

    if torch.is_tensor(labels):
        while labels.dim() > 2: labels = labels.squeeze(0)
        gt = labels.cpu().numpy()
    else:
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
# ║  SOURCE MODEL TRAINING & PHASE 0 STATS                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_or_load_source():
    model = smp.Unet(
        encoder_name=CFG.ENCODER,
        encoder_weights=CFG.ENCODER_WEIGHTS,
        classes=CFG.NUM_CLASSES,
        in_channels=3
    ).to(CFG.DEVICE)

    # Check for existing pretrained checkpoint
    for ckpt_cand in [
        CFG.CKPT_PATH,
        "/kaggle/working/cxr_best_model.pth",
        "/kaggle/working/cxr_sz2mc_best_model.pth",
    ]:
        if os.path.exists(ckpt_cand):
            print(f"  ✓ Loaded source model checkpoint from {ckpt_cand}")
            ckpt = torch.load(ckpt_cand, map_location=CFG.DEVICE)
            state = ckpt["state"] if "state" in ckpt else ckpt
            model.load_state_dict(state)
            return model

    # If absent, train on Montgomery
    print(f"  [INFO] Training source model on {CFG.SOURCE} from scratch...")
    train_pairs, val_pairs = build_source_pairs(CFG.SOURCE)
    train_loader = DataLoader(CXRLungDataset(train_pairs, transform=train_tfm()),
                              batch_size=CFG.TRAIN_BS, shuffle=True, num_workers=CFG.NUM_WORKERS)
    val_loader   = DataLoader(CXRLungDataset(val_pairs, transform=val_tfm()),
                              batch_size=CFG.TRAIN_BS, shuffle=False, num_workers=CFG.NUM_WORKERS)

    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scaler    = GradScaler(enabled=CFG.USE_AMP)

    best_val_dice = -1.0
    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        model.train()
        for imgs, masks, _ in train_loader:
            imgs  = imgs.to(CFG.DEVICE, non_blocking=True)
            masks = masks.to(CFG.DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast(enabled=CFG.USE_AMP):
                loss = criterion(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_res = []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(CFG.DEVICE, non_blocking=True)
                with autocast(enabled=CFG.USE_AMP):
                    probs = F.softmax(model(imgs), dim=1)
                for p, m in zip(probs.cpu(), masks.cpu()):
                    val_res.append(evaluate_binary(p, m))

        vd = aggregate_metrics(val_res)["dice"][0]
        if vd > best_val_dice:
            best_val_dice = vd
            torch.save({"state": model.state_dict()}, CFG.CKPT_PATH)

    print(f"  ✓ Finished training. Best Val Dice = {best_val_dice:.4f}")
    ckpt = torch.load(CFG.CKPT_PATH, map_location=CFG.DEVICE)
    model.load_state_dict(ckpt["state"])
    return model


def collect_source_stats(model):
    for stats_cand in [
        CFG.STATS_PATH,
        "/kaggle/working/cxr_source_stats.pth",
        "/kaggle/working/cxr_sz2mc_source_stats.pth",
    ]:
        if os.path.exists(stats_cand):
            print(f"  ✓ Loaded Phase 0 source prototype statistics from {stats_cand}")
            return torch.load(stats_cand, map_location=CFG.DEVICE)

    print(f"  STEP 2 — Phase 0: Collect Class-Conditional Statistics ({CFG.SOURCE})")
    model.eval()
    K = CFG.NUM_CLASSES
    bn_mods = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]

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
    train_pairs, _ = build_source_pairs(CFG.SOURCE)
    loader = DataLoader(CXRLungDataset(train_pairs, transform=val_tfm()),
                        batch_size=16, shuffle=False, num_workers=CFG.NUM_WORKERS)

    with torch.no_grad():
        for imgs, msks, _ in tqdm(loader, desc="  Collecting Source Stats"):
            state["labels"] = msks
            with autocast(enabled=CFG.USE_AMP):
                _ = model(imgs.to(CFG.DEVICE))
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
            smu[k] = g  * tgt_mu[k] + (1.0 - g)  * self.src_cls_mean[k]
            sva[k] = gs * tgt_va[k] + (1.0 - gs) * self.src_cls_var[k]
        return smu, sva

    def _prior_match(self, cls_mu, cls_va, pi_hat=None):
        pi = self.src_prior if pi_hat is None else (0.5 * self.src_prior + 0.5 * pi_hat)
        pi = pi / pi.sum().clamp(min=1e-6)
        pm_mu = (pi[:, None] * cls_mu).sum(0)
        within  = (pi[:, None] * cls_va).sum(0)
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
            src_mu = self.bn.running_mean.detach()
            src_va = self.bn.running_var.detach()

            tgt_mu, tgt_va, Nk, img_mu, img_va = self._cls_stats(x, mask)
            pi_hat = (Nk / Nk.sum().clamp(min=1e-6)) if Nk.sum() > 0 else self.src_prior
            shr_mu, shr_va = self._shrink(tgt_mu, tgt_va, Nk, img_mu, img_va, H, W)
            pm_mu, pm_va   = self._prior_match(shr_mu, shr_va, pi_hat=pi_hat)

            Dc = self._w2_dist(pm_mu, pm_va, src_mu, src_va)
            span = self.amax - self.amin
            alpha = torch.clamp(self.amin + span * torch.tanh(Dc / (1.5 * self.tau)), self.amin, self.amax)

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


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  INFERENCE FUNCTIONS & BILATERAL ANATOMICAL REFINEMENT                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def refine_lung_mask(mask_binary, min_size=150):
    mask_np = mask_binary.squeeze().cpu().numpy().astype(bool)
    if not mask_np.any():
        return mask_binary

    H, W = mask_np.shape
    mid_x = W // 2
    left_filled  = ndi.binary_fill_holes(mask_np[:, :mid_x])
    right_filled = ndi.binary_fill_holes(mask_np[:, mid_x:])
    filled = np.hstack([left_filled, right_filled])

    labeled, num_features = ndi.label(filled)
    if num_features > 2:
        sizes = ndi.sum(filled, labeled, range(1, num_features + 1))
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
def _predict_pmsa_single(model, image, em_iters=CFG.EM_ITERS, src_model=None, p_alpha=None):
    if p_alpha is not None:
        probs = p_alpha
    elif src_model is not None:
        probs = predict_alpha_bn(copy.deepcopy(src_model).to(image.device), image)
    else:
        probs = predict_source_only(model, image)

    set_adapt(model, True)
    for _ in range(em_iters):
        p_lung = probs[:, 1:2]; p_bg = probs[:, 0:1]
        soft_m = torch.cat([p_bg, p_lung], dim=1)
        set_mask(model, soft_m)
        probs = F.softmax(model(image), dim=1)
    set_adapt(model, False); set_mask(model, None)
    return probs

@torch.no_grad()
def predict_pmsa(model, image, em_iters=CFG.EM_ITERS, src_model=None, use_tta=CFG.USE_TTA, p_alpha=None):
    if p_alpha is None and src_model is not None:
        p_alpha = predict_alpha_bn(copy.deepcopy(src_model).to(image.device), image)

    p_orig = _predict_pmsa_single(model, image, em_iters, src_model, p_alpha)
    if use_tta:
        img_flip   = torch.flip(image, dims=[3])
        p_alp_flip = torch.flip(p_alpha, dims=[3]) if p_alpha is not None else None
        p_flip     = _predict_pmsa_single(model, img_flip, em_iters, src_model, p_alp_flip)
        p_pmsa     = 0.5 * p_orig + 0.5 * torch.flip(p_flip, dims=[3])
    else:
        p_pmsa = p_orig

    if p_alpha is not None:
        p_lung_fused = (0.70 * p_pmsa[:, 1:2] + 0.30 * p_alpha[:, 1:2]).clamp(0.0, 1.0)
    else:
        p_lung_fused = p_pmsa[:, 1:2]

    raw_mask = (p_lung_fused > 0.50)
    is_lung  = refine_lung_mask(raw_mask, min_size=150)
    p_lung_clean = torch.where(is_lung, p_lung_fused.clamp(min=0.55), p_lung_fused.clamp(max=0.45))
    return torch.cat([1.0 - p_lung_clean, p_lung_clean], dim=1)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TARGET EVALUATION HARNESS                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def evaluate_target_domain(target_name, test_pairs, src_model, pmsa_model):
    print("\n" + "═" * 65)
    print(f"  Evaluating Target: {target_name} ({len(test_pairs)} test cases)")
    print("═" * 65)

    test_ds = CXRLungDataset(test_pairs, transform=val_tfm())
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=CFG.NUM_WORKERS)

    method_res = defaultdict(list)
    src_model.eval()

    for images, labels, _ in tqdm(test_loader, desc=f"  Evaluating {target_name}"):
        images = images.to(CFG.DEVICE)

        p_src   = predict_source_only(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        method_res["Source Only"].append(evaluate_binary(p_src.cpu(), labels))

        p_adabn = predict_adabn(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        method_res["AdaBN"].append(evaluate_binary(p_adabn.cpu(), labels))

        p_alpha = predict_alpha_bn(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        method_res["α-BN (α=0.5)"].append(evaluate_binary(p_alpha.cpu(), labels))

        pm = copy.deepcopy(pmsa_model).to(CFG.DEVICE)
        p_pmsa  = predict_pmsa(pm, images, src_model=src_model, p_alpha=p_alpha)
        method_res["PMSA-BN (Ours)"].append(evaluate_binary(p_pmsa.cpu(), labels))

    aggregated = {m: aggregate_metrics(r) for m, r in method_res.items()}

    print("\n" + "═" * 58)
    print(f"  [{target_name}] Benchmark Summary: Lung Field Segmentation")
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

    # Statistical significance testing
    pmsa_d = [x["dice"] for x in method_res["PMSA-BN (Ours)"]]
    adabn_d = [x["dice"] for x in method_res["AdaBN"]]
    t_stat, p_val = scipy_stats.ttest_rel(pmsa_d, adabn_d)
    w_stat, w_val = scipy_stats.wilcoxon(pmsa_d, adabn_d)
    print(f"  Statistical Significance vs. AdaBN:")
    print(f"  - Paired t-test    : t = {t_stat:.4f}, p = {p_val:.4e}")
    print(f"  - Wilcoxon signed  : W = {w_stat:.1f}, p = {w_val:.4e}")

    # Plot Comparison Bar Chart
    methods = list(aggregated.keys())
    dice_vals = [aggregated[m]["dice"][0] for m in methods]
    dice_errs = [aggregated[m]["dice"][1] for m in methods]
    colors = ["#95a5a6", "#e74c3c", "#f39c12", "#27ae60"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(methods, dice_vals, yerr=dice_errs, capsize=4, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    ax.set_ylim(max(0.5, min(dice_vals) - 0.08), min(1.0, max(dice_vals) + 0.05))
    ax.set_ylabel("Lung Dice ↑", fontsize=11)
    ax.set_title(f"Cross-Domain Adaptation: {CFG.SOURCE} → {target_name}", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f"{h:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(CFG.OUT_DIR, f"fig_{target_name.lower().replace('-', '_')}_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved figure → {fig_path}")

    return aggregated


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE EXECUTION                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   PMSA-BN — Multi-Target Chest Radiography Benchmark Suite           ║")
    print(f"║   Source Domain : {CFG.SOURCE:<51}║")
    print(f"║   Target Domains: {', '.join(CFG.TARGETS):<51}║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")

    # Step 1: Source Model
    src_model = train_or_load_source()

    # Step 2: Prototype Stats
    stats = collect_source_stats(src_model)

    # Initialize PMSA model
    pmsa_model = copy.deepcopy(src_model)
    pmsa_model = replace_bn(pmsa_model)
    inject_stats(pmsa_model, stats)

    # Step 3: Sequential Target Evaluation
    all_target_results = {}

    for target in CFG.TARGETS:
        if target.lower() == "jsrt":
            pairs = discover_jsrt_pairs()
            dataset_hint = "raddar/nodules-in-chest-xrays"
        elif target.lower() in ("covid-qu-ex", "covid"):
            pairs = discover_covid_qu_ex_pairs()
            dataset_hint = "tawsifurrahman/covid19-radiography-database"
        else:
            pairs = []
            dataset_hint = "unknown"

        if len(pairs) == 0:
            print(f"\n  [WARN] Target '{target}' dataset not found in /kaggle/input.")
            print(f"  👉 Please attach '{dataset_hint}' to your notebook to evaluate on {target}.")
            continue

        agg = evaluate_target_domain(target, pairs, src_model, pmsa_model)
        all_target_results[target] = agg

    # Step 4: Export Master Summary
    if len(all_target_results) > 0:
        out_json = os.path.join(CFG.OUT_DIR, "multi_target_radiography_results.json")
        with open(out_json, "w") as f:
            json.dump(all_target_results, f, indent=2)
        print(f"\n  ✓ All multi-target results exported → {out_json}")

        print("\n" + "═" * 70)
        print(f"  MASTER MULTI-TARGET RADIOGRAPHY SUMMARY TABLE (Source: {CFG.SOURCE})")
        print("═" * 70)
        print(f"  {'Target Domain':<15} │ {'Source Only':>12} │ {'AdaBN':>12} │ {'α-BN (0.5)':>12} │ {'PMSA-BN (Ours)':>14}")
        print("  " + "─" * 66)
        for tgt, res in all_target_results.items():
            so  = f"{res['Source Only']['dice'][0]:.4f}"
            ada = f"{res['AdaBN']['dice'][0]:.4f}"
            alp = f"{res['α-BN (α=0.5)']['dice'][0]:.4f}"
            pm  = f"{res['PMSA-BN (Ours)']['dice'][0]:.4f} ★"
            print(f"  {tgt:<15} │ {so:>12} │ {ada:>12} │ {alp:>12} │ {pm:>14}")
        print("═" * 70 + "\n")
    else:
        print("\n  [INFO] No external targets were evaluated. Attach either JSRT or COVID-QU-Ex to evaluate.")

if __name__ == "__main__":
    main()
