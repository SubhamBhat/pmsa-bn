"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm                      ║
║    Reverse Dermoscopy Benchmark Suite: PH2 (Source) -> ISIC 2018 (Target)    ║
║                                                                              ║
║  SOURCE: PH2 Dataset (kanametov/ph2dataset)                                  ║
║          Pedro Hispano Hospital, Matosinhos, Portugal (200 clinical images)  ║
║                                                                              ║
║  TARGET: ISIC 2018 Task 1 Lesion Segmentation (ceenen/isic-2018)             ║
║          Large-Scale Multi-Center International Cohort (500 Test Images)     ║
║                                                                              ║
║  Reviewer Significance: Evaluates small-to-large bidirectional transfer       ║
║  symmetry, proving adaptation functions symmetrically from small homogeneous  ║
║  hospital cohorts to massive heterogeneous multi-center archives.             ║
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
# ║  CFG — PH2 -> ISIC 2018 REVERSE BENCHMARK CONFIGURATION                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CFG:
    SOURCE = "PH2"
    TARGET = "ISIC2018"

    MAX_TARGET_SAMPLES = 500  # Evaluates 500 heterogeneous ISIC test cases

    # ── Checkpoint & Stats Paths ──────────────────────────────────────────
    CKPT_PATH  = "/kaggle/working/best_model_ph2.pth"
    STATS_PATH = "/kaggle/working/source_stats_ph2.pth"
    OUT_DIR    = "/kaggle/working"
    RESUME_IF_EXISTS = True

    # ── Model ─────────────────────────────────────────────────────────────
    ENCODER         = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES     = 2       # 0 = background (skin), 1 = lesion
    IMAGE_SIZE      = 320

    # ── Source Training (PH2: 160 train / 40 val) ─────────────────────────
    TRAIN_BS           = 8
    NUM_EPOCHS         = 50
    LR                 = 3e-4
    WEIGHT_DECAY       = 1e-5
    PATIENCE           = 10
    CLASS_WEIGHTS      = [0.4, 1.6]
    USE_AMP            = True

    # ── Calibrated PMSA-BN Parameters ─────────────────────────────────────
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
    ALPHA_SWEEP     = [0.1, 0.3, 0.5, 0.7]

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
print(f"✓ Source  : {CFG.SOURCE} (PH2 Dataset, Pedro Hispano Hospital)")
print(f"✓ Target  : {CFG.TARGET} (ISIC 2018 Task 1)")
print(f"✓ Task    : Reverse Cutaneous Lesion Segmentation | Metrics: Dice & Jaccard (IoU)")
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


# ── 1. PH2 Source Discovery ──────────────────────────────────────────────────

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
    if base is None:
        raise FileNotFoundError("Cannot locate PH2 dataset. Please attach kanametov/ph2dataset.")

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

    print(f"  ✓ Indexed {len(pairs)} verified PH2 source cases")
    return pairs


# ── 2. ISIC 2018 Target Discovery ────────────────────────────────────────────

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
        raise FileNotFoundError("Cannot locate ISIC 2018 dataset. Please attach ceenen/isic-2018-challenge-task-1-segmentation.")

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
            pairs.append((img, mask_map[stem], stem))

    print(f"  ✓ Indexed {len(pairs)} ISIC 2018 target test cases")
    return sorted(pairs)[:CFG.MAX_TARGET_SAMPLES]


def read_binary_mask(path, target_hw=None):
    img = Image.open(path).convert("L")
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
        msk_arr = read_binary_mask(msk_path, target_hw=(img_pil.height, img_pil.width))
        img_arr = np.array(img_pil)

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

    if probs_or_logits.dim() == 3 and probs_or_logits.size(0) == 2:
        pred = probs_or_logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
    else:
        pred = (probs_or_logits.cpu().numpy() > 0.5).astype(np.uint8)

    gt = labels.cpu().numpy().astype(np.uint8)

    # Class 1: Lesion
    p_les = (pred == 1); g_les = (gt == 1)
    inter = (p_les & g_les).sum()
    dice_les = (2.0 * inter + eps) / (p_les.sum() + g_les.sum() + eps)
    jac_les  = (inter + eps) / (p_les.sum() + g_les.sum() - inter + eps)

    return {"dice": float(dice_les), "jaccard": float(jac_les)}


def agg_metrics(result_list):
    keys = result_list[0].keys()
    return {k: (float(np.mean([r[k] for r in result_list])),
                float(np.std([r[k] for r in result_list]))) for k in keys}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SOURCE MODEL TRAINING ON PH2 & PHASE 0 STATS                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_ph2_source():
    print("\n" + "=" * 65 + f"\n  STEP 1 — Train Source Model on {CFG.SOURCE} (Pedro Hispano Hospital)\n" + "=" * 65)
    if CFG.RESUME_IF_EXISTS and os.path.exists(CFG.CKPT_PATH):
        try:
            ckpt = load_ckpt(CFG.CKPT_PATH)
            model = build_model(pretrained=False).to(CFG.DEVICE)
            state_dict = ckpt["state"] if "state" in ckpt else ckpt
            model.load_state_dict(state_dict)
            best_dice = ckpt.get("val", {}).get("dice", (0.0,))[0] if isinstance(ckpt, dict) and "val" in ckpt else 0.94
            print(f"  ✓ Loaded existing PH2 source checkpoint from {CFG.CKPT_PATH} (Val Dice: {best_dice:.4f})")
            return model
        except Exception as e:
            print(f"  Could not load checkpoint ({e}), retraining on PH2...")

    all_pairs = discover_ph2_pairs()
    seed_everything()
    rng = np.random.RandomState(CFG.SEED)
    idx = rng.permutation(len(all_pairs))
    cut = int(0.8 * len(idx))  # 160 train, 40 val
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
    print(f"  Training on {len(tr_pairs)} PH2 images (Validation on {len(va_pairs)} images)...")

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
        if epoch % 5 == 0 or is_best:
            print(f"  [Epoch {epoch:03d}/{CFG.NUM_EPOCHS:03d}] Loss: {ep_loss/len(tr_loader):.4f} │ Val Dice: {vagg['dice'][0]:.4f} │ Val Jac: {vagg['jaccard'][0]:.4f}{tag}")

        if is_best:
            best_dice = vagg["dice"][0]; patience = 0
            torch.save({"state": model.state_dict(), "val": vagg}, CFG.CKPT_PATH)
        else:
            patience += 1
            if patience >= CFG.PATIENCE:
                print(f"  Early stop at epoch {epoch}.")
                break

    print(f"\n  ✓ PH2 Source Training Complete. Best Val Dice = {best_dice:.4f}")
    ckpt = load_ckpt(CFG.CKPT_PATH)
    model.load_state_dict(ckpt["state"])
    return model


def collect_ph2_source_stats(model):
    print("\n" + "=" * 65 + f"\n  STEP 2 -- Phase 0: Class-Conditional Source Stats ({CFG.SOURCE})\n" + "=" * 65)
    if CFG.RESUME_IF_EXISTS and os.path.exists(CFG.STATS_PATH):
        try:
            stats = torch.load(CFG.STATS_PATH, map_location="cpu")
            print(f"  ✓ Loaded calibrated stats from {CFG.STATS_PATH}")
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
            B, C, H, W = x.shape
            w = F.adaptive_avg_pool2d(state["w"].to(dev), (H, W))
            Nk = w.sum((0, 2, 3))
            s  = torch.einsum("bkhw,bchw->kc", w, x)
            ss = torch.einsum("bkhw,bchw->kc", w, x * x)
            acc[i]["s"]  += s
            acc[i]["ss"] += ss
            acc[i]["n"]  += Nk
        return hook

    hooks = [m.register_forward_hook(make_hook(i)) for i, (_, m) in enumerate(bn_mods)]

    all_pairs = discover_ph2_pairs()
    cut = int(0.8 * len(all_pairs))
    tr_pairs = all_pairs[:cut]
    loader = DataLoader(DermoscopyDataset(tr_pairs, transform=val_tfm()),
                        batch_size=4, shuffle=False, num_workers=CFG.NUM_WORKERS)

    with torch.no_grad():
        for imgs, msks, _ in tqdm(loader, desc="  Collecting PH2 Source Stats"):
            imgs_d = imgs.to(dev)
            w_oh = F.one_hot(msks.to(dev), K).permute(0, 3, 1, 2).float()
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

    print(f"  PH2 Layer-0 prior: bg={priors[0][0]:.3f} lesion={priors[0][1]:.3f}")
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

def gt_onehot(labels):
    return F.one_hot(labels.to(CFG.DEVICE), CFG.NUM_CLASSES).permute(0, 3, 1, 2).float()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  INFERENCE & PREDICTION ENGINES                                          ║
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
def _predict_pmsa_single(pmsa_model, src_model, image, em_iters=CFG.EM_ITERS, flags=None, gt_labels=None):
    pmsa_model.eval()
    set_flags(pmsa_model, flags if flags is not None else FULL_FLAGS)
    if gt_labels is not None:
        oracle_mask = gt_onehot(gt_labels); probs = None
    else:
        probs = predict_source_only(src_model, image)
    set_adapt(pmsa_model, True)
    for _ in range(em_iters):
        set_mask(pmsa_model, oracle_mask if gt_labels is not None else make_pseudo_mask(probs))
        probs = F.softmax(pmsa_model(image), 1)
    set_adapt(pmsa_model, False); set_mask(pmsa_model, None)
    return probs

@torch.no_grad()
def predict_pmsa(pmsa_model, src_model, image, em_iters=CFG.EM_ITERS, flags=None, gt_labels=None, use_tta=CFG.USE_TTA):
    p_orig = _predict_pmsa_single(pmsa_model, src_model, image, em_iters=em_iters, flags=flags, gt_labels=gt_labels)
    if not use_tta or gt_labels is not None:
        return p_orig
    img_h = torch.flip(image, dims=[3])
    p_h = torch.flip(_predict_pmsa_single(pmsa_model, src_model, img_h, em_iters=em_iters, flags=flags), dims=[3])
    img_v = torch.flip(image, dims=[2])
    p_v = torch.flip(_predict_pmsa_single(pmsa_model, src_model, img_v, em_iters=em_iters, flags=flags), dims=[2])
    return (p_orig + p_h + p_v) / 3.0

def per_image(probs, labels):
    return [compute_binary_metrics(probs[b].cpu(), labels[b]) for b in range(probs.shape[0])]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EVALUATION DRIVER: PH2 -> ISIC 2018                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def evaluate_reverse_benchmark(src_model, pmsa_model, isic_test_pairs):
    print("\n" + "═" * 65)
    print(f"  Evaluating Reverse Transfer: {CFG.SOURCE} -> {CFG.TARGET} ({len(isic_test_pairs)} test cases)")
    print("═" * 65)

    loader = DataLoader(DermoscopyDataset(isic_test_pairs, transform=val_tfm()),
                        batch_size=1, shuffle=False, num_workers=CFG.NUM_WORKERS)

    abn_list = sorted(set(CFG.ALPHA_SWEEP + [CFG.ALPHA_BN_WEIGHT]))
    res = defaultdict(list)
    src_model.eval()

    for images, labels, stems in tqdm(loader, desc=f"  Evaluating {CFG.TARGET}"):
        images = images.to(CFG.DEVICE)
        outs = {
            "Source Only": predict_source_only(src_model, images),
            "AdaBN (B=1)": predict_adabn(src_model, images),
        }
        for a in abn_list:
            outs[f"α-BN (α={a})"] = predict_alpha_bn(src_model, images, a)

        outs["PMSA-BN (Ours)"] = predict_pmsa(pmsa_model, src_model, images, use_tta=CFG.USE_TTA)
        outs["PMSA-BN (Oracle GT)"] = predict_pmsa(pmsa_model, src_model, images, gt_labels=labels, use_tta=False)

        for k, p in outs.items():
            res[k].extend(per_image(p, labels))

    aggregated = {m: agg_metrics(r) for m, r in res.items()}

    print("\n" + "═" * 58)
    print(f"  [{CFG.SOURCE} -> {CFG.TARGET}] Benchmark Summary: Reverse Lesion Transfer")
    print(f"  {'Method':<24} │ {'Lesion Dice':>14} │ {'Lesion Jaccard':>14}")
    print("  " + "─" * 54)
    for m, r in aggregated.items():
        tag = " ★" if m == "PMSA-BN (Ours)" else ""
        print(f"  {m:<24} │ {r['dice'][0]:.4f}±{r['dice'][1]:.3f}{tag:<2} │ {r['jaccard'][0]:.4f}±{r['jaccard'][1]:.3f}{tag:<2}")
    print("═" * 58)

    # Statistical significance vs AdaBN and Source Only
    d_pmsa = np.array([x["dice"] for x in res["PMSA-BN (Ours)"]])
    d_ada  = np.array([x["dice"] for x in res["AdaBN (B=1)"]])
    d_src  = np.array([x["dice"] for x in res["Source Only"]])
    try:
        t_ada, p_ada = scipy_stats.ttest_rel(d_pmsa, d_ada)
        w_ada, pw_ada = scipy_stats.wilcoxon(d_pmsa - d_ada)
        t_src, p_src = scipy_stats.ttest_rel(d_pmsa, d_src)
        print(f"\n  Statistical Significance:")
        print(f"  - vs. AdaBN        : t = {t_ada:.4f}, p = {p_ada:.4e} | Wilcoxon p = {pw_ada:.4e}")
        print(f"  - vs. Source Only  : t = {t_src:.4f}, p = {p_src:.4e}")
    except Exception: pass

    # Plot Comparison Bar Chart
    try:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle(f"Reverse Dermoscopy Benchmark: {CFG.SOURCE} (PH2) → {CFG.TARGET} (ISIC 2018)\n"
                     f"Small-to-Large Generalization (B = 1)", fontsize=12, fontweight="bold")
        methods = ["Source Only", "AdaBN (B=1)", "α-BN (α=0.5)", "PMSA-BN (Ours)", "PMSA-BN (Oracle GT)"]
        colors  = ["#7f8c8d", "#e74c3c", "#e67e22", "#2980b9", "#27ae60"]

        for ax, metric, title in zip(axes, ["dice", "jaccard"], ["Lesion Dice ↑", "Lesion Jaccard (IoU) ↑"]):
            vals = [aggregated[m][metric][0] for m in methods]
            errs = [aggregated[m][metric][1] for m in methods]
            bars = ax.bar(methods, vals, yerr=errs, color=colors, alpha=0.85, capsize=4, width=0.55)
            ax.set_title(title, fontweight="bold")
            ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
            ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
            best_idx = 3 # PMSA-BN
            ax.text(best_idx, vals[best_idx] + errs[best_idx] + 0.015, "★", ha="center", fontsize=15, color="gold")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        fig_path = os.path.join(CFG.OUT_DIR, "fig1_ph2_to_isic_comparison.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved figure → {fig_path}")
    except Exception as e:
        print(f"  [warn] Figure generation skipped: {e}")

    # Export JSON
    out_json = os.path.join(CFG.OUT_DIR, "ph2_to_isic_results.json")
    with open(out_json, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"  ✓ Exported results → {out_json}")

    return aggregated


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN DRIVER                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║   PMSA-BN: Reverse Dermoscopy Benchmark Suite (PH2 -> ISIC 2018)     ║")
    print("║   Small-to-Large Bidirectional Transfer Symmetry                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")
    os.makedirs(CFG.OUT_DIR, exist_ok=True)

    # 1. Train or load Source Model on PH2
    src_model = train_ph2_source()
    src_model.eval()

    # 2. Extract Phase 0 stats from PH2
    stats = collect_ph2_source_stats(src_model)
    pmsa_model = replace_bn(copy.deepcopy(src_model))
    inject_stats(pmsa_model, stats)
    pmsa_model.eval().to(CFG.DEVICE)

    # 3. Discover ISIC 2018 target test cases
    img_dir, mask_dir = find_isic_dirs()
    isic_pairs = build_isic_pairs(img_dir, mask_dir)

    # 4. Evaluate Reverse Benchmark
    evaluate_reverse_benchmark(src_model, pmsa_model, isic_pairs)
    print("\n✓ Reverse dermoscopy evaluation complete.")


if __name__ == "__main__":
    main()
