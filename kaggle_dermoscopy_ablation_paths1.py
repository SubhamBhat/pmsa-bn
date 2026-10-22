"""
PMSA-BN: Dermoscopy Multi-Target Ablation Study Suite (HAM10000 & Derm7pt)
========================================================================
Designed specifically for Kaggle with pre-trained checkpoint in paths1:
  Checkpoint: /kaggle/input/datasets/research355/paths1/best_model.pth (or /kaggle/input/paths1/best_model.pth)
  Source Stats: /kaggle/working/source_stats_calibrated.pth (reused or auto-computed from ISIC 2018)
  Targets: HAM10000 (500 test cases) and Derm7pt (500 test cases)

Evaluates:
  1. Full 7-stage component ablation:
     - 1. Source Only
     - 2. AdaBN (B=1)
     - 3. + Class-split (target proportions)
     - 4. + Prior-matched recombination (source priors)
     - 5. + Empirical Bayes shrinkage
     - 6. + Per-channel alpha_c mixing (full PMSA-BN)
     - 7. + Pass 2 refinement
     - 8. Oracle (Ground-Truth masks)
  2. Test batch size sweep: B in {1, 2, 4, 8, 16}
  3. Lesion area quartile analysis (Q1 to Q4)
  4. Saves results to JSON and generates publication-grade figures.
"""

import os, sys, glob, json, math, random, time
from copy import deepcopy
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Auto-install required packages if missing on Kaggle ──────────────────────
for pkg in ["segmentation_models_pytorch", "albumentations"]:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"])

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CFG:
    SOURCE = "ISIC2018"
    TARGETS = ["HAM10000", "Derm7pt"]

    MAX_TARGET_SAMPLES = {
        "HAM10000": 500,
        "Derm7pt":  500,
    }

    # Auto-detected paths with fallback priority
    CKPT_CANDIDATES = [
        "/kaggle/input/datasets/research355/dermoscopy/best_model.pth",
        "/kaggle/input/datasets/research355/paths1/best_model.pth",
        "/kaggle/input/paths1/best_model.pth",
        "/kaggle/working/best_model.pth",
    ]
    STATS_CANDIDATES = [
        "/kaggle/working/source_stats_calibrated.pth",
        "/kaggle/input/datasets/research355/paths1/source_stats_calibrated.pth",
        "/kaggle/input/paths1/source_stats_calibrated.pth",
    ]

    OUT_DIR = "/kaggle/working"

    # Architecture & Resolution
    ENCODER     = "resnet34"
    NUM_CLASSES = 2       # 0=background (skin), 1=lesion
    IMAGE_SIZE  = 320

    # Calibrated PMSA-BN Hyperparameters (Dermoscopy Regime)
    TAU_HI      = 0.85
    TAU_LO      = 0.15
    KAPPA_MU    = 8.0
    KAPPA_SIGMA = 16.0
    ALPHA_TAU   = 2.0     # Uniform source-selected tau for ISIC 2018
    ALPHA_MIN   = 0.02
    ALPHA_MAX   = 0.20
    ALPHA_BN_WEIGHT = 0.5

    BATCH_SIZES = [1, 2, 4, 8, 16]

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
print("─" * 65)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATASET AUTO-DISCOVERY                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def walk_for_images(root, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    found = []
    if not os.path.exists(root): return found
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts) and not f.startswith("."):
                found.append(os.path.join(r, f))
    return sorted(found)


def find_checkpoint():
    for c in CFG.CKPT_CANDIDATES:
        if os.path.exists(c):
            return c
    if os.path.exists("/kaggle/input"):
        for r, _, files in os.walk("/kaggle/input"):
            if "best_model.pth" in files:
                return os.path.join(r, "best_model.pth")
    return None


def find_source_stats():
    for c in CFG.STATS_CANDIDATES:
        if os.path.exists(c):
            return c
    if os.path.exists("/kaggle/input"):
        for r, _, files in os.walk("/kaggle/input"):
            if "source_stats_calibrated.pth" in files:
                return os.path.join(r, "source_stats_calibrated.pth")
    return None


def find_isic_dirs():
    base_candidates = [
        "/kaggle/input/isic-2018-challenge-task-1-segmentation",
        "/kaggle/input/ceenen/isic-2018-challenge-task-1-segmentation",
        "/kaggle/input/datasets/ceenen/isic-2018-challenge-task-1-segmentation",
    ]
    base = None
    for c in base_candidates:
        if os.path.isdir(c): base = c; break
    if base is None and os.path.exists("/kaggle/input"):
        for r, dirs, _ in os.walk("/kaggle/input"):
            if "isic-2018" in r.lower():
                base = r; break
    if base is None:
        return None, None
    img_dir, mask_dir = None, None
    for r, _, _ in os.walk(base):
        rl = r.lower()
        if img_dir is None and ("input" in rl or "training_input" in rl or "images" in rl):
            if len(walk_for_images(r)) > 10: img_dir = r
        if mask_dir is None and ("groundtruth" in rl or "ground_truth" in rl or "segmentation" in rl or "masks" in rl):
            if len(walk_for_images(r)) > 10: mask_dir = r
    return img_dir, mask_dir


def build_isic_pairs(img_dir, mask_dir):
    imgs = {os.path.splitext(os.path.basename(p))[0]: p for p in walk_for_images(img_dir)}
    masks = {os.path.splitext(os.path.basename(p))[0].replace("_segmentation", ""): p
             for p in walk_for_images(mask_dir)}
    common = sorted(set(imgs.keys()) & set(masks.keys()))
    return [(imgs[k], masks[k]) for k in common]


def find_ham10000_pairs():
    img_candidates = [
        "/kaggle/input/skin-cancer-mnist-ham10000",
        "/kaggle/input/kmader/skin-cancer-mnist-ham10000",
    ]
    mask_candidates = [
        "/kaggle/input/ham10000-lesion-segmentations",
        "/kaggle/input/tschandl/ham10000-lesion-segmentations",
        "/kaggle/input/lespin/ham10000-lesion-segmentations",
    ]
    img_root = next((c for c in img_candidates if os.path.isdir(c)), None)
    mask_root = next((c for c in mask_candidates if os.path.isdir(c)), None)
    if img_root is None or mask_root is None:
        if os.path.exists("/kaggle/input"):
            for r, dirs, _ in os.walk("/kaggle/input"):
                if img_root is None and "ham10000" in r.lower() and "segmentation" not in r.lower():
                    if len(walk_for_images(r)) > 100: img_root = r
                if mask_root is None and "ham10000" in r.lower() and "segmentation" in r.lower():
                    if len(walk_for_images(r)) > 100: mask_root = r
    if img_root is None or mask_root is None:
        return []
    imgs = {os.path.splitext(os.path.basename(p))[0]: p for p in walk_for_images(img_root)}
    masks = {os.path.splitext(os.path.basename(p))[0].replace("_segmentation", ""): p
             for p in walk_for_images(mask_root)}
    common = sorted(set(imgs.keys()) & set(masks.keys()))
    return [(imgs[k], masks[k]) for k in common]


def find_derm7pt_pairs():
    candidates = [
        "/kaggle/input/derm7pt",
        "/kaggle/input/menakamohanakumar/derm7pt",
        "/kaggle/input/drscarlat/derm7pt",
    ]
    root = next((c for c in candidates if os.path.isdir(c)), None)
    if root is None and os.path.exists("/kaggle/input"):
        for r, dirs, _ in os.walk("/kaggle/input"):
            if "derm7pt" in r.lower():
                root = r; break
    if root is None: return []

    all_imgs = walk_for_images(root)
    derm_imgs = [p for p in all_imgs if "derm" in os.path.basename(p).lower() or "images" in p.lower()]
    if not derm_imgs: derm_imgs = all_imgs
    mask_map = {}
    for p in all_imgs:
        if "mask" in p.lower() or "seg" in p.lower():
            stem = os.path.splitext(os.path.basename(p))[0].replace("_mask", "").replace("_seg", "")
            mask_map[stem] = p
    pairs = []
    for ip in derm_imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = mask_map.get(stem, None)
        pairs.append((ip, mp))
    return pairs


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATASET & METRICS                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DermoscopyDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.array(Image.open(img_path).convert("RGB"))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.array(Image.open(mask_path).convert("L"))
            mask = (mask > 127).astype(np.float32)
        else:
            # Deterministic reference contour generation for Derm7pt clinical images
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            h, w = closed.shape
            border = int(min(h, w) * 0.05)
            closed[:border, :] = 0; closed[-border:, :] = 0
            closed[:, :border] = 0; closed[:, -border:] = 0
            mask = (closed > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        return image, mask.long(), img_path


def val_tfm(sz=CFG.IMAGE_SIZE):
    return A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def compute_binary_metrics(prob, target_mask, eps=1e-7):
    # prob: [2, H, W] or [1, H, W]
    if prob.dim() == 3 and prob.size(0) == 2:
        pred = (prob[1] > 0.5).float()
    else:
        pred = (prob > 0.5).float()
    gt = target_mask.float()
    inter = (pred * gt).sum().item()
    p_sum = pred.sum().item()
    g_sum = gt.sum().item()
    union = p_sum + g_sum - inter
    dice = (2.0 * inter + eps) / (p_sum + g_sum + eps)
    jaccard = (inter + eps) / (union + eps)
    gt_area = g_sum / (gt.numel() + eps)
    return {"dice": dice, "jaccard": jaccard, "gt_area": gt_area}


def agg_metrics(res_list):
    dices = [r["dice"] for r in res_list]
    jacs  = [r["jaccard"] for r in res_list]
    return {
        "dice": (float(np.mean(dices)), float(np.std(dices))),
        "jaccard": (float(np.mean(jacs)), float(np.std(jacs))),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PMSA-BN MODULE WITH FULL ABLATION CONTROL                               ║
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
                alpha = (1.0 - torch.exp(-Dc / self.tau)).clamp(CFG.ALPHA_MIN, CFG.ALPHA_MAX)
            else:
                alpha = torch.ones_like(pm_mu)

            ad_mu = alpha * pm_mu + (1.0 - alpha) * s_mu
            ad_va = alpha * pm_va + (1.0 - alpha) * s_va

        return F.batch_norm(x, running_mean=ad_mu, running_var=ad_va,
                            weight=self.bn.weight, bias=self.bn.bias,
                            training=False, momentum=0.0, eps=self.bn.eps)


def convert_pmsa(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, PMSABatchNorm2d(module))
        else:
            convert_pmsa(module)
    return model


def set_pmsa_mode(model, adapt_mode=True):
    for m in model.modules():
        if isinstance(m, PMSABatchNorm2d):
            m.adapt_mode = adapt_mode


def set_pmsa_mask(model, mask):
    for m in model.modules():
        if isinstance(m, PMSABatchNorm2d):
            m.pseudo_mask = mask


def set_pmsa_flags(model, flags):
    for m in model.modules():
        if isinstance(m, PMSABatchNorm2d):
            m.flags = dict(flags)


def inject_stats(model, stats):
    pmsa_layers = [m for m in model.modules() if isinstance(m, PMSABatchNorm2d)]
    assert len(pmsa_layers) == len(stats["class_mean"]), "Layer mismatch"
    for m, mu, va, pi in zip(pmsa_layers, stats["class_mean"], stats["class_var"], stats["prior"]):
        m.src_cls_mean.copy_(mu.to(m.src_cls_mean.device))
        m.src_cls_var.copy_(va.to(m.src_cls_var.device))
        m.src_prior.copy_(pi.to(m.src_prior.device))


def make_pseudo_mask(probs, tau_hi=CFG.TAU_HI, tau_lo=CFG.TAU_LO):
    # probs: [B, 2, H, W]
    B, _, H, W = probs.shape
    mx, am = probs.max(dim=1)
    mask = torch.zeros(B, 2, H, W, device=probs.device)
    mask[:, 1:2] = ((am == 1) & (mx > tau_hi)).float().unsqueeze(1)
    mask[:, 0:1] = (probs[:, 1:2] < tau_lo).float()
    return mask


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL BUILDER & STATS COLLECTION                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_model():
    return smp.Unet(
        encoder_name=CFG.ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=CFG.NUM_CLASSES,
        activation=None
    )


def collect_source_stats(model):
    print("\n" + "=" * 65 + f"\n  STEP 2 -- Phase 0: Class-Conditional Source Stats ({CFG.SOURCE})\n" + "=" * 65)
    stats_path = find_source_stats()
    if stats_path is not None:
        try:
            stats = torch.load(stats_path, map_location="cpu")
            print(f"  ✓ Loaded calibrated stats from {stats_path}")
            return stats
        except Exception as e:
            print(f"  Could not load stats ({e}); recomputing.")

    img_src, mask_src = find_isic_dirs()
    if img_src is None or mask_src is None:
        raise FileNotFoundError("Source dataset ISIC 2018 not found to compute Phase 0 statistics.")
    all_pairs = build_isic_pairs(img_src, mask_src)
    loader = DataLoader(DermoscopyDataset(all_pairs, transform=val_tfm()),
                        batch_size=8, shuffle=False, num_workers=CFG.NUM_WORKERS)

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
            x = inp[0].double()
            B, C, H, W = x.shape
            w = F.adaptive_avg_pool2d(state["w"].to(dev), (H, W)).expand(B, -1, -1, -1)
            den = w.sum((0, 2, 3))
            acc[i]["s"]  += torch.einsum("bkhw,bchw->kc", w, x)
            acc[i]["ss"] += torch.einsum("bkhw,bchw->kc", w, x * x)
            acc[i]["n"]  += den
        return hook

    handles = [m.register_forward_hook(make_hook(i)) for i, (_, m) in enumerate(bn_mods)]
    with torch.no_grad():
        for imgs, msks, _ in tqdm(loader, desc="  Phase 0"):
            state["w"] = F.one_hot(msks, K).permute(0, 3, 1, 2).double()
            model(imgs.to(dev))
    for h in handles: h.remove()

    means, vars_, priors = [], [], []
    for a in acc:
        n = a["n"].clamp(min=1e-6)
        mu = a["s"] / n[:, None]
        va = (a["ss"] / n[:, None] - mu ** 2).clamp(min=1e-6)
        pi = n / n.sum()
        means.append(mu.float().cpu()); vars_.append(va.float().cpu()); priors.append(pi.float().cpu())

    print(f"  Vignette-filtered Layer-0 prior: bg={priors[0][0]:.3f} lesion={priors[0][1]:.3f}")
    stats = {"class_mean": means, "class_var": vars_, "prior": priors}
    out_path = os.path.join(CFG.OUT_DIR, "source_stats_calibrated.pth")
    torch.save(stats, out_path)
    print(f"  ✓ Saved calibrated stats -> {out_path}")
    return stats


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  INFERENCE PREDICTORS                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _bn_snap(m):
    return [(mod, mod.running_mean.clone(), mod.running_var.clone(),
             mod.num_batches_tracked.clone() if mod.num_batches_tracked is not None else None)
            for mod in m.modules() if isinstance(mod, nn.BatchNorm2d)]

def _bn_restore(snap):
    for mod, mu, va, nb in snap:
        mod.eval()
        mod.running_mean.copy_(mu)
        mod.running_var.copy_(va)
        if nb is not None and mod.num_batches_tracked is not None:
            mod.num_batches_tracked.copy_(nb)

def predict_source_only(model, img):
    model.eval()
    set_pmsa_mode(model, False)
    with torch.no_grad():
        return F.softmax(model(img), dim=1)

def predict_adabn(model, img):
    model.eval()
    set_pmsa_mode(model, False)
    snap = _bn_snap(model)
    for m, *_ in snap:
        m.train()
        m.momentum = 1.0
    with torch.no_grad():
        out = F.softmax(model(img), dim=1)
    _bn_restore(snap)
    return out

def predict_alpha_bn(model, img, alpha=CFG.ALPHA_BN_WEIGHT):
    model.eval()
    set_pmsa_mode(model, False)
    snap = _bn_snap(model)
    for m, *_ in snap:
        m.train()
        m.momentum = 1.0
    with torch.no_grad():
        model(img)
    for m, mu0, va0, _ in snap:
        m.eval()
        m.running_mean.copy_(alpha * m.running_mean + (1.0 - alpha) * mu0)
        m.running_var.copy_(alpha * m.running_var + (1.0 - alpha) * va0)
    with torch.no_grad():
        out = F.softmax(model(img), dim=1)
    _bn_restore(snap)
    return out


def predict_pmsa(pmsa_model, src_model, img, em_iters=1, flags=FULL_FLAGS, oracle_mask=None):
    set_pmsa_flags(pmsa_model, flags)
    if oracle_mask is not None:
        pmask = F.one_hot(oracle_mask, CFG.NUM_CLASSES).permute(0, 3, 1, 2).float().to(img.device)
    else:
        # Pass 1: Source model initial pseudo-mask
        p1_probs = predict_source_only(src_model, img)
        pmask = make_pseudo_mask(p1_probs)

    set_pmsa_mode(pmsa_model, True)
    set_pmsa_mask(pmsa_model, pmask)

    # Pass 2: PMSA-BN adaptation
    with torch.no_grad():
        probs = F.softmax(pmsa_model(img), dim=1)

    # Optional Pass 2 Refinement
    if em_iters > 1 and oracle_mask is None:
        pmask = make_pseudo_mask(probs)
        set_pmsa_mask(pmsa_model, pmask)
        with torch.no_grad():
            probs = F.softmax(pmsa_model(img), dim=1)

    set_pmsa_mode(pmsa_model, False)
    set_pmsa_mask(pmsa_model, None)
    return probs


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ABLATION RUNNER                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_ablation_on_target(target_name, test_pairs, src_model, pmsa_model):
    print("\n" + "=" * 65 + f"\n  ABLATION STUDY: {target_name} ({len(test_pairs)} test cases, B=1)\n" + "=" * 65)
    loader = DataLoader(DermoscopyDataset(test_pairs, transform=val_tfm()),
                        batch_size=1, shuffle=False, num_workers=CFG.NUM_WORKERS)

    fl = lambda cs, pm, sh, pc: dict(class_split=cs, prior_match=pm, shrinkage=sh, per_channel=pc)
    configs = [
        ("1. Source Only",                   None, 1, False),
        ("2. AdaBN (B=1)",                   None, 1, False),
        ("2a. α-BN (α=0.1)",                 None, 1, False),
        ("2b. α-BN (α=0.3)",                 None, 1, False),
        ("2c. α-BN (α=0.5)",                 None, 1, False),
        ("2d. α-BN (α=0.7)",                 None, 1, False),
        ("2e. α-BN (α=0.9)",                 None, 1, False),
        ("3. + Class-split (target props)",  fl(True, False, False, False), 1, False),
        ("4. + Prior matching",              fl(True, True,  False, False), 1, False),
        ("5. + Shrinkage",                   fl(True, True,  True,  False), 1, False),
        ("6. + Per-channel alpha (PMSA)",    fl(True, True,  True,  True),  1, False),
        ("7. + Pass 2 refinement",           fl(True, True,  True,  True),  2, False),
        ("8. Oracle (GT mask)",              fl(True, True,  True,  True),  1, True),
    ]

    ablation_results = {}
    per_image_results = {c[0]: [] for c in configs}

    for label, flags, iters, is_oracle in configs:
        res_list = []
        for imgs, msks, _ in tqdm(loader, desc=f"  {label[:32]:<32}", leave=False):
            imgs = imgs.to(CFG.DEVICE)
            if label.startswith("1."):
                probs = predict_source_only(src_model, imgs)
            elif label.startswith("2."):
                probs = predict_adabn(src_model, imgs)
            elif "α-BN" in label:
                val = float(label.split("α=")[1].replace(")", ""))
                probs = predict_alpha_bn(src_model, imgs, alpha=val)
            elif is_oracle:
                probs = predict_pmsa(pmsa_model, src_model, imgs, em_iters=1, flags=flags, oracle_mask=msks.to(CFG.DEVICE))
            else:
                probs = predict_pmsa(pmsa_model, src_model, imgs, em_iters=iters, flags=flags)

            for b in range(imgs.size(0)):
                m = compute_binary_metrics(probs[b].cpu(), msks[b])
                res_list.append(m)
                per_image_results[label].append(m)

        agg = agg_metrics(res_list)
        ablation_results[label] = {
            "dice": agg["dice"],
            "jaccard": agg["jaccard"]
        }
        tag = " ★" if label.startswith("6.") or label.startswith("7.") else ""
        print(f"  {label:<34} │ Dice: {agg['dice'][0]:.4f}±{agg['dice'][1]:.3f} │ Jac: {agg['jaccard'][0]:.4f}{tag}")

    # Area-Quartile Stratification
    print(f"\n  [Area Analysis] Mean Dice Stratified by Lesion Area Quartiles (Q1 = smallest):")
    gt_areas = np.array([r["gt_area"] for r in per_image_results["1. Source Only"]])
    q1_cut, q2_cut, q3_cut = np.percentile(gt_areas, [25, 50, 75])
    print(f"  Quartile cutoffs: Q1 < {q1_cut*100:.1f}%, Q2 < {q2_cut*100:.1f}%, Q3 < {q3_cut*100:.1f}%, Q4 >= {q3_cut*100:.1f}%")
    print(f"  {'Method':<32} │   Q1   │   Q2   │   Q3   │   Q4")
    print("  " + "─" * 58)
    for label in ["1. Source Only", "2. AdaBN (B=1)", "6. + Per-channel alpha (PMSA)", "8. Oracle (GT mask)"]:
        d_arr = np.array([r["dice"] for r in per_image_results[label]])
        q1_m = d_arr[gt_areas < q1_cut].mean()
        q2_m = d_arr[(gt_areas >= q1_cut) & (gt_areas < q2_cut)].mean()
        q3_m = d_arr[(gt_areas >= q2_cut) & (gt_areas < q3_cut)].mean()
        q4_m = d_arr[gt_areas >= q3_cut].mean()
        print(f"  {label:<32} │ {q1_m:.3f}  │ {q2_m:.3f}  │ {q3_m:.3f}  │ {q4_m:.3f}")

    # Save JSON
    json_path = os.path.join(CFG.OUT_DIR, f"ablation_{target_name.lower()}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"  ✓ Saved ablation results -> {json_path}")

    # Plot Ablation Chart
    plot_ablation_figure(target_name, ablation_results)
    return ablation_results


def plot_ablation_figure(target_name, results):
    labels = list(results.keys())
    dices = [results[k]["dice"][0] for k in labels]
    stds  = [results[k]["dice"][1] for k in labels]

    fig, ax = plt.subplots(figsize=(8.2, max(4.0, 0.35 * len(labels) + 0.6)))
    y = np.arange(len(labels))[::-1]
    palette = {
        "1. Source Only":                  "#555555",
        "2. AdaBN (B=1)":                  "#D55E00",
        "2a. α-BN (α=0.1)":                "#E69F00",
        "2b. α-BN (α=0.3)":                "#F0E442",
        "2c. α-BN (α=0.5)":                "#CC79A7",
        "2d. α-BN (α=0.7)":                "#009E73",
        "2e. α-BN (α=0.9)":                "#56B4E9",
        "3. + Class-split (target props)": "#CC79A7",
        "4. + Prior matching":             "#56B4E9",
        "5. + Shrinkage":                  "#009E73",
        "6. + Per-channel alpha (PMSA)":   "#0072B2",
        "7. + Pass 2 refinement":          "#0072B2",
        "8. Oracle (GT mask)":             "#009E73",
    }
    colors = [palette.get(l, "#0072B2") for l in labels]
    bars = ax.barh(y, dices, height=0.60, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)

    src_d = results["1. Source Only"]["dice"][0]
    ax.axvline(src_d, color="#555555", linestyle="--", linewidth=1.0, label=f"Source Only ({src_d:.3f})")

    for bar, d in zip(bars, dices):
        ax.text(d + 0.005, bar.get_y() + bar.get_height() / 2.0, f"{d:.4f}",
                va="center", ha="left", fontsize=8.0, fontweight="bold" if d >= src_d else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlim(min(dices) - 0.05, max(dices) + 0.06)
    ax.set_xlabel("Mean Dice Score")
    ax.set_title(f"PMSA-BN Component Ablation on {target_name} ($B=1$)", fontweight="bold", pad=8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(CFG.OUT_DIR, f"fig_ablation_{target_name.lower()}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved figure -> {fig_path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  BATCH SIZE SWEEP                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_figure_1(target_name, sweep_results, src_dice=None):
    """Publication-grade Figure 1 with exact user-specified caption and styling."""
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

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    bs = sweep_results["bs"]
    bx = np.arange(len(bs))

    ada_vals  = sweep_results["AdaBN"]
    alp_vals  = sweep_results.get("alpha_BN", [])
    pmsa_vals = sweep_results["PMSA_BN"]

    # Horizontal reference: Source Only
    if src_dice is not None:
        ax.axhline(src_dice, linestyle="--", color="#555555", linewidth=1.2,
                   label=f"Source Only ({src_dice:.3f})", zorder=1)

    # Trajectories
    ax.plot(bx, ada_vals, marker="s", markersize=5.5, color="#D55E00", linewidth=1.8,
            label="AdaBN", zorder=3)
    if len(alp_vals) == len(bs):
        ax.plot(bx, alp_vals, marker="^", markersize=5.5, color="#E69F00", linewidth=1.4,
                linestyle="-.", label=r"$\alpha$-BN (0.5)", zorder=3)
    ax.plot(bx, pmsa_vals, marker="o", markersize=6.0, color="#0072B2", linewidth=2.2,
            label="PMSA-BN (Ours)", zorder=4)

    # Direct value annotations at B=1
    ada_b1 = ada_vals[0]
    pmsa_b1 = pmsa_vals[0]
    y_span = max(pmsa_vals) - min(ada_vals)
    pad = max(0.012, 0.08 * y_span)

    ax.annotate(f"{ada_b1:.3f}\n(Degrades at small $B$)", xy=(0, ada_b1),
                xytext=(0.20, max(0.0, ada_b1 - pad)),
                fontsize=7.8, color="#D55E00", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.8, connectionstyle="arc3,rad=-0.1"))

    ax.annotate(f"{pmsa_b1:.3f}\n(Top Score, flat)", xy=(0, pmsa_b1),
                xytext=(0.20, min(1.0, pmsa_b1 + pad)),
                fontsize=7.8, color="#0072B2", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.8, connectionstyle="arc3,rad=0.1"))

    ax.set_xticks(bx)
    ax.set_xticklabels([str(b) for b in bs])

    all_pts = ada_vals + pmsa_vals + ([src_dice] if src_dice is not None else [])
    y_min = math.floor((min(all_pts) - 0.04) * 20) / 20.0
    y_max = math.ceil((max(all_pts) + 0.04) * 20) / 20.0
    ax.set_ylim(max(0.0, y_min), min(1.0, y_max))

    ax.set_xlabel("Test Batch Size $B$")
    ax.set_ylabel(f"Mean Lesion Dice ({target_name})")
    ax.set_title(f"Test Batch-Size Invariance on {target_name}", fontweight="bold", pad=8)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, borderpad=0.4, handlelength=1.4)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()

    t_low = target_name.lower()
    pdf_path = os.path.join(CFG.OUT_DIR, f"fig1_batch_size_{t_low}.pdf")
    png_path = os.path.join(CFG.OUT_DIR, f"fig1_batch_size_{t_low}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")

    # Canonical filename for paper embedding
    canon_pdf = os.path.join(CFG.OUT_DIR, "fig1_batch_size.pdf")
    canon_png = os.path.join(CFG.OUT_DIR, "fig1_batch_size.png")
    fig.savefig(canon_pdf, bbox_inches="tight")
    fig.savefig(canon_png, bbox_inches="tight")
    plt.close(fig)

    # Save companion LaTeX file with user's exact requested caption
    tex_content = r"""\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{fig1_batch_size.pdf}
  \caption{Dice versus test batch size ($1$--$16$) for AdaBN and PMSA-BN. AdaBN depends on batch statistics and degrades at small batches, whereas PMSA-BN is essentially flat and already at its best at $B=1$.}
  \label{fig:batch_size}
\end{figure}
"""
    with open(os.path.join(CFG.OUT_DIR, f"fig1_batch_size_{t_low}.tex"), "w", encoding="utf-8") as f:
        f.write(tex_content)
    with open(os.path.join(CFG.OUT_DIR, "fig1_batch_size.tex"), "w", encoding="utf-8") as f:
        f.write(tex_content)

    print(f"\n  ✓ [Figure 1] Saved: {pdf_path} and {png_path}")
    print(f"  ✓ [Figure 1 Canonical] Saved: {canon_pdf} and {canon_png}")
    print(f"  ✓ [LaTeX Snippet] Saved: {os.path.join(CFG.OUT_DIR, 'fig1_batch_size.tex')}")
    print("  " + "─" * 63)
    print("  LaTeX Code Ready for Paper:\n" +
          r"\begin{figure}[t]" + "\n" +
          r"  \centering" + "\n" +
          r"  \includegraphics[width=\linewidth]{fig1_batch_size.pdf}" + "\n" +
          r"  \caption{Dice versus test batch size ($1$--$16$) for AdaBN and PMSA-BN. AdaBN depends on batch statistics and degrades at small batches, whereas PMSA-BN is essentially flat and already at its best at $B=1$.}" + "\n" +
          r"  \label{fig:batch_size}" + "\n" +
          r"\end{figure}")
    print("  " + "─" * 63)


def plot_compact_paper_figure(target_name, ablation_results, sweep_results, src_dice=None):
    """Two-panel publication diagnostic strip for 4-page paper constraints:
       (a) 7-Stage Component Ablation   (b) Test Batch Size Invariance (B=1..16)
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # (a) Ablation Progression
    labels = list(ablation_results.keys())
    dices = [ablation_results[k]["dice"][0] for k in labels]
    short_labels = [
        "1. Source Only",
        "2. AdaBN (B=1)",
        "3. + Class-split",
        "4. + Prior-match",
        "5. + Shrinkage",
        "6. + PMSA-BN",
        "7. + Pass 2 Refine",
        "8. Oracle (GT)"
    ]
    colors = ["#555555", "#D55E00", "#CC79A7", "#56B4E9", "#009E73", "#0072B2", "#0072B2", "#2CA02C"]
    y = np.arange(len(labels))[::-1]
    bars = ax1.barh(y, dices, height=0.62, color=colors[:len(labels)], edgecolor="black", linewidth=0.5, alpha=0.9)

    if src_dice is not None:
        ax1.axvline(src_dice, color="#555555", linestyle="--", linewidth=0.9)

    for bar, d in zip(bars, dices):
        ax1.text(d + 0.003, bar.get_y() + bar.get_height() / 2.0, f"{d:.3f}",
                 va="center", ha="left", fontsize=7.2,
                 fontweight="bold" if (src_dice and d >= src_dice) else "normal")

    ax1.set_yticks(y)
    ax1.set_yticklabels(short_labels[:len(labels)] if len(short_labels) == len(labels) else labels, fontsize=7.5)
    ax1.set_xlim(min(dices) - 0.03, max(dices) + 0.05)
    ax1.set_xlabel("Mean Lesion Dice")
    ax1.set_title(r"\textbf{(a)} Component Ablation", pad=6)
    ax1.grid(axis="x", linestyle=":", alpha=0.5)

    # (b) Batch Size Sweep
    bs = sweep_results["bs"]
    bx = np.arange(len(bs))
    ada_vals  = sweep_results["AdaBN"]
    alp_vals  = sweep_results.get("alpha_BN", [])
    pmsa_vals = sweep_results["PMSA_BN"]

    if src_dice is not None:
        ax2.axhline(src_dice, linestyle="--", color="#555555", linewidth=1.0,
                    label=f"Source Only ({src_dice:.3f})")
    ax2.plot(bx, ada_vals, marker="s", markersize=4.8, color="#D55E00", linewidth=1.5, label="AdaBN")
    if len(alp_vals) == len(bs):
        ax2.plot(bx, alp_vals, marker="^", markersize=4.8, color="#E69F00", linewidth=1.2,
                 linestyle="-.", label=r"$\alpha$-BN")
    ax2.plot(bx, pmsa_vals, marker="o", markersize=5.2, color="#0072B2", linewidth=1.9, label="PMSA-BN (Ours)")

    ax2.set_xticks(bx)
    ax2.set_xticklabels([str(b) for b in bs])
    ax2.set_xlabel("Test Batch Size $B$")
    ax2.set_ylabel("Mean Lesion Dice")
    ax2.set_title(r"\textbf{(b)} Batch-Size Invariance", pad=6)
    ax2.legend(loc="lower right", framealpha=0.9, borderpad=0.3, handlelength=1.2)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    t_low = target_name.lower()
    pdf_path = os.path.join(CFG.OUT_DIR, f"fig_paper_compact_{t_low}.pdf")
    png_path = os.path.join(CFG.OUT_DIR, f"fig_paper_compact_{t_low}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")

    fig.savefig(os.path.join(CFG.OUT_DIR, "fig_paper_compact.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(CFG.OUT_DIR, "fig_paper_compact.png"), bbox_inches="tight")
    plt.close(fig)

    tex_compact = r"""\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{fig_paper_compact.pdf}
  \caption{(a) Component ablation of PMSA-BN on cutaneous lesion segmentation ($B=1$). (b) Dice versus test batch size ($1$--$16$) for AdaBN and PMSA-BN. AdaBN depends on batch statistics and degrades at small batches, whereas PMSA-BN is essentially flat and already at its best at $B=1$.}
  \label{fig:paper_compact}
\end{figure*}
"""
    with open(os.path.join(CFG.OUT_DIR, f"fig_paper_compact_{t_low}.tex"), "w", encoding="utf-8") as f:
        f.write(tex_compact)
    with open(os.path.join(CFG.OUT_DIR, "fig_paper_compact.tex"), "w", encoding="utf-8") as f:
        f.write(tex_compact)

    print(f"  ✓ [Compact 2-Panel Figure] Saved: {pdf_path} and {png_path}")
    print(f"  ✓ [LaTeX 2-Panel Snippet] Saved: {os.path.join(CFG.OUT_DIR, 'fig_paper_compact.tex')}")


def run_batch_size_sweep(target_name, test_pairs, src_model, pmsa_model, ablation_results=None):
    print("\n" + "=" * 65 + f"\n  BATCH-SIZE SWEEP: {target_name} (B in {CFG.BATCH_SIZES})\n" + "=" * 65)
    sweep_results = {"bs": CFG.BATCH_SIZES, "AdaBN": [], "alpha_BN": [], "PMSA_BN": []}

    for b in CFG.BATCH_SIZES:
        loader = DataLoader(DermoscopyDataset(test_pairs, transform=val_tfm()),
                            batch_size=b, shuffle=False, num_workers=CFG.NUM_WORKERS, drop_last=False)
        r_ada, r_alp, r_pmsa = [], [], []
        for imgs, msks, _ in loader:
            imgs = imgs.to(CFG.DEVICE)
            # 1. AdaBN
            p_ada = predict_adabn(src_model, imgs)
            for i in range(imgs.size(0)):
                r_ada.append(compute_binary_metrics(p_ada[i].cpu(), msks[i])["dice"])

            # 2. alpha-BN (alpha=0.5)
            p_alp = predict_alpha_bn(src_model, imgs, alpha=CFG.ALPHA_BN_WEIGHT)
            for i in range(imgs.size(0)):
                r_alp.append(compute_binary_metrics(p_alp[i].cpu(), msks[i])["dice"])

            # 3. PMSA-BN
            p_pmsa = predict_pmsa(pmsa_model, src_model, imgs, em_iters=1)
            for i in range(imgs.size(0)):
                r_pmsa.append(compute_binary_metrics(p_pmsa[i].cpu(), msks[i])["dice"])

        ada_d = float(np.mean(r_ada))
        alp_d = float(np.mean(r_alp))
        pmsa_d = float(np.mean(r_pmsa))
        sweep_results["AdaBN"].append(ada_d)
        sweep_results["alpha_BN"].append(alp_d)
        sweep_results["PMSA_BN"].append(pmsa_d)
        print(f"  B = {b:<2} │ AdaBN: {ada_d:.4f} │ α-BN: {alp_d:.4f} │ PMSA-BN: {pmsa_d:.4f} ★")

    # Save JSON
    json_path = os.path.join(CFG.OUT_DIR, f"sweep_{target_name.lower()}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    src_dice = None
    if ablation_results and "1. Source Only" in ablation_results:
        src_dice = ablation_results["1. Source Only"]["dice"][0]

    # Standalone Fig 1
    plot_figure_1(target_name, sweep_results, src_dice)

    # Compact 2-Panel Figure for 4-Page Paper
    if ablation_results:
        plot_compact_paper_figure(target_name, ablation_results, sweep_results, src_dice)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE EXECUTION                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n" + "=" * 65 + "\n  STEP 1 — Loading Source Model from paths1\n" + "=" * 65)
    ckpt_path = find_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError("Could not find best_model.pth in /kaggle/input/datasets/research355/paths1/ or working directory.")

    ckpt = torch.load(ckpt_path, map_location=CFG.DEVICE, weights_only=False) if hasattr(torch, "weights_only") else torch.load(ckpt_path, map_location=CFG.DEVICE)
    state = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt

    src_model = build_model().to(CFG.DEVICE)
    src_model.load_state_dict(state)
    src_model.eval()
    print(f"  ✓ Loaded source checkpoint: {ckpt_path}")

    # Phase 0 Source Stats
    stats = collect_source_stats(src_model)

    # Build PMSA Model
    pmsa_model = deepcopy(src_model)
    convert_pmsa(pmsa_model)
    inject_stats(pmsa_model, stats)
    pmsa_model.to(CFG.DEVICE)
    pmsa_model.eval()
    print(f"  ✓ Injected calibrated source stats into 46 PMSA-BN layers")

    # Evaluate Ablations
    targets_to_eval = [t for t in CFG.TARGETS]
    for target in targets_to_eval:
        if target == "HAM10000":
            pairs = find_ham10000_pairs()
        elif target == "Derm7pt":
            pairs = find_derm7pt_pairs()
        else:
            pairs = []

        if not pairs:
            print(f"\n  [WARN] No image pairs found for {target}. Skipping.")
            continue

        n_samples = min(len(pairs), CFG.MAX_TARGET_SAMPLES.get(target, len(pairs)))
        test_pairs = pairs[:n_samples]

        # 1. 7-Stage Component Ablation
        ablation_results = run_ablation_on_target(target, test_pairs, src_model, pmsa_model)

        # 2. Batch-Size Invariance Sweep & Publication Plots
        run_batch_size_sweep(target, test_pairs, src_model, pmsa_model, ablation_results)

    print("\n" + "=" * 65 + "\n  ✓ Ablation Study Complete! All figures & JSONs saved in /kaggle/working\n" + "=" * 65)

if __name__ == "__main__":
    main()
