"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm                      ║
║    Dual-Source Polyp Benchmark: Kvasir & ClinicDB -> ETIS-LaribPolypDB      ║
║                                                                              ║
║  EVALUATION TARGET:                                                          ║
║    ETIS-LaribPolypDB (196 challenging unseen images from Paris, France)      ║
║                                                                              ║
║  EXPERIMENTS EXECUTED SEQUENTIALLY:                                          ║
║    1. Benchmark A: Kvasir-SEG  ──► ETIS-LaribPolypDB                         ║
║    2. Benchmark B: ClinicDB    ──► ETIS-LaribPolypDB                         ║
║                                                                              ║
║  MODELS REUSED AUTOMATICALLY:                                                ║
║    - /kaggle/working/best_model_polyp.pth (Kvasir source model)              ║
║    - /kaggle/working/best_model_c2k.pth   (ClinicDB source model)            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Install ──────────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "segmentation-models-pytorch", "albumentations"], check=False)

# ── Imports ──────────────────────────────────────────────────────────────────
import os, glob, copy, json, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CFG — Hyperparameters & Configuration                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CFG:
    OUT_DIR         = "/kaggle/working"
    ENCODER         = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES     = 2        # 0=background mucosa, 1=polyp
    IMAGE_SIZE      = 352      # PraNet / Polyp-PVT standard benchmark resolution

    # Training parameters (only used if checkpoints are not found)
    TRAIN_BS        = 8
    NUM_EPOCHS      = 50
    LR              = 1e-4
    WEIGHT_DECAY    = 1e-5
    CLASS_WEIGHTS   = [0.2, 1.8]

    # PMSA-BN Hyperparameters (Validated Calibrated Formulation)
    TAU_HI          = 0.50
    KAPPA_MU        = 16.0     # Bayesian shrinkage pseudo-count (mean)
    KAPPA_SIGMA     = 32.0     # Bayesian shrinkage pseudo-count (variance)
    ALPHA_TAU       = 1.0      # Temperature τ for Wasserstein shift sensitivity
    ALPHA_MIN       = 0.0      # Invariant channels stay on robust source statistics
    ALPHA_MAX       = 0.45     # Upper bound avoids single-image sample variance explosion
    EM_ITERS        = 1

    ALPHA_BN_WEIGHT = 0.5
    SEED            = 42
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS     = 2


def seed_everything(seed=CFG.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

seed_everything()
print(f"✓ Device: {CFG.DEVICE}  |  PyTorch: {torch.__version__}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATASET AUTO-DISCOVERY                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def auto_find_dir(keywords, candidate_paths=None):
    if candidate_paths:
        for p in candidate_paths:
            if os.path.exists(p): return p
    search_root = "/kaggle/input"
    if os.path.exists(search_root):
        for root, dirs, _ in os.walk(search_root):
            for d in dirs:
                full_d = os.path.join(root, d)
                if all(kw.lower() in full_d.lower() for kw in keywords):
                    return full_d
    return None

def find_checkpoint(model_name):
    cands = [
        os.path.join(CFG.OUT_DIR, model_name),
        os.path.join("/kaggle/working", model_name),
    ]
    for c in cands:
        if os.path.exists(c): return c
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if model_name in files:
                return os.path.join(root, model_name)
    return None

def get_kvasir_base():
    cands = [
        "/kaggle/input/datasets/debeshjha1/kvasirseg/Kvasir-SEG",
        "/kaggle/input/kvasirseg/Kvasir-SEG",
        "/kaggle/input/kvasir-seg/Kvasir-SEG",
    ]
    return auto_find_dir(["kvasir"], cands)

def get_clinicdb_base():
    cands = [
        "/kaggle/input/datasets/balraj98/cvcclinicdb/PNG",
        "/kaggle/input/cvcclinicdb/PNG",
        "/kaggle/input/cvc-clinicdb/PNG",
    ]
    return auto_find_dir(["clinicdb"], cands)

def get_etis_base():
    cands = [
        "/kaggle/input/datasets/nguyenvoquocduong/etis-laribpolypdb",
        "/kaggle/input/etis-laribpolypdb",
        "/kaggle/input/datasets/balraj98/etislaribpolypdb",
        "/kaggle/input/etislaribpolypdb",
    ]
    return auto_find_dir(["etis"], cands)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TRANSFORMS                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_compose(transforms):
    try:
        return A.Compose(transforms, is_check_shapes=False)
    except TypeError:
        return A.Compose(transforms)

def train_tfm(sz=CFG.IMAGE_SIZE):
    return make_compose([
        A.Resize(sz, sz),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def val_tfm(sz=CFG.IMAGE_SIZE):
    return make_compose([
        A.Resize(sz, sz),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATASET CLASSES                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class KvasirDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.transform = transform
        base = get_kvasir_base()
        if not base: raise FileNotFoundError("Kvasir-SEG directory not found.")
        img_dir = os.path.join(base, "images") if os.path.exists(os.path.join(base, "images")) else os.path.join(base, "Kvasir-SEG", "images")
        msk_dir = os.path.join(base, "masks")  if os.path.exists(os.path.join(base, "masks"))  else os.path.join(base, "Kvasir-SEG", "masks")
        all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(all_imgs))
        split_idx = int(0.8 * len(all_imgs))
        chosen_idx = idx[:split_idx] if split == "train" else idx[split_idx:]
        self.samples = []
        for i in chosen_idx:
            ip = all_imgs[i]
            stem = os.path.splitext(os.path.basename(ip))[0]
            mp = os.path.join(msk_dir, stem + ".jpg")
            if not os.path.exists(mp): mp = os.path.join(msk_dir, stem + ".png")
            if os.path.exists(mp): self.samples.append((ip, mp, stem))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        ip, mp, stem = self.samples[idx]
        image = np.array(Image.open(ip).convert("RGB"))
        raw = np.array(Image.open(mp).convert("L"))
        if raw.shape[:2] != image.shape[:2]:
            raw = np.array(Image.fromarray(raw).resize((image.shape[1], image.shape[0]), resample=Image.NEAREST))
        mask = (raw > 127).astype(np.int64)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"].long()
        return image, mask, stem


class ClinicDBDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.transform = transform
        base = get_clinicdb_base()
        if not base: raise FileNotFoundError("CVC-ClinicDB directory not found.")
        img_dir = os.path.join(base, "Original") if os.path.exists(os.path.join(base, "Original")) else base
        msk_dir = os.path.join(base, "Ground Truth") if os.path.exists(os.path.join(base, "Ground Truth")) else base
        all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(all_imgs))
        split_idx = int(0.8 * len(all_imgs))
        chosen_idx = idx[:split_idx] if split == "train" else idx[split_idx:]
        self.samples = []
        for i in chosen_idx:
            ip = all_imgs[i]
            stem = os.path.splitext(os.path.basename(ip))[0]
            mp = os.path.join(msk_dir, stem + ".png")
            if os.path.exists(mp): self.samples.append((ip, mp, stem))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        ip, mp, stem = self.samples[idx]
        image = np.array(Image.open(ip).convert("RGB"))
        raw = np.array(Image.open(mp).convert("L"))
        if raw.shape[:2] != image.shape[:2]:
            raw = np.array(Image.fromarray(raw).resize((image.shape[1], image.shape[0]), resample=Image.NEAREST))
        mask = (raw > 127).astype(np.int64)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"].long()
        return image, mask, stem


class ETISDataset(Dataset):
    """
    ETIS-LaribPolypDB Dataset (196 challenging test colonoscopy frames).
    Target domain for zero-shot test-time adaptation.
    """
    def __init__(self, transform=None):
        self.transform = transform
        base = get_etis_base()
        if not base:
            raise FileNotFoundError(
                "ETIS-LaribPolypDB directory not found under /kaggle/input.\n"
                "Please ensure nguyenvoquocduong/etis-laribpolypdb is added to the notebook inputs."
            )

        # Locate images directory
        img_dirs = [os.path.join(base, "images"), os.path.join(base, "Images"), base]
        img_dir = next((d for d in img_dirs if os.path.isdir(d) and len(glob.glob(os.path.join(d, "*.*"))) > 0), None)

        # Locate masks directory
        msk_dirs = [os.path.join(base, "masks"), os.path.join(base, "Masks"),
                    os.path.join(base, "ground_truth"), os.path.join(base, "Ground Truth")]
        msk_dir = next((d for d in msk_dirs if os.path.isdir(d)), None)

        if not img_dir or not msk_dir:
            # Recursive search fallback
            all_imgs = [p for p in glob.glob(os.path.join(base, "**", "*.*"), recursive=True)
                        if "mask" not in p.lower() and p.lower().endswith((".png", ".jpg", ".tif", ".bmp"))]
            all_msks = [p for p in glob.glob(os.path.join(base, "**", "*.*"), recursive=True)
                        if "mask" in p.lower() and p.lower().endswith((".png", ".jpg", ".tif", ".bmp"))]
        else:
            all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.*")))
            all_msks = sorted(glob.glob(os.path.join(msk_dir, "*.*")))

        # Match images and masks by numeric/alphanumeric stem
        msk_map = {}
        for mp in all_msks:
            m_stem = os.path.splitext(os.path.basename(mp))[0].lower().replace("p", "").replace("_mask", "").replace("mask", "")
            msk_map[m_stem] = mp
            msk_map[os.path.splitext(os.path.basename(mp))[0].lower()] = mp

        self.samples = []
        for ip in all_imgs:
            stem = os.path.splitext(os.path.basename(ip))[0]
            clean_stem = stem.lower().replace("p", "").replace("_image", "")
            mp = msk_map.get(clean_stem) or msk_map.get(stem.lower())
            if mp and os.path.exists(mp):
                self.samples.append((ip, mp, stem))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"Failed to match images and masks in {base}.")
        print(f"  Target ETIS-Larib: {len(self.samples)} test images successfully indexed (base: {base})")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ip, mp, stem = self.samples[idx]
        image = np.array(Image.open(ip).convert("RGB"))
        raw   = np.array(Image.open(mp).convert("L"))

        # Explicit shape-matching before augmentation
        if raw.shape[:2] != image.shape[:2]:
            raw = np.array(Image.fromarray(raw).resize(
                (image.shape[1], image.shape[0]), resample=Image.NEAREST
            ))

        mask = (raw > 127).astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"].long()
        return image, mask, stem


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EVALUATION METRICS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def dice_score(pred_bin, target_bin, eps=1e-6):
    inter = (pred_bin * target_bin).sum()
    total = pred_bin.sum() + target_bin.sum()
    return float((2.0 * inter + eps) / (total + eps))

def iou_score(pred_bin, target_bin, eps=1e-6):
    inter = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - inter
    return float((inter + eps) / (union + eps))

def evaluate_sample(probs, label):
    pred = (probs.argmax(dim=0) == 1).numpy().astype(np.float32)
    tgt  = (label == 1).numpy().astype(np.float32)
    return {
        "dice": dice_score(pred, tgt),
        "iou":  iou_score(pred, tgt),
    }

def agg(res_list):
    out = {}
    for k in res_list[0].keys():
        vals = [r[k] for r in res_list]
        out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL BUILDER                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_model():
    return smp.Unet(
        encoder_name=CFG.ENCODER,
        encoder_weights=CFG.ENCODER_WEIGHTS,
        in_channels=3,
        classes=CFG.NUM_CLASSES
    )

def train_source(source_name, ckpt_path, stats_path):
    print("\n" + "═"*60)
    print(f"  Training Source Model on {source_name}")
    print("═"*60)
    model = build_model().to(CFG.DEVICE)
    ds_cls = KvasirDataset if source_name == "Kvasir" else ClinicDBDataset
    tr_ds = ds_cls("train", train_tfm())
    va_ds = ds_cls("val",   val_tfm())

    tr_loader = DataLoader(tr_ds, batch_size=CFG.TRAIN_BS, shuffle=True,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=CFG.TRAIN_BS*2, shuffle=False,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True)

    weights = torch.tensor(CFG.CLASS_WEIGHTS, device=CFG.DEVICE)
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    dice_l  = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    criterion = lambda p, y: 0.5 * ce_loss(p, y) + 0.5 * dice_l(p, y)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.NUM_EPOCHS, eta_min=1e-6)

    best_dice = 0.0
    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for imgs, lbls, _ in tr_loader:
            imgs, lbls = imgs.to(CFG.DEVICE), lbls.to(CFG.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward(); optimizer.step()
            ep_loss += loss.item()
        scheduler.step()

        model.eval(); val_res = []
        with torch.no_grad():
            for imgs, lbls, _ in va_loader:
                probs = F.softmax(model(imgs.to(CFG.DEVICE)), dim=1)
                for b in range(imgs.size(0)):
                    val_res.append(evaluate_sample(probs[b].cpu(), lbls[b]))
        vagg = agg(val_res)
        is_best = vagg["dice"][0] > best_dice
        if is_best:
            best_dice = vagg["dice"][0]
            torch.save({"state": model.state_dict(), "val": vagg}, ckpt_path)
        print(f"  [{source_name} Epoch {epoch:02d}/{CFG.NUM_EPOCHS}] Loss: {ep_loss/len(tr_loader):.4f} │ Val Dice: {vagg['dice'][0]:.4f}")

    ckpt = torch.load(ckpt_path, map_location=CFG.DEVICE)
    model.load_state_dict(ckpt["state"])
    return model

def collect_source_stats(model, source_name, stats_path):
    print(f"\n  Collecting Phase-0 Statistics for {source_name}...")
    model.eval()
    K = CFG.NUM_CLASSES
    bn_mods = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]

    accum = [{"sum_k":    torch.zeros(K, m.num_features),
              "sum_sq_k": torch.zeros(K, m.num_features),
              "count_k":  torch.zeros(K)} for _, m in bn_mods]
    global_count = torch.zeros(K)
    state = {"labels": None}

    def make_hook(idx):
        def hook(mod, inp, out):
            x = inp[0].detach().cpu()
            lbl = state["labels"]
            if lbl is None: return
            B, C, H, W = x.shape
            lbl_d = F.interpolate(lbl.float().unsqueeze(1), (H, W), mode="nearest").squeeze(1).long()
            for k in range(K):
                mk = (lbl_d == k).float(); n = mk.sum()
                if n < 1: continue
                mk4 = mk.unsqueeze(1)
                accum[idx]["sum_k"][k]    += (x * mk4).sum([0,2,3])
                accum[idx]["sum_sq_k"][k] += (x**2 * mk4).sum([0,2,3])
                accum[idx]["count_k"][k]  += n
        return hook

    hooks = [m.register_forward_hook(make_hook(i)) for i, (_, m) in enumerate(bn_mods)]
    ds_cls = KvasirDataset if source_name == "Kvasir" else ClinicDBDataset
    loader = DataLoader(ds_cls("train", val_tfm()), batch_size=4, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)

    with torch.no_grad():
        for imgs, lbls, _ in tqdm(loader, desc=f"  Phase 0 ({source_name})", leave=False):
            state["labels"] = lbls
            _ = model(imgs.to(CFG.DEVICE))
            for k in range(K):
                global_count[k] += (lbls == k).float().sum()

    for h in hooks: h.remove()

    class_means, class_vars = [], []
    for ac in accum:
        mu = torch.zeros(K, ac["sum_k"].shape[1])
        va = torch.ones (K, ac["sum_k"].shape[1])
        for k in range(K):
            n = ac["count_k"][k].item()
            if n > 0:
                mu[k] = ac["sum_k"][k] / n
                ex2   = ac["sum_sq_k"][k] / n
                va[k] = (ex2 - mu[k]**2).clamp(min=1e-6)
        class_means.append(mu)
        class_vars.append(va)

    prior = global_count / global_count.sum().clamp(min=1e-6)
    stats = {"class_mean": class_means, "class_var": class_vars, "prior": prior}
    torch.save(stats, stats_path)
    print(f"  ✓ Saved {source_name} stats → {stats_path} (Prior π: bg={prior[0]:.3f}, polyp={prior[1]:.3f})")
    return stats


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PMSA-BN MODULE (Resolution-Invariant Formulation)                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class PMSABatchNorm2d(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d,
                 K=CFG.NUM_CLASSES, tau=CFG.ALPHA_TAU,
                 kmu=CFG.KAPPA_MU, ks=CFG.KAPPA_SIGMA,
                 amin=CFG.ALPHA_MIN, amax=CFG.ALPHA_MAX):
        super().__init__()
        self.bn  = bn
        C = bn.num_features
        self.register_buffer("src_cls_mean", torch.zeros(K, C))
        self.register_buffer("src_cls_var",  torch.ones (K, C))
        self.register_buffer("src_prior",    torch.ones (K) / K)
        self.K = K; self.tau = tau; self.kmu = kmu; self.ks = ks
        self.amin = amin; self.amax = amax
        self.adapt_mode  = False
        self.pseudo_mask = None
        self.last_Dc = None; self.last_alpha = None

    def _cls_stats(self, x, mask):
        B, C, H, W = x.shape
        mu = torch.zeros(self.K, C, device=x.device)
        va = torch.zeros(self.K, C, device=x.device)
        Nk = torch.zeros(self.K, device=x.device)

        tgt_img_mu = x.mean([0, 2, 3])
        tgt_img_va = x.var([0, 2, 3], unbiased=False)

        for k in range(self.K):
            mk = mask[:, k:k+1]; Nk[k] = mk.sum()
            if Nk[k] > 1e-4:
                mu[k] = (mk * x).sum([0, 2, 3]) / Nk[k]
                va[k] = (mk * (x - mu[k].view(1, C, 1, 1))**2).sum([0, 2, 3]) / Nk[k]
            else:
                mu[k] = tgt_img_mu
                va[k] = tgt_img_va
        return mu, va, Nk, tgt_img_mu, tgt_img_va

    def _shrink(self, tgt_mu, tgt_va, Nk, tgt_img_mu, tgt_img_va, H, W):
        smu = torch.zeros_like(tgt_mu)
        sva = torch.zeros_like(tgt_va)

        src_global_mu = (self.src_prior[:, None] * self.src_cls_mean).sum(0)
        scale_factor  = (352.0 * 352.0) / float(H * W)

        for k in range(self.K):
            n_eff = Nk[k].item() * scale_factor
            g  = n_eff / (n_eff + self.kmu)
            gs = max(0.0, n_eff - 1.0) / (max(0.0, n_eff - 1.0) + self.ks)

            anchor_mu = tgt_img_mu + (self.src_cls_mean[k] - src_global_mu)
            smu[k] = g * tgt_mu[k] + (1.0 - g) * anchor_mu
            sva[k] = gs * tgt_va[k] + (1.0 - gs) * tgt_img_va
        return smu, sva

    def _prior_match(self, cls_mu, cls_va, Nk):
        pi_tgt = Nk / Nk.sum().clamp(min=1e-6)
        pi = 0.5 * pi_tgt + 0.5 * self.src_prior

        pm_mu   = (pi[:, None] * cls_mu).sum(0)
        within  = (pi[:, None] * cls_va).sum(0)
        diff    = cls_mu - pm_mu[None, :]
        between = (pi[:, None] * (diff**2)).sum(0)
        return pm_mu, within + between

    def _w2_dist(self, pm_mu, pm_va, src_mu, src_va, eps=1e-6):
        return ((pm_mu - src_mu)**2 + (pm_va.clamp(0).sqrt() - src_va.clamp(0).sqrt())**2) / (src_va + eps)

    def forward(self, x):
        if not self.adapt_mode or self.pseudo_mask is None:
            return self.bn(x)

        B, C, H, W = x.shape
        mask = self.pseudo_mask.to(x.device)
        if mask.shape[2] != H or mask.shape[3] != W:
            mask = F.interpolate(mask, (H, W), mode="bilinear", align_corners=False)
        mask = mask / mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        mask = mask.expand(B, -1, -1, -1)

        with torch.no_grad():
            src_mu = self.bn.running_mean.detach()
            src_va = self.bn.running_var.detach()

            tgt_mu, tgt_va, Nk, img_mu, img_va = self._cls_stats(x, mask)
            shr_mu, shr_va = self._shrink(tgt_mu, tgt_va, Nk, img_mu, img_va, H, W)
            pm_mu, pm_va   = self._prior_match(shr_mu, shr_va, Nk)

            Dc = self._w2_dist(pm_mu, pm_va, src_mu, src_va)
            span = self.amax - self.amin
            alpha = torch.clamp(self.amin + span * torch.tanh(Dc / (1.5 * self.tau)), self.amin, self.amax)

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
    means = stats["class_mean"]; vars_ = stats["class_var"]; prior = stats["prior"]
    dev = next(model.parameters()).device
    for lyr, mu, va in zip(pmsa, means, vars_):
        lyr.src_cls_mean = mu.to(dev)
        lyr.src_cls_var  = va.to(dev)
        lyr.src_prior    = prior.to(dev)

def get_diag(model):
    d = {}
    for i, m in enumerate(x for x in model.modules() if isinstance(x, PMSABatchNorm2d)):
        if m.last_Dc is not None:
            d[i] = {"Dc": m.last_Dc, "alpha": m.last_alpha}
    return d


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  INFERENCE BASELINES & PMSA-BN                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def predict_source_only(model, image):
    model.eval()
    return F.softmax(model(image), dim=1)

@torch.no_grad()
def predict_adabn(model, image):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train(); m.momentum = 1.0
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
def predict_pmsa(model, image, em_iters=CFG.EM_ITERS, src_model=None):
    if src_model is not None:
        probs = predict_alpha_bn(copy.deepcopy(src_model).to(image.device), image)
    else:
        probs = predict_alpha_bn(copy.deepcopy(model).to(image.device), image)

    set_adapt(model, True)
    diag_list = []
    for _ in range(em_iters):
        # Continuous soft responsibilities on probability simplex
        set_mask(model, probs)
        probs = F.softmax(model(image), dim=1)
        diag_list.append(get_diag(model))

    set_adapt(model, False)
    set_mask(model, None)
    return probs, diag_list


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EVALUATION SUITE                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def evaluate_source_on_etis(src_model, pmsa_model, loader, source_name):
    print("\n" + "═"*65)
    print(f"  Evaluating {source_name} ──► ETIS-LaribPolypDB (196 Images)")
    print("═"*65)

    method_res = defaultdict(list)
    all_diag   = []
    src_model.eval()

    for images, labels, stems in tqdm(loader, desc=f"  {source_name} -> ETIS"):
        images = images.to(CFG.DEVICE)

        # 1. Source Only
        p = predict_source_only(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        method_res["Source Only"].append(evaluate_sample(p.squeeze(0).cpu(), labels.squeeze(0)))

        # 2. AdaBN
        p = predict_adabn(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        method_res["AdaBN"].append(evaluate_sample(p.squeeze(0).cpu(), labels.squeeze(0)))

        # 3. α-BN
        abn_m = copy.deepcopy(src_model).to(CFG.DEVICE)
        p = predict_alpha_bn(abn_m, images)
        method_res["α-BN (α=0.5)"].append(evaluate_sample(p.squeeze(0).cpu(), labels.squeeze(0)))

        # 4. PMSA-BN
        pm = copy.deepcopy(pmsa_model).to(CFG.DEVICE)
        p, diags = predict_pmsa(pm, images, src_model=src_model)
        method_res["PMSA-BN (Ours)"].append(evaluate_sample(p.squeeze(0).cpu(), labels.squeeze(0)))
        if diags: all_diag.append(diags[-1])

    aggregated = {m: agg(r) for m, r in method_res.items()}
    best_dice = max(res["dice"][0] for res in aggregated.values())
    best_iou  = max(res["iou"][0]  for res in aggregated.values())

    print("\n" + "╔"+"═"*58+"╗")
    print(f"║  RESULTS: {source_name} ──► ETIS-LaribPolypDB{' '*(30-len(source_name))}║")
    print("╠"+"═"*24+"╦"+"═"*16+"╦"+"═"*16+"╣")
    print(f"║ {'Method':<22} ║ {'Polyp Dice':^14} ║ {'Polyp IoU':^14} ║")
    print("╠"+"═"*24+"╬"+"═"*16+"╬"+"═"*16+"╣")
    for m, res in aggregated.items():
        d_val, d_std = res["dice"]
        i_val, i_std = res["iou"]
        d_star = " ★" if d_val == best_dice else "  "
        i_star = " ★" if i_val == best_iou  else "  "
        print(f"║ {m:<22} ║ {d_val:>6.4f}±{d_std:.3f}{d_star} ║ {i_val:>6.4f}±{i_std:.3f}{i_star} ║")
    print("╚"+"═"*24+"╩"+"═"*16+"╩"+"═"*16+"╝")

    return aggregated, all_diag


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURES & SUMMARY                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_bar_comparison(agg_kvasir, agg_clinicdb, path):
    methods = list(agg_kvasir.keys())
    colors  = ["#95a5a6", "#e74c3c", "#e67e22", "#2980b9"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (agg_data, title) in zip(axes, [
        (agg_kvasir,   "Source: Kvasir-SEG → Target: ETIS-Larib"),
        (agg_clinicdb, "Source: CVC-ClinicDB → Target: ETIS-Larib")
    ]):
        dice_vals = [agg_data[m]["dice"][0] for m in methods]
        dice_errs = [agg_data[m]["dice"][1] for m in methods]
        x = np.arange(len(methods))
        ax.bar(x, dice_vals, color=colors, alpha=0.85, edgecolor="black", lw=0.7)
        ax.errorbar(x, dice_vals, yerr=dice_errs, fmt="none", color="black", capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        best = np.argmax(dice_vals)
        ax.text(x[best], dice_vals[best] + dice_errs[best] + 0.01, "★", ha="center", fontsize=15, color="gold")

    plt.suptitle("PMSA-BN Cross-Domain Generalization on ETIS-LaribPolypDB Benchmark", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN EXECUTION                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n" + "╔"+"═"*62+"╗")
    print("║   PMSA-BN: Dual-Source Evaluation on ETIS-LaribPolypDB       ║")
    print("║   Sources: Kvasir-SEG & CVC-ClinicDB ──► Target: ETIS-Larib  ║")
    print("╚"+"═"*62+"╝")

    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    etis_ds = ETISDataset(transform=val_tfm())
    etis_loader = DataLoader(etis_ds, batch_size=1, shuffle=False,
                             num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # ─────────────────────────────────────────────────────────────────────────
    # BENCHMARK 1: Kvasir-SEG ──► ETIS-LaribPolypDB
    # ─────────────────────────────────────────────────────────────────────────
    print("\n>>> Setting up Benchmark 1: Kvasir ──► ETIS...")
    kvasir_ckpt  = find_checkpoint("best_model_polyp.pth")
    kvasir_stats = find_checkpoint("source_stats_polyp.pth")

    if kvasir_ckpt and kvasir_stats:
        print(f"  ✓ Reusing Kvasir model checkpoint: {kvasir_ckpt}")
        kvasir_src = build_model().to(CFG.DEVICE)
        ck = torch.load(kvasir_ckpt, map_location=CFG.DEVICE)
        kvasir_src.load_state_dict(ck["state"] if "state" in ck else ck)
        k_stats = torch.load(kvasir_stats, map_location="cpu")
    else:
        kvasir_ckpt  = os.path.join(CFG.OUT_DIR, "best_model_polyp.pth")
        kvasir_stats = os.path.join(CFG.OUT_DIR, "source_stats_polyp.pth")
        kvasir_src   = train_source("Kvasir", kvasir_ckpt, kvasir_stats)
        k_stats      = collect_source_stats(kvasir_src, "Kvasir", kvasir_stats)

    kvasir_pmsa = replace_bn(copy.deepcopy(kvasir_src))
    inject_stats(kvasir_pmsa, k_stats)
    kvasir_pmsa.eval().to(CFG.DEVICE)

    agg_kvasir, diags_k = evaluate_source_on_etis(kvasir_src, kvasir_pmsa, etis_loader, "Kvasir")

    # ─────────────────────────────────────────────────────────────────────────
    # BENCHMARK 2: CVC-ClinicDB ──► ETIS-LaribPolypDB
    # ─────────────────────────────────────────────────────────────────────────
    print("\n>>> Setting up Benchmark 2: ClinicDB ──► ETIS...")
    clinic_ckpt  = find_checkpoint("best_model_c2k.pth")
    clinic_stats = find_checkpoint("source_stats_c2k.pth")

    if clinic_ckpt and clinic_stats:
        print(f"  ✓ Reusing ClinicDB model checkpoint: {clinic_ckpt}")
        clinic_src = build_model().to(CFG.DEVICE)
        ck = torch.load(clinic_ckpt, map_location=CFG.DEVICE)
        clinic_src.load_state_dict(ck["state"] if "state" in ck else ck)
        c_stats = torch.load(clinic_stats, map_location="cpu")
    else:
        clinic_ckpt  = os.path.join(CFG.OUT_DIR, "best_model_c2k.pth")
        clinic_stats = os.path.join(CFG.OUT_DIR, "source_stats_c2k.pth")
        clinic_src   = train_source("ClinicDB", clinic_ckpt, clinic_stats)
        c_stats      = collect_source_stats(clinic_src, "ClinicDB", clinic_stats)

    clinic_pmsa = replace_bn(copy.deepcopy(clinic_src))
    inject_stats(clinic_pmsa, c_stats)
    clinic_pmsa.eval().to(CFG.DEVICE)

    agg_clinic, diags_c = evaluate_source_on_etis(clinic_src, clinic_pmsa, etis_loader, "ClinicDB")

    # ─────────────────────────────────────────────────────────────────────────
    # MASTER COMPARATIVE SUMMARY & FIGURE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("  🏆 MASTER COMPARISON: BOTH SOURCES EVALUATED ON ETIS-LARIB (196 IMAGES)")
    print("═"*70)
    print(f"  {'Method':<20} │ {'Kvasir -> ETIS Dice':^21} │ {'ClinicDB -> ETIS Dice':^23}")
    print("  " + "─"*68)
    for m in agg_kvasir.keys():
        kd, ks = agg_kvasir[m]["dice"]
        cd, cs = agg_clinic[m]["dice"]
        print(f"  {m:<20} │ {kd:>6.4f} ± {ks:.3f}          │ {cd:>6.4f} ± {cs:.3f}")
    print("═"*70)

    plot_bar_comparison(agg_kvasir, agg_clinic, os.path.join(CFG.OUT_DIR, "fig_etis_dual_benchmark.png"))

    summary = {
        "kvasir_to_etis": {m: {k: [round(x, 4) for x in v] for k, v in r.items()} for m, r in agg_kvasir.items()},
        "clinicdb_to_etis": {m: {k: [round(x, 4) for x in v] for k, v in r.items()} for m, r in agg_clinic.items()}
    }
    with open(os.path.join(CFG.OUT_DIR, "results_etis_dual_benchmark.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved results summary → {os.path.join(CFG.OUT_DIR, 'results_etis_dual_benchmark.json')}")

if __name__ == "__main__":
    main()
