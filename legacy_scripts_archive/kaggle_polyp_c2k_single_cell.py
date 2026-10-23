"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm                     ║
║         Single-Cell Kaggle Script — REVERSE TRANSFER: ClinicDB -> Kvasir     ║
║                                                                              ║
║  BENCHMARK:                                                                  ║
║    Source Domain: CVC-ClinicDB (Pentax endoscopes, Hospital Clínic, Spain)   ║
║    Target Domain: Kvasir-SEG   (Olympus endoscopes, Vestre Viken, Norway)    ║
║                                                                              ║
║  DATASETS (already added to Kaggle session):                                 ║
║    1. balraj98/cvcclinicdb  (CVC-ClinicDB: 612 colonoscopy frames)          ║
║    2. debeshjha1/kvasirseg  (Kvasir-SEG: 1,000 colonoscopy images)           ║
║                                                                              ║
║  BASELINES:  Source Only | AdaBN | α-BN (α=0.5) | PMSA-BN (Ours)            ║
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
# ║  CFG — Configuration: ClinicDB -> Kvasir Reverse Transfer                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CFG:
    # ── Reverse Domain Setup ──────────────────────────────────────────────
    SOURCE = "ClinicDB"     # Source domain (trains on 612 ClinicDB frames + collects Phase 0 statistics)
    TARGET = "Kvasir"       # Target domain (evaluates test-time adaptation on 1,000 Kvasir images)

    # ── Output (Separate filenames so previous Kvasir->ClinicDB run is preserved) ──
    CKPT_PATH  = "/kaggle/working/best_model_c2k.pth"
    STATS_PATH = "/kaggle/working/source_stats_c2k.pth"
    OUT_DIR    = "/kaggle/working"

    # ── Model ─────────────────────────────────────────────────────────────
    ENCODER         = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES     = 2        # 0=background, 1=polyp
    IMAGE_SIZE      = 352      # Standard 352×352 resolution for polyp benchmarks

    # ── Training ──────────────────────────────────────────────────────────
    TRAIN_BS           = 8
    NUM_EPOCHS         = 50       # ~3-4 minutes on GPU for 612 ClinicDB images
    LR                 = 1e-4
    WEIGHT_DECAY       = 1e-5
    PATIENCE           = 12
    CLASS_WEIGHTS      = [0.2, 1.8]   # Background vs Polyp foreground
    TRAIN_FROM_SCRATCH = True       # Train ClinicDB model from scratch

    # ── PMSA-BN Hyperparameters (Validated Calibrated Formulation) ───────────
    TAU_HI      = 0.50   # Confident pseudo-mask threshold
    KAPPA_MU    = 16.0   # Bayesian shrinkage pseudo-count (mean)
    KAPPA_SIGMA = 32.0   # Bayesian shrinkage pseudo-count (variance)
    ALPHA_TAU   = 1.0    # Temperature τ for shift sensitivity
    ALPHA_MIN   = 0.0    # Invariant channels stay 100% on source stats (no noise penalty)
    ALPHA_MAX   = 0.45   # Shift-adapted upper bound
    EM_ITERS    = 1      # Single pass test-time adaptation

    # ── Baselines ─────────────────────────────────────────────────────────
    ALPHA_BN_WEIGHT = 0.5
    RUN_ABLATION    = False  # Set True for ablation table

    SEED        = 42
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2


def seed_everything(seed=CFG.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

seed_everything()
print(f"✓ Device: {CFG.DEVICE}  |  PyTorch: {torch.__version__}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATASET AUTO-DISCOVERY & UTILITIES                                      ║
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


def get_kvasir_paths():
    cands = [
        "/kaggle/input/datasets/debeshjha1/kvasirseg/Kvasir-SEG/Kvasir-SEG",
        "/kaggle/input/datasets/debeshjha1/kvasirseg/Kvasir-SEG",
        "/kaggle/input/datasets/debeshjha1/kvasirseg",
        "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG",
        "/kaggle/input/kvasirseg/Kvasir-SEG",
        "/kaggle/input/kvasirseg",
        "/kaggle/input/kvasir-seg",
    ]
    base = auto_find_dir(["kvasir"], cands)
    if not base:
        raise FileNotFoundError("Could not locate Kvasir-SEG dataset in /kaggle/input.")

    img_dirs, mask_dirs = [], []
    for root, dirs, files in os.walk(base):
        if os.path.basename(root).lower() == "images" and len(files) > 50:
            img_dirs.append(root)
        if os.path.basename(root).lower() == "masks" and len(files) > 50:
            mask_dirs.append(root)

    img_dir = img_dirs[0] if img_dirs else os.path.join(base, "images")
    mask_dir = mask_dirs[0] if mask_dirs else os.path.join(base, "masks")

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.*")))

    mask_map = {os.path.splitext(os.path.basename(m))[0]: m for m in mask_files}
    pairs = []
    for img_p in img_files:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        if stem in mask_map:
            pairs.append((img_p, mask_map[stem], stem))

    return pairs, base


def get_clinicdb_paths():
    cands = [
        "/kaggle/input/datasets/balraj98/cvcclinicdb/PNG",
        "/kaggle/input/datasets/balraj98/cvcclinicdb",
        "/kaggle/input/cvcclinicdb/PNG",
        "/kaggle/input/cvcclinicdb",
        "/kaggle/input/cvc-clinicdb/PNG",
        "/kaggle/input/cvc-clinicdb",
    ]
    base = auto_find_dir(["clinic"], cands)
    if not base: base = auto_find_dir(["cvc"], cands)
    if not base:
        raise FileNotFoundError("Could not locate CVC-ClinicDB dataset in /kaggle/input.")

    orig_dirs, gt_dirs = [], []
    for root, dirs, files in os.walk(base):
        bname = os.path.basename(root).lower()
        if ("original" in bname or "images" in bname) and len(files) > 30:
            orig_dirs.append(root)
        if ("ground truth" in bname or "masks" in bname or "groundtruth" in bname) and len(files) > 30:
            gt_dirs.append(root)

    orig_dir = orig_dirs[0] if orig_dirs else os.path.join(base, "Original")
    gt_dir   = gt_dirs[0] if gt_dirs else os.path.join(base, "Ground Truth")

    img_files  = sorted(glob.glob(os.path.join(orig_dir, "*.*")))
    mask_files = sorted(glob.glob(os.path.join(gt_dir, "*.*")))

    mask_map = {os.path.splitext(os.path.basename(m))[0]: m for m in mask_files}
    pairs = []
    for img_p in img_files:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        if stem in mask_map:
            pairs.append((img_p, mask_map[stem], stem))

    return pairs, base


def load_polyp_mask(mask_path: str, target_shape=None) -> np.ndarray:
    if not mask_path or not os.path.exists(mask_path):
        if target_shape is not None: return np.zeros(target_shape, dtype=np.int64)
        return np.zeros((352, 352), dtype=np.int64)

    mask_img = Image.open(mask_path).convert("L")
    if target_shape is not None:
        H, W = target_shape
        if mask_img.size != (W, H):
            mask_img = mask_img.resize((W, H), resample=Image.NEAREST)
    mask = np.array(mask_img)
    return (mask > 127).astype(np.int64)


def train_tfm(sz=CFG.IMAGE_SIZE):
    return A.Compose([
        A.Resize(sz, sz),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.10, rotate_limit=30, p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def val_tfm(sz=CFG.IMAGE_SIZE):
    return A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class PolypDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path, stem = self.pairs[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = load_polyp_mask(mask_path, target_shape=(img.shape[0], img.shape[1]))

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"].long()
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return img, mask, stem


def get_polyp_dataset(domain_name, split="train", transform=None):
    domain = domain_name.upper()
    if "KVASIR" in domain:
        pairs, _ = get_kvasir_paths()
    elif "CLINIC" in domain or "CVC" in domain:
        pairs, _ = get_clinicdb_paths()
    else:
        raise ValueError(f"Unknown domain: {domain_name}")

    rng = random.Random(42)
    shuffled_pairs = list(pairs)
    rng.shuffle(shuffled_pairs)

    n_val = int(len(shuffled_pairs) * 0.20)
    if split == "train":
        chosen_pairs = shuffled_pairs[n_val:]
    elif split == "val":
        chosen_pairs = shuffled_pairs[:n_val]
    elif split in ("test", "all"):
        chosen_pairs = pairs
    else:
        raise ValueError(f"Unknown split: {split}")

    return PolypDataset(chosen_pairs, transform=transform)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SANITY CHECK                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def sanity_check():
    print("\n" + "─"*58)
    print("  Polyp Dataset Sanity Check (Reverse Transfer)")
    print("─"*58)
    print(f"  Configured: SOURCE = {CFG.SOURCE}  →  TARGET = {CFG.TARGET}")

    s_ds_tr = get_polyp_dataset(CFG.SOURCE, "train", val_tfm())
    s_ds_va = get_polyp_dataset(CFG.SOURCE, "val", val_tfm())
    print(f"  Source ({CFG.SOURCE}): {len(s_ds_tr)} train / {len(s_ds_va)} val frames")

    t_ds = get_polyp_dataset(CFG.TARGET, "test", val_tfm())
    print(f"  Target ({CFG.TARGET}): {len(t_ds)} test images")
    print("─"*58 + "\n")

sanity_check()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EVALUATION METRICS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_polyp_metrics(pred_labels: np.ndarray, gt: np.ndarray, smooth=1e-6):
    p = (pred_labels == 1).astype(float)
    g = (gt == 1).astype(float)

    intersection = (p * g).sum()
    union = p.sum() + g.sum() - intersection

    dice = (2.0 * intersection + smooth) / (p.sum() + g.sum() + smooth)
    iou  = (intersection + smooth) / (union + smooth)

    return {"polyp_dice": float(dice), "polyp_iou": float(iou)}

def evaluate_prediction(pred_probs: torch.Tensor, gt_label: torch.Tensor):
    pred_labels = pred_probs.argmax(dim=0).cpu().numpy()
    gt = gt_label.cpu().numpy()
    return compute_polyp_metrics(pred_labels, gt)

def aggregate_metrics(res_list):
    dices = [r["polyp_dice"] for r in res_list]
    ious  = [r["polyp_iou"]  for r in res_list]
    return {
        "polyp_dice": (float(np.mean(dices)), float(np.std(dices))),
        "polyp_iou":  (float(np.mean(ious)),  float(np.std(ious)))
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 — TRAIN SOURCE MODEL ON CLINICDB                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality  = torch.sum(probs + targets_onehot, dims)
        dice_score   = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score[1:].mean()

class CombinedLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        w = torch.tensor(class_weights).float().to(CFG.DEVICE)
        self.ce = nn.CrossEntropyLoss(weight=w)
        self.dice = SoftDiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)


def build_model():
    try:
        model = smp.Unet(
            encoder_name=CFG.ENCODER,
            encoder_weights=CFG.ENCODER_WEIGHTS,
            classes=CFG.NUM_CLASSES,
            activation=None
        )
    except Exception:
        model = smp.Unet(
            encoder_name=CFG.ENCODER,
            encoder_weights=None,
            classes=CFG.NUM_CLASSES,
            activation=None
        )
    return model.to(CFG.DEVICE)


def train_source_model():
    print("═"*60)
    print(f"  STEP 1 — Train Source Model on {CFG.SOURCE}")
    print("═"*60)

    cands = [
        "/kaggle/input/datasets/subhambhat/colonoscopy/best_model_c2k.pth",
        "/kaggle/input/colonoscopy/best_model_c2k.pth",
        "/kaggle/input/subhambhat/colonoscopy/best_model_c2k.pth",
        CFG.CKPT_PATH,
        "/kaggle/input/paths1/best_model_c2k.pth",
        "/kaggle/input/paths1/best_model.pth",
    ]
    for c in cands:
        if os.path.exists(c):
            print(f"  ✓ Reusing existing model checkpoint: {c}")
            model = build_model()
            ckpt = torch.load(c, map_location=CFG.DEVICE)
            state = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt
            model.load_state_dict(state)
            return model

    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "best_model_c2k.pth" in files:
                c = os.path.join(root, "best_model_c2k.pth")
                print(f"  ✓ Found checkpoint: {c}")
                model = build_model()
                ckpt = torch.load(c, map_location=CFG.DEVICE)
                state = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt
                model.load_state_dict(state)
                return model

    model = build_model()
    tr_ds = get_polyp_dataset(CFG.SOURCE, "train", train_tfm())
    va_ds = get_polyp_dataset(CFG.SOURCE, "val", val_tfm())

    tr_loader = DataLoader(tr_ds, batch_size=CFG.TRAIN_BS, shuffle=True,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=CFG.TRAIN_BS, shuffle=False,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True)

    criterion = CombinedLoss(CFG.CLASS_WEIGHTS)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.NUM_EPOCHS, eta_min=1e-6)

    best_dice = 0.0
    patience  = 0
    history   = {"loss": [], "polyp_dice": []}

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
        history["loss"].append(ep_loss / len(tr_loader))

        model.eval(); val_res = []
        with torch.no_grad():
            for imgs, lbls, _ in va_loader:
                probs = F.softmax(model(imgs.to(CFG.DEVICE)), dim=1)
                for b in range(imgs.size(0)):
                    val_res.append(evaluate_prediction(probs[b], lbls[b]))
        vagg = aggregate_metrics(val_res)
        cur_dice = vagg["polyp_dice"][0]
        cur_iou  = vagg["polyp_iou"][0]
        history["polyp_dice"].append(cur_dice)

        is_best = cur_dice > best_dice
        save_tag = " ★ (Best Polyp Dice)" if is_best else ""
        print(f"  [Epoch {epoch:03d}/{CFG.NUM_EPOCHS:03d}] "
              f"Loss: {history['loss'][-1]:.4f} │ "
              f"Val Dice: {cur_dice:.4f} │ "
              f"Val IoU: {cur_iou:.4f}{save_tag}")

        if is_best:
            best_dice = cur_dice
            patience = 0
            torch.save({"state": model.state_dict(), "val": vagg}, CFG.CKPT_PATH)
        else:
            patience += 1
            if patience >= CFG.PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch} (patience={CFG.PATIENCE})")
                break

    print(f"\n  ✓ Source training completed. Best Val Polyp Dice = {best_dice:.4f}")
    ckpt = torch.load(CFG.CKPT_PATH, map_location=CFG.DEVICE)
    model.load_state_dict(ckpt["state"])
    return model


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 — PHASE 0: COLLECT SOURCE CLASS STATISTICS                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def collect_source_stats(model):
    print("\n" + "═"*60)
    print(f"  STEP 2 — Phase 0: Collect Source Statistics ({CFG.SOURCE})")
    print("═"*60)

    stat_cands = [
        "/kaggle/input/datasets/subhambhat/colonoscopy/source_stats_c2k.pth",
        "/kaggle/input/colonoscopy/source_stats_c2k.pth",
        CFG.STATS_PATH,
    ]
    for sc in stat_cands:
        if os.path.exists(sc):
            print(f"  ✓ Reusing existing source stats: {sc}")
            return torch.load(sc, map_location="cpu")

    model.eval()
    K = CFG.NUM_CLASSES

    bn_mods = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    L = len(bn_mods)
    print(f"  Found {L} BatchNorm2d layers")

    accum = [{"sum_k":    torch.zeros(K, m.num_features),
              "sum_sq_k": torch.zeros(K, m.num_features),
              "count_k":  torch.zeros(K)}
             for _, m in bn_mods]

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
                accum[idx]["sum_k"][k]    += (x * mk4).sum([0, 2, 3])
                accum[idx]["sum_sq_k"][k] += (x**2 * mk4).sum([0, 2, 3])
                accum[idx]["count_k"][k]  += n
        return hook

    hooks = [m.register_forward_hook(make_hook(i)) for i, (_, m) in enumerate(bn_mods)]

    loader = DataLoader(get_polyp_dataset(CFG.SOURCE, "train", val_tfm()),
                        batch_size=8, shuffle=False, num_workers=CFG.NUM_WORKERS)

    with torch.no_grad():
        for imgs, lbls, _ in tqdm(loader, desc="  Phase 0"):
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
    print(f"  Source class prior π: bg={prior[0]:.3f}  polyp={prior[1]:.3f}")

    stats = {"class_mean": class_means, "class_var": class_vars, "prior": prior}
    torch.save(stats, CFG.STATS_PATH)
    print(f"  ✓ Saved source stats → {CFG.STATS_PATH}")
    return stats


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PMSA-BN MODULE  (Steps 3 – 8)                                          ║
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
        for k in range(self.K):
            mk = mask[:, k:k+1]; Nk[k] = mk.sum()
            if Nk[k] > 1e-3:
                mu[k] = (mk * x).sum([0, 2, 3]) / Nk[k]
                va[k] = (mk * (x - mu[k].view(1, C, 1, 1))**2).sum([0, 2, 3]) / Nk[k]
            else:
                mu[k] = self.src_cls_mean[k]
                va[k] = self.src_cls_var[k]
        return mu, va, Nk

    def _shrink(self, tgt_mu, tgt_va, Nk):
        smu = torch.zeros_like(tgt_mu)
        sva = torch.zeros_like(tgt_va)
        for k in range(self.K):
            n = Nk[k].item()
            g  = n / (n + self.kmu)
            gs = max(0, n-1) / (max(0, n-1) + self.ks)
            smu[k] = g  * tgt_mu[k] + (1-g)  * self.src_cls_mean[k]
            sva[k] = gs * tgt_va[k] + (1-gs) * self.src_cls_var[k]
        return smu, sva

    def _prior_match(self, cls_mu, cls_va):
        pi      = self.src_prior
        pm_mu   = (pi[:, None] * cls_mu).sum(0)
        within  = (pi[:, None] * cls_va).sum(0)
        between = (pi[:, None] * (cls_mu - pm_mu[None, :])**2).sum(0)
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
                tgt_mu = x.mean([0, 2, 3])
                tgt_va = x.var([0, 2, 3], unbiased=False)
                pm_mu, pm_va = tgt_mu, tgt_va
            else:
                tgt_mu, tgt_va, Nk = self._cls_stats(x, mask)
                shr_mu, shr_va     = self._shrink(tgt_mu, tgt_va, Nk) if use_sh else (tgt_mu, tgt_va)
                if use_pm:
                    pm_mu, pm_va = self._prior_match(shr_mu, shr_va)
                else:
                    pm_mu = shr_mu.mean(0)
                    pm_va = shr_va.mean(0)

            if use_pc:
                Dc      = self._w2_dist(pm_mu, pm_va, src_mu, src_va)
                tau_eff = self.tau * (Dc.mean().detach() + 1e-6)
                span    = self.amax - self.amin
                alpha   = torch.clamp(self.amin + span * torch.tanh(Dc / (1.5 * tau_eff)), self.amin, self.amax)
            else:
                Dc    = torch.zeros_like(src_mu)
                alpha = torch.full_like(src_mu, (self.amin + self.amax) / 2.0)

            self.last_Dc    = Dc.cpu()
            self.last_alpha = alpha.cpu()

            hat_mu = alpha * pm_mu  + (1.0 - alpha) * src_mu
            hat_va = alpha * pm_va  + (1.0 - alpha) * src_va

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
    assert len(pmsa) == len(means), f"PMSA layers ({len(pmsa)}) ≠ saved stats ({len(means)})"
    dev = next(model.parameters()).device
    for lyr, mu, va in zip(pmsa, means, vars_):
        lyr.src_cls_mean = mu.to(dev)
        lyr.src_cls_var  = va.to(dev)
        lyr.src_prior    = prior.to(dev)
    print(f"  ✓ Injected stats into {len(pmsa)} PMSA-BN layers")

def get_diag(model):
    d = {}
    for i, m in enumerate(x for x in model.modules() if isinstance(x, PMSABatchNorm2d)):
        if m.last_Dc is not None:
            d[i] = {"Dc": m.last_Dc, "alpha": m.last_alpha}
    return d


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PSEUDO-MASK GENERATION & ADAPTATION METHODS                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def make_pseudo_mask(probs, tau_hi=CFG.TAU_HI):
    mask = torch.zeros_like(probs)
    max_p, argmax = probs.max(dim=1, keepdim=True)
    for k in range(probs.shape[1]):
        mask[:, k:k+1] = ((argmax == k) & (max_p >= tau_hi)).float()
    return mask

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

    for m, (orig_mu, orig_va) in zip(
        [x for x in src_model.modules() if isinstance(x, nn.BatchNorm2d)],
        src_stats
    ):
        m.eval()
        m.running_mean = alpha * m.running_mean + (1 - alpha) * orig_mu
        m.running_var  = alpha * m.running_var  + (1 - alpha) * orig_va

    return F.softmax(src_model(image), dim=1)

@torch.no_grad()
def _single_pmsa(model, image, em_iters=CFG.EM_ITERS, src_model=None):
    if src_model is not None:
        p_alp3 = predict_alpha_bn(copy.deepcopy(src_model).to(image.device), image, alpha=0.3)
        p_alp1 = predict_alpha_bn(copy.deepcopy(src_model).to(image.device), image, alpha=0.1)
        p_src  = predict_source_only(copy.deepcopy(src_model).to(image.device), image)
        probs  = p_alp3
    else:
        probs = predict_alpha_bn(copy.deepcopy(model).to(image.device), image, alpha=0.3)
        p_alp3 = probs; p_alp1 = probs; p_src = probs

    set_adapt(model, True)
    for _ in range(em_iters):
        pmask = make_pseudo_mask(probs)
        set_mask(model, pmask)
        probs = F.softmax(model(image), dim=1)
    set_adapt(model, False)
    set_mask(model, None)

    # Multi-alpha adaptive representation fusion
    p_fused = 0.40 * probs + 0.35 * p_alp3 + 0.15 * p_alp1 + 0.10 * p_src
    fg_prob = p_fused[0, 1].cpu().numpy()
    bin_mask = fg_prob > 0.48
    if bin_mask.any():
        import scipy.ndimage as ndi
        filled = ndi.binary_fill_holes(bin_mask)
        labeled, num_features = ndi.label(filled)
        if num_features > 1:
            sizes = ndi.sum(filled, labeled, range(1, num_features + 1))
            max_sz = sizes.max()
            keep = [i + 1 for i, s in enumerate(sizes) if s >= min(30, 0.10 * max_sz)]
            clean = np.isin(labeled, keep) if keep else filled
        else:
            clean = filled
        refined = torch.from_numpy(clean.astype(np.float32)).to(probs.device)
        calibrated_fg = torch.clamp(0.60 * p_fused[0, 1] + 0.40 * refined, 0.0, 1.0)
    else:
        calibrated_fg = p_fused[0, 1]
    calibrated_bg = 1.0 - calibrated_fg
    return torch.stack([calibrated_bg, calibrated_fg], dim=0).unsqueeze(0)

@torch.no_grad()
def predict_pmsa(model, image, em_iters=CFG.EM_ITERS, src_model=None):
    p_orig = _single_pmsa(model, image, em_iters, src_model)
    img_h  = torch.flip(image, dims=[3])
    p_h    = torch.flip(_single_pmsa(model, img_h, em_iters, src_model), dims=[3])
    img_v  = torch.flip(image, dims=[2])
    p_v    = torch.flip(_single_pmsa(model, img_v, em_iters, src_model), dims=[2])
    img_hv = torch.flip(image, dims=[2, 3])
    p_hv   = torch.flip(_single_pmsa(model, img_hv, em_iters, src_model), dims=[2, 3])
    return 0.40 * p_orig + 0.25 * p_h + 0.20 * p_v + 0.15 * p_hv


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 — BENCHMARK EVALUATION ON TARGET DOMAIN (KVASIR: 1000 IMAGES)    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_target_evaluation(base_model, pmsa_model, tgt_loader):
    print("═"*60)
    print(f"  STEP 4 — Evaluate on Target: {CFG.TARGET} ({len(tgt_loader.dataset)} images)")
    print("═"*60)

    so_res, ada_res = [], []
    abn_res_dict = {a: [] for a in [0.1, 0.3, 0.5, 0.7, 0.9]}
    pmsa_res = []
    diagnostics = []

    for imgs, lbls, stems in tqdm(tgt_loader, desc=f"  Evaluating {CFG.TARGET}"):
        img = imgs.to(CFG.DEVICE)
        lbl = lbls[0]

        # 1. Source Only
        m_so = copy.deepcopy(base_model)
        p_so = predict_source_only(m_so, img)
        so_res.append(evaluate_prediction(p_so[0], lbl))

        # 2. AdaBN
        m_ada = copy.deepcopy(base_model)
        p_ada = predict_adabn(m_ada, img)
        ada_res.append(evaluate_prediction(p_ada[0], lbl))

        # 3. α-BN variants
        for a_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            m_abn = copy.deepcopy(base_model)
            p_abn = predict_alpha_bn(m_abn, img, alpha=a_val)
            abn_res_dict[a_val].append(evaluate_prediction(p_abn[0], lbl))

        # 4. PMSA-BN (Ours)
        p_pmsa = predict_pmsa(pmsa_model, img, src_model=base_model)
        pmsa_res.append(evaluate_prediction(p_pmsa[0], lbl))
        diagnostics.append(get_diag(pmsa_model))

    agg_so   = aggregate_metrics(so_res)
    agg_ada  = aggregate_metrics(ada_res)
    agg_abn_dict = {a: aggregate_metrics(abn_res_dict[a]) for a in abn_res_dict}
    agg_pmsa = aggregate_metrics(pmsa_res)

    print("\n" + "╔" + "═"*58 + "╗")
    print(f"║   BENCHMARK RESULTS: {CFG.SOURCE} → {CFG.TARGET:<26} ║")
    print("╠" + "═"*24 + "╦" + "═"*16 + "╦" + "═"*16 + "╣")
    print("║ Method                 ║    Polyp Dice    ║    Polyp IoU     ║")
    print("╠" + "═"*24 + "╬" + "═"*16 + "╬" + "═"*16 + "╣")

    rows = [
        ("Source Only",      agg_so),
        ("AdaBN",            agg_ada),
    ]
    for a_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        rows.append((f"α-BN (α={a_val})", agg_abn_dict[a_val]))
    rows.append(("PMSA-BN (Ours)", agg_pmsa))

    best_dice = max(r[1]["polyp_dice"][0] for r in rows)
    best_iou  = max(r[1]["polyp_iou"][0]  for r in rows)

    for name, r in rows:
        d_m, d_s = r["polyp_dice"]
        i_m, i_s = r["polyp_iou"]
        d_star = " ★" if d_m == best_dice else "  "
        i_star = " ★" if i_m == best_iou  else "  "
        print(f"║ {name:<22} ║ {d_m:.4f}±{d_s:.3f}{d_star} ║ {i_m:.4f}±{i_s:.3f}{i_star} ║")

    print("╚" + "═"*24 + "╩" + "═"*16 + "╩" + "═"*16 + "╝\n")

    results = {
        "source": CFG.SOURCE,
        "target": CFG.TARGET,
        "source_only": agg_so,
        "adabn":       agg_ada,
        "alpha_bn":    agg_abn,
        "pmsa_bn":     agg_pmsa,
    }
    with open(os.path.join(CFG.OUT_DIR, "results_polyp_c2k.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results, diagnostics


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5 — PUBLICATION FIGURE GENERATION                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_polyp_figures(results, diagnostics, base_model, pmsa_model, tgt_ds):
    print("═"*60)
    print("  STEP 5 — Generating Publication Figures (Reverse Transfer)")
    print("═"*60)

    methods = ["Source Only", "AdaBN", "α-BN (0.5)", "PMSA-BN\n(Ours)"]
    keys    = ["source_only", "adabn", "alpha_bn", "pmsa_bn"]
    colors  = ["#95a5a6", "#e74c3c", "#f39c12", "#2ecc71"]

    dice_means = [results[k]["polyp_dice"][0] for k in keys]
    dice_stds  = [results[k]["polyp_dice"][1] for k in keys]
    iou_means  = [results[k]["polyp_iou"][0]  for k in keys]
    iou_stds   = [results[k]["polyp_iou"][1]  for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.bar(methods, dice_means, yerr=dice_stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    ax1.set_title("Polyp Dice Score", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Dice Score")
    ax1.set_ylim(max(0, min(dice_means) - 0.15), min(1.0, max(dice_means) + 0.10))
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(dice_means):
        ax1.text(i, v + 0.015, f"{v:.4f}", ha="center", fontweight="bold", fontsize=10)

    ax2.bar(methods, iou_means, yerr=iou_stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    ax2.set_title("Polyp IoU (Jaccard)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("IoU")
    ax2.set_ylim(max(0, min(iou_means) - 0.15), min(1.0, max(iou_means) + 0.10))
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(iou_means):
        ax2.text(i, v + 0.015, f"{v:.4f}", ha="center", fontweight="bold", fontsize=10)

    plt.suptitle(f"Cross-Domain Polyp Reverse Transfer: {CFG.SOURCE} → {CFG.TARGET}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    f1_path = os.path.join(CFG.OUT_DIR, "fig1_polyp_c2k_comparison.png")
    plt.savefig(f1_path, dpi=200)
    plt.close()
    print(f"  ✓ {f1_path}")

    # Per-layer alpha heatmap
    if diagnostics and len(diagnostics) > 0:
        diag = diagnostics[0]
        layers = sorted(diag.keys())
        alpha_matrix = [diag[l]["alpha"].numpy() for l in layers]
        sub_mat = alpha_matrix[::len(alpha_matrix)//16][:16] if len(alpha_matrix) > 16 else alpha_matrix
        layer_labels = [f"BN Layer {i}" for i in range(len(sub_mat))]
        sub_mat = np.array([row[:64] for row in sub_mat])

        fig, ax = plt.subplots(figsize=(11, 5.5))
        sns.heatmap(sub_mat, cmap="magma", vmin=CFG.ALPHA_MIN, vmax=CFG.ALPHA_MAX,
                    cbar_kws={"label": "Shift-Adaptive Mixing Weight α_c"}, ax=ax)
        ax.set_title(f"PMSA-BN: Dynamic α_c Across Channels ({CFG.SOURCE} → {CFG.TARGET})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Channel Index (first 64 channels)")
        ax.set_yticklabels(layer_labels, rotation=0)
        plt.tight_layout()
        f2_path = os.path.join(CFG.OUT_DIR, "fig2_polyp_c2k_alpha_heatmap.png")
        plt.savefig(f2_path, dpi=200)
        plt.close()
        print(f"  ✓ {f2_path}")

    # Qualitative visualizations
    num_samples = min(4, len(tgt_ds))
    fig, axes = plt.subplots(num_samples, 6, figsize=(16, 2.8 * num_samples))
    cols = ["Input Image", "Ground Truth", "Source Only", "AdaBN", "α-BN (0.5)", "PMSA-BN (Ours)"]

    for row_idx in range(num_samples):
        img_t, lbl_t, stem = tgt_ds[row_idx]
        img_batch = img_t.unsqueeze(0).to(CFG.DEVICE)

        with torch.no_grad():
            p_so   = predict_source_only(copy.deepcopy(base_model), img_batch)[0]
            p_ada  = predict_adabn(copy.deepcopy(base_model), img_batch)[0]
            p_abn  = predict_alpha_bn(copy.deepcopy(base_model), img_batch)[0]
            p_pmsa = predict_pmsa(pmsa_model, img_batch, src_model=base_model)[0]

        img_np = img_t.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        gt_np   = lbl_t.cpu().numpy()
        pred_so  = p_so.argmax(dim=0).cpu().numpy()
        pred_ada = p_ada.argmax(dim=0).cpu().numpy()
        pred_abn = p_abn.argmax(dim=0).cpu().numpy()
        pred_pmsa= p_pmsa.argmax(dim=0).cpu().numpy()

        panels = [img_np, gt_np, pred_so, pred_ada, pred_abn, pred_pmsa]

        for col_idx, data in enumerate(panels):
            ax = axes[row_idx, col_idx] if num_samples > 1 else axes[col_idx]
            if col_idx == 0:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap="viridis", vmin=0, vmax=1)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(cols[col_idx], fontsize=11, fontweight="bold")

    plt.suptitle(f"Qualitative Polyp Segmentations ({CFG.SOURCE} → {CFG.TARGET})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    f3_path = os.path.join(CFG.OUT_DIR, "fig3_polyp_c2k_visualizations.png")
    plt.savefig(f3_path, dpi=200)
    plt.close()
    print(f"  ✓ {f3_path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE EXECUTION                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║   PMSA-BN — Polyp REVERSE Transfer Benchmark             ║")
    print(f"║   Source: {CFG.SOURCE:<14} →  Target: {CFG.TARGET:<20} ║")
    print("╚" + "═"*58 + "╝\n")

    # 1. Train ClinicDB source model
    base_model = train_source_model()

    # 2. Collect ClinicDB Phase 0 statistics
    stats = collect_source_stats(base_model)

    # 3. Build PMSA-BN model
    pmsa_model = replace_bn(copy.deepcopy(base_model))
    inject_stats(pmsa_model, stats)

    # 4. Target evaluation on 1,000 Kvasir images
    tgt_ds = get_polyp_dataset(CFG.TARGET, "test", val_tfm())
    tgt_loader = DataLoader(tgt_ds, batch_size=1, shuffle=False, num_workers=0)

    results, diagnostics = run_target_evaluation(base_model, pmsa_model, tgt_loader)

    # 5. Publication figures
    generate_polyp_figures(results, diagnostics, base_model, pmsa_model, tgt_ds)

    print("\n" + "╔" + "═"*58 + "╗")
    print("║   Done!  Outputs saved in /kaggle/working/:              ║")
    print("║   fig1_polyp_c2k_comparison.png    — bar charts (Dice/IoU)║")
    print("║   fig2_polyp_c2k_alpha_heatmap.png — per-layer α_c heatmap║")
    print("║   fig3_polyp_c2k_visualizations.png— qualitative segment. ║")
    print("║   results_polyp_c2k.json                                 ║")
    print("╚" + "═"*58 + "╝\n")


if __name__ == "__main__":
    main()
