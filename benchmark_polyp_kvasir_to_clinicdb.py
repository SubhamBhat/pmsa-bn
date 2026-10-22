"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm                     ║
║         Single-Cell Kaggle Script — Endoscopic Polyp Cross-Domain TTA       ║
║                                                                              ║
║  DATASETS (add both as Kaggle inputs):                                       ║
║    1. debeshjha1/kvasirseg  (Kvasir-SEG: 1,000 colonoscopy images)           ║
║    2. balraj98/cvcclinicdb  (CVC-ClinicDB: 612 colonoscopy frames)          ║
║                                                                              ║
║  BENCHMARK:                                                                  ║
║    Source Domain: Kvasir-SEG  (Olympus endoscopes, Vestre Viken, Norway)    ║
║    Target Domain: CVC-ClinicDB (Pentax endoscopes, Hospital Clínic, Spain)   ║
║    (or vice versa by changing CFG.SOURCE and CFG.TARGET)                    ║
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
# ║  CFG — Configuration & Domain Switcher                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CFG:
    # ── Source & Target Domain Selection ──────────────────────────────────
    # Choose from: "Kvasir", "ClinicDB"
    SOURCE = "Kvasir"       # Source domain (trains from scratch + collects Phase 0 statistics)
    TARGET = "ClinicDB"     # Target domain (evaluates test-time adaptation methods)

    # ── Output ────────────────────────────────────────────────────────────
    CKPT_PATH  = "/kaggle/working/best_model_polyp.pth"
    STATS_PATH = "/kaggle/working/source_stats_polyp.pth"
    OUT_DIR    = "/kaggle/working"

    # ── Model ─────────────────────────────────────────────────────────────
    ENCODER         = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES     = 2        # 0=background (mucosa), 1=polyp
    IMAGE_SIZE      = 352      # Standard 352×352 resolution for polyp benchmarks (PraNet)

    # ── Training ──────────────────────────────────────────────────────────
    TRAIN_BS           = 8
    NUM_EPOCHS         = 50       # 50 epochs for full convergence (~4-5 mins on GPU)
    LR                 = 1e-4
    WEIGHT_DECAY       = 1e-5
    PATIENCE           = 12
    CLASS_WEIGHTS      = [0.2, 1.8]   # Background (dominant) vs Polyp (sparse foreground)
    TRAIN_FROM_SCRATCH = False      # Set False to reuse /kaggle/working/best_model_polyp.pth

    # ── PMSA-BN Hyperparameters ───────────────────────────────────────────
    TAU_HI      = 0.50   # confident pseudo-mask threshold
    KAPPA_MU    = 16.0   # Bayesian shrinkage pseudo-count (mean)
    KAPPA_SIGMA = 32.0   # Bayesian shrinkage pseudo-count (variance)
    ALPHA_TAU   = 1.0    # temperature τ for shift sensitivity
    ALPHA_MIN   = 0.0    # zero-shift channels stay 100% on robust source stats
    ALPHA_MAX   = 0.45   # upper bound for shift-adapted channels (avoids single-image variance explosion)
    EM_ITERS    = 1      # single refinement pass (avoids confirmation bias at B=1)

    # ── Baselines ─────────────────────────────────────────────────────────
    ALPHA_BN_WEIGHT = 0.5
    RUN_ABLATION    = False  # Suspended for rapid iteration; set True for final paper ablation table

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
    """Search candidate paths first, then recursively search /kaggle/input."""
    if candidate_paths:
        for p in candidate_paths:
            if os.path.exists(p):
                return p
    search_root = "/kaggle/input"
    if os.path.exists(search_root):
        for root, dirs, _ in os.walk(search_root):
            for d in dirs:
                full_d = os.path.join(root, d)
                if all(kw.lower() in full_d.lower() for kw in keywords):
                    return full_d
    return None


def get_kvasir_paths():
    """
    Auto-discovers Kvasir-SEG image and mask files.
    Candidate paths cover:
      - /kaggle/input/datasets/debeshjha1/kvasirseg/Kvasir-SEG/Kvasir-SEG/
      - /kaggle/input/kvasirseg/Kvasir-SEG/
      - /kaggle/input/kvasir-seg/
    """
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
        raise FileNotFoundError("Could not locate Kvasir-SEG dataset in /kaggle/input. "
                                "Please add 'debeshjha1/kvasirseg' to your Kaggle session.")

    # Search for images folder
    img_dirs = []
    mask_dirs = []
    for root, dirs, files in os.walk(base):
        if os.path.basename(root).lower() == "images" and len(files) > 50:
            img_dirs.append(root)
        if os.path.basename(root).lower() == "masks" and len(files) > 50:
            mask_dirs.append(root)

    img_dir = img_dirs[0] if img_dirs else os.path.join(base, "images")
    mask_dir = mask_dirs[0] if mask_dirs else os.path.join(base, "masks")

    # Match images with masks by stem
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
    """
    Auto-discovers CVC-ClinicDB image and mask files.
    Candidate paths cover:
      - /kaggle/input/datasets/balraj98/cvcclinicdb/PNG/
      - /kaggle/input/cvcclinicdb/PNG/
      - /kaggle/input/balraj98/cvcclinicdb
    """
    cands = [
        "/kaggle/input/datasets/balraj98/cvcclinicdb/PNG",
        "/kaggle/input/datasets/balraj98/cvcclinicdb",
        "/kaggle/input/cvcclinicdb/PNG",
        "/kaggle/input/cvcclinicdb",
        "/kaggle/input/cvc-clinicdb/PNG",
        "/kaggle/input/cvc-clinicdb",
    ]
    base = auto_find_dir(["clinic"], cands)
    if not base:
        base = auto_find_dir(["cvc"], cands)
    if not base:
        raise FileNotFoundError("Could not locate CVC-ClinicDB dataset in /kaggle/input. "
                                "Please add 'balraj98/cvcclinicdb' to your Kaggle session.")

    # Locate Original and Ground Truth folders
    orig_dirs = []
    gt_dirs = []
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


# ── Binary Mask Loader Helper ────────────────────────────────────────────────

def load_polyp_mask(mask_path: str, target_shape=None) -> np.ndarray:
    """
    Loads binary polyp mask: 0 = background, 1 = polyp.
    Handles any resolution, PNG/JPG formats, and compression noise.
    """
    if not mask_path or not os.path.exists(mask_path):
        if target_shape is not None:
            return np.zeros(target_shape, dtype=np.int64)
        return np.zeros((352, 352), dtype=np.int64)

    mask_img = Image.open(mask_path).convert("L")
    if target_shape is not None:
        H, W = target_shape
        if mask_img.size != (W, H):
            mask_img = mask_img.resize((W, H), resample=Image.NEAREST)
    mask = np.array(mask_img)

    # Threshold: values > 127 are polyp foreground
    label = (mask > 127).astype(np.int64)
    return label


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATASETS & TRANSFORMS                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

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
    """Unified Dataset for Kvasir-SEG and CVC-ClinicDB."""
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

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
    """
    Returns dataset instance with deterministic 80/20 train/val split for source domain,
    or complete set for target evaluation.
    """
    domain = domain_name.upper()
    if "KVASIR" in domain:
        pairs, _ = get_kvasir_paths()
    elif "CLINIC" in domain or "CVC" in domain:
        pairs, _ = get_clinicdb_paths()
    else:
        raise ValueError(f"Unknown domain: {domain_name}. Choose 'Kvasir' or 'ClinicDB'.")

    # Deterministic split (80% train, 20% val)
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
    print("  Polyp Dataset Sanity Check")
    print("─"*58)
    print(f"  Configured: SOURCE = {CFG.SOURCE}  →  TARGET = {CFG.TARGET}")

    # Check Source
    try:
        s_ds_tr = get_polyp_dataset(CFG.SOURCE, "train", val_tfm())
        s_ds_va = get_polyp_dataset(CFG.SOURCE, "val", val_tfm())
        print(f"  Source ({CFG.SOURCE}): {len(s_ds_tr)} train / {len(s_ds_va)} val images")
        img, lbl, stem = s_ds_tr[0]
        u, c = np.unique(lbl.numpy(), return_counts=True)
        dist = {int(k): f"{v/lbl.numel()*100:.1f}%" for k, v in zip(u, c)}
        print(f"  Source train OK │ classes: {dist} │ sample: {stem}")
    except Exception as e:
        print(f"  [ERROR] Source ({CFG.SOURCE}) check failed: {e}")

    # Check Target
    try:
        t_ds = get_polyp_dataset(CFG.TARGET, "test", val_tfm())
        print(f"  Target ({CFG.TARGET}): {len(t_ds)} test images")
        img, lbl, stem = t_ds[0]
        u, c = np.unique(lbl.numpy(), return_counts=True)
        dist = {int(k): f"{v/lbl.numel()*100:.1f}%" for k, v in zip(u, c)}
        print(f"  Target test OK │ classes: {dist} │ sample: {stem}")
    except Exception as e:
        print(f"  [ERROR] Target ({CFG.TARGET}) check failed: {e}")

    print("─"*58 + "\n")

sanity_check()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EVALUATION METRICS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_polyp_metrics(pred_labels: np.ndarray, gt: np.ndarray, smooth=1e-6):
    """
    Computes standard Polyp Segmentation benchmarks:
      - Polyp Dice Score: 2 |P ∩ G| / (|P| + |G|)
      - Polyp IoU (Jaccard): |P ∩ G| / (|P ∪ G|)
    """
    p = (pred_labels == 1).astype(float)
    g = (gt == 1).astype(float)

    intersection = (p * g).sum()
    union = p.sum() + g.sum() - intersection

    dice = (2.0 * intersection + smooth) / (p.sum() + g.sum() + smooth)
    iou  = (intersection + smooth) / (union + smooth)

    return {"polyp_dice": float(dice), "polyp_iou": float(iou)}


def evaluate_prediction(pred_probs: torch.Tensor, gt_label: torch.Tensor):
    """Evaluates a single model prediction probability tensor [K, H, W] vs ground truth."""
    pred_labels = pred_probs.argmax(dim=0).cpu().numpy()
    gt = gt_label.cpu().numpy()
    return compute_polyp_metrics(pred_labels, gt)


def aggregate_metrics(res_list):
    """Computes mean and std over list of metric dicts."""
    dices = [r["polyp_dice"] for r in res_list]
    ious  = [r["polyp_iou"]  for r in res_list]
    return {
        "polyp_dice": (float(np.mean(dices)), float(np.std(dices))),
        "polyp_iou":  (float(np.mean(ious)),  float(np.std(ious)))
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 — TRAIN SOURCE MODEL                                             ║
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
        # Average over foreground classes (class 1: polyp)
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
    """Instantiates ResNet34-UNet for 2-class polyp segmentation."""
    try:
        model = smp.Unet(
            encoder_name=CFG.ENCODER,
            encoder_weights=CFG.ENCODER_WEIGHTS,
            classes=CFG.NUM_CLASSES,
            activation=None
        )
    except Exception:
        print("  [WARN] Downloading weights failed. Initializing random encoder.")
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

    # Check if existing checkpoint can be reused
    if not CFG.TRAIN_FROM_SCRATCH and os.path.exists(CFG.CKPT_PATH):
        print(f"  ✓ Reusing existing model checkpoint: {CFG.CKPT_PATH}")
        model = build_model()
        ckpt = torch.load(CFG.CKPT_PATH, map_location=CFG.DEVICE)
        model.load_state_dict(ckpt["state"])
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

        # Validation pass
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
            x = inp[0].detach().cpu()   # [B, C, H, W]
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
    """
    Drop-in replacement for nn.BatchNorm2d for test-time polyp adaptation.
    Canonical 8-step Bayesian formulation.
    """
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
        self.pseudo_mask = None   # [1, K, H, W]
        self.last_Dc = None; self.last_alpha = None

    def _cls_stats(self, x, mask):
        """Step 3: Class-conditional statistics for background and polyp"""
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
                # Robust fallback to source class prior when polyp is tiny/absent
                mu[k] = self.src_cls_mean[k]
                va[k] = self.src_cls_var[k]
        return mu, va, Nk

    def _shrink(self, tgt_mu, tgt_va, Nk):
        """Step 4: Empirical Bayes shrinkage toward source priors"""
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
        """Step 5: Prior-matched recombination removing label-proportion confound"""
        pi      = self.src_prior                                  # [K]
        pm_mu   = (pi[:, None] * cls_mu).sum(0)                   # [C]
        within  = (pi[:, None] * cls_va).sum(0)
        between = (pi[:, None] * (cls_mu - pm_mu[None, :])**2).sum(0)
        return pm_mu, within + between

    def _w2_dist(self, pm_mu, pm_va, src_mu, src_va, eps=1e-6):
        """Step 6: Scale-free 2-Wasserstein distance per channel"""
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
                Dc    = self._w2_dist(pm_mu, pm_va, src_mu, src_va)
                span  = self.amax - self.amin
                alpha = torch.clamp(self.amin + span * torch.tanh(Dc / (1.5 * self.tau)), self.amin, self.amax)
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
    """Generates high-confidence pseudo-mask for polyp class."""
    mask = torch.zeros_like(probs)
    max_p, argmax = probs.max(dim=1, keepdim=True)
    for k in range(probs.shape[1]):
        mask[:, k:k+1] = ((argmax == k) & (max_p >= tau_hi)).float()
    return mask


@torch.no_grad()
def predict_source_only(model, image):
    """Baseline 1 — Source Only (unadapted)."""
    model.eval()
    return F.softmax(model(image), dim=1)


@torch.no_grad()
def predict_adabn(model, image):
    """Baseline 2 — AdaBN (single image test statistics, momentum=1)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train(); m.momentum = 1.0
    _ = model(image)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()
    return F.softmax(model(image), dim=1)


@torch.no_grad()
def predict_alpha_bn(src_model, image, alpha=CFG.ALPHA_BN_WEIGHT):
    """Baseline 3 — α-BN (global fixed α=0.5 blending)."""
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
def predict_pmsa(model, image, em_iters=CFG.EM_ITERS, src_model=None):
    """
    PMSA-BN (Ours) — 2-pass test-time adaptation algorithm.
    Pass 1: α-BN warm-start coarse prediction to initialize pseudo-mask
    Pass 2: Adapted forward pass through PMSA-BN layers (Steps 3–8)
    """
    # Pass 1: Warm-start
    if src_model is not None:
        probs = predict_alpha_bn(copy.deepcopy(src_model).to(image.device), image)
    else:
        probs = predict_alpha_bn(copy.deepcopy(model).to(image.device), image)

    # Pass 2: PMSA-BN adaptation
    set_adapt(model, True)
    for _ in range(em_iters):
        pmask = make_pseudo_mask(probs)
        set_mask(model, pmask)
        probs = F.softmax(model(image), dim=1)
    set_adapt(model, False)
    set_mask(model, None)
    return probs


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 — BENCHMARK EVALUATION ON TARGET DOMAIN                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_target_evaluation(base_model, pmsa_model, tgt_loader):
    print("═"*60)
    print(f"  STEP 4 — Evaluate on Target: {CFG.TARGET} ({len(tgt_loader.dataset)} images)")
    print("═"*60)

    so_res, ada_res, abn_res, pmsa_res = [], [], [], []
    diagnostics = []

    for imgs, lbls, stems in tqdm(tgt_loader, desc=f"  Evaluating {CFG.TARGET}"):
        img  = imgs.to(CFG.DEVICE)   # [1, 3, H, W]
        lbl  = lbls[0]               # [H, W]

        # 1. Source Only
        m_so = copy.deepcopy(base_model)
        p_so = predict_source_only(m_so, img)
        so_res.append(evaluate_prediction(p_so[0], lbl))

        # 2. AdaBN
        m_ada = copy.deepcopy(base_model)
        p_ada = predict_adabn(m_ada, img)
        ada_res.append(evaluate_prediction(p_ada[0], lbl))

        # 3. α-BN (α=0.5)
        m_abn = copy.deepcopy(base_model)
        p_abn = predict_alpha_bn(m_abn, img, alpha=CFG.ALPHA_BN_WEIGHT)
        abn_res.append(evaluate_prediction(p_abn[0], lbl))

        # 4. PMSA-BN (Ours)
        p_pmsa = predict_pmsa(pmsa_model, img, src_model=base_model)
        pmsa_res.append(evaluate_prediction(p_pmsa[0], lbl))
        diagnostics.append(get_diag(pmsa_model))

    agg_so   = aggregate_metrics(so_res)
    agg_ada  = aggregate_metrics(ada_res)
    agg_abn  = aggregate_metrics(abn_res)
    agg_pmsa = aggregate_metrics(pmsa_res)

    print("\n" + "╔" + "═"*58 + "╗")
    print(f"║   BENCHMARK RESULTS: {CFG.SOURCE} → {CFG.TARGET:<26} ║")
    print("╠" + "═"*24 + "╦" + "═"*16 + "╦" + "═"*16 + "╣")
    print("║ Method                 ║    Polyp Dice    ║    Polyp IoU     ║")
    print("╠" + "═"*24 + "╬" + "═"*16 + "╬" + "═"*16 + "╣")

    rows = [
        ("Source Only",    agg_so),
        ("AdaBN",          agg_ada),
        ("α-BN (α=0.5)",   agg_abn),
        ("PMSA-BN (Ours)", agg_pmsa),
    ]

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
    with open(os.path.join(CFG.OUT_DIR, "results_polyp.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results, diagnostics


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5 — PUBLICATION FIGURE GENERATION                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_polyp_figures(results, diagnostics, base_model, pmsa_model, tgt_ds):
    print("═"*60)
    print("  STEP 5 — Generating Publication Figures")
    print("═"*60)

    # ── Figure 1: Performance Comparison Bar Chart ─────────────────────────
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

    plt.suptitle(f"Cross-Domain Polyp Adaptation: {CFG.SOURCE} → {CFG.TARGET}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    f1_path = os.path.join(CFG.OUT_DIR, "fig1_polyp_comparison.png")
    plt.savefig(f1_path, dpi=200)
    plt.close()
    print(f"  ✓ {f1_path}")

    # ── Figure 2: Dynamic α_c Distribution Heatmap ─────────────────────────
    if diagnostics and len(diagnostics) > 0:
        diag = diagnostics[0]
        layers = sorted(diag.keys())
        alpha_matrix = []
        for l in layers:
            alpha_matrix.append(diag[l]["alpha"].numpy())

        # Select a representative subset of layers if many
        if len(alpha_matrix) > 16:
            step = len(alpha_matrix) // 16
            sub_mat = alpha_matrix[::step][:16]
            layer_labels = [f"BN Layer {i*step}" for i in range(len(sub_mat))]
        else:
            sub_mat = alpha_matrix
            layer_labels = [f"BN Layer {i}" for i in range(len(sub_mat))]

        # Truncate channels to first 64 for clean visualization
        sub_mat = np.array([row[:64] for row in sub_mat])

        fig, ax = plt.subplots(figsize=(11, 5.5))
        sns.heatmap(sub_mat, cmap="magma", vmin=CFG.ALPHA_MIN, vmax=CFG.ALPHA_MAX,
                    cbar_kws={"label": "Shift-Adaptive Mixing Weight α_c"}, ax=ax)
        ax.set_title("PMSA-BN: Dynamic Shift-Adaptive Mixing Weight (α_c) Across Channels",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Channel Index (first 64 channels)")
        ax.set_yticklabels(layer_labels, rotation=0)
        plt.tight_layout()
        f2_path = os.path.join(CFG.OUT_DIR, "fig2_polyp_alpha_heatmap.png")
        plt.savefig(f2_path, dpi=200)
        plt.close()
        print(f"  ✓ {f2_path}")

    # ── Figure 3: Qualitative Segmentation Visualizations ──────────────────
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

        # Convert image tensor back to displayable RGB
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
    f3_path = os.path.join(CFG.OUT_DIR, "fig3_polyp_visualizations.png")
    plt.savefig(f3_path, dpi=200)
    plt.close()
    print(f"  ✓ {f3_path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  OPTIONAL ABLATION STUDY                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_ablation_study(base_model, stats, tgt_loader):
    if not CFG.RUN_ABLATION:
        print("\n  [INFO] Ablation study suspended to save time. (Set CFG.RUN_ABLATION = True to re-enable)")
        return

    print("\n" + "═"*60)
    print("  STEP 6 — Full PMSA-BN Ablation Study")
    print("═"*60)

    ABLATION_CONFIGS = [
        ("Source Only",        {"class_split": False, "shrinkage": False, "prior_match": False, "per_channel": False, "_mode": "so"}),
        ("AdaBN",              {"class_split": False, "shrinkage": False, "prior_match": False, "per_channel": False, "_mode": "adabn"}),
        ("+ Class-Split",      {"class_split": True,  "shrinkage": False, "prior_match": False, "per_channel": False}),
        ("+ Prior Matching",  {"class_split": True,  "shrinkage": False, "prior_match": True,  "per_channel": False}),
        ("+ Shrinkage (κ)",    {"class_split": True,  "shrinkage": True,  "prior_match": True,  "per_channel": False}),
        ("Full PMSA-BN",       {"class_split": True,  "shrinkage": True,  "prior_match": True,  "per_channel": True}),
    ]

    ablation_res = {}
    for name, cfg_flags in ABLATION_CONFIGS:
        mode = cfg_flags.get("_mode", "pmsa")
        cur_results = []
        if mode == "pmsa":
            m_test = replace_bn(copy.deepcopy(base_model))
            inject_stats(m_test, stats)
            for m in m_test.modules():
                if isinstance(m, PMSABatchNorm2d): m.ablation_flags = cfg_flags

        for imgs, lbls, _ in tqdm(tgt_loader, desc=f"  Ablation: {name}"):
            img = imgs.to(CFG.DEVICE); lbl = lbls[0]
            if mode == "so":
                probs = predict_source_only(copy.deepcopy(base_model), img)
            elif mode == "adabn":
                probs = predict_adabn(copy.deepcopy(base_model), img)
            else:
                probs = predict_pmsa(m_test, img, src_model=base_model)
            cur_results.append(evaluate_prediction(probs[0], lbl))

        agg = aggregate_metrics(cur_results)
        ablation_res[name] = agg
        print(f"  {name:<22} │ Dice: {agg['polyp_dice'][0]:.4f}±{agg['polyp_dice'][1]:.3f} │ IoU: {agg['polyp_iou'][0]:.4f}±{agg['polyp_iou'][1]:.3f}")

    with open(os.path.join(CFG.OUT_DIR, "ablation_polyp.json"), "w") as f:
        json.dump(ablation_res, f, indent=2)

    # Plot ablation figure
    names = list(ablation_res.keys())
    dices = [ablation_res[n]["polyp_dice"][0] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(names, dices, color="cornflowerblue", edgecolor="black", height=0.6)
    ax.set_xlabel("Polyp Dice Score", fontweight="bold")
    ax.set_title("Component-Wise Ablation of PMSA-BN", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    f4_path = os.path.join(CFG.OUT_DIR, "fig4_polyp_ablation.png")
    plt.savefig(f4_path, dpi=200)
    plt.close()
    print(f"  ✓ {f4_path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE EXECUTION                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║   PMSA-BN — Polyp Cross-Domain TTA Benchmark             ║")
    print(f"║   Source: {CFG.SOURCE:<14} →  Target: {CFG.TARGET:<20} ║")
    print("╚" + "═"*58 + "╝\n")

    # 1. Train source model
    base_model = train_source_model()

    # 2. Collect Phase 0 statistics
    stats = collect_source_stats(base_model)

    # 3. Build PMSA-BN model
    pmsa_model = replace_bn(copy.deepcopy(base_model))
    inject_stats(pmsa_model, stats)

    # 4. Target evaluation
    tgt_ds = get_polyp_dataset(CFG.TARGET, "test", val_tfm())
    tgt_loader = DataLoader(tgt_ds, batch_size=1, shuffle=False, num_workers=0)

    results, diagnostics = run_target_evaluation(base_model, pmsa_model, tgt_loader)

    # 5. Publication figures
    generate_polyp_figures(results, diagnostics, base_model, pmsa_model, tgt_ds)

    # 6. Ablation study
    run_ablation_study(base_model, stats, tgt_loader)

    print("\n" + "╔" + "═"*58 + "╗")
    print("║   Done!  All outputs generated in /kaggle/working/       ║")
    print("║   fig1_polyp_comparison.png    — bar charts (Dice & IoU) ║")
    print("║   fig2_polyp_alpha_heatmap.png — per-layer α_c heatmap   ║")
    print("║   fig3_polyp_visualizations.png— qualitative predictions ║")
    print("║   results_polyp.json                                     ║")
    print("╚" + "═"*58 + "╝\n")


if __name__ == "__main__":
    main()
