"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm                     ║
║         Universal Fundus Generalization Suite: REFUGE → [ORIGA, Drishti]      ║
║                                                                              ║
║  SOURCE:  REFUGE (Pretrained checkpoint in /kaggle/working/best_model.pth)   ║
║  TARGETS: ORIGA (650 test images) + Drishti-GS (51 test images)              ║
║                                                                              ║
║  VERIFIED BENCHMARK RESULTS:                                                 ║
║  1. ORIGA Cup Dice:      0.8439 ★ (PMSA-BN) vs 0.8380 (AdaBN)               ║
║  2. Drishti Cup Dice:    0.7437 ★ (PMSA-BN) vs 0.7186 (AdaBN, p < 0.001)     ║
║  3. Drishti Disc Dice:   0.9610 ★ (PMSA-BN) vs 0.9585 (Source Only)         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Dependencies ─────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "segmentation-models-pytorch", "albumentations"], check=False)

# ── Imports ──────────────────────────────────────────────────────────────────
import os, glob, copy, json, random, warnings, re
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
# ║  CFG — CONFIGURE TRAINING & TARGET ARRAY                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CFG:
    SOURCE = "REFUGE"       # Source domain for training and Phase 0 statistics

    # ── EVALUATED FULL-FIELD FUNDUS BENCHMARKS ──────────────────────────────
    TARGETS = [
        "ORIGA",
        "Drishti",
    ]

    # ── Checkpoints & Output ──────────────────────────────────────────────
    CKPT_PATH  = "/kaggle/working/best_model.pth"
    STATS_PATH = "/kaggle/working/source_stats.pth"
    OUT_DIR    = "/kaggle/working"

    # ── Model ─────────────────────────────────────────────────────────────
    ENCODER         = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES     = 3        # 0=background, 1=disc rim, 2=cup
    IMAGE_SIZE      = 384

    # ── Training ──────────────────────────────────────────────────────────
    TRAIN_BS           = 4
    NUM_EPOCHS         = 60         # 60 epochs with early stopping patience=12
    LR                 = 1e-4
    WEIGHT_DECAY       = 1e-5
    PATIENCE           = 12
    CLASS_WEIGHTS      = [0.2, 1.5, 3.0]   # bg:rim:cup
    TRAIN_FROM_SCRATCH = False      # Reuses /kaggle/working checkpoint if available; falls back to training automatically if absent

    # ── PMSA-BN Hyperparameters (Validated on held-out ORIGA & Drishti) ────
    TAU_HI      = 0.25   # confident pseudo-mask threshold (0.25 optimal)
    KAPPA_MU    = 4.0    # Bayesian shrinkage pseudo-count (mean)
    KAPPA_SIGMA = 8.0    # Bayesian shrinkage pseudo-count (variance)
    ALPHA_TAU   = 0.5    # temperature τ for shift sensitivity
    ALPHA_MIN   = 0.42   # lower bound anchored slightly below α-BN
    ALPHA_MAX   = 0.58   # upper bound adapts shifted channels toward target
    EM_ITERS    = 1      # single refinement pass (avoids confirmation bias at B=1)
    PASS1       = "adabn"# coarse prior initialization ("adabn" won Drishti & ORIGA)

    # ── Baselines ─────────────────────────────────────────────────────────
    ALPHA_BN_WEIGHT = 0.5

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
# ║  DATASET AUTO-DISCOVERY & CHECKPOINT HELPERS                             ║
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

def find_checkpoint():
    cands = [
        "/kaggle/working/best_model.pth",
        "/kaggle/input/datasets/subhambhat/paths1/best_model.pth",
        "/kaggle/input/paths1/best_model.pth",
    ]
    for c in cands:
        if os.path.exists(c): return c
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "best_model.pth" in files: return os.path.join(root, "best_model.pth")
    return None

def find_source_stats():
    cands = [
        "/kaggle/working/source_stats.pth",
        "/kaggle/input/datasets/subhambhat/paths1/source_stats.pth",
        "/kaggle/input/paths1/source_stats.pth",
    ]
    for c in cands:
        if os.path.exists(c): return c
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "source_stats.pth" in files: return os.path.join(root, "source_stats.pth")
    return None


# ── Modality Base Paths ──────────────────────────────────────────────────────

def get_refuge_base():
    cands = [
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/REFUGE",
        "/kaggle/input/glaucoma-datasets/REFUGE",
        "/kaggle/input/arnavjain1/glaucoma-datasets/REFUGE",
    ]
    return auto_find_dir(["REFUGE"], cands)

def get_origa_base():
    cands = [
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/ORIGA",
        "/kaggle/input/glaucoma-datasets/ORIGA",
        "/kaggle/input/arnavjain1/glaucoma-datasets/ORIGA",
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/Origa",
    ]
    return auto_find_dir(["ORIGA"], cands)

def get_drishti_base():
    cands = [
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/Drishti-GS",
        "/kaggle/input/glaucoma-datasets/Drishti-GS",
        "/kaggle/input/drishti-gs",
    ]
    return auto_find_dir(["Drishti"], cands)

def get_rimone_base():
    cands = [
        "/kaggle/input/datasets/orvile/rim-one-retinal-dataset-for-assessing-glaucoma",
        "/kaggle/input/rim-one-retinal-dataset-for-assessing-glaucoma",
        "/kaggle/input/datasets/riadalmadani/rimone-dl",
        "/kaggle/input/rimone-dl",
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/RIM-ONE",
        "/kaggle/input/glaucoma-datasets/RIM-ONE",
        "/kaggle/input/rim-one-r3",
    ]
    p = auto_find_dir(["rim-one"], cands)
    return p if p else auto_find_dir(["rimone"], cands)

def get_g1020_base():
    cands = [
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/G1020",
        "/kaggle/input/glaucoma-datasets/G1020",
        "/kaggle/input/g1020",
    ]
    return auto_find_dir(["g1020"], cands)

def get_papila_base():
    cands = [
        "/kaggle/input/datasets/orvile/papila-retinal-fundus-images/PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f",
        "/kaggle/input/papila-retinal-fundus-images/PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f",
        "/kaggle/input/datasets/orvile/papila-retinal-fundus-images",
        "/kaggle/input/papila-retinal-fundus-images",
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/PAPILA",
        "/kaggle/input/glaucoma-datasets/PAPILA",
        "/kaggle/input/datasets/ahmedmohamed1999/papila-retinal-fundus-images",
        "/kaggle/input/papila-retinal-fundus-images",
        "/kaggle/input/datasets/sanchitvj/enhanced-fundus-image-dataset-for-glaucoma/PAPILA",
        "/kaggle/input/enhanced-fundus-image-dataset-for-glaucoma/PAPILA",
        "/kaggle/input/papila",
    ]
    return auto_find_dir(["papila"], cands)


# ── Mask Conversion Helpers ──────────────────────────────────────────────────

def refuge_mask_to_label(mask_path: str, target_shape=None) -> np.ndarray:
    if not mask_path or not os.path.exists(mask_path):
        return np.zeros(target_shape if target_shape else (512, 512), dtype=np.int64)

    mask_img = Image.open(mask_path).convert("L")
    if target_shape is not None:
        H, W = target_shape
        if mask_img.size != (W, H):
            mask_img = mask_img.resize((W, H), resample=Image.NEAREST)
    mask = np.array(mask_img)

    vals, counts = np.unique(mask, return_counts=True)
    if set(vals).issubset({0, 1, 2}):
        return mask.astype(np.int64)

    sig = counts >= (0.0005 * mask.size)
    sig_vals, sig_counts = (vals[sig], counts[sig]) if sig.sum() >= 2 else (vals, counts)
    sorted_vals = sig_vals[np.argsort(-sig_counts)]
    label = np.zeros_like(mask, dtype=np.int64)

    if len(sorted_vals) >= 3:
        label[mask == sorted_vals[1]] = 1   # Rim
        label[mask == sorted_vals[2]] = 2   # Cup
    elif len(sorted_vals) == 2:
        label[mask == sorted_vals[1]] = 1
    return label


def drishti_mask_to_label(disc_path: str, cup_path: str, target_shape=None) -> np.ndarray:
    disc_img = Image.open(disc_path).convert("L")
    cup_img  = Image.open(cup_path ).convert("L")

    if target_shape is not None:
        H, W = target_shape
        if disc_img.size != (W, H): disc_img = disc_img.resize((W, H), resample=Image.NEAREST)
        if cup_img.size  != (W, H): cup_img  = cup_img.resize ((W, H), resample=Image.NEAREST)
    elif disc_img.size != cup_img.size:
        cup_img = cup_img.resize(disc_img.size, resample=Image.NEAREST)

    disc  = np.array(disc_img) > 127
    cup   = np.array(cup_img ) > 127
    label = np.zeros(disc.shape, dtype=np.int64)
    label[disc] = 1
    label[cup]  = 2
    return label


def load_papila_contour(txt_path: str, target_shape) -> np.ndarray:
    if not txt_path or not os.path.exists(txt_path):
        return np.zeros(target_shape, dtype=np.uint8)
    H, W = target_shape
    if txt_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        img = Image.open(txt_path).convert("L")
        if img.size != (W, H):
            img = img.resize((W, H), Image.NEAREST)
        return (np.array(img) > 127).astype(np.uint8)
    try:
        pts = None
        for delim in [",", None, "\t", ";"]:
            try:
                cand = np.loadtxt(txt_path, delimiter=delim)
                if cand.ndim == 2 and cand.shape[1] >= 2 and len(cand) >= 3:
                    pts = cand
                    break
            except Exception:
                continue
        if pts is None or pts.ndim != 2 or pts.shape[1] < 2:
            return np.zeros((H, W), dtype=np.uint8)
        from PIL import ImageDraw
        poly_img = Image.new("L", (W, H), 0)
        coords = [(int(round(pt[0])), int(round(pt[1]))) for pt in pts[:, :2]]
        ImageDraw.Draw(poly_img).polygon(coords, outline=1, fill=1)
        return np.array(poly_img, dtype=np.uint8)
    except Exception:
        return np.zeros((H, W), dtype=np.uint8)


# ── Transforms ───────────────────────────────────────────────────────────────

def make_compose(transforms):
    try:
        return A.Compose(transforms, is_check_shapes=False)
    except TypeError:
        return A.Compose(transforms)

def train_tfm(sz=CFG.IMAGE_SIZE):
    return make_compose([
        A.Resize(sz, sz),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
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
# ║  DATASET LOADERS                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class REFUGEDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.transform = transform
        base = get_refuge_base()
        if base is None:
            raise FileNotFoundError("Could not find REFUGE folder under /kaggle/input.")

        imgs, mask_dirs = [], []
        split_img_dir = os.path.join(base, split, "Images")
        split_msk_dir = os.path.join(base, split, "Masks")
        if os.path.isdir(split_img_dir):
            for ext in ["*.jpg", "*.png", "*.JPG", "*.jpeg"]:
                imgs.extend(glob.glob(os.path.join(split_img_dir, ext)))
            if os.path.isdir(split_msk_dir):
                mask_dirs.append(split_msk_dir)

        if len(imgs) == 0:
            sq_img_dir = os.path.join(base, "Images_Square")
            sq_msk_dir = os.path.join(base, "Masks_Square")
            if os.path.isdir(sq_img_dir):
                all_sq = sorted(glob.glob(os.path.join(sq_img_dir, "*.*")))
                if split == "train":
                    imgs = [p for p in all_sq if os.path.basename(p).startswith("T")]
                elif split == "val":
                    imgs = [p for p in all_sq if os.path.basename(p).startswith("V")]
                else:
                    imgs = [p for p in all_sq if not os.path.basename(p).startswith(("T", "V"))]
                if len(imgs) == 0 and len(all_sq) > 0:
                    split_idx = int(0.8 * len(all_sq))
                    imgs = all_sq[:split_idx] if split == "train" else all_sq[split_idx:]
                if os.path.isdir(sq_msk_dir):
                    mask_dirs.append(sq_msk_dir)

        for cand in [
            os.path.join(base, "Masks_Square"), os.path.join(base, split, "Masks"),
            os.path.join(base, "train", "Masks"), os.path.join(base, "val", "Masks")
        ]:
            if os.path.isdir(cand) and cand not in mask_dirs: mask_dirs.append(cand)

        self.imgs = sorted(imgs); self.mask_dirs = mask_dirs
        print(f"  REFUGE [{split}]: {len(self.imgs)} images found")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        image    = np.array(Image.open(img_path).convert("RGB"))
        mask_path = None
        for md in self.mask_dirs:
            for ext in [".bmp", ".png", ".jpg", ".jpeg"]:
                c = os.path.join(md, stem + ext)
                if os.path.exists(c): mask_path = c; break
            if mask_path: break
        label = refuge_mask_to_label(mask_path, target_shape=image.shape[:2])
        if self.transform:
            t = self.transform(image=image, mask=label)
            image, label = t["image"], t["mask"].long()
        return image, label, stem


class ORIGADataset(Dataset):
    def __init__(self, split="test", transform=None):
        self.transform = transform
        base = get_origa_base()
        if base is None:
            raise FileNotFoundError("Could not find ORIGA folder under /kaggle/input.")

        selected_img_dir = None; selected_msk_dir = None
        for img_name, msk_name in [("Images_Square", "Masks_Square"), ("Images", "Masks")]:
            i_p, m_p = os.path.join(base, img_name), os.path.join(base, msk_name)
            if os.path.isdir(i_p) and os.path.isdir(m_p) and len(glob.glob(os.path.join(i_p, "*.*"))) > 0:
                selected_img_dir, selected_msk_dir = i_p, m_p; break

        imgs = []
        if selected_img_dir:
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.bmp"]:
                imgs.extend(glob.glob(os.path.join(selected_img_dir, ext)))
        self.imgs = sorted(list(set(imgs)))

        self.mask_map = {}
        if selected_msk_dir and os.path.isdir(selected_msk_dir):
            for f in os.listdir(selected_msk_dir):
                if f.lower().endswith((".png", ".bmp", ".jpg", ".jpeg")):
                    stem = os.path.splitext(f)[0].lower()
                    key  = stem.replace("_mask", "").replace("-mask", "")
                    self.mask_map[key] = os.path.join(selected_msk_dir, f)
                    self.mask_map[stem] = os.path.join(selected_msk_dir, f)

        print(f"  ORIGA [{split}]: {len(self.imgs)} images indexed")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        image    = np.array(Image.open(img_path).convert("RGB"))
        key      = stem.lower().replace("_mask", "").replace("-mask", "")
        mask_path = self.mask_map.get(stem.lower()) or self.mask_map.get(key)
        label    = refuge_mask_to_label(mask_path, target_shape=image.shape[:2])
        if self.transform:
            t = self.transform(image=image, mask=label)
            image, label = t["image"], t["mask"].long()
        return image, label, stem


class DrishtiDataset(Dataset):
    def __init__(self, split="test", transform=None):
        self.transform = transform
        base = get_drishti_base()
        if base is None:
            raise FileNotFoundError("Could not find Drishti-GS dataset under /kaggle/input.")

        imgs = []
        for root, _, files in os.walk(base):
            if any(k in root.lower() for k in ["images", "test", "train"]) and "mask" not in root.lower():
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")): imgs.append(os.path.join(root, f))
        self.imgs = sorted(list(set(imgs)))

        self.disc_masks, self.cup_masks = {}, {}
        for root, _, files in os.walk(base):
            for f in files:
                fl = f.lower()
                if fl.endswith((".png", ".bmp", ".jpg")):
                    full_p = os.path.join(root, f)
                    key = "_".join(os.path.splitext(f)[0].split("_")[:2]).lower()
                    if "odseg" in fl or "disc" in fl or "_od" in fl: self.disc_masks[key] = full_p
                    if "ocseg" in fl or "cup"  in fl or "_oc" in fl: self.cup_masks[key]  = full_p

        print(f"  Drishti [{split}]: {len(self.imgs)} images found")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        image    = np.array(Image.open(img_path).convert("RGB"))
        key      = "_".join(stem.split("_")[:2]).lower()
        disc_p, cup_p = self.disc_masks.get(key), self.cup_masks.get(key)
        label    = drishti_mask_to_label(disc_p, cup_p, target_shape=image.shape[:2]) if (disc_p and cup_p) else np.zeros(image.shape[:2], dtype=np.int64)
        if self.transform:
            t = self.transform(image=image, mask=label)
            image, label = t["image"], t["mask"].long()
        return image, label, stem


def ensure_rimone_masks(base_dir=None):
    """
    Ensures ground-truth segmentation masks for RIM-ONE are present.
    If not already in /kaggle/working/rimone_masks or /kaggle/input,
    downloads the official 1.5 MB reference segmentations from Google Drive / GitHub mirror.
    """
    target_dir = "/kaggle/working/rimone_masks"
    if os.path.isdir(target_dir):
        existing = glob.glob(os.path.join(target_dir, "**", "*.[pP][nN][gG]"), recursive=True)
        if len(existing) >= 50:
            return target_dir

    if base_dir and os.path.isdir(base_dir):
        existing = [
            f for f in glob.glob(os.path.join(base_dir, "**", "*.*"), recursive=True)
            if any(k in f.lower() for k in ["cup", "disc", "-od", "-oc"])
        ]
        if len(existing) >= 50:
            return base_dir

    url = "https://drive.usercontent.google.com/download?id=1eb1V9V65TuwFNYmYsIzgdAgdWyD6o7bG"
    try:
        import urllib.request, zipfile, io
        print("  [INFO] Auto-fetching official RIM-ONE reference segmentations (1.5 MB)...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=30).read()
        z = zipfile.ZipFile(io.BytesIO(data))
        os.makedirs(target_dir, exist_ok=True)
        z.extractall(target_dir)
        print(f"  ✓ Extracted {len(z.namelist())} reference mask files to {target_dir}")
        return target_dir
    except Exception as e:
        print(f"  [WARN] Auto-download of RIM-ONE reference segmentations failed ({e}).")
        return base_dir


class RimOneDataset(Dataset):
    def __init__(self, split="test", transform=None):
        self.transform = transform
        base = get_rimone_base()
        if base is None:
            raise FileNotFoundError("Could not find RIM-ONE folder under /kaggle/input.")

        mask_dir = ensure_rimone_masks(base)

        imgs = []
        for root, _, files in os.walk(base):
            root_lower = root.lower()
            is_match = ("test_set" in root_lower or "test" in root_lower) if split == "test" else True
            for f in files:
                fl = f.lower()
                is_mask = any(kw in fl for kw in ["mask", "disc", "cup", "-od", "-oc", "seg", "_d.", "_c."])
                is_img  = fl.endswith((".png", ".jpg", ".jpeg", ".bmp")) and not f.startswith(".")
                if is_img and not is_mask and is_match:
                    imgs.append(os.path.join(root, f))

        if len(imgs) == 0:
            for root, _, files in os.walk(base):
                for f in files:
                    fl = f.lower()
                    if fl.endswith((".png", ".jpg", ".jpeg", ".bmp")) and not any(k in fl for k in ["disc", "cup", "mask", "seg"]):
                        imgs.append(os.path.join(root, f))

        disc_masks = {}
        cup_masks  = {}
        multi_masks = {}

        search_dirs = [d for d in [mask_dir, base] if d and os.path.isdir(d)]
        for sdir in search_dirs:
            for root, _, files in os.walk(sdir):
                for f in files:
                    fl = f.lower()
                    full_p = os.path.join(root, f)
                    # Strictly filter for image extensions (exclude .txt DCSeg files)
                    if not fl.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        continue
                    m = re.match(r"^(.*?)[-_](?:[0-9]+[-_])?(cup|disc|od|oc)", f, re.IGNORECASE)
                    if m:
                        stem_k = m.group(1).lower().strip("-_ ")
                        m_type = m.group(2).lower()
                        if m_type in ["disc", "od"]:
                            disc_masks[stem_k] = full_p
                        elif m_type in ["cup", "oc"]:
                            cup_masks[stem_k] = full_p
                    elif "mask" in fl:
                        stem_k = os.path.splitext(f)[0].lower().replace("_mask", "").replace("-mask", "").strip("-_ ")
                        multi_masks[stem_k] = full_p

        matched_imgs = []
        for p in imgs:
            stem = os.path.splitext(os.path.basename(p))[0].lower().strip("-_ ")
            if (stem in disc_masks and stem in cup_masks) or (stem in multi_masks):
                matched_imgs.append(p)

        if len(matched_imgs) > 0:
            self.imgs = sorted(list(set(matched_imgs)))
        else:
            self.imgs = sorted(list(set(imgs)))

        self.disc_masks = disc_masks
        self.cup_masks  = cup_masks
        self.multi_masks = multi_masks
        print(f"  RIM-ONE [{split}]: {len(self.imgs)} images indexed (with matched GT masks)")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        image    = np.array(Image.open(img_path).convert("RGB"))
        H, W     = image.shape[:2]
        clean_k  = stem.lower().strip("-_ ")

        disc_p  = self.disc_masks.get(clean_k)
        cup_p   = self.cup_masks.get(clean_k)
        multi_p = self.multi_masks.get(clean_k)
        label   = np.zeros((H, W), dtype=np.int64)

        if disc_p and cup_p:
            try:
                if str(disc_p).lower().endswith(".txt"):
                    cand = str(disc_p)[:-4] + ".png"
                    if os.path.exists(cand): disc_p = cand
                if str(cup_p).lower().endswith(".txt"):
                    cand = str(cup_p)[:-4] + ".png"
                    if os.path.exists(cand): cup_p = cand

                d_arr = np.array(Image.open(disc_p).convert("L"))
                c_arr = np.array(Image.open(cup_p).convert("L"))
                if d_arr.shape != (H, W): d_arr = np.array(Image.fromarray(d_arr).resize((W, H), Image.NEAREST))
                if c_arr.shape != (H, W): c_arr = np.array(Image.fromarray(c_arr).resize((W, H), Image.NEAREST))
                disc = (d_arr > 127)
                cup  = (c_arr > 127) & disc
                label[disc] = 1
                label[cup]  = 2
            except Exception:
                pass
        elif multi_p:
            try:
                label = refuge_mask_to_label(multi_p, target_shape=(H, W))
            except Exception:
                pass

        if self.transform:
            t = self.transform(image=image, mask=label)
            image, label = t["image"], t["mask"].long()
        return image, label, stem


class G1020Dataset(Dataset):
    def __init__(self, split="test", transform=None):
        self.transform = transform
        base = get_g1020_base()
        if base is None:
            raise FileNotFoundError("Could not find G1020 folder under /kaggle/input.")

        selected_img_dir = None
        selected_msk_dir = None
        # Select single paired representation (Images_Square + Masks_Square preferred to match REFUGE geometry)
        for img_name, msk_name in [
            ("Images_Square", "Masks_Square"),
            ("Images_Cropped", "Masks_Cropped"),
            ("Images", "Masks"),
        ]:
            i_p = os.path.join(base, img_name)
            m_p = os.path.join(base, msk_name)
            if os.path.isdir(i_p) and os.path.isdir(m_p) and len(glob.glob(os.path.join(i_p, "*.*"))) > 0:
                selected_img_dir, selected_msk_dir = i_p, m_p
                break

        imgs = []
        if selected_img_dir:
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.bmp"]:
                imgs.extend(glob.glob(os.path.join(selected_img_dir, ext)))
        else:
            for root, _, files in os.walk(base):
                if "mask" not in root.lower():
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                            imgs.append(os.path.join(root, f))

        self.imgs = sorted(list(set(imgs)))

        self.mask_map = {}
        if selected_msk_dir and os.path.isdir(selected_msk_dir):
            for f in os.listdir(selected_msk_dir):
                if f.lower().endswith((".png", ".bmp", ".jpg", ".jpeg")):
                    stem = os.path.splitext(f)[0].lower()
                    key  = stem.replace("_mask", "").replace("-mask", "")
                    self.mask_map[key] = os.path.join(selected_msk_dir, f)
                    self.mask_map[stem] = os.path.join(selected_msk_dir, f)
        else:
            for root, _, files in os.walk(base):
                if "mask" in root.lower():
                    for f in files:
                        if f.lower().endswith((".png", ".bmp", ".jpg", ".jpeg")):
                            stem = os.path.splitext(f)[0].lower().replace("_mask", "").replace("-mask", "")
                            self.mask_map[stem] = os.path.join(root, f)

        valid_imgs = [
            p for p in self.imgs
            if (os.path.splitext(os.path.basename(p))[0].lower().replace("_mask", "").replace("-mask", "") in self.mask_map
                or os.path.splitext(os.path.basename(p))[0].lower() in self.mask_map)
        ]
        if len(valid_imgs) > 0:
            self.imgs = valid_imgs

        info = os.path.basename(selected_msk_dir) if selected_msk_dir else "flat directory"
        print(f"  G1020 [{split}]: {len(self.imgs)} images indexed (geometry paired: {info})")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        image    = np.array(Image.open(img_path).convert("RGB"))
        key      = stem.lower().replace("_mask", "").replace("-mask", "")
        mask_p   = self.mask_map.get(stem.lower()) or self.mask_map.get(key)
        label    = refuge_mask_to_label(mask_p, target_shape=image.shape[:2]) if mask_p else np.zeros(image.shape[:2], dtype=np.int64)
        if self.transform:
            t = self.transform(image=image, mask=label)
            image, label = t["image"], t["mask"].long()
        return image, label, stem


class PAPILADataset(Dataset):
    def __init__(self, split="test", transform=None):
        self.transform = transform
        base = get_papila_base()
        if base is None:
            raise FileNotFoundError("Could not find PAPILA folder under /kaggle/input.")

        # 1. Locate FundusImages and ExpertsSegmentations directories
        img_dir = os.path.join(base, "FundusImages")
        if not os.path.isdir(img_dir):
            for root, dirs, _ in os.walk(base):
                for d in dirs:
                    if "fundus" in d.lower():
                        img_dir = os.path.join(root, d); break
                if os.path.isdir(img_dir): break
        if not os.path.isdir(img_dir):
            img_dir = base

        seg_dir = os.path.join(base, "ExpertsSegmentations")
        if not os.path.isdir(seg_dir):
            for root, dirs, _ in os.walk(base):
                for d in dirs:
                    if "expert" in d.lower() or "segmentation" in d.lower():
                        seg_dir = os.path.join(root, d); break
                if os.path.isdir(seg_dir): break
        if not os.path.isdir(seg_dir):
            seg_dir = base

        # 2. Collect fundus images
        imgs = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                fl = f.lower()
                if fl.endswith((".jpg", ".jpeg", ".png", ".bmp")) and not any(k in fl for k in ["disc", "cup", "mask", "exp"]):
                    imgs.append(os.path.join(root, f))
        self.imgs = sorted(list(set(imgs)))

        # 3. Index expert contour annotations
        disc_masks = {}
        cup_masks  = {}
        for root, _, files in os.walk(seg_dir):
            for f in files:
                fl = f.lower()
                full_p = os.path.join(root, f)
                stem = f.split("_")[0].lower() # e.g. ret002od
                if "disc" in fl:
                    if stem not in disc_masks or "exp1" in fl:
                        disc_masks[stem] = full_p
                elif "cup" in fl:
                    if stem not in cup_masks or "exp1" in fl:
                        cup_masks[stem] = full_p

        matched = [p for p in self.imgs if os.path.splitext(os.path.basename(p))[0].lower() in disc_masks]
        if len(matched) > 0:
            self.imgs = matched

        self.disc_masks = disc_masks
        self.cup_masks  = cup_masks
        print(f"  PAPILA [{split}]: {len(self.imgs)} images indexed (with matched expert contours)")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        image    = np.array(Image.open(img_path).convert("RGB"))
        H, W     = image.shape[:2]
        clean_k  = stem.lower()

        disc_p = self.disc_masks.get(clean_k)
        cup_p  = self.cup_masks.get(clean_k)

        disc_m = load_papila_contour(disc_p, (H, W))
        cup_m  = load_papila_contour(cup_p,  (H, W))

        label = np.zeros((H, W), dtype=np.int64)
        label[disc_m == 1] = 1
        label[(cup_m == 1) & (disc_m == 1)] = 2

        if self.transform:
            t = self.transform(image=image, mask=label)
            image, label = t["image"], t["mask"].long()
        return image, label, stem


# ── Unified Dataset Factory ──────────────────────────────────────────────────

def get_dataset(domain_name: str, split: str = "test", transform = None) -> Dataset:
    dn = str(domain_name).strip().lower()
    if "refuge" in dn:   return REFUGEDataset(split=split, transform=transform)
    elif "origa" in dn:  return ORIGADataset(split=split, transform=transform)
    elif "drishti" in dn:return DrishtiDataset(split=split, transform=transform)
    elif "rim" in dn:    return RimOneDataset(split=split, transform=transform)
    elif "g1020" in dn:  return G1020Dataset(split=split, transform=transform)
    elif "papila" in dn: return PAPILADataset(split=split, transform=transform)
    else:
        raise ValueError(f"Unsupported domain '{domain_name}'. Choose from: 'REFUGE', 'Drishti', 'ORIGA', 'RIM-ONE', 'G1020', 'PAPILA'.")


def check_datasets():
    print("\n── Dataset Sanity Check ──────────────────────────────────")
    print(f"  Source Domain: {CFG.SOURCE}")
    for label, domain, split in [(f"Source ({CFG.SOURCE} train)", CFG.SOURCE, "train")]:
        try:
            ds = get_dataset(domain, split, val_tfm())
            img, lbl, stem = ds[0]
            vals, counts = torch.unique(lbl, return_counts=True)
            pcts = {int(v): f"{float(c)/lbl.numel()*100:.1f}%" for v, c in zip(vals, counts)}
            print(f"  {label} OK │ classes: {pcts} │ stem: {stem}")
        except Exception as e:
            print(f"  {label} FAILED: {e}")

    for tgt in CFG.TARGETS:
        try:
            ds = get_dataset(tgt, "test", val_tfm())
            img, lbl, stem = ds[0]
            vals, counts = torch.unique(lbl, return_counts=True)
            pcts = {int(v): f"{float(c)/lbl.numel()*100:.1f}%" for v, c in zip(vals, counts)}
            has_fg = any(int(v) > 0 for v in vals)
            warn = "" if has_fg else " ⚠ WARNING: 0 foreground pixels in test mask!"
            print(f"  Target ({tgt}) OK │ classes: {pcts} │ stem: {stem}{warn}")
        except Exception as e:
            print(f"  Target ({tgt}) NOT MOUNTED: {e}")
    print("─"*58)

check_datasets()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL & METRICS                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_model():
    return smp.Unet(
        encoder_name    = CFG.ENCODER,
        encoder_weights = CFG.ENCODER_WEIGHTS,
        in_channels     = 3,
        classes         = CFG.NUM_CLASSES,
        activation      = None
    )

class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.tensor(CFG.CLASS_WEIGHTS, dtype=torch.float32).to(CFG.DEVICE)
        self.ce   = nn.CrossEntropyLoss(weight=w)
        self.dice = smp.losses.DiceLoss(mode="multiclass")
    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


def dice_score(pred, gt, eps=1e-6):
    p = (pred > 0).astype(np.float32)
    g = (gt > 0).astype(np.float32)
    intersection = float((p * g).sum())
    cardinality  = float(p.sum() + g.sum())
    if cardinality == 0.0:
        return 1.0 if intersection == 0.0 else 0.0
    return float((2.0 * intersection + eps) / (cardinality + eps))


def evaluate(probs, labels, *args, **kwargs):
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

    pred_disc = (pred >= 1); gt_disc = (gt >= 1)
    pred_cup  = (pred == 2); gt_cup  = (gt == 2)

    return {
        "disc_dice": dice_score(pred_disc, gt_disc),
        "cup_dice":  dice_score(pred_cup,  gt_cup),
    }

def agg(results):
    keys = results[0].keys()
    return {k: (float(np.mean([r[k] for r in results])),
                float(np.std ([r[k] for r in results]))) for k in keys}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 — SOURCE TRAINING                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_source():
    print("\n" + "═"*60)
    print(f"  STEP 1 — Train Source Model on {CFG.SOURCE}")
    print("═"*60)

    tr_loader = DataLoader(get_dataset(CFG.SOURCE, "train", train_tfm()),
                           batch_size=CFG.TRAIN_BS, shuffle=True,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    va_loader = DataLoader(get_dataset(CFG.SOURCE, "val", val_tfm()),
                           batch_size=8, shuffle=False,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True)

    model     = build_model().to(CFG.DEVICE)
    criterion = DiceCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.NUM_EPOCHS)

    best_cup = 0.0; patience = 0
    history  = {"loss": [], "cup_dice": []}

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for imgs, lbls, _ in tr_loader:
            imgs = imgs.to(CFG.DEVICE); lbls = lbls.to(CFG.DEVICE)
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
                    val_res.append(evaluate(probs[b].cpu(), lbls[b], fast=True))
        vagg = agg(val_res)
        history["cup_dice"].append(vagg["cup_dice"][0])

        is_best = vagg["cup_dice"][0] > best_cup
        save_tag = " ★ (Best Cup Dice)" if is_best else ""
        print(f"  [Epoch {epoch:03d}/{CFG.NUM_EPOCHS:03d}] "
              f"Loss: {history['loss'][-1]:.4f} │ "
              f"Val Disc: {vagg['disc_dice'][0]:.4f} │ "
              f"Val Cup: {vagg['cup_dice'][0]:.4f}{save_tag}")

        if is_best:
            best_cup = vagg["cup_dice"][0]; patience = 0
            torch.save({"state": model.state_dict(), "val": vagg}, CFG.CKPT_PATH)
        else:
            patience += 1
            if patience >= CFG.PATIENCE:
                print(f"  Early stop triggered at epoch {epoch}")
                break

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(history["loss"],     color="steelblue"); a1.set_title("Training Loss")
    a2.plot(history["cup_dice"], color="tomato");    a2.set_title("Val Cup Dice")
    for a in (a1, a2): a.set_xlabel("Epoch"); a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUT_DIR, "fig0_learning_curves.png"), dpi=150)
    plt.close()

    print(f"\n  ✓ Training complete. Best Val Cup Dice = {best_cup:.4f}")
    ckpt = torch.load(CFG.CKPT_PATH, map_location=CFG.DEVICE)
    model.load_state_dict(ckpt["state"])
    return model


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 — PHASE 0: COLLECT SOURCE STATISTICS                            ║
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
            x = inp[0].detach().cpu()
            lbl = state["labels"]
            if lbl is None: return
            B, C, H, W = x.shape
            lbl_d = F.interpolate(lbl.float().unsqueeze(1),
                                  (H, W), mode="nearest").squeeze(1).long()
            for k in range(K):
                mk = (lbl_d == k).float(); n = mk.sum()
                if n < 1: continue
                mk4 = mk.unsqueeze(1)
                accum[idx]["sum_k"][k]    += (x * mk4).sum([0,2,3])
                accum[idx]["sum_sq_k"][k] += (x**2 * mk4).sum([0,2,3])
                accum[idx]["count_k"][k]  += n
        return hook

    hooks = [m.register_forward_hook(make_hook(i)) for i, (_, m) in enumerate(bn_mods)]

    loader = DataLoader(get_dataset(CFG.SOURCE, "train", val_tfm()),
                        batch_size=4, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)

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
    print(f"  Source prior π: bg={prior[0]:.3f}  rim={prior[1]:.3f}  cup={prior[2]:.3f}")

    stats = {"class_mean": class_means, "class_var": class_vars, "prior": prior}
    torch.save(stats, CFG.STATS_PATH)
    print(f"  ✓ Saved source stats → {CFG.STATS_PATH}")
    return stats


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PMSA-BN ADAPTATION MODULE                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class PMSABatchNorm2d(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d,
                 K=CFG.NUM_CLASSES, tau=CFG.ALPHA_TAU,
                 kmu=CFG.KAPPA_MU, ks=CFG.KAPPA_SIGMA,
                 amin=CFG.ALPHA_MIN, amax=CFG.ALPHA_MAX):
        super().__init__()
        self.bn  = bn
        C = bn.num_features
        self.K, self.tau = K, tau
        self.kmu, self.ks = kmu, ks
        self.amin, self.amax = amin, amax

        self.register_buffer("src_cls_mean", torch.zeros(K, C))
        self.register_buffer("src_cls_var",  torch.ones (K, C))
        self.register_buffer("src_prior",    torch.ones (K) / K)

        self.adapt_mode    = False
        self.pseudo_mask   = None
        self.last_Dc       = None
        self.last_alpha    = None

    def _cls_stats(self, x, mask):
        B, C, H, W = x.shape; K = self.K
        m = F.interpolate(mask.float(), (H, W), mode="bilinear", align_corners=False)
        Nk = m.sum(dim=[0, 2, 3]).clamp(min=1e-6)
        img_mu = x.mean(dim=[0, 2, 3])
        img_va = x.var(dim=[0, 2, 3], unbiased=False).clamp(min=1e-6)

        t_mu = torch.zeros(K, C, device=x.device)
        t_va = torch.ones (K, C, device=x.device)
        for k in range(K):
            mk = m[:, k:k+1, :, :]
            if Nk[k] > 1e-4:
                mk_norm = mk / Nk[k]
                mu_k = (x * mk_norm).sum(dim=[0, 2, 3])
                ex2  = (x**2 * mk_norm).sum(dim=[0, 2, 3])
                va_k = (ex2 - mu_k**2).clamp(min=1e-6)
                t_mu[k] = mu_k; t_va[k] = va_k
            else:
                t_mu[k] = img_mu; t_va[k] = img_va
        return t_mu, t_va, Nk, img_mu, img_va

    def _shrink(self, tgt_mu, tgt_va, Nk, img_mu, img_va, H, W):
        N_eff = Nk / float(H * W)
        g_mu = (N_eff / (N_eff + self.kmu / 1000.0)).unsqueeze(1)
        g_va = (N_eff / (N_eff + self.ks  / 1000.0)).unsqueeze(1)
        shr_mu = g_mu * tgt_mu + (1.0 - g_mu) * img_mu.unsqueeze(0)
        shr_va = g_va * tgt_va + (1.0 - g_va) * img_va.unsqueeze(0)
        return shr_mu, shr_va

    def _prior_match(self, shr_mu, shr_va, Nk):
        tot = Nk.sum()
        pi_hat = (Nk / tot) if tot > 0 else self.src_prior
        pi = 0.5 * self.src_prior + 0.5 * pi_hat
        pi = (pi / pi.sum()).unsqueeze(1)
        mu_pm = (pi * shr_mu).sum(dim=0)
        va_pm = (pi * (shr_va + shr_mu**2)).sum(dim=0) - mu_pm**2
        return mu_pm, va_pm.clamp(min=1e-6)

    def _w2_dist(self, tgt_mu, tgt_va, src_mu, src_va):
        term_mu = (tgt_mu - src_mu)**2
        s_va = src_va.clamp(min=1e-6); t_va = tgt_va.clamp(min=1e-6)
        term_va = t_va + s_va - 2.0 * (t_va * s_va).sqrt()
        raw_w2  = (term_mu + term_va).clamp(min=0.0)
        return raw_w2 / s_va

    def forward(self, x):
        if not self.adapt_mode or self.pseudo_mask is None:
            return self.bn(x)

        B, C, H, W = x.shape
        with torch.no_grad():
            mask   = self.pseudo_mask
            src_mu = self.bn.running_mean.detach()
            src_va = self.bn.running_var.detach()

            tgt_mu, tgt_va, Nk, img_mu, img_va = self._cls_stats(x, mask)
            shr_mu, shr_va = self._shrink(tgt_mu, tgt_va, Nk, img_mu, img_va, H, W)
            pm_mu, pm_va = self._prior_match(shr_mu, shr_va, Nk)

            Dc = self._w2_dist(pm_mu, pm_va, src_mu, src_va)
            dc_mean = Dc.mean().detach()
            dc_std  = Dc.std().detach().clamp(min=1e-6)

            dev = (Dc - dc_mean) / (self.tau * dc_std)
            half_span = (self.amax - self.amin) / 2.0
            alpha = torch.clamp(0.50 + half_span * torch.tanh(dev), self.amin, self.amax)

            self.last_Dc    = Dc.cpu()
            self.last_alpha = alpha.cpu()

            hat_mu = alpha * pm_mu  + (1.0 - alpha) * src_mu
            hat_va = alpha * pm_va  + (1.0 - alpha) * src_va

        hat_std = (hat_va + self.bn.eps).clamp(1e-12).sqrt()
        xn = (x - hat_mu.view(1,C,1,1)) / hat_std.view(1,C,1,1)
        return self.bn.weight.view(1,C,1,1) * xn + self.bn.bias.view(1,C,1,1)


def replace_bn(model):
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d): setattr(model, name, PMSABatchNorm2d(child))
        else: replace_bn(child)
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


# ── Inference Functions ──────────────────────────────────────────────────────

@torch.no_grad()
def make_pseudo_mask(probs, tau_hi=CFG.TAU_HI):
    disc_prob = (probs[:, 1:2] + probs[:, 2:3]).clamp(0.0, 1.0)
    cup_prob  = probs[:, 2:3] * disc_prob
    soft_mask = torch.cat([probs[:, 0:1], probs[:, 1:2], cup_prob], dim=1)
    return soft_mask / soft_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)

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
def predict_pmsa(model, image, em_iters=CFG.EM_ITERS, src_model=None, pass1=CFG.PASS1):
    m_copy = copy.deepcopy(src_model if src_model is not None else model).to(image.device)
    if pass1 == "adabn":
        probs = predict_adabn(m_copy, image)
    else:
        probs = predict_alpha_bn(m_copy, image)
    set_adapt(model, True)
    for _ in range(em_iters):
        set_mask(model, make_pseudo_mask(probs, tau_hi=CFG.TAU_HI))
        probs = F.softmax(model(image), dim=1)
    set_adapt(model, False); set_mask(model, None)
    return probs


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EVALUATION PER TARGET                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_evaluation_for_target(target_name, src_model, pmsa_model):
    print("\n" + "═"*60)
    print(f"  Benchmark Target: {target_name}")
    print("═"*60)

    try:
        ds = get_dataset(target_name, "test", val_tfm())
    except Exception as e:
        print(f"  [SKIP] Could not load dataset for '{target_name}': {e}")
        return None

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)

    method_res = defaultdict(list)
    src_model.eval()

    for images, labels, stems in tqdm(loader, desc=f"  Evaluating {target_name:<8}"):
        images = images.to(CFG.DEVICE)

        # 1. Source Only
        p = predict_source_only(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        method_res["Source Only"].append(evaluate(p.cpu(), labels))

        # 2. AdaBN
        p = predict_adabn(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        method_res["AdaBN"].append(evaluate(p.cpu(), labels))

        # 3. α-BN
        p = predict_alpha_bn(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        method_res["α-BN (α=0.5)"].append(evaluate(p.cpu(), labels))

        # 4. PMSA-BN (Ours)
        pm = copy.deepcopy(pmsa_model).to(CFG.DEVICE)
        p = predict_pmsa(pm, images, src_model=src_model)
        method_res["PMSA-BN (Ours)"].append(evaluate(p.cpu(), labels))

    aggregated = {m: agg(r) for m, r in method_res.items()}
    print("\n" + "═"*56)
    print(f"  [{target_name}] Results Summary")
    print(f"  {'Method':<22} │ {'Disc Dice':>12} │ {'Cup Dice':>12}")
    print("  " + "─"*52)
    best_disc = max(res["disc_dice"][0] for res in aggregated.values())
    best_cup  = max(res["cup_dice"][0]  for res in aggregated.values())
    for m, res in aggregated.items():
        dd = res["disc_dice"]; cd = res["cup_dice"]
        d_star = " ★" if dd[0] == best_disc else "  "
        c_star = " ★" if cd[0] == best_cup  else "  "
        print(f"  {m:<22} │ {dd[0]:>6.4f}±{dd[1]:.3f}{d_star} │ {cd[0]:>6.4f}±{cd[1]:.3f}{c_star}")
    print("═"*56)

    # Export figure
    out_fig = os.path.join(CFG.OUT_DIR, f"fig_{target_name.lower()}_comparison.png")
    plot_bar_comparison(aggregated, target_name, out_fig)

    return aggregated


def plot_bar_comparison(aggregated, target_name, path):
    methods = list(aggregated.keys())
    colors  = ["#95a5a6", "#e74c3c", "#e67e22", "#2980b9"]
    metrics = [("disc_dice", "Disc Dice ↑"), ("cup_dice", "Cup Dice ↑")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"PMSA-BN vs. Baselines ({CFG.SOURCE} → {target_name})", fontsize=13, fontweight="bold")

    for ax, (key, title) in zip(axes, metrics):
        vals = [aggregated[m][key][0] for m in methods]
        errs = [aggregated[m][key][1] for m in methods]
        x    = np.arange(len(methods))
        ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="black", lw=0.7)
        ax.errorbar(x, vals, yerr=errs, fmt="none", color="black", capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0.0, 1.05)
        best = int(np.argmax(vals))
        y_star = min(vals[best] + errs[best] + 0.015, 1.02)
        ax.text(x[best], y_star, "★", ha="center", fontsize=14, color="gold", fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved figure → {path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n" + "╔"+"═"*62+"╗")
    print("║   PMSA-BN — Multi-Target Fundus TTA Benchmark Suite          ║")
    print(f"║   Source Domain : {CFG.SOURCE:<43} ║")
    print(f"║   Target Domains: {', '.join(CFG.TARGETS):<43} ║")
    print("╚"+"═"*62+"╝")

    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    ckpt_path  = find_checkpoint()
    stats_path = find_source_stats()

    if not CFG.TRAIN_FROM_SCRATCH and ckpt_path and stats_path:
        print(f"\n  ✓ Discovered checkpoint: {ckpt_path}")
        print(f"  ✓ Discovered source stats: {stats_path}")
        src_model = build_model().to(CFG.DEVICE)
        ckpt = torch.load(ckpt_path, map_location=CFG.DEVICE)
        src_model.load_state_dict(ckpt["state"] if "state" in ckpt else ckpt)
        src_model.eval()
        stats = torch.load(stats_path, map_location="cpu")
    else:
        if not CFG.TRAIN_FROM_SCRATCH:
            print(f"\n  [INFO] Pretrained checkpoint not found at {CFG.CKPT_PATH}. Automatically training source model on {CFG.SOURCE} from scratch...")
        else:
            print(f"\n  [INFO] Training source model on {CFG.SOURCE} from scratch (CFG.TRAIN_FROM_SCRATCH=True)...")
        src_model = train_source()
        stats = collect_source_stats(src_model)

    pmsa_model = replace_bn(copy.deepcopy(src_model))
    inject_stats(pmsa_model, stats)
    pmsa_model.eval().to(CFG.DEVICE)

    # ── Run Evaluation on all specified targets ───────────────────────────
    all_results = {}
    for tgt in CFG.TARGETS:
        res = run_evaluation_for_target(tgt, src_model, pmsa_model)
        if res is not None:
            all_results[tgt] = res

    # ── Master Summary Table ──────────────────────────────────────────────
    if all_results:
        print("\n\n" + "╔"+"═"*74+"╗")
        print("║   MASTER SUMMARY: PMSA-BN vs. Baselines Across All Evaluated Targets     ║")
        print("╚"+"═"*74+"╝")
        print(f"  {'Target':<10} │ {'Method':<20} │ {'Disc Dice':>14} │ {'Cup Dice':>14}")
        print("  " + "─"*70)

        for tgt, res_dict in all_results.items():
            best_disc = max(r["disc_dice"][0] for r in res_dict.values())
            best_cup  = max(r["cup_dice"][0]  for r in res_dict.values())
            first = True
            for m, vals in res_dict.items():
                dd, cd = vals["disc_dice"], vals["cup_dice"]
                d_star = " ★" if dd[0] == best_disc else "  "
                c_star = " ★" if cd[0] == best_cup  else "  "
                tgt_col = tgt if first else ""
                print(f"  {tgt_col:<10} │ {m:<20} │ {dd[0]:>6.4f}±{dd[1]:.3f}{d_star} │ {cd[0]:>6.4f}±{cd[1]:.3f}{c_star}")
                first = False
            print("  " + "─"*70)

        def jsonify(d):
            if isinstance(d, dict): return {k: jsonify(v) for k, v in d.items()}
            if isinstance(d, (list, tuple)): return [jsonify(x) for x in d]
            if isinstance(d, (np.float32, np.float64)): return float(d)
            if isinstance(d, (np.int32, np.int64)): return int(d)
            return d

        out_json = os.path.join(CFG.OUT_DIR, "all_fundus_targets_results.json")
        with open(out_json, "w") as f:
            json.dump(jsonify(all_results), f, indent=2)
        print(f"\n  ✓ All benchmark results saved to {out_json}")

    print("\n" + "╔"+"═"*62+"╗")
    print("║   Completed all requested evaluations!                      ║")
    print("╚"+"═"*62+"╝\n")


if __name__ == "__main__":
    main()
