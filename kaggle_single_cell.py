"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm                     ║
║         Single-Cell Kaggle Script — REFUGE -> ORIGA Target                   ║
║                                                                              ║
║  DATASET (add as Kaggle input):                                              ║
║  Glaucoma Fundus Imaging Datasets (arnavjain1/glaucoma-datasets)            ║
║  Contains both:                                                              ║
║    - REFUGE (Source domain for training + validation)                        ║
║    - ORIGA  (Target domain for test-time adaptation evaluation)              ║
║                                                                              ║
║  BASELINES:  Source Only | AdaBN | α-BN | PMSA-BN (Ours)                    ║
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
# ║  CFG — All paths confirmed from actual Kaggle folder structure           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CFG:
    # ── Source & Target Domain Selection ──────────────────────────────────
    # Choose from: "REFUGE", "ORIGA", "Drishti"
    SOURCE = "REFUGE"       # Source domain (trains from scratch + collects Phase 0 statistics)
    TARGET = "ORIGA"        # Target domain (evaluates test-time adaptation methods)

    # ── Output ────────────────────────────────────────────────────────────
    CKPT_PATH  = "/kaggle/working/best_model.pth"
    STATS_PATH = "/kaggle/working/source_stats.pth"
    OUT_DIR    = "/kaggle/working"


    # ── Model ─────────────────────────────────────────────────────────────
    ENCODER         = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES     = 3        # 0=background  1=disc rim  2=cup
    IMAGE_SIZE      = 384

    # ── Training ──────────────────────────────────────────────────────────
    TRAIN_BS           = 4
    NUM_EPOCHS         = 60         # set to 5 for quick smoke test, 60 for full run
    LR                 = 1e-4
    WEIGHT_DECAY       = 1e-5
    PATIENCE           = 12
    CLASS_WEIGHTS      = [0.2, 1.5, 3.0]   # bg:rim:cup  (cup is rare → upweight)
    TRAIN_FROM_SCRATCH = False      # Reuses /kaggle/working/best_model.pth (saves 15 min!)

    # ── PMSA-BN hyperparameters ───────────────────────────────────────────
    TAU_HI      = 0.50   # confident pseudo-mask threshold
    KAPPA_MU    = 8.0    # Bayesian shrinkage pseudo-count (mean)
    KAPPA_SIGMA = 16.0   # Bayesian shrinkage pseudo-count (variance)
    ALPHA_TAU   = 1.0    # temperature τ for shift sensitivity
    ALPHA_MIN   = 0.42   # lower bound anchored slightly below α-BN baseline
    ALPHA_MAX   = 0.58   # upper bound adapts shifted channels toward target
    EM_ITERS    = 1      # single refinement pass (avoids confirmation bias at B=1)

    # ── Baselines ─────────────────────────────────────────────────────────
    ALPHA_BN_WEIGHT = 0.5
    RUN_ABLATION    = False  # Suspended for rapid iteration; set True for final paper table

    SEED        = 42
    seed        = 42
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
    # Search /kaggle/input
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
            if "best_model.pth" in files:
                return os.path.join(root, "best_model.pth")
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
            if "source_stats.pth" in files:
                return os.path.join(root, "source_stats.pth")
    return None


def get_refuge_base():
    cands = [
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/REFUGE",
        "/kaggle/input/glaucoma-datasets/REFUGE",
        "/kaggle/input/arnavjain1/glaucoma-datasets/REFUGE",
    ]
    p = auto_find_dir(["REFUGE"], cands)
    if p: return p
    # Try finding glaucoma-datasets
    p = auto_find_dir(["glaucoma", "dataset"])
    if p:
        cand = os.path.join(p, "REFUGE")
        if os.path.isdir(cand): return cand
    return None


def get_drishti_base():
    cands = [
        "/kaggle/input/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation",
        "/kaggle/input/drishtigs-retina-dataset-for-onh-segmentation",
        "/kaggle/input/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation",
    ]
    p = auto_find_dir(["drishti"], cands)
    return p


def get_origa_base():
    cands = [
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/ORIGA",
        "/kaggle/input/glaucoma-datasets/ORIGA",
        "/kaggle/input/arnavjain1/glaucoma-datasets/ORIGA",
    ]
    for c in cands:
        if os.path.isdir(c):
            return c
    p = auto_find_dir(["ORIGA"], cands)
    if p: return p
    # Try finding glaucoma-datasets then ORIGA subfolder
    p = auto_find_dir(["glaucoma", "dataset"])
    if p:
        cand = os.path.join(p, "ORIGA")
        if os.path.isdir(cand): return cand
    if os.path.exists("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input"):
            for d in dirs:
                if d.lower() == "origa":
                    return os.path.join(root, d)
    return None



# ── Mask conversion helpers ──────────────────────────────────────────────────

def refuge_mask_to_label(mask_path: str, target_shape=None) -> np.ndarray:
    """
    Robust multi-format mask to 3-class label converter.
    Uses pixel area frequency ranking:
      - Largest region (>80% of pixels)  --> Background (0)
      - Intermediate region (~2-5%)       --> Disc rim   (1)
      - Smallest region (<2%)             --> Optic Cup  (2)
    Immune to color inversion (whether background is 0 or 255) and compression noise.
    """
    if not mask_path or not os.path.exists(mask_path):
        if target_shape is not None:
            return np.zeros(target_shape, dtype=np.int64)
        return np.zeros((512, 512), dtype=np.int64)

    mask_img = Image.open(mask_path).convert("L")
    if target_shape is not None:
        H, W = target_shape
        if mask_img.size != (W, H):
            mask_img = mask_img.resize((W, H), resample=Image.NEAREST)
    mask = np.array(mask_img)

    vals, counts = np.unique(mask, return_counts=True)
    if set(vals).issubset({0, 1, 2}):
        return mask.astype(np.int64)

    # Filter out compression fringe / noise (< 0.05% of image)
    sig = counts >= (0.0005 * mask.size)
    if sig.sum() >= 2:
        sig_vals   = vals[sig]
        sig_counts = counts[sig]
    else:
        sig_vals, sig_counts = vals, counts

    sorted_vals = sig_vals[np.argsort(-sig_counts)]  # descending by area
    label = np.zeros_like(mask, dtype=np.int64)

    if len(sorted_vals) >= 3:
        # sorted_vals[0] is Background (largest area) -> stays 0
        label[mask == sorted_vals[1]] = 1   # Disc rim (intermediate)
        label[mask == sorted_vals[2]] = 2   # Optic Cup (smallest)
    elif len(sorted_vals) == 2:
        label[mask == sorted_vals[1]] = 1

    return label



def drishti_mask_to_label(disc_path: str, cup_path: str, target_shape=None) -> np.ndarray:
    """Convert separate Drishti binary disc + cup mask PNGs to 3-class label."""
    disc_img = Image.open(disc_path).convert("L")
    cup_img  = Image.open(cup_path ).convert("L")

    if target_shape is not None:
        H, W = target_shape
        if disc_img.size != (W, H):
            disc_img = disc_img.resize((W, H), resample=Image.NEAREST)
        if cup_img.size != (W, H):
            cup_img = cup_img.resize((W, H), resample=Image.NEAREST)
    elif disc_img.size != cup_img.size:
        cup_img = cup_img.resize(disc_img.size, resample=Image.NEAREST)

    disc  = np.array(disc_img) > 127
    cup   = np.array(cup_img ) > 127
    label = np.zeros(disc.shape, dtype=np.int64)
    label[disc] = 1   # disc rim (will be overwritten by cup below)
    label[cup]  = 2   # cup
    return label


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
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.15, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std =(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def val_tfm(sz=CFG.IMAGE_SIZE):
    return make_compose([
        A.Resize(sz, sz),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std =(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  REFUGE DATASET (Robust Multi-Layout Support)                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class REFUGEDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.transform = transform
        base = get_refuge_base()
        if base is None:
            raise FileNotFoundError(
                "Could not find REFUGE folder under /kaggle/input.\n"
                "Please verify that arnavjain1/glaucoma-datasets is added as input."
            )

        imgs = []
        mask_dirs = []

        # Layout 1: REFUGE/train/Images and REFUGE/train/Masks
        split_img_dir = os.path.join(base, split, "Images")
        split_msk_dir = os.path.join(base, split, "Masks")
        if os.path.isdir(split_img_dir):
            for ext in ["*.jpg", "*.png", "*.JPG", "*.jpeg"]:
                imgs.extend(glob.glob(os.path.join(split_img_dir, ext)))
            if os.path.isdir(split_msk_dir):
                mask_dirs.append(split_msk_dir)

        # Layout 2: REFUGE/Images_Square and REFUGE/Masks_Square (Flat images)
        if len(imgs) == 0:
            sq_img_dir = os.path.join(base, "Images_Square")
            sq_msk_dir = os.path.join(base, "Masks_Square")
            if os.path.isdir(sq_img_dir):
                all_sq = sorted(glob.glob(os.path.join(sq_img_dir, "*.*")))
                # In REFUGE: Txxxx is Train, Vxxxx is Val
                if split == "train":
                    imgs = [p for p in all_sq if os.path.basename(p).startswith("T")]
                elif split == "val":
                    imgs = [p for p in all_sq if os.path.basename(p).startswith("V")]
                else:
                    imgs = [p for p in all_sq if not os.path.basename(p).startswith(("T", "V"))]

                # Fallback if no T/V prefix found: split by percentage
                if len(imgs) == 0 and len(all_sq) > 0:
                    split_idx = int(0.8 * len(all_sq))
                    imgs = all_sq[:split_idx] if split == "train" else all_sq[split_idx:]

                if os.path.isdir(sq_msk_dir):
                    mask_dirs.append(sq_msk_dir)

        # Additional mask fallback directories
        for cand in [
            os.path.join(base, "Masks_Square"),
            os.path.join(base, "train", "Masks"),
            os.path.join(base, "val", "Masks"),
            os.path.join(base, "train", "gts"),
            os.path.join(base, "val", "gts"),
        ]:
            if os.path.isdir(cand) and cand not in mask_dirs:
                mask_dirs.append(cand)

        self.imgs = sorted(imgs)
        self.mask_dirs = mask_dirs

        if len(self.imgs) == 0:
            raise FileNotFoundError(
                f"No REFUGE images found for split '{split}' in {base}."
            )
        print(f"  REFUGE [{split}]: {len(self.imgs)} images found (base: {base})")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        image    = np.array(Image.open(img_path).convert("RGB"))

        # Search matching mask by stem
        mask_path = None
        for md in self.mask_dirs:
            for ext in [".bmp", ".png", ".jpg", ".jpeg"]:
                c = os.path.join(md, stem + ext)
                if os.path.exists(c):
                    mask_path = c
                    break
            if mask_path: break

        label = refuge_mask_to_label(mask_path, target_shape=image.shape[:2]) if mask_path \
                else np.zeros(image.shape[:2], dtype=np.int64)

        if label.shape[:2] != image.shape[:2]:
            label = np.array(Image.fromarray(label.astype(np.uint8)).resize(
                (image.shape[1], image.shape[0]), resample=Image.NEAREST
            )).astype(np.int64)

        if self.transform:
            t = self.transform(image=image, mask=label)
            image = t["image"]; label = t["mask"].long()

        return image, label, stem


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DRISHTI DATASET (Robust Multi-Layout Support)                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DrishtiDataset(Dataset):
    def __init__(self, split="test", transform=None):
        self.transform = transform
        base = get_drishti_base()
        if base is None:
            raise FileNotFoundError(
                "Could not find Drishti-GS dataset folder under /kaggle/input.\n"
                "Please verify that drishtigs-retina-dataset-for-onh-segmentation is added as input."
            )

        imgs = []
        # Walk and collect image files belonging to the split
        for root, dirs, files in os.walk(base):
            root_lower = root.lower()
            if "images" in root_lower:
                match_split = (split.lower() in root_lower) or \
                              ("test" in root_lower if split == "test" else ("train" in root_lower or "training" in root_lower))
                if match_split:
                    for f in files:
                        if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.startswith("."):
                            imgs.append(os.path.join(root, f))

        # Fallback: if no images found by split name, collect all drishti images
        if len(imgs) == 0:
            for root, dirs, files in os.walk(base):
                if "images" in root.lower():
                    for f in files:
                        if "drishti" in f.lower() and f.lower().endswith((".png", ".jpg", ".jpeg")):
                            imgs.append(os.path.join(root, f))

        self.imgs = sorted(list(set(imgs)))

        # Pre-index GT masks of THIS split only (train/test share IDs like drishtiGS_001)
        self.disc_masks = {}
        self.cup_masks  = {}
        want = "test" if split == "test" else "train"
        for root, dirs, files in os.walk(base):
            rl = root.lower()
            if "gt" in rl and want in rl:
                for f in files:
                    fl = f.lower()
                    if fl.endswith((".png", ".bmp", ".jpg")):
                        full_p = os.path.join(root, f)
                        key = "_".join(os.path.splitext(f)[0].split("_")[:2]).lower()
                        if "odseg" in fl or "disc" in fl or "_od" in fl:
                            self.disc_masks[key] = full_p
                        if "ocseg" in fl or "cup"  in fl or "_oc" in fl:
                            self.cup_masks[key] = full_p

        if len(self.imgs) == 0:
            raise FileNotFoundError(
                f"No Drishti images found for split '{split}' under {base}."
            )
        print(f"  Drishti [{split}]: {len(self.imgs)} images, "
              f"{len(self.disc_masks)} disc / {len(self.cup_masks)} cup masks")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        image    = np.array(Image.open(img_path).convert("RGB"))

        key = "_".join(stem.split("_")[:2]).lower()
        disc_path = self.disc_masks.get(key)
        cup_path  = self.cup_masks.get(key)

        if disc_path and cup_path:
            label = drishti_mask_to_label(disc_path, cup_path, target_shape=image.shape[:2])
        else:
            label = np.zeros(image.shape[:2], dtype=np.int64)

        if label.shape[:2] != image.shape[:2]:
            label = np.array(Image.fromarray(label.astype(np.uint8)).resize(
                (image.shape[1], image.shape[0]), resample=Image.NEAREST
            )).astype(np.int64)

        if self.transform:
            t = self.transform(image=image, mask=label)
            image = t["image"]; label = t["mask"].long()

        return image, label, stem


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ORIGA DATASET                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ORIGADataset(Dataset):
    """
    ORIGA fundus dataset — single-file masks in Masks/ (or Masks_Square/).
    Mask encoding: rank-by-area (same as REFUGE) — largest region = background,
    intermediate = disc rim (1), smallest = cup (2).
    Images: ORIGA/Images/001.jpg  → Masks: ORIGA/Masks/001.png (or .bmp)
    """
    def __init__(self, split="test", transform=None):
        self.transform = transform
        self.split     = split
        base = get_origa_base()
        if base is None:
            raise FileNotFoundError(
                "Could not find ORIGA folder under /kaggle/input.\n"
                "Please verify that arnavjain1/glaucoma-datasets is added as input."
            )

        # Match corresponding image and mask folders
        selected_img_dir = None
        selected_msk_dir = None
        for img_name, msk_name in [
            ("Images_Square",  "Masks_Square"),
            ("Images_Cropped", "Masks_Cropped"),
            ("Images",         "Masks"),
        ]:
            i_p = os.path.join(base, img_name)
            m_p = os.path.join(base, msk_name)
            if os.path.isdir(i_p) and os.path.isdir(m_p):
                imgs_test = glob.glob(os.path.join(i_p, "*.*"))
                if len(imgs_test) > 0:
                    selected_img_dir = i_p
                    selected_msk_dir = m_p
                    break

        if selected_img_dir is None:
            imgs = []
            for root, _, files in os.walk(base):
                if "mask" not in root.lower():
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg", ".png")):
                            imgs.append(os.path.join(root, f))
            self.imgs = sorted(list(set(imgs)))
            selected_msk_dir = os.path.join(base, "Masks")
        else:
            imgs = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.bmp"]:
                imgs.extend(glob.glob(os.path.join(selected_img_dir, ext)))
            self.imgs = sorted(list(set(imgs)))

        # Split support if used as source domain
        if split == "train":
            split_idx = int(0.8 * len(self.imgs))
            self.imgs = self.imgs[:split_idx]
        elif split == "val":
            split_idx = int(0.8 * len(self.imgs))
            self.imgs = self.imgs[split_idx:]

        # Build mask lookup: stem -> mask_path
        self.mask_map = {}
        search_msk_dirs = [selected_msk_dir] if selected_msk_dir and os.path.isdir(selected_msk_dir) else []
        for msk_subdir in ["Masks_Square", "Masks_Cropped", "Masks"]:
            p = os.path.join(base, msk_subdir)
            if os.path.isdir(p) and p not in search_msk_dirs:
                search_msk_dirs.append(p)

        for md in search_msk_dirs:
            for f in os.listdir(md):
                if f.lower().endswith((".png", ".bmp", ".jpg", ".jpeg")):
                    stem = os.path.splitext(f)[0].lower()
                    key  = stem.replace("_mask", "").replace("-mask", "")
                    if key not in self.mask_map:
                        self.mask_map[key] = os.path.join(md, f)
                    if stem not in self.mask_map:
                        self.mask_map[stem] = os.path.join(md, f)

        n_masks = 0
        for p in self.imgs:
            stem = os.path.splitext(os.path.basename(p))[0].lower()
            key  = stem.replace("_mask", "").replace("-mask", "")
            if stem in self.mask_map or key in self.mask_map:
                n_masks += 1

        if len(self.imgs) == 0:
            raise FileNotFoundError(f"No ORIGA images found for split '{split}' under {base}.")
        print(f"  ORIGA [{split}]: {len(self.imgs)} images, {n_masks} with masks "
              f"(folder: {os.path.basename(selected_img_dir or base)})")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        raw_stem = os.path.splitext(os.path.basename(img_path))[0]
        stem     = raw_stem.lower()
        image    = np.array(Image.open(img_path).convert("RGB"))

        key = stem.replace("_mask", "").replace("-mask", "")
        mask_path = self.mask_map.get(stem) or self.mask_map.get(key)
        label = refuge_mask_to_label(mask_path, target_shape=image.shape[:2]) \
                if mask_path else np.zeros(image.shape[:2], dtype=np.int64)

        if label.shape[:2] != image.shape[:2]:
            label = np.array(Image.fromarray(label.astype(np.uint8)).resize(
                (image.shape[1], image.shape[0]), resample=Image.NEAREST
            )).astype(np.int64)

        if self.transform:
            t = self.transform(image=image, mask=label)
            image = t["image"]; label = t["mask"].long()

        return image, label, raw_stem


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RIM-ONE DATASET (Target Domain — Stereo / Clinical)                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

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
    if p: return p
    p = auto_find_dir(["rimone"], cands)
    if p: return p
    return None

class RimOneDataset(Dataset):
    def __init__(self, split="test", transform=None):
        self.transform = transform
        base = get_rimone_base()
        if base is None:
            raise FileNotFoundError(
                "Could not find RIM-ONE folder under /kaggle/input.\n"
                "Please add 'orvile/rim-one-retinal-dataset-for-assessing-glaucoma' or 'riadalmadani/rimone-dl'."
            )

        imgs = []
        disc_masks, cup_masks, multi_masks = {}, {}, {}

        for root, dirs, files in os.walk(base):
            root_lower = root.lower()
            is_split_match = ("test_set" in root_lower or "test" in root_lower) if split == "test" else ("training_set" in root_lower or "train" in root_lower)
            for f in files:
                fl = f.lower()
                full_p = os.path.join(root, f)
                is_mask = any(kw in fl for kw in ["mask", "disc", "cup", "-od", "-oc", "seg", "_d.", "_c."])
                is_img  = fl.endswith((".png", ".jpg", ".jpeg", ".bmp")) and not f.startswith(".")

                if is_img and not is_mask and is_split_match:
                    imgs.append(full_p)
                elif is_mask:
                    stem_clean = os.path.splitext(f)[0].lower()
                    for term in ["-disc", "-cup", "_disc", "_cup", "disc", "cup", "-od", "-oc", "_d", "_c", "_mask"]:
                        stem_clean = stem_clean.replace(term, "")
                    stem_clean = stem_clean.strip("-_ ")
                    if any(k in fl for k in ["disc", "-od", "_d."]):
                        disc_masks[stem_clean] = full_p
                    elif any(k in fl for k in ["cup", "-oc", "_c."]):
                        cup_masks[stem_clean] = full_p
                    else:
                        multi_masks[stem_clean] = full_p

        if len(imgs) == 0:
            for root, dirs, files in os.walk(base):
                for f in files:
                    fl = f.lower()
                    if fl.endswith((".png", ".jpg", ".jpeg", ".bmp")) and not any(k in fl for k in ["disc", "cup", "mask"]):
                        imgs.append(os.path.join(root, f))

        self.imgs = sorted(list(set(imgs)))
        self.disc_masks  = disc_masks
        self.cup_masks   = cup_masks
        self.multi_masks = multi_masks

        print(f"  RIM-ONE [{split}]: {len(self.imgs)} images, "
              f"{len(disc_masks)} disc / {len(cup_masks)} cup / {len(multi_masks)} combined masks")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        image    = np.array(Image.open(img_path).convert("RGB"))
        H, W     = image.shape[:2]

        clean_k = stem.lower().strip("-_ ")
        disc_p  = self.disc_masks.get(clean_k)
        cup_p   = self.cup_masks.get(clean_k)
        multi_p = self.multi_masks.get(clean_k)

        label = np.zeros((H, W), dtype=np.int64)
        if disc_p and cup_p:
            d_arr = np.array(Image.open(disc_p).convert("L"))
            c_arr = np.array(Image.open(cup_p).convert("L"))
            if d_arr.shape != (H, W): d_arr = np.array(Image.fromarray(d_arr).resize((W, H), Image.NEAREST))
            if c_arr.shape != (H, W): c_arr = np.array(Image.fromarray(c_arr).resize((W, H), Image.NEAREST))
            disc = (d_arr > 127).astype(np.uint8)
            cup  = (c_arr > 127).astype(np.uint8) * disc
            label[disc == 1] = 1
            label[cup  == 1] = 2
        elif multi_p:
            label = refuge_mask_to_label(multi_p, target_shape=(H, W))

        if self.transform:
            t = self.transform(image=image, mask=label)
            image, label = t["image"], t["mask"].long()

        return image, label, stem


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  G1020 DATASET (Target Domain — Large Benchmark)                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_g1020_base():
    cands = [
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/G1020",
        "/kaggle/input/glaucoma-datasets/G1020",
        "/kaggle/input/g1020-dataset",
        "/kaggle/input/g1020",
    ]
    p = auto_find_dir(["g1020"], cands)
    return p

class G1020Dataset(Dataset):
    def __init__(self, split="test", transform=None):
        self.transform = transform
        base = get_g1020_base()
        if base is None:
            raise FileNotFoundError("Could not find G1020 folder under /kaggle/input.")

        imgs = []
        for root, _, files in os.walk(base):
            if "mask" not in root.lower():
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        imgs.append(os.path.join(root, f))
        self.imgs = sorted(list(set(imgs)))

        self.mask_map = {}
        for root, _, files in os.walk(base):
            if "mask" in root.lower():
                for f in files:
                    if f.lower().endswith((".png", ".bmp", ".jpg")):
                        stem = os.path.splitext(f)[0].lower().replace("_mask", "").replace("-mask", "")
                        self.mask_map[stem] = os.path.join(root, f)

        print(f"  G1020 [{split}]: {len(self.imgs)} images, {len(self.mask_map)} masks found")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        image    = np.array(Image.open(img_path).convert("RGB"))
        key      = stem.lower().replace("_mask", "").replace("-mask", "")
        mask_p   = self.mask_map.get(key)
        label    = refuge_mask_to_label(mask_p, target_shape=image.shape[:2]) if mask_p else np.zeros(image.shape[:2], dtype=np.int64)

        if self.transform:
            t = self.transform(image=image, mask=label)
            image, label = t["image"], t["mask"].long()

        return image, label, stem


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAPILA DATASET (Target Domain — Multi-Expert Annotations)               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_papila_base():
    cands = [
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/PAPILA",
        "/kaggle/input/glaucoma-datasets/PAPILA",
        "/kaggle/input/papila-dataset",
        "/kaggle/input/papila",
    ]
    p = auto_find_dir(["papila"], cands)
    return p

class PAPILADataset(Dataset):
    def __init__(self, split="test", transform=None):
        self.transform = transform
        base = get_papila_base()
        if base is None:
            raise FileNotFoundError("Could not find PAPILA folder under /kaggle/input.")

        imgs = []
        for root, _, files in os.walk(base):
            if "mask" not in root.lower() and "expert" not in root.lower():
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        imgs.append(os.path.join(root, f))
        self.imgs = sorted(list(set(imgs)))

        self.disc_masks, self.cup_masks = {}, {}
        for root, _, files in os.walk(base):
            for f in files:
                fl = f.lower()
                full_p = os.path.join(root, f)
                stem = os.path.splitext(f)[0].lower()
                if "od" in fl or "disc" in fl:
                    self.disc_masks[stem] = full_p
                elif "oc" in fl or "cup" in fl:
                    self.cup_masks[stem] = full_p

        print(f"  PAPILA [{split}]: {len(self.imgs)} images found")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        image    = np.array(Image.open(img_path).convert("RGB"))
        H, W     = image.shape[:2]

        disc_p = self.disc_masks.get(stem.lower())
        cup_p  = self.cup_masks.get(stem.lower())
        if disc_p and cup_p:
            label = drishti_mask_to_label(disc_p, cup_p, target_shape=(H, W))
        else:
            label = np.zeros((H, W), dtype=np.int64)

        if self.transform:
            t = self.transform(image=image, mask=label)
            image, label = t["image"], t["mask"].long()

        return image, label, stem


# ── Unified Dataset Factory ──────────────────────────────────────────────────

def get_dataset(domain_name: str, split: str = "test", transform = None) -> Dataset:
    """
    Unified dataset factory to switch source/target domains cleanly:
      - 'REFUGE'   (Source domain or Target benchmark)
      - 'Drishti'  (Target benchmark 1 — 51/101 images)
      - 'ORIGA'    (Target benchmark 2 — 650 images)
      - 'RIM-ONE'  (Target benchmark 3 — 159/466 images)
      - 'G1020'    (Target benchmark 4 — 1020 images)
      - 'PAPILA'   (Target benchmark 5 — 470 images)
    """
    dn = str(domain_name).strip().lower()
    if "refuge" in dn:
        return REFUGEDataset(split=split, transform=transform)
    elif "origa" in dn:
        return ORIGADataset(split=split, transform=transform)
    elif "drishti" in dn:
        return DrishtiDataset(split=split, transform=transform)
    elif "rim" in dn:
        return RimOneDataset(split=split, transform=transform)
    elif "g1020" in dn:
        return G1020Dataset(split=split, transform=transform)
    elif "papila" in dn:
        return PAPILADataset(split=split, transform=transform)
    else:
        raise ValueError(
            f"Unsupported domain '{domain_name}'. Please select from: 'REFUGE', 'Drishti', 'ORIGA', 'RIM-ONE', 'G1020', 'PAPILA'."
        )


# ── Quick dataset sanity check ────────────────────────────────────────────────

def check_datasets():
    print("\n── Dataset Sanity Check ──────────────────────────────────")
    print(f"  Configured: SOURCE = {CFG.SOURCE}  →  TARGET = {CFG.TARGET}")
    for label, get_ds in [
        (f"Source ({CFG.SOURCE} train)", lambda: get_dataset(CFG.SOURCE, "train", val_tfm())),
        (f"Source ({CFG.SOURCE} val)",   lambda: get_dataset(CFG.SOURCE, "val",   val_tfm())),
        (f"Target ({CFG.TARGET} test)",  lambda: get_dataset(CFG.TARGET, "test",  val_tfm())),
    ]:
        try:
            ds = get_ds()
            img, lbl, stem = ds[0]
            vals, counts = torch.unique(lbl, return_counts=True)
            pcts = {int(v): f"{float(c)/lbl.numel()*100:.1f}%" for v, c in zip(vals, counts)}
            print(f"  {label} OK │ classes: {pcts} │ stem: {stem}")
        except Exception as e:
            print(f"  {label} FAILED: {e}")
    print("─"*58)

check_datasets()




# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_model():
    return smp.Unet(
        encoder_name    = CFG.ENCODER,
        encoder_weights = CFG.ENCODER_WEIGHTS,
        in_channels     = 3,
        classes         = CFG.NUM_CLASSES,
        activation      = None   # raw logits
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LOSS                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.tensor(CFG.CLASS_WEIGHTS, dtype=torch.float32).to(CFG.DEVICE)
        self.ce   = nn.CrossEntropyLoss(weight=w)
        self.dice = smp.losses.DiceLoss(mode="multiclass")
    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  METRICS                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def dice_score(pred, gt, eps=1e-6):
    p = (pred > 0).astype(np.float32)
    g = (gt > 0).astype(np.float32)
    intersection = float((p * g).sum())
    cardinality  = float(p.sum() + g.sum())
    if cardinality == 0.0:
        return 1.0 if intersection == 0.0 else 0.0
    return float((2.0 * intersection + eps) / (cardinality + eps))

def evaluate(probs, labels, *args, **kwargs):
    """
    Computes segmentation metrics: Disc Dice and Cup Dice.
    Robust to [1, C, H, W], [C, H, W], [1, H, W], [H, W], torch tensors, and numpy arrays.
    """
    if torch.is_tensor(probs):
        while probs.dim() > 3:
            probs = probs.squeeze(0)
        pred = probs.argmax(dim=0).cpu().numpy()
    elif isinstance(probs, np.ndarray):
        while probs.ndim > 3:
            probs = probs.squeeze(0)
        pred = probs.argmax(axis=0)

    if torch.is_tensor(labels):
        while labels.dim() > 2:
            labels = labels.squeeze(0)
        gt = labels.cpu().numpy()
    elif isinstance(labels, np.ndarray):
        while labels.ndim > 2:
            labels = labels.squeeze(0)
        gt = labels

    pred_disc = (pred >= 1)
    gt_disc   = (gt >= 1)
    pred_cup  = (pred == 2)
    gt_cup    = (gt == 2)

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
    # Validation uses batch_size=8 for 8x faster parallel GPU inference (zero accuracy impact)
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
        # ── train ────────────────────────────────────────────────────────
        model.train(); ep_loss = 0
        for imgs, lbls, _ in tr_loader:
            imgs = imgs.to(CFG.DEVICE); lbls = lbls.to(CFG.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward(); optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        history["loss"].append(ep_loss / len(tr_loader))

        # ── val (fast parallel GPU inference) ───────────────────────────
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
                print(f"  Early stop triggered at epoch {epoch} (no improvement for {CFG.PATIENCE} epochs)")
                break

    # Learning curve figure
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(history["loss"],     color="steelblue"); a1.set_title("Training Loss")
    a2.plot(history["cup_dice"], color="tomato");    a2.set_title("Val Cup Dice")
    for a in (a1, a2): a.set_xlabel("Epoch"); a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUT_DIR, "fig0_learning_curves.png"), dpi=150)
    plt.close()

    print(f"\n  ✓ Training done.  Best Val Cup Dice = {best_cup:.4f}")
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

    bn_mods = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.BatchNorm2d)]
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
            x = inp[0].detach().cpu()   # [B, C, H, W]  — input to BN
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

    hooks = [m.register_forward_hook(make_hook(i))
             for i, (_, m) in enumerate(bn_mods)]

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
# ║  PMSA-BN MODULE  (Steps 3 – 8)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class PMSABatchNorm2d(nn.Module):
    """
    Drop-in replacement for nn.BatchNorm2d that performs PMSA-BN adaptation.

    Steps implemented in forward():
      3. Class-conditional target stats (with target-anchor fallback)
      4. Scale-invariant Bayesian shrinkage (resolution-independent)
      5. Prior-matched recombination (smooth blend with target prior)
      6. Scale-free 2-Wasserstein dist D_c per channel
      7. Centered shift-adaptive channel mixing weight α_c
      8. Mix statistics & normalise
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
        self.last_Dc = None; self.last_alpha = None   # for figures

    # ── Step 3 ───────────────────────────────────────────────────────────

    def _cls_stats(self, x, mask):
        B, C, H, W = x.shape
        mu = torch.zeros(self.K, C, device=x.device)
        va = torch.zeros(self.K, C, device=x.device)
        Nk = torch.zeros(self.K, device=x.device)

        # Global spatial target statistics (domain anchor)
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

    # ── Step 4 ───────────────────────────────────────────────────────────

    def _shrink(self, tgt_mu, tgt_va, Nk, tgt_img_mu, tgt_img_va, H, W):
        """
        Resolution-invariant Bayesian shrinkage:
        Shrinks class-conditional estimates toward target image prior anchors:
          mu_anchor[k] = tgt_img_mu + (src_cls_mean[k] - src_global_mean)
        This preserves target domain lighting while regularizing class offsets.
        """
        smu = torch.zeros_like(tgt_mu)
        sva = torch.zeros_like(tgt_va)

        src_global_mu = (self.src_prior[:, None] * self.src_cls_mean).sum(0)
        scale_factor = (384.0 * 384.0) / float(H * W)

        for k in range(self.K):
            n_eff = Nk[k].item() * scale_factor
            g  = n_eff / (n_eff + self.kmu)
            gs = max(0.0, n_eff - 1.0) / (max(0.0, n_eff - 1.0) + self.ks)

            anchor_mu = tgt_img_mu + (self.src_cls_mean[k] - src_global_mu)
            smu[k] = g * tgt_mu[k] + (1.0 - g) * anchor_mu
            sva[k] = gs * tgt_va[k] + (1.0 - gs) * tgt_img_va
        return smu, sva

    # ── Step 5 ───────────────────────────────────────────────────────────

    def _prior_match(self, cls_mu, cls_va, Nk):
        """
        Prior-Matched Statistics (PMS):
        Smooth blend between empirical target prior and source prior.
        Law of Total Variance eliminates label-proportion confound.
        """
        pi_tgt = Nk / Nk.sum().clamp(min=1e-6)
        pi = 0.5 * pi_tgt + 0.5 * self.src_prior

        pm_mu   = (pi[:, None] * cls_mu).sum(0)                   # [C]
        within  = (pi[:, None] * cls_va).sum(0)
        diff    = cls_mu - pm_mu[None, :]
        between = (pi[:, None] * (diff**2)).sum(0)
        return pm_mu, within + between

    # ── Step 6 ───────────────────────────────────────────────────────────

    def _w2_dist(self, pm_mu, pm_va, src_mu, src_va, eps=1e-6):
        """
        Scale-free normalised 2-Wasserstein distance per channel:
          D_c = [(μ_PM − μ_src)² + (σ_PM − σ_src)²] / (σ²_src + ε)
        """
        return ((pm_mu - src_mu)**2 + (pm_va.clamp(0).sqrt() - src_va.clamp(0).sqrt())**2) \
               / (src_va + eps)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, x):
        if not self.adapt_mode or self.pseudo_mask is None:
            return self.bn(x)        # training-time: standard BN

        B, C, H, W = x.shape
        mask = self.pseudo_mask.to(x.device)
        if mask.shape[2] != H or mask.shape[3] != W:
            mask = F.interpolate(mask, (H, W), mode="bilinear", align_corners=False)
        mask = mask / mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
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
                tgt_mu, tgt_va, Nk, img_mu, img_va = self._cls_stats(x, mask)    # Step 3
                shr_mu, shr_va = self._shrink(tgt_mu, tgt_va, Nk, img_mu, img_va, H, W) if use_sh else (tgt_mu, tgt_va)  # Step 4
                if use_pm:
                    pm_mu, pm_va = self._prior_match(shr_mu, shr_va, Nk) # Step 5
                else:
                    pm_mu = shr_mu.mean(0)
                    pm_va = shr_va.mean(0)

            if use_pc:
                Dc = self._w2_dist(pm_mu, pm_va, src_mu, src_va)        # Step 6
                dc_mean = Dc.mean().detach()
                dc_std  = Dc.std().detach().clamp(min=1e-6)

                # Center-anchored dynamic modulation around alpha_0 = 0.50
                dev = (Dc - dc_mean) / (self.tau * dc_std)
                half_span = (self.amax - self.amin) / 2.0
                alpha = torch.clamp(0.50 + half_span * torch.tanh(dev), self.amin, self.amax) # Step 7
            else:
                Dc    = torch.zeros_like(src_mu)
                alpha = torch.full_like(src_mu, (self.amin + self.amax) / 2.0)   # fixed global α=0.5

            self.last_Dc    = Dc.cpu()
            self.last_alpha = alpha.cpu()

            hat_mu = alpha * pm_mu  + (1.0 - alpha) * src_mu               # Step 8
            hat_va = alpha * pm_va  + (1.0 - alpha) * src_va

        hat_std = (hat_va + self.bn.eps).clamp(1e-12).sqrt()
        xn = (x - hat_mu.view(1,C,1,1)) / hat_std.view(1,C,1,1)
        return self.bn.weight.view(1,C,1,1) * xn + self.bn.bias.view(1,C,1,1)


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    assert len(pmsa) == len(means), \
        f"PMSA layers ({len(pmsa)}) ≠ saved stat entries ({len(means)})"
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
# ║  PSEUDO-MASK (Step 2)                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def make_pseudo_mask(probs, tau_hi=CFG.TAU_HI):
    """
    Soft posterior responsibilities with anatomical consistency.
    - Continuous spatial responsibilities prevent zero-count collapse.
    - Anatomical gating: optic cup probability is bounded by total disc presence.
    - Normalised on the probability simplex so Σ_k P(k|x) = 1 at every pixel.
    """
    disc_prob = (probs[:, 1:2] + probs[:, 2:3]).clamp(0.0, 1.0)
    cup_prob  = probs[:, 2:3] * disc_prob
    rim_prob  = probs[:, 1:2]
    bg_prob   = probs[:, 0:1]

    soft_mask = torch.cat([bg_prob, rim_prob, cup_prob], dim=1)
    soft_mask = soft_mask / soft_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return soft_mask


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  BASELINES + PMSA-BN PREDICT                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def predict_source_only(model, image):
    """Baseline 1 — no adaptation, source statistics only."""
    model.eval()
    return F.softmax(model(image), dim=1)


@torch.no_grad()
def predict_adabn(model, image):
    """
    Baseline 2 — AdaBN.
    Sets ALL BN running stats to target stats (momentum = 1).
    No class splitting, no prior matching, no per-channel weighting.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train(); m.momentum = 1.0
    _ = model(image)   # running stats now = target stats
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()
    return F.softmax(model(image), dim=1)


@torch.no_grad()
def predict_alpha_bn(src_model, image, alpha=CFG.ALPHA_BN_WEIGHT):
    """
    Baseline 3 — α-BN.
    Single global mixing weight α for ALL channels:
      running_stat = α·target + (1−α)·source
    """
    # 1. Snapshot original source running stats
    src_stats = [(m.running_mean.clone(), m.running_var.clone())
                 for m in src_model.modules() if isinstance(m, nn.BatchNorm2d)]

    # 2. Update running stats to target via train-mode pass
    for m in src_model.modules():
        if isinstance(m, nn.BatchNorm2d): m.train(); m.momentum = 1.0
    _ = src_model(image)

    # 3. Blend target stats with original source stats
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
    Pass 1: α-BN coarse prediction to initialize pseudo-mask
    Pass 2: Adapted forward pass through PMSA-BN layers (Steps 3–8)
    """
    if src_model is not None:
        probs = predict_alpha_bn(copy.deepcopy(src_model).to(image.device), image)
    else:
        probs = predict_alpha_bn(copy.deepcopy(model).to(image.device), image)

    set_adapt(model, True)
    diag_list = []
    for _ in range(em_iters):
        mask = make_pseudo_mask(probs)   # Step 2
        set_mask(model, mask)
        probs = F.softmax(model(image), dim=1)   # Steps 3–8
        diag_list.append(get_diag(model))

    set_adapt(model, False)
    set_mask(model, None)
    return probs, diag_list


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 — EVALUATE ALL METHODS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_evaluation(src_model, pmsa_model):
    print("\n" + "═"*60)
    print(f"  STEP 3 — Evaluate All Methods on {CFG.TARGET} (Target Domain)")
    print("═"*60)

    loader = DataLoader(get_dataset(CFG.TARGET, "test", val_tfm()),
                        batch_size=1, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)



    method_res = defaultdict(list)
    all_diag   = []
    true_cdrs  = []; pred_cdrs = defaultdict(list)

    def cdr(pred_np):
        d = (pred_np >= 1).sum(); c = (pred_np == 2).sum()
        return float(c) / max(float(d), 1.0)

    src_model.eval()

    for images, labels, stems in tqdm(loader, desc="  Evaluating"):
        images = images.to(CFG.DEVICE)
        gt_np  = labels.squeeze().numpy()
        true_cdrs.append(cdr(gt_np))

        # 1. Source Only
        p = predict_source_only(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        method_res["Source Only"].append(evaluate(p.cpu(), labels))
        pred_cdrs["Source Only"].append(cdr(p.argmax(1).squeeze().cpu().numpy()))

        # 2. AdaBN
        p = predict_adabn(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        method_res["AdaBN"].append(evaluate(p.cpu(), labels))
        pred_cdrs["AdaBN"].append(cdr(p.argmax(1).squeeze().cpu().numpy()))

        # 3. α-BN
        abn_m = copy.deepcopy(src_model).to(CFG.DEVICE)
        p = predict_alpha_bn(abn_m, images)
        method_res["α-BN (α=0.5)"].append(evaluate(p.cpu(), labels))
        pred_cdrs["α-BN (α=0.5)"].append(cdr(p.argmax(1).squeeze().cpu().numpy()))

        # 4. PMSA-BN
        pm = copy.deepcopy(pmsa_model).to(CFG.DEVICE)
        p, diags = predict_pmsa(pm, images, src_model=src_model)
        method_res["PMSA-BN (Ours)"].append(evaluate(p.cpu(), labels))
        pred_cdrs["PMSA-BN (Ours)"].append(cdr(p.argmax(1).squeeze().cpu().numpy()))
        if diags: all_diag.append(diags[-1])

    # ── Print table ───────────────────────────────────────────────────────
    aggregated = {m: agg(r) for m, r in method_res.items()}
    print("\n" + "═"*56)
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

    return aggregated, all_diag, true_cdrs, dict(pred_cdrs)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 — ABLATION                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_ablation(src_model, pmsa_model):
    print("\n" + "═"*60)
    print("  STEP 4 — Ablation Study (Each Row +1 Component)")
    print("═"*60)

    loader = DataLoader(get_dataset(CFG.TARGET, "test", val_tfm()),
                        batch_size=1, shuffle=False, num_workers=0)

    configs = [
        # label                         cs     pm     sh     pc
        ("1. Source Only",              False, False, False, False),
        ("2. AdaBN",                    None,  None,  None,  None),
        ("3. + Class-Split Stats",      True,  False, False, False),
        ("4. + Prior Matching",         True,  True,  False, False),
        ("5. + Shrinkage (B=1 safe)",   True,  True,  True,  False),
        ("6. PMSA-BN (Full Method)",    True,  True,  True,  True),
    ]

    abl_res = {}
    for (label, cs, pm_flag, sh, pc) in configs:
        res = []
        for images, labels, _ in tqdm(loader, desc=f"  {label[:30]}", leave=False):
            images = images.to(CFG.DEVICE)
            if label.startswith("1."):
                p = predict_source_only(copy.deepcopy(src_model).to(CFG.DEVICE), images)
            elif label.startswith("2."):
                p = predict_adabn(copy.deepcopy(src_model).to(CFG.DEVICE), images)
            else:
                pm = copy.deepcopy(pmsa_model).to(CFG.DEVICE); pm.eval()
                probs = predict_alpha_bn(copy.deepcopy(src_model).to(CFG.DEVICE), images)

                set_adapt(pm, True)
                mask = make_pseudo_mask(probs)
                set_mask(pm, mask)
                probs = _ablation_step(pm, images, cs, pm_flag, sh, pc)
                set_adapt(pm, False)
                p = probs
            res.append(evaluate(p.cpu(), labels))

        r = agg(res)
        abl_res[label] = r
        dd = r["disc_dice"]; cd = r["cup_dice"]
        print(f"  {label:<32} │ Disc {dd[0]:.4f} │ Cup {cd[0]:.4f}")

    return abl_res


def _ablation_step(model, image, cs, pm_flag, sh, pc):
    """Single adapted forward with selective component toggles."""
    for lyr in model.modules():
        if isinstance(lyr, PMSABatchNorm2d):
            lyr.ablation_flags = {
                "class_split": cs,
                "prior_match": pm_flag,
                "shrinkage":   sh,
                "per_channel": pc,
            }
    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)
    return probs


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5 — PAPER FIGURES                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_bar_comparison(aggregated, path):
    methods = list(aggregated.keys())
    colors  = ["#95a5a6", "#e74c3c", "#e67e22", "#2980b9"]
    metrics = [("disc_dice", "Disc Dice ↑"),
               ("cup_dice",  "Cup Dice ↑")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"PMSA-BN vs. Baselines  ({CFG.SOURCE} → {CFG.TARGET})",
                 fontsize=13, fontweight="bold")

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
        ax.text(x[best], y_star, "★",
                ha="center", fontsize=14, color="gold", fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")


def plot_alpha_heatmap(all_diag, path, max_layers=20, max_ch=64):
    if not all_diag: return
    lkeys = sorted(all_diag[0].keys())[:max_layers]
    mat   = np.zeros((len(lkeys), max_ch))
    for diag in all_diag:
        for i, lk in enumerate(lkeys):
            if lk in diag:
                a = diag[lk]["alpha"].numpy(); C = min(len(a), max_ch)
                mat[i, :C] += a[:C]
    mat /= max(len(all_diag), 1)

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(mat, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
                xticklabels=8,
                cbar_kws={"label": r"$\alpha_c$  (0=source, 1=target)"})
    ax.set_xlabel("Channel index $c$", fontsize=11)
    ax.set_ylabel("Layer (encoder → decoder)", fontsize=11)
    ax.set_title(r"Per-Layer Per-Channel Mixing Weights $\alpha_c$" + "\n"
                 "Early layers adapt more; deep layers stay close to source", fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")


def plot_cdr_scatter(true_cdrs, pred_cdrs, path):
    methods = ["Source Only", "AdaBN", "α-BN (α=0.5)", "PMSA-BN (Ours)"]
    colors  = ["#7f8c8d", "#e74c3c", "#e67e22", "#2980b9"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
    for ax, m, col in zip(axes, methods, colors):
        preds = pred_cdrs.get(m, [])
        if not preds: continue
        trues = true_cdrs[:len(preds)]
        ax.scatter(trues, preds, alpha=0.55, s=22, color=col, edgecolors="none")
        ax.plot([0,1],[0,1], "k--", lw=1.2)
        if len(trues) > 1 and np.std(trues) > 1e-4 and np.std(preds) > 1e-4:
            r = float(np.corrcoef(trues, preds)[0,1])
            c = np.polyfit(trues, preds, 1)
            xs = np.linspace(min(trues), max(trues), 80)
            ax.plot(xs, np.polyval(c,xs), color=col, lw=1.5, alpha=0.8)
        else:
            r = 0.0
        ax.set_title(f"{m}\nr = {r:.3f}", fontsize=10)
        ax.set_xlabel("True CDR"); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Predicted CDR")
    fig.suptitle("Predicted vs. True Cup-to-Disc Ratio", fontsize=12,
                 fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")


def plot_ablation(abl_res, path):
    labels = list(abl_res.keys())
    disc_m = [abl_res[l]["disc_dice"][0] for l in labels]
    cup_m  = [abl_res[l]["cup_dice"][0]  for l in labels]
    y      = np.arange(len(labels))
    cols   = plt.cm.Blues_r(np.linspace(0.15, 0.75, len(labels)))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, vals, title in zip(
        axes,
        [disc_m, cup_m],
        ["Disc Dice ↑ (Ablation)", "Cup Dice ↑ (Ablation)"]
    ):
        ax.barh(y, vals, color=cols, edgecolor="black", lw=0.5, alpha=0.88)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11); ax.grid(axis="x", alpha=0.3)
        best = np.argmax(vals)
        ax.get_yticklabels()[best].set_fontweight("bold")
        ax.get_yticklabels()[best].set_color("#2980b9")

    plt.suptitle("Ablation — Each Row Adds One Component of PMSA-BN",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")




# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n" + "╔"+"═"*58+"╗")
    print("║   PMSA-BN — Kaggle Proof of Concept                     ║")
    print(f"║   Source: {CFG.SOURCE:<12}  →  Target: {CFG.TARGET:<15} ║")
    print("╚"+"═"*58+"╝")



    os.makedirs(CFG.OUT_DIR, exist_ok=True)

    # ── Step 1 & 2: Train source model & collect statistics ───────────────
    ckpt_path  = find_checkpoint()
    stats_path = find_source_stats()
    if not CFG.TRAIN_FROM_SCRATCH and ckpt_path and stats_path:
        print(f"\n  ✓ Discovered and reusing checkpoint: {ckpt_path}")
        print(f"  ✓ Discovered and reusing source stats: {stats_path}")
        src_model = build_model().to(CFG.DEVICE)
        ckpt = torch.load(ckpt_path, map_location=CFG.DEVICE)
        src_model.load_state_dict(ckpt["state"] if "state" in ckpt else ckpt)
        src_model.eval()
        stats = torch.load(stats_path, map_location="cpu")
    else:
        src_model = train_source()
        stats = collect_source_stats(src_model)


    # Build PMSA model
    pmsa_model = replace_bn(copy.deepcopy(src_model))
    inject_stats(pmsa_model, stats)
    pmsa_model.eval().to(CFG.DEVICE)

    # ── Step 3: Evaluate ─────────────────────────────────────────────────
    aggregated, all_diag, true_cdrs, pred_cdrs = run_evaluation(src_model, pmsa_model)

    # ── Step 4: Ablation ─────────────────────────────────────────────────
    if getattr(CFG, "RUN_ABLATION", False):
        abl_res = run_ablation(src_model, pmsa_model)
    else:
        print("\n  [INFO] Ablation study suspended to save time. (Set CFG.RUN_ABLATION = True to re-enable)")
        abl_res = {}

    # ── Step 5: Figures ──────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 5 — Generating Figures")
    print("═"*60)

    plot_bar_comparison(aggregated,
        os.path.join(CFG.OUT_DIR, "fig1_comparison.png"))
    plot_alpha_heatmap(all_diag,
        os.path.join(CFG.OUT_DIR, "fig2_alpha_heatmap.png"))
    plot_cdr_scatter(true_cdrs, pred_cdrs,
        os.path.join(CFG.OUT_DIR, "fig3_cdr_scatter.png"))
    if abl_res:
        plot_ablation(abl_res,
            os.path.join(CFG.OUT_DIR, "fig4_ablation.png"))

    # ── Save JSON ────────────────────────────────────────────────────────
    def jsonify(d):
        if isinstance(d, dict):  return {k: jsonify(v) for k,v in d.items()}
        if isinstance(d, (list, tuple)): return [jsonify(x) for x in d]
        if isinstance(d, (np.float32, np.float64)): return float(d)
        if isinstance(d, (np.int32,  np.int64)):   return int(d)
        return d
    with open(os.path.join(CFG.OUT_DIR, "results.json"),  "w") as f:
        json.dump(jsonify(aggregated), f, indent=2)
    with open(os.path.join(CFG.OUT_DIR, "ablation.json"), "w") as f:
        json.dump(jsonify(abl_res),    f, indent=2)

    print("\n" + "╔"+"═"*58+"╗")
    print("║   Done!  All outputs in /kaggle/working/                 ║")
    print("║   fig1_comparison.png    — bar chart, 4 methods          ║")
    print("║   fig2_alpha_heatmap.png — per-layer α_c                 ║")
    print("║   fig3_cdr_scatter.png   — CDR scatter (4 panels)        ║")
    print("║   fig4_ablation.png      — ablation bars                  ║")
    print("║   results.json  |  ablation.json                         ║")
    print("╚"+"═"*58+"╝")


main()
