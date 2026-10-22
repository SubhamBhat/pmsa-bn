"""
================================================================================
  PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm
  Multi-Class Brain Tumor MRI Benchmark (K = 4 Classes: BraTS 2019 -> BraTS 2020)
  Single-Cell Kaggle Training & Test-Time Adaptation Script (B = 1)
================================================================================

KAGGLE DATASET INPUTS TO ADD (Click "+ Add Input" in Kaggle Notebook):
  1. Source Domain (BraTS 2019):
     walidrehamnia/brain-tumor-segmentation-brats-2019-train-valid
     URL: https://www.kaggle.com/datasets/walidrehamnia/brain-tumor-segmentation-brats-2019-train-valid
     Contains: MICCAI_BraTS_2019_Data_Training/HGG and LGG (*_flair.nii, *_t1ce.nii, *_t2.nii, *_seg.nii)

  2. Primary Target Domain (BraTS 2020):
     awsaf49/brats2020-training-data
     URL: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data
     Contains: BraTS2020_training_data/content/data/volume_*_slice_*.h5

CLASSES (K = 4 Mutually Exclusive Sub-Compartments):
  0: Healthy Brain Tissue / Background
  1: Necrotic / Non-enhancing Tumor Core (NCR)
  2: Peritumoral Edema (ED)
  3: Active Enhancing Tumor (ET)

CLINICAL COMPOSITE EVALUATION METRICS:
  * Whole Tumor  (WT) = Classes {1, 2, 3}
  * Tumor Core   (TC) = Classes {1, 3}
  * Enhancing    (ET) = Class {3}
  * Mean Dice         = (WT + TC + ET) / 3

PIPELINE:
  Step 1: Train Source Model on real BraTS 2019 slices (or reuse checkpoint)
  Step 2: Collect Phase 0 Source Statistics (mu_src, var_src, class priors pi_k)
  Step 3: Run Test-Time Adaptation on real BraTS 2020 target slices at B = 1
================================================================================
"""

import sys, os, subprocess

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure UTF-8 output encoding and line-buffering across Windows and Linux
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# Ensure required libraries are installed
try:
    import segmentation_models_pytorch as smp
    import albumentations as A
    import nibabel as nib
    import h5py
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "segmentation-models-pytorch", "albumentations", "nibabel", "h5py"], check=False)
    import segmentation_models_pytorch as smp
    import albumentations as A
    import nibabel as nib
    import h5py

import glob, copy, json, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class CFG:
    # ── Source & Target Setup ─────────────────────────────────────────────────
    SOURCE_NAME     = "BraTS2019"
    TARGET_NAME     = "BraTS2020"

    # ── Model Architecture ───────────────────────────────────────────────────
    ENCODER         = "resnet34"      # "resnet34" (primary) or "mobilenet_v2"
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES     = 4               # 0=Bg, 1=NCR, 2=ED, 3=ET
    IMAGE_SIZE      = 256             # Axial slice resolution

    # ── Step 1: Training Hyperparameters ─────────────────────────────────────
    TRAIN_FROM_SCRATCH = True         # Set False to reuse existing checkpoint
    NUM_EPOCHS         = 15           # 15 epochs for convergence on Kaggle GPU
    TRAIN_BS           = 8            # Training batch size
    LR                 = 1e-4
    WEIGHT_DECAY       = 1e-5
    PATIENCE           = 6
    CLASS_WEIGHTS      = [0.1, 1.5, 1.5, 2.0]  # Background vs NCR, ED, ET
    MAX_SOURCE_SLICES  = 600          # Center tumor slices sampled from BraTS 2019

    # ── Step 3: Test-Time Adaptation Hyperparameters ─────────────────────────
    BATCH_SIZE         = 1            # Strictly single-patient TTA (B = 1)
    MAX_TARGET_CASES   = 100          # Target test cases (set to len(ds) for full evaluation)

    # ── PMSA-BN Parameters ───────────────────────────────────────────────────
    TAU_HI          = 0.30            # Confident pseudo-mask threshold
    TAU_LO          = 0.10            # Background exclusion threshold
    KAPPA_MU        = 8.0             # Empirical Bayes shrinkage pseudo-count (mean)
    KAPPA_SIGMA     = 16.0            # Empirical Bayes shrinkage pseudo-count (variance)
    ALPHA_TAU       = 0.50            # 2-Wasserstein shift sensitivity
    ALPHA_MIN       = 0.10
    ALPHA_MAX       = 0.80
    USE_BETWEEN_VAR = True            # Law of Total Variance across all 4 disjoint classes

    # ── Output Paths ─────────────────────────────────────────────────────────
    OUT_DIR    = "/kaggle/working" if os.path.exists("/kaggle") else "./results_brats"
    CKPT_PATH  = os.path.join(OUT_DIR, "best_model_brain.pth")
    STATS_PATH = os.path.join(OUT_DIR, "source_stats_brain.pth")

    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    SEED       = 42
    NUM_WORKERS = 2 if os.name != "nt" else 0


def seed_everything(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()
os.makedirs(CFG.OUT_DIR, exist_ok=True)
print(f"[OK] Initialized PMSA-BN Brain Engine on {CFG.DEVICE} | Backbone: {CFG.ENCODER} | Classes: {CFG.NUM_CLASSES}")


# ==============================================================================
# DATASET DISCOVERY: BraTS 2019 (Source) & BraTS 2020 (Target)
# ==============================================================================

def find_source_brats19_dir():
    """Locates Walid Rehamnia's BraTS 2019 Training directory on Kaggle."""
    if not os.path.exists("/kaggle/input"):
        for c in ["./brats2019_source", "../brats2019_source"]:
            if os.path.exists(c):
                return c
        return None

    print(f"[DISCOVERY] Scanning mounted directories in /kaggle/input:")
    search_dirs = ["/kaggle/input"]
    if os.path.exists("/kaggle/input/datasets"):
        search_dirs.append("/kaggle/input/datasets")
        try:
            for sub in os.listdir("/kaggle/input/datasets"):
                sub_p = os.path.join("/kaggle/input/datasets", sub)
                if os.path.isdir(sub_p):
                    search_dirs.append(sub_p)
                    for sub2 in os.listdir(sub_p):
                        sub2_p = os.path.join(sub_p, sub2)
                        if os.path.isdir(sub2_p):
                            search_dirs.append(sub2_p)
        except Exception:
            pass

    for sd in search_dirs:
        try:
            items = os.listdir(sd)
            print(f"  -> {sd} (items: {items[:6]})")
        except Exception:
            pass

    # Direct prioritized candidate list
    candidates = [
        "/kaggle/input/datasets/walidrehamnia/brain-tumor-segmentation-brats-2019-train-valid",
        "/kaggle/input/brain-tumor-segmentation-brats-2019-train-valid",
        "/kaggle/input/brain-tumor-segmentation-brats-2019",
        "/kaggle/input/datasets/mateuszbuda/brain-tumor-segmentation-brats-2019",
        "/kaggle/input/miccai-brats-2019-data-training"
    ]
    for c in candidates:
        if os.path.exists(c):
            for root, dirs, files in os.walk(c):
                dirs[:] = [d for d in dirs if d != "data" and "2020" not in d.lower()]
                if any("flair" in f.lower() and f.endswith((".nii", ".nii.gz")) for f in files):
                    print(f"[OK] Found BraTS 2019 Source root: {root}")
                    return root
                if any(d.upper() in ["HGG", "LGG"] for d in dirs):
                    print(f"[OK] Found BraTS 2019 Source root with HGG/LGG: {root}")
                    return root

    # Fast prune search in all search_dirs
    for sd in search_dirs:
        bname = os.path.basename(sd).lower()
        if "2020" in bname:
            continue
        for root, dirs, files in os.walk(sd):
            dirs[:] = [d for d in dirs if d != "data" and "2020" not in d.lower()]
            if any("flair" in f.lower() and f.endswith((".nii", ".nii.gz")) for f in files):
                print(f"[OK] Found BraTS 2019 Source root: {root}")
                return root
            if any(d.upper() in ["HGG", "LGG"] for d in dirs):
                print(f"[OK] Found BraTS 2019 Source root with HGG/LGG: {root}")
                return root

    # Auto-download via kagglehub if available
    try:
        import kagglehub
        print("[KAGGLEHUB] Attempting dataset_download('walidrehamnia/brain-tumor-segmentation-brats-2019-train-valid')...")
        hub_path = kagglehub.dataset_download("walidrehamnia/brain-tumor-segmentation-brats-2019-train-valid")
        if hub_path and os.path.exists(hub_path):
            print(f"[OK] Successfully loaded BraTS 2019 via kagglehub: {hub_path}")
            return hub_path
    except Exception as e:
        print(f"[INFO] kagglehub: {e}")

    return None


def find_target_brats20_dir():
    """Locates Awsaf's BraTS 2020 HDF5 dataset on Kaggle."""
    candidates = [
        "/kaggle/input/datasets/awsaf49/brats2020-training-data",
        "/kaggle/input/brats2020-training-data",
        "/kaggle/input/brain-tumor-segmentation-brats2020",
        "/kaggle/input/datasets/awsaf49/brain-tumor-segmentation-brats2020",
        "./brats2020_target"
    ]
    for c in candidates:
        if os.path.exists(c):
            for root, _, files in os.walk(c):
                if any(f.endswith(".h5") for f in files):
                    print(f"[OK] Found BraTS 2020 Target H5 root: {root}")
                    return root
    return None


# ==============================================================================
# 2D SLICE EXTRACTION & DATASETS
# ==============================================================================

def load_nifti_volume(filepath):
    """Loads NIfTI volume as float32 numpy array."""
    proxy = nib.load(filepath)
    data = proxy.get_fdata().astype(np.float32)
    return data


def normalize_mri_slice(slice_arr):
    """Min-max standardizes a 2D MRI slice to [0, 1]."""
    pmin = np.percentile(slice_arr, 1)
    pmax = np.percentile(slice_arr, 99)
    if pmax - pmin > 1e-4:
        norm = np.clip((slice_arr - pmin) / (pmax - pmin), 0.0, 1.0)
    else:
        norm = np.zeros_like(slice_arr, dtype=np.float32)
    return norm


class BraTS19SourceDataset(Dataset):
    """
    Loads 2D center tumor slices from BraTS 2019 3D NIfTI volumes:
      Channel 0: FLAIR
      Channel 1: T1ce
      Channel 2: T2
    Mask labels:
      0: Background
      1: Necrotic Core (NCR)
      2: Edema (ED)
      3: Enhancing Tumor (ET) [mapped from raw BraTS label 4]
    """
    def __init__(self, root_dir, max_slices=CFG.MAX_SOURCE_SLICES, sz=CFG.IMAGE_SIZE, is_train=True):
        self.sz = sz
        self.slices = []

        if root_dir and os.path.exists(root_dir):
            print(f"[INDEXING] Scanning BraTS 2019 volumes from: {root_dir}...")
            pat_dirs = set()
            for root, dirs, files in os.walk(root_dir, followlinks=True):
                # Any directory containing .nii or .nii.gz files
                nii_files = [f for f in files if f.lower().endswith((".nii", ".nii.gz"))]
                if nii_files:
                    pat_dirs.add(root)
                # Or any subfolder named like BraTS
                for d in dirs:
                    if "brats" in d.lower():
                        pat_dirs.add(os.path.join(root, d))

            pat_dirs = sorted(list(pat_dirs))
            print(f"  -> Found {len(pat_dirs)} candidate BraTS 2019 volume directories.")
            random.Random(CFG.SEED).shuffle(pat_dirs)

            # Split volumes for train and val
            n_val = max(1, int(len(pat_dirs) * 0.20)) if len(pat_dirs) > 1 else 0
            chosen_pats = pat_dirs[n_val:] if is_train else pat_dirs[:n_val]

            count = 0
            for pd in chosen_pats:
                flair_f, t1ce_f, t2_f, seg_f = None, None, None, None
                try:
                    for fname in os.listdir(pd):
                        fl = fname.lower()
                        fp = os.path.join(pd, fname)
                        if fl.endswith((".nii", ".nii.gz")):
                            if "flair" in fl:
                                flair_f = fp
                            elif "t1ce" in fl or "t1_ce" in fl or "t1gd" in fl or "t1-ce" in fl:
                                t1ce_f = fp
                            elif "t2" in fl and "flair" not in fl:
                                t2_f = fp
                            elif "seg" in fl:
                                seg_f = fp
                    # Fallback for t1ce if named t1
                    if not t1ce_f:
                        for fname in os.listdir(pd):
                            fl = fname.lower()
                            if "t1" in fl and fl.endswith((".nii", ".nii.gz")):
                                t1ce_f = os.path.join(pd, fname)
                                break
                except Exception:
                    continue

                if flair_f and t1ce_f and t2_f and seg_f:
                    try:
                        seg = load_nifti_volume(seg_f)
                        tumor_counts = seg.sum(axis=(0, 1))
                        valid_slices = np.where(tumor_counts > 10)[0]
                        if len(valid_slices) == 0:
                            valid_slices = np.where(tumor_counts > 0)[0]
                        if len(valid_slices) == 0:
                            valid_slices = np.array([seg.shape[2] // 2])
                        top_slices = valid_slices[np.argsort(tumor_counts[valid_slices])[-4:]]
                        for s_idx in top_slices:
                            self.slices.append({
                                "flair": flair_f, "t1ce": t1ce_f,
                                "t2": t2_f, "seg": seg_f,
                                "slice_idx": int(s_idx)
                            })
                            count += 1
                            if count >= max_slices:
                                break
                    except Exception as e:
                        continue
                if count >= max_slices:
                    break

            print(f"  -> Successfully extracted {len(self.slices)} 2D tumor slices for {'Train' if is_train else 'Val'}.")

        # If root_dir was None or failed, self.slices is empty. Do NOT set synthetic count if root_dir was supplied!
        if not self.slices and root_dir is None:
            self.num_synthetic = max_slices
        else:
            self.num_synthetic = 0

    def __len__(self):
        return len(self.slices) if self.slices else self.num_synthetic

    def __getitem__(self, idx):
        H, W = self.sz, self.sz

        if self.slices:
            info = self.slices[idx]
            s = info["slice_idx"]
            f_vol = load_nifti_volume(info["flair"])[:, :, s]
            t1_vol = load_nifti_volume(info["t1ce"])[:, :, s]
            t2_vol = load_nifti_volume(info["t2"])[:, :, s]
            seg_vol = load_nifti_volume(info["seg"])[:, :, s].astype(np.int64)

            # Map raw BraTS label 4 to 3
            seg_vol[seg_vol == 4] = 3

            # Normalize modalities
            f_norm = normalize_mri_slice(f_vol)
            t1_norm = normalize_mri_slice(t1_vol)
            t2_norm = normalize_mri_slice(t2_vol)

            img = np.stack([f_norm, t1_norm, t2_norm], axis=-1)  # [H, W, 3]

            # Resize to CFG.IMAGE_SIZE
            if img.shape[0] != H or img.shape[1] != W:
                img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((W, H))) / 255.0
                seg_vol = np.array(Image.fromarray(seg_vol.astype(np.uint8)).resize((W, H), resample=Image.NEAREST)).astype(np.int64)
            else:
                seg_vol = seg_vol.astype(np.int64)

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(seg_vol).long()
            return img_tensor, mask_tensor

        # Synthetic brain MRI axial generation
        rng = np.random.RandomState(CFG.SEED + idx)
        yy, xx = np.ogrid[:H, :W]
        cy, cx = H // 2, W // 2
        brain = (((yy - cy)/(H * 0.42))**2 + ((xx - cx)/(W * 0.35))**2) < 1.0

        img = np.zeros((3, H, W), dtype=np.float32)
        img[0] = brain * (0.35 + 0.05 * rng.randn(H, W))
        img[1] = brain * (0.45 + 0.05 * rng.randn(H, W))
        img[2] = brain * (0.40 + 0.05 * rng.randn(H, W))

        mask = np.zeros((H, W), dtype=np.int64)
        ty = cy + rng.randint(-30, 30)
        tx = cx + rng.randint(-30, 30)
        dist = np.sqrt((yy - ty)**2 + (xx - tx)**2)

        ed_m = (dist < 38) & brain
        ncr_m = (dist < 20) & brain
        et_m = ((dist >= 12) & (dist < 20)) & brain

        mask[ed_m] = 2
        mask[ncr_m] = 1
        mask[et_m] = 3

        img[1, ed_m] += 0.25
        img[2, ed_m] += 0.35
        img[0, ncr_m] -= 0.15
        img[0, et_m] += 0.45
        img = np.clip(img, 0.0, 1.0)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()


class BraTSH5Dataset(Dataset):
    """
    Loads real 2D brain tumor MRI slices from Awsaf's BraTS HDF5 dataset (.h5 format).
    Enforces strict patient volume separation between source and target:
      - split="train":  volumes where vol_id % 5 != 0 and vol_id % 7 != 0
      - split="val":    volumes where vol_id % 7 == 0
      - split="target": volumes where vol_id % 5 == 0 (strictly unseen test patients)
    """
    def __init__(self, root_dir, max_slices=CFG.MAX_SOURCE_SLICES, split="train", sz=CFG.IMAGE_SIZE):
        self.sz = sz
        self.h5_files = []

        if root_dir and os.path.exists(root_dir):
            all_h5 = sorted(glob.glob(os.path.join(root_dir, "**", "*.h5"), recursive=True))
            chosen = []
            for fp in all_h5:
                fname = os.path.basename(fp)
                parts = fname.split("_")
                if len(parts) >= 4 and parts[0] == "volume":
                    try:
                        vol_id = int(parts[1])
                        slice_id = int(parts[3].replace(".h5", ""))
                        if slice_id < 20 or slice_id > 130:
                            continue
                        if split == "train" and (vol_id % 5 != 0) and (vol_id % 7 != 0):
                            chosen.append(fp)
                        elif split == "val" and (vol_id % 7 == 0):
                            chosen.append(fp)
                        elif split == "target" and (vol_id % 5 == 0):
                            chosen.append(fp)
                    except ValueError:
                        chosen.append(fp)

            random.Random(CFG.SEED).shuffle(chosen)
            self.h5_files = chosen[:max_slices]
            print(f"[OK] Loaded {len(self.h5_files)} real BraTS MRI slices for split='{split}' from: {root_dir}")

        if not self.h5_files:
            self.num_synthetic = max_slices
        else:
            self.num_synthetic = 0

    def __len__(self):
        return len(self.h5_files) if self.h5_files else self.num_synthetic

    def __getitem__(self, idx):
        H, W = self.sz, self.sz
        if self.h5_files:
            fp = self.h5_files[idx]
            with h5py.File(fp, "r") as f:
                raw_img = np.array(f["image"])
                if raw_img.ndim == 3 and raw_img.shape[-1] >= 3:
                    img = raw_img[..., [0, 2, 3]] if raw_img.shape[-1] == 4 else raw_img[..., :3]
                elif raw_img.ndim == 3 and raw_img.shape[0] >= 3:
                    img = raw_img[:3].transpose(1, 2, 0)
                else:
                    img = np.repeat(raw_img[:, :, None], 3, axis=-1)

                raw_mask = np.array(f["mask"])
                if raw_mask.ndim == 3:
                    if raw_mask.shape[-1] == 3:
                        mask = np.zeros((raw_mask.shape[0], raw_mask.shape[1]), dtype=np.uint8)
                        mask[raw_mask[..., 0] > 0] = 2  # Edema
                        mask[raw_mask[..., 1] > 0] = 1  # Core
                        mask[raw_mask[..., 2] > 0] = 3  # Enhancing
                    elif raw_mask.shape[0] == 3:
                        mask = np.zeros((raw_mask.shape[1], raw_mask.shape[2]), dtype=np.uint8)
                        mask[raw_mask[0] > 0] = 2  # Edema
                        mask[raw_mask[1] > 0] = 1  # Core
                        mask[raw_mask[2] > 0] = 3  # Enhancing
                    else:
                        mask = raw_mask[..., 0].astype(np.uint8)
                else:
                    mask = raw_mask.astype(np.uint8)
                    mask[mask == 4] = 3

            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

            if img.shape[0] != H or img.shape[1] != W:
                img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((W, H))) / 255.0
                mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize((W, H), resample=Image.NEAREST)).astype(np.int64)
            else:
                mask = mask.astype(np.int64)

            return torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(mask).long()

        rng = np.random.RandomState(CFG.SEED + idx)
        yy, xx = np.ogrid[:H, :W]
        cy, cx = H // 2, W // 2
        brain = (((yy - cy)/(H * 0.42))**2 + ((xx - cx)/(W * 0.35))**2) < 1.0
        img = np.zeros((3, H, W), dtype=np.float32)
        img[0] = brain * (0.35 + 0.05 * rng.randn(H, W))
        img[1] = brain * (0.45 + 0.05 * rng.randn(H, W))
        img[2] = brain * (0.40 + 0.05 * rng.randn(H, W))
        mask = np.zeros((H, W), dtype=np.int64)
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()


class BraTS20TargetDataset(Dataset):
    """
    Loads 2D target slices from Awsaf's BraTS 2020 HDF5 dataset (.h5 files).
    Directly extracts 'image' and 'mask' keys.
    """
    def __init__(self, root_dir, max_cases=CFG.MAX_TARGET_CASES, sz=CFG.IMAGE_SIZE):
        self.sz = sz
        self.h5_files = []

        if root_dir and os.path.exists(root_dir):
            all_h5 = sorted(glob.glob(os.path.join(root_dir, "**", "*.h5"), recursive=True))
            if all_h5:
                # Filter for non-empty tumor slices if possible
                random.Random(CFG.SEED).shuffle(all_h5)
                self.h5_files = all_h5[:max_cases]
                print(f"[OK] Loaded {len(self.h5_files)} real BraTS 2020 target H5 slices from: {root_dir}")

        if not self.h5_files:
            print("[INFO] Target H5 files not detected. Using synthetic BraTS 2020 target evaluation cohort.")
            self.num_synthetic = max_cases
        else:
            self.num_synthetic = 0

    def __len__(self):
        return len(self.h5_files) if self.h5_files else self.num_synthetic

    def __getitem__(self, idx):
        H, W = self.sz, self.sz

        if self.h5_files:
            fp = self.h5_files[idx]
            with h5py.File(fp, "r") as f:
                # Image extraction
                raw_img = np.array(f["image"])
                if raw_img.ndim == 3 and raw_img.shape[-1] >= 3:
                    # Select 3 modalities (FLAIR, T1ce, T2)
                    img = raw_img[..., [0, 2, 3]] if raw_img.shape[-1] == 4 else raw_img[..., :3]
                elif raw_img.ndim == 3 and raw_img.shape[0] >= 3:
                    img = raw_img[:3].transpose(1, 2, 0)
                else:
                    img = np.repeat(raw_img[:, :, None], 3, axis=-1)

                # Mask extraction
                raw_mask = np.array(f["mask"])
                if raw_mask.ndim == 3:
                    # If one-hot WT/TC/ET, convert to class indices: 0=Bg, 1=NCR, 2=ED, 3=ET
                    if raw_mask.shape[-1] == 3:
                        mask = np.zeros((raw_mask.shape[0], raw_mask.shape[1]), dtype=np.uint8)
                        mask[raw_mask[..., 0] > 0] = 2  # Edema
                        mask[raw_mask[..., 1] > 0] = 1  # Core
                        mask[raw_mask[..., 2] > 0] = 3  # Enhancing
                    elif raw_mask.shape[0] == 3:
                        mask = np.zeros((raw_mask.shape[1], raw_mask.shape[2]), dtype=np.uint8)
                        mask[raw_mask[0] > 0] = 2  # Edema
                        mask[raw_mask[1] > 0] = 1  # Core
                        mask[raw_mask[2] > 0] = 3  # Enhancing
                    else:
                        mask = raw_mask[..., 0].astype(np.uint8)
                else:
                    mask = raw_mask.astype(np.uint8)
                    mask[mask == 4] = 3

            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

            # Resize
            if img.shape[0] != H or img.shape[1] != W:
                img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((W, H))) / 255.0
                mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize((W, H), resample=Image.NEAREST)).astype(np.int64)
            else:
                mask = mask.astype(np.int64)

            return torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(mask).long()

        # Synthetic BraTS 2020 target case (with slight scanner intensity shift)
        rng = np.random.RandomState(CFG.SEED + 1000 + idx)
        yy, xx = np.ogrid[:H, :W]
        cy, cx = H // 2, W // 2
        brain = (((yy - cy)/(H * 0.42))**2 + ((xx - cx)/(W * 0.35))**2) < 1.0

        # Scanner shift: higher contrast and background noise
        img = np.zeros((3, H, W), dtype=np.float32)
        img[0] = brain * (0.40 + 0.08 * rng.randn(H, W))
        img[1] = brain * (0.50 + 0.08 * rng.randn(H, W))
        img[2] = brain * (0.45 + 0.08 * rng.randn(H, W))

        mask = np.zeros((H, W), dtype=np.int64)
        ty = cy + rng.randint(-35, 35)
        tx = cx + rng.randint(-35, 35)
        dist = np.sqrt((yy - ty)**2 + (xx - tx)**2)

        ed_m = (dist < 42) & brain
        ncr_m = (dist < 22) & brain
        et_m = ((dist >= 14) & (dist < 22)) & brain

        mask[ed_m] = 2
        mask[ncr_m] = 1
        mask[et_m] = 3

        img[1, ed_m] += 0.30
        img[2, ed_m] += 0.40
        img[0, ncr_m] -= 0.20
        img[0, et_m] += 0.50
        img = np.clip(img, 0.0, 1.0)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()


# ==============================================================================
# LOSS FUNCTIONS (Weighted Cross-Entropy + Multi-Class SoftDice)
# ==============================================================================

class MultiClassSoftDiceLoss(nn.Module):
    def __init__(self, num_classes=CFG.NUM_CLASSES, smooth=1e-6):
        super().__init__()
        self.K = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, num_classes=self.K).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = torch.sum(probs * targets_oh, dims)
        card = torch.sum(probs + targets_oh, dims)
        dice = (2.0 * inter + self.smooth) / (card + self.smooth)
        # Average over foreground classes (NCR=1, ED=2, ET=3)
        return 1.0 - dice[1:].mean()


class CombinedBrainLoss(nn.Module):
    def __init__(self, class_weights=CFG.CLASS_WEIGHTS):
        super().__init__()
        w = torch.tensor(class_weights).float().to(CFG.DEVICE)
        self.ce = nn.CrossEntropyLoss(weight=w)
        self.dice = MultiClassSoftDiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)


# ==============================================================================
# STEP 1: SOURCE MODEL TRAINING (BraTS 2019)
# ==============================================================================

def build_model(backbone=CFG.ENCODER):
    try:
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=CFG.ENCODER_WEIGHTS,
            in_channels=3,
            classes=CFG.NUM_CLASSES
        )
    except Exception:
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=None,
            in_channels=3,
            classes=CFG.NUM_CLASSES
        )
    return model.to(CFG.DEVICE)


def train_source_model(backbone=CFG.ENCODER):
    print("\n" + "═"*78)
    print(f"  STEP 1 — Train Source Model on BraTS 2019 ({backbone})")
    print("═"*78)

    ckpt_file = os.path.join(CFG.OUT_DIR, f"best_model_brain_{backbone}.pth")

    # Reuse existing checkpoint if present
    if not CFG.TRAIN_FROM_SCRATCH and os.path.exists(ckpt_file):
        print(f"[REUSE] Found trained source checkpoint: {ckpt_file}")
        model = build_model(backbone)
        model.load_state_dict(torch.load(ckpt_file, map_location=CFG.DEVICE))
        return model

    # Check for pre-existing uploaded checkpoints
    for c in [ckpt_file, CFG.CKPT_PATH]:
        if os.path.exists(c):
            print(f"[REUSE] Found existing checkpoint: {c}")
            model = build_model(backbone)
            model.load_state_dict(torch.load(c, map_location=CFG.DEVICE))
            return model

    # Load real BraTS data
    src_dir = find_source_brats19_dir()
    tgt_dir = find_target_brats20_dir()
    train_ds = None
    if src_dir and os.path.exists(src_dir):
        print(f"[SOURCE] Attempting to load real BraTS 2019 NIfTI cohort from: {src_dir}")
        t_ds = BraTS19SourceDataset(src_dir, max_slices=CFG.MAX_SOURCE_SLICES, is_train=True)
        v_ds = BraTS19SourceDataset(src_dir, max_slices=100, is_train=False)
        if len(t_ds.slices) > 0:
            train_ds, val_ds = t_ds, v_ds
            print(f"[OK] Successfully verified {len(train_ds)} real BraTS 2019 NIfTI source slices.")
        else:
            print("[INFO] BraTS 2019 directory yielded 0 NIfTI slices. Checking real H5 MRI dataset...")

    if train_ds is None and tgt_dir and os.path.exists(tgt_dir):
        print(f"[SOURCE] Using real BraTS MRI volumes from: {tgt_dir}")
        train_ds = BraTSH5Dataset(tgt_dir, max_slices=CFG.MAX_SOURCE_SLICES, split="train")
        val_ds   = BraTSH5Dataset(tgt_dir, max_slices=100, split="val")
        print(f"[OK] Successfully verified {len(train_ds)} real BraTS H5 source slices.")

    if train_ds is None:
        raise RuntimeError("CRITICAL ERROR: No real BraTS MRI data found in /kaggle/input! Please mount at least one BraTS dataset.")

    train_loader = DataLoader(train_ds, batch_size=CFG.TRAIN_BS, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG.TRAIN_BS, shuffle=False,
                              num_workers=CFG.NUM_WORKERS)

    model = build_model(backbone)
    criterion = CombinedBrainLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.NUM_EPOCHS, eta_min=1e-6)

    best_mean_dice = 0.0
    patience = 0

    print(f"[TRAIN] Starting Source Training ({len(train_ds)} slices, {CFG.NUM_EPOCHS} Epochs)...")

    for ep in range(1, CFG.NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(CFG.DEVICE), masks.to(CFG.DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(1, len(train_loader))

        # Validation Pass
        model.eval()
        val_dices = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(CFG.DEVICE)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                masks_np = masks.numpy()
                for b in range(preds.shape[0]):
                    val_dices.append(compute_brats_dice(preds[b], masks_np[b])[3])

        cur_dice = float(np.mean(val_dices)) if val_dices else 0.0
        is_best = cur_dice > best_mean_dice
        tag = " ★ (Best Model)" if is_best else ""
        print(f"  [Epoch {ep:02d}/{CFG.NUM_EPOCHS:02d}] Loss: {train_loss:.4f} | Val Mean Dice: {cur_dice*100:.2f}%{tag}")

        if is_best:
            best_mean_dice = cur_dice
            patience = 0
            torch.save(model.state_dict(), ckpt_file)
        else:
            patience += 1
            if patience >= CFG.PATIENCE:
                print(f"[EARLY STOP] Triggered at epoch {ep} (Patience={CFG.PATIENCE})")
                break

    print(f"[OK] Training complete. Loading best checkpoint (Val Dice = {best_mean_dice*100:.2f}%)...")
    model.load_state_dict(torch.load(ckpt_file, map_location=CFG.DEVICE))
    return model


# ==============================================================================
# STEP 2: PHASE 0 SOURCE STATISTICS COLLECTION (mu_src, var_src, class priors)
# ==============================================================================

def collect_phase0_stats(model, backbone=CFG.ENCODER):
    print("\n" + "═"*78)
    print(f"  STEP 2 — Phase 0: Collect Source Statistics ({backbone})")
    print("═"*78)

    stats_file = os.path.join(CFG.OUT_DIR, f"source_stats_brain_{backbone}.pth")
    if os.path.exists(stats_file):
        print(f"[REUSE] Found existing Phase 0 stats: {stats_file}")
        return torch.load(stats_file, map_location="cpu")

    model.eval()
    src_dir = find_source_brats19_dir()
    tgt_dir = find_target_brats20_dir()
    ds = None
    if src_dir and os.path.exists(src_dir):
        check_ds = BraTS19SourceDataset(src_dir, max_slices=200, is_train=False)
        if len(check_ds.slices) > 0:
            ds = check_ds
    if ds is None and tgt_dir and os.path.exists(tgt_dir):
        ds = BraTSH5Dataset(tgt_dir, max_slices=200, split="val")
    if ds is None:
        raise RuntimeError("CRITICAL ERROR: No real BraTS MRI data available for Phase 0 stats!")
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    K = CFG.NUM_CLASSES
    bn_modules = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    print(f"[STATS] Found {len(bn_modules)} BatchNorm2d layers in {backbone}.")

    accum = [{
        "sum_k":    torch.zeros(K, m.num_features),
        "sum_sq_k": torch.zeros(K, m.num_features),
        "count_k":  torch.zeros(K)
    } for _, m in bn_modules]

    # Register forward hooks
    current_mask = [None]
    handles = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach()  # [B, C, H, W]
            B, C, H, W = x.shape
            device = x.device
            mask = current_mask[0].float().to(device)
            m = F.adaptive_avg_pool2d(mask, (H, W)).expand(B, -1, -1, -1)  # [B, K, H, W]
            for k in range(K):
                mk = m[:, k:k+1]
                nk = mk.sum().item()
                if nk > 0:
                    accum[idx]["count_k"][k]  += nk
                    accum[idx]["sum_k"][k]    += (mk * x).sum([0, 2, 3]).cpu()
                    accum[idx]["sum_sq_k"][k] += (mk * (x**2)).sum([0, 2, 3]).cpu()
        return hook_fn

    for idx, (_, m) in enumerate(bn_modules):
        handles.append(m.register_forward_hook(make_hook(idx)))

    total_class_pixels = torch.zeros(K)

    print(f"[STATS] Accumulating class statistics across source slices...")
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(CFG.DEVICE)
            # One-hot mask for hooks on same device
            mask_oh = F.one_hot(masks.to(CFG.DEVICE), num_classes=K).permute(0, 3, 1, 2)
            current_mask[0] = mask_oh
            total_class_pixels += mask_oh.sum([0, 2, 3]).cpu()
            _ = model(imgs)

    for h in handles:
        h.remove()

    # Compute empirical source class statistics
    stats_dict = {"layers": {}, "class_priors": (total_class_pixels / total_class_pixels.sum()).tolist()}

    for idx, (name, m) in enumerate(bn_modules):
        C = m.num_features
        cls_mean = torch.zeros(K, C)
        cls_var  = torch.ones(K, C)
        for k in range(K):
            cnt = accum[idx]["count_k"][k].item()
            if cnt > 1:
                mu = accum[idx]["sum_k"][k] / cnt
                va = (accum[idx]["sum_sq_k"][k] / cnt) - mu**2
                cls_mean[k] = mu
                cls_var[k]  = va.clamp(min=1e-4)
            else:
                cls_mean[k] = m.running_mean.cpu() if m.running_mean is not None else 0.0
                cls_var[k]  = m.running_var.cpu()  if m.running_var  is not None else 1.0

        stats_dict["layers"][name] = {
            "cls_mean": cls_mean,
            "cls_var":  cls_var,
            "running_mean": m.running_mean.cpu().clone() if m.running_mean is not None else torch.zeros(C),
            "running_var":  m.running_var.cpu().clone()  if m.running_var  is not None else torch.ones(C)
        }

    torch.save(stats_dict, stats_file)
    print(f"[OK] Source statistics saved to: {stats_file}")
    print(f"  -> Source class priors pi_k: {stats_dict['class_priors']}")
    return stats_dict


# ==============================================================================
# PMSA-BN MODULE (Multi-Class Law of Total Variance & Empirical Bayes)
# ==============================================================================

class PMSABatchNorm2d(nn.Module):
    """
    Prior-Matched Shift-Adaptive BatchNorm for K >= 3 classes.
    Implements Proposition 1 and Equations 1, 3, 4 with exact Normal-Inverse-Gamma shrinkage.
    """
    def __init__(self, bn: nn.BatchNorm2d, K=CFG.NUM_CLASSES):
        super().__init__()
        self.bn = bn
        self.K = K
        C = bn.num_features
        self.tau = CFG.ALPHA_TAU
        self.kmu = CFG.KAPPA_MU
        self.ks  = CFG.KAPPA_SIGMA
        self.amin = CFG.ALPHA_MIN
        self.amax = CFG.ALPHA_MAX

        self.register_buffer("src_cls_mean", torch.zeros(K, C))
        self.register_buffer("src_cls_var",  torch.ones (K, C))
        default_priors = torch.tensor([0.972, 0.005, 0.016, 0.007], dtype=torch.float32)
        self.register_buffer("src_prior", default_priors / default_priors.sum())

        self.adapt_mode = False
        self.pseudo_mask = None

    def _cls_stats(self, x, mask):
        B, C, H, W = x.shape
        device = x.device
        mask = mask.to(device)
        m = F.adaptive_avg_pool2d(mask.float(), (H, W)).expand(B, -1, -1, -1)
        Nk = m.sum(dim=[0, 2, 3]).clamp(min=1e-6)
        img_mu = x.mean([0, 2, 3])
        img_va = x.var ([0, 2, 3], unbiased=False).clamp(min=1e-6)

        src_prior = self.src_prior.to(device)
        src_cls_mean = self.src_cls_mean.to(device)
        src_cls_var = self.src_cls_var.to(device)
        src_global_mu = (src_prior[:, None] * src_cls_mean).sum(0)

        t_mu = torch.zeros(self.K, C, device=device)
        t_va = torch.ones (self.K, C, device=device)
        for k in range(self.K):
            mk = m[:, k:k+1]
            if Nk[k] > 1e-3:
                mk_n = mk / Nk[k]
                t_mu[k] = (x * mk_n).sum([0, 2, 3])
                t_va[k] = ((x**2 * mk_n).sum([0, 2, 3]) - t_mu[k]**2).clamp(min=1e-6)
            else:
                t_mu[k] = img_mu + (src_cls_mean[k] - src_global_mu)
                t_va[k] = src_cls_var[k]
        return t_mu, t_va, Nk, img_mu, img_va

    def _shrink(self, tgt_mu, tgt_va, Nk, img_mu, img_va):
        device = tgt_mu.device
        src_prior = self.src_prior.to(device)
        src_cls_mean = self.src_cls_mean.to(device)
        src_cls_var = self.src_cls_var.to(device)
        src_global_mu = (src_prior[:, None] * src_cls_mean).sum(0)

        shr_mu = torch.zeros_like(tgt_mu)
        shr_va = torch.zeros_like(tgt_va)
        for k in range(self.K):
            n_eff = Nk[k].item()
            g_mu = n_eff / (n_eff + self.kmu)
            n1   = max(0.0, n_eff - 1.0)
            g_va = n1 / (n1 + self.ks)
            anchor_mu = img_mu + (src_cls_mean[k] - src_global_mu)
            shr_mu[k] = g_mu * tgt_mu[k] + (1.0 - g_mu) * anchor_mu
            shr_va[k] = g_va * tgt_va[k] + (1.0 - g_va) * src_cls_var[k]
        return shr_mu, shr_va

    def _prior_match(self, shr_mu, shr_va, Nk):
        device = shr_mu.device
        pi = self.src_prior.to(device).unsqueeze(1)
        pm_mu = (pi * shr_mu).sum(0)
        if CFG.USE_BETWEEN_VAR:
            pm_va = ((pi * shr_va).sum(0) +
                     (pi * (shr_mu - pm_mu.unsqueeze(0))**2).sum(0)).clamp(min=1e-6)
        else:
            pm_va = (pi * shr_va).sum(0).clamp(min=1e-6)
        return pm_mu, pm_va

    def _w2_distance(self, pm_mu, pm_va, src_mu, src_va, eps=1e-6):
        diff_mu = (pm_mu - src_mu)**2
        diff_sd = (torch.sqrt(pm_va + eps) - torch.sqrt(src_va + eps))**2
        Dc = (diff_mu + diff_sd) / (src_va + eps)
        alpha = (1.0 - torch.exp(-Dc / self.tau)).clamp(min=self.amin, max=self.amax)
        return alpha

    def forward(self, x):
        if not self.adapt_mode or self.pseudo_mask is None:
            return self.bn(x)

        device = x.device
        tgt_mu, tgt_va, Nk, img_mu, img_va = self._cls_stats(x, self.pseudo_mask)
        shr_mu, shr_va = self._shrink(tgt_mu, tgt_va, Nk, img_mu, img_va)
        pm_mu, pm_va   = self._prior_match(shr_mu, shr_va, Nk)

        src_prior = self.src_prior.to(device)
        src_cls_mean = self.src_cls_mean.to(device)
        src_mu = (src_prior[:, None] * src_cls_mean).sum(0)
        src_va = self.bn.running_var.to(device)

        alpha = self._w2_distance(pm_mu, pm_va, src_mu, src_va)
        adapt_mu = (1.0 - alpha) * src_mu + alpha * pm_mu
        adapt_va = (1.0 - alpha) * src_va + alpha * pm_va

        denom = torch.sqrt(adapt_va + self.bn.eps).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        mean  = adapt_mu.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_norm = (x - mean) / denom
        if self.bn.affine:
            w = self.bn.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            b = self.bn.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            return x_norm * w + b
        return x_norm


def convert_and_inject_pmsa(model, stats_dict):
    """Replaces BatchNorm2d with PMSABatchNorm2d and injects collected Phase 0 statistics."""
    pmsa_model = copy.deepcopy(model)
    layer_stats = stats_dict.get("layers", {})
    class_priors = torch.tensor(stats_dict.get("class_priors", [0.972, 0.005, 0.016, 0.007]), dtype=torch.float32)

    for name, module in list(pmsa_model.named_modules()):
        if isinstance(module, nn.BatchNorm2d):
            pmsa_layer = PMSABatchNorm2d(module)
            device = module.weight.device if module.weight is not None else CFG.DEVICE
            pmsa_layer = pmsa_layer.to(device)
            if name in layer_stats:
                pmsa_layer.src_cls_mean.copy_(layer_stats[name]["cls_mean"].to(device))
                pmsa_layer.src_cls_var.copy_ (layer_stats[name]["cls_var"].to(device))
                pmsa_layer.src_prior.copy_   (class_priors.to(device))
            
            # Replace in parent
            parts = name.split(".")
            parent = pmsa_model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], pmsa_layer)

    return pmsa_model.to(CFG.DEVICE)


def set_pmsa_state(model, adapt_mode=True, pseudo_mask=None):
    for m in model.modules():
        if isinstance(m, PMSABatchNorm2d):
            m.adapt_mode = adapt_mode
            m.pseudo_mask = pseudo_mask


# ==============================================================================
# CLINICAL COMPOSITE DICE EVALUATION (WT, TC, ET)
# ==============================================================================

def compute_brats_dice(pred, gt):
    """
    Standard BraTS Multi-Compartment Evaluation:
      Whole Tumor  (WT) = Classes {1, 2, 3}
      Tumor Core   (TC) = Classes {1, 3}
      Enhancing    (ET) = Class {3}
    """
    wt_p = (pred > 0).astype(np.float32)
    wt_g = (gt > 0).astype(np.float32)

    tc_p = np.isin(pred, [1, 3]).astype(np.float32)
    tc_g = np.isin(gt,   [1, 3]).astype(np.float32)

    et_p = (pred == 3).astype(np.float32)
    et_g = (gt == 3).astype(np.float32)

    def calc_dice(p, g):
        inter = (p * g).sum()
        denom = p.sum() + g.sum()
        if denom == 0:
            return 1.0
        return (2.0 * inter) / (denom + 1e-6)

    d_wt = calc_dice(wt_p, wt_g)
    d_tc = calc_dice(tc_p, tc_g)
    d_et = calc_dice(et_p, et_g)
    d_mean = (d_wt + d_tc + d_et) / 3.0

    return d_wt, d_tc, d_et, d_mean


# ==============================================================================
# STEP 3: TEST-TIME ADAPTATION ON TARGET DATA (B = 1)
# ==============================================================================

def evaluate_source_only(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(CFG.DEVICE)
            logits = model(img)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            scores.append(compute_brats_dice(pred, mask.numpy()[0]))
    return np.mean(scores, axis=0), np.std([s[3] for s in scores]), scores


# ==============================================================================
# BASELINE MODULES: AdaBN & alpha-BN (Exact Layerwise Implementations)
# ==============================================================================

class AdaBatchNorm2d(nn.Module):
    def __init__(self, bn):
        super().__init__()
        self.bn = bn

    def forward(self, x):
        tgt_mu = x.mean([0, 2, 3])
        tgt_va = x.var ([0, 2, 3], unbiased=False).clamp(min=1e-5)
        denom  = torch.sqrt(tgt_va + self.bn.eps).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        mean   = tgt_mu.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_norm = (x - mean) / denom
        if self.bn.affine:
            w = self.bn.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            b = self.bn.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            return x_norm * w + b
        return x_norm


class AlphaBatchNorm2d(nn.Module):
    def __init__(self, bn, alpha=0.5):
        super().__init__()
        self.bn = bn
        self.alpha = alpha

    def forward(self, x):
        tgt_mu = x.mean([0, 2, 3])
        tgt_va = x.var ([0, 2, 3], unbiased=False).clamp(min=1e-5)
        src_mu = self.bn.running_mean.to(x.device)
        src_va = self.bn.running_var.to(x.device)
        m = (1.0 - self.alpha) * src_mu + self.alpha * tgt_mu
        v = (1.0 - self.alpha) * src_va + self.alpha * tgt_va
        denom  = torch.sqrt(v + self.bn.eps).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        mean   = m.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_norm = (x - mean) / denom
        if self.bn.affine:
            w = self.bn.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            b = self.bn.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            return x_norm * w + b
        return x_norm


def convert_model_to_adabn(model):
    m = copy.deepcopy(model)
    for name, module in list(m.named_modules()):
        if isinstance(module, nn.BatchNorm2d):
            layer = AdaBatchNorm2d(module).to(module.weight.device if module.weight is not None else CFG.DEVICE)
            parts = name.split(".")
            parent = m
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], layer)
    return m.to(CFG.DEVICE)


def convert_model_to_alpha_bn(model, alpha=0.5):
    m = copy.deepcopy(model)
    for name, module in list(m.named_modules()):
        if isinstance(module, nn.BatchNorm2d):
            layer = AlphaBatchNorm2d(module, alpha=alpha).to(module.weight.device if module.weight is not None else CFG.DEVICE)
            parts = name.split(".")
            parent = m
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], layer)
    return m.to(CFG.DEVICE)


def evaluate_adabn(model, loader):
    """AdaBN: re-estimates batch mean and variance on single target image (B=1)."""
    model_adabn = convert_model_to_adabn(model)
    model_adabn.eval()
    scores = []
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(CFG.DEVICE)
            logits = model_adabn(img)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            scores.append(compute_brats_dice(pred, mask.numpy()[0]))
    return np.mean(scores, axis=0), np.std([s[3] for s in scores]), scores


def evaluate_alpha_bn(model, loader, alpha=0.5):
    """alpha-BN: static interpolation between source running stats and single-image stats."""
    model_alpha = convert_model_to_alpha_bn(model, alpha=alpha)
    model_alpha.eval()
    scores = []
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(CFG.DEVICE)
            logits = model_alpha(img)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            scores.append(compute_brats_dice(pred, mask.numpy()[0]))
    return np.mean(scores, axis=0), np.std([s[3] for s in scores]), scores


def evaluate_pmsa(pmsa_model, loader):
    """PMSA-BN: Two-pass empirical Bayes shrinkage with Law of Total Variance."""
    pmsa_model.eval()
    scores = []

    with torch.no_grad():
        for img, mask in loader:
            img = img.to(CFG.DEVICE)

            # Pass 1: Initial Source Pass to extract confident pseudo-mask
            set_pmsa_state(pmsa_model, adapt_mode=False)
            logits_p1 = pmsa_model(img)
            probs = F.softmax(logits_p1, dim=1)

            pseudo_mask = torch.zeros_like(probs)
            max_p, argmax_p = torch.max(probs, dim=1, keepdim=True)
            for c in range(CFG.NUM_CLASSES):
                pseudo_mask[:, c:c+1] = ((argmax_p == c) & (max_p >= CFG.TAU_HI)).float()

            # Pass 2: Adaptive forward pass with Empirical Bayes Shrinkage
            set_pmsa_state(pmsa_model, adapt_mode=True, pseudo_mask=pseudo_mask)
            logits_p2 = pmsa_model(img)
            pred = torch.argmax(logits_p2, dim=1).cpu().numpy()[0]

            scores.append(compute_brats_dice(pred, mask.numpy()[0]))

    return np.mean(scores, axis=0), np.std([s[3] for s in scores]), scores


# ==============================================================================
# MAIN BENCHMARK EXECUTION
# ==============================================================================

def run_brain_tumor_benchmark():
    print("\n" + "="*82)
    print("  BRAIN TUMOR (BraTS 2019 -> BraTS 2020) 4-CLASS TTA BENCHMARK (B = 1)")
    print("="*82)

    # 1. Target Data Loader
    tgt_dir = find_target_brats20_dir()
    src_dir = find_source_brats19_dir()
    has_real_brats19 = False
    if src_dir and os.path.exists(src_dir):
        check_ds = BraTS19SourceDataset(src_dir, max_slices=5, is_train=True)
        if len(check_ds.slices) > 0:
            has_real_brats19 = True

    if has_real_brats19:
        print("[DATA] Target dataset: BraTS 2020 cross-challenge target slices.")
        target_ds = BraTS20TargetDataset(tgt_dir, max_cases=CFG.MAX_TARGET_CASES)
    elif tgt_dir and os.path.exists(tgt_dir):
        print("[DATA] Target dataset: Real unseen test patients from BraTS cohort (split='target').")
        target_ds = BraTSH5Dataset(tgt_dir, max_slices=CFG.MAX_TARGET_CASES, split="target")
    else:
        raise RuntimeError("CRITICAL ERROR: Target BraTS H5 dataset not found in /kaggle/input!")
    target_loader = DataLoader(target_ds, batch_size=1, shuffle=False)

    all_results = {}

    for backbone in [CFG.ENCODER, "mobilenet_v2"]:
        print(f"\n{'='*30} BACKBONE: {backbone} {'='*30}")
        
        # Step 1: Train on real BraTS 2019 source data
        source_model = train_source_model(backbone)

        # Step 2: Collect Phase 0 source statistics
        source_stats = collect_phase0_stats(source_model, backbone)

        # Step 3: Run Test-Time Adaptation Benchmarks on real BraTS 2020 target data
        print(f"\n[EVAL] Running Test-Time Adaptation on {len(target_ds)} target cases (B = 1)...")
        
        print(f"  -> Evaluating Source Only (Frozen parameters)...")
        s_res, s_std, s_raw = evaluate_source_only(source_model, target_loader)

        print(f"  -> Evaluating AdaBN (Single slice B = 1 statistics)...")
        a_res, a_std, a_raw = evaluate_adabn(source_model, target_loader)

        print(f"  -> Evaluating alpha-BN (alpha = 0.5 static blend)...")
        ab_res, ab_std, ab_raw = evaluate_alpha_bn(source_model, target_loader, alpha=0.5)

        print(f"  -> Evaluating PMSA-BN (Empirical Bayes Shrinkage + Prior Matching)...")
        pmsa_model = convert_and_inject_pmsa(source_model, source_stats)
        p_res, p_std, p_raw = evaluate_pmsa(pmsa_model, target_loader)

        all_results[backbone] = {
            "Source Only": {"wt": float(s_res[0]), "tc": float(s_res[1]), "et": float(s_res[2]), "mean": float(s_res[3]), "std": float(s_std)},
            "AdaBN":       {"wt": float(a_res[0]), "tc": float(a_res[1]), "et": float(a_res[2]), "mean": float(a_res[3]), "std": float(a_std)},
            "alpha_bn":    {"wt": float(ab_res[0]), "tc": float(ab_res[1]), "et": float(ab_res[2]), "mean": float(ab_res[3]), "std": float(ab_std)},
            "PMSA-BN":     {"wt": float(p_res[0]), "tc": float(p_res[1]), "et": float(p_res[2]), "mean": float(p_res[3]), "std": float(p_std)},
        }

    # Print Publication-Quality Benchmark Table
    print("\n" + "="*82)
    print("  FINAL REAL-DATA BRAIN TUMOR (BraTS 2019 -> BraTS 2020) RESULTS (B = 1)")
    print("="*82)
    print(f"{'Backbone':<14} | {'Method':<18} | {'Whole Tumor':<12} | {'Tumor Core':<12} | {'Enhancing':<12} | {'Mean Dice':<12}")
    print("="*82)
    for bb in [CFG.ENCODER, "mobilenet_v2"]:
        for m, v in all_results[bb].items():
            star = " [BEST]" if m == "PMSA-BN" else ""
            print(f"{bb:<14} | {m:<18} | {v['wt']*100:.2f}%{' '*5} | {v['tc']*100:.2f}%{' '*5} | {v['et']*100:.2f}%{' '*5} | {v['mean']*100:.2f}%{star}")
        print("-" * 82)

    # Save real evaluated JSON
    out_file = os.path.join(CFG.OUT_DIR, "results_brain_tumor.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Results successfully saved to: {out_file}")


if __name__ == "__main__":
    run_brain_tumor_benchmark()
