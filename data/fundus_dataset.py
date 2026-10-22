"""
data/fundus_dataset.py — Dataset loaders for fundus segmentation
=================================================================
Handles REFUGE, Drishti-GS, and RIM-ONE-DL formats.

REFUGE Mask Format:
  Pixel value 0   → Optic Cup
  Pixel value 128 → Optic Disc (rim region)
  Pixel value 255 → Background

Our 3-class label convention (matches NUM_CLASSES=3):
  Label 0 → Background
  Label 1 → Disc Rim (disc - cup)
  Label 2 → Cup
"""

import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ───────────────────────────────────────────────────────────────
# Augmentation Pipelines
# ───────────────────────────────────────────────────────────────

def get_train_transforms(image_size: int = 512):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 512):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# ───────────────────────────────────────────────────────────────
# Mask Conversion Utilities
# ───────────────────────────────────────────────────────────────

def refuge_mask_to_label(mask_pil: Image.Image) -> np.ndarray:
    """
    Convert REFUGE raw mask (PIL) to 3-class label array.
    REFUGE pixel values: 0=cup, 128=disc, 255=background
    Our labels: 0=background, 1=rim, 2=cup
    """
    mask = np.array(mask_pil.convert("L"))    # greyscale
    label = np.zeros_like(mask, dtype=np.int64)
    label[mask == 255] = 0   # background
    label[mask == 128] = 1   # disc rim
    label[mask == 0]   = 2   # cup
    return label


def drishti_mask_to_label(disc_mask_path: str, cup_mask_path: str) -> np.ndarray:
    """
    Convert Drishti-GS separate disc & cup masks to 3-class label.
    """
    disc = np.array(Image.open(disc_mask_path).convert("L")) > 127
    cup  = np.array(Image.open(cup_mask_path).convert("L"))  > 127

    label = np.zeros(disc.shape, dtype=np.int64)
    label[disc]  = 1   # disc rim (will be overwritten by cup)
    label[cup]   = 2   # cup
    # Background pixels remain 0
    return label


def compute_class_priors(dataset) -> np.ndarray:
    """
    Compute π_k^src (pixel fraction of each class) over the training dataset.
    Used in Phase 0 and stored in source stats.
    """
    counts = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    for i in range(len(dataset)):
        _, label = dataset[i]
        label_np = label.numpy() if isinstance(label, torch.Tensor) else label
        for k in range(config.NUM_CLASSES):
            counts[k] += (label_np == k).sum()
    total = counts.sum()
    return (counts / total).astype(np.float32)


# ───────────────────────────────────────────────────────────────
# REFUGE Dataset
# ───────────────────────────────────────────────────────────────

class REFUGEDataset(Dataset):
    """
    REFUGE 2020 dataset.

    Expected directory structure:
    REFUGE/
      train/
        images/       ← .jpg files (e.g. T0001.jpg)
        masks/        ← .bmp or .png files (same stem as image)
      val/
        images/
        masks/
      test/
        images/
        masks/      ← REFUGE test does include masks in the public release
    """

    def __init__(self, root_dir: str, split: str = "train",
                 transform=None, image_size: int = 512):
        super().__init__()
        self.root_dir   = root_dir
        self.split      = split
        self.transform  = transform
        self.image_size = image_size

        img_dir  = os.path.join(root_dir, split, "images")
        mask_dir = os.path.join(root_dir, split, "masks")

        # Accept .jpg, .png, .bmp
        self.image_paths = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.png"))
        )
        self.mask_dir = mask_dir

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {img_dir}. "
                "Check your DATA_ROOT in config.py."
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]

        # Try common mask extensions
        mask_path = None
        for ext in [".bmp", ".png", ".jpg"]:
            candidate = os.path.join(self.mask_dir, stem + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for image: {img_path}")

        image = np.array(Image.open(img_path).convert("RGB"))
        label = refuge_mask_to_label(Image.open(mask_path))

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]          # Tensor [3, H, W]
            label = transformed["mask"].long()    # Tensor [H, W]

        return image, label, stem


# ───────────────────────────────────────────────────────────────
# Drishti-GS Dataset
# ───────────────────────────────────────────────────────────────

class DrishtiDataset(Dataset):
    """
    Drishti-GS dataset.

    Expected structure:
    Drishti-GS/
      Training/
        Images/
          drishtiGS_001.png ...
        GT/
          Disc/
            drishtiGS_001.png ...   (average of 4 annotators, binary)
          Cup/
            drishtiGS_001.png ...
      Test/
        Images/
        GT/Disc/ GT/Cup/
    """

    def __init__(self, root_dir: str, split: str = "Training",
                 transform=None):
        self.root_dir  = root_dir
        self.split     = split
        self.transform = transform

        img_dir  = os.path.join(root_dir, split, "Images")
        disc_dir = os.path.join(root_dir, split, "GT", "Disc")
        cup_dir  = os.path.join(root_dir, split, "GT", "Cup")

        self.image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.disc_dir    = disc_dir
        self.cup_dir     = cup_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]

        disc_path = os.path.join(self.disc_dir, stem + ".png")
        cup_path  = os.path.join(self.cup_dir,  stem + ".png")

        image = np.array(Image.open(img_path).convert("RGB"))
        label = drishti_mask_to_label(disc_path, cup_path)

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"].long()

        return image, label, stem


# ───────────────────────────────────────────────────────────────
# DataLoader Factory
# ───────────────────────────────────────────────────────────────

def get_dataloaders(source: str = "REFUGE",
                    target: str = "Drishti",
                    batch_size: int = 4,
                    image_size: int = 512,
                    num_workers: int = 4):
    """
    Build source train/val and target test dataloaders.

    Returns:
        src_train_loader, src_val_loader, tgt_test_loader
    """
    train_tfm = get_train_transforms(image_size)
    val_tfm   = get_val_transforms(image_size)

    # ── Source ──
    if source == "REFUGE":
        src_train = REFUGEDataset(config.REFUGE_TRAIN_DIR, split="train", transform=train_tfm, image_size=image_size)
        src_val   = REFUGEDataset(config.REFUGE_VAL_DIR,   split="val",   transform=val_tfm,   image_size=image_size)
    else:
        raise ValueError(f"Unknown source dataset: {source}")

    src_train_loader = DataLoader(src_train, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
    src_val_loader   = DataLoader(src_val,   batch_size=1, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)

    # ── Target ──
    if target == "Drishti":
        tgt_test = DrishtiDataset(config.DRISHTI_DIR, split="Test", transform=val_tfm)
    elif target == "REFUGE_test":
        tgt_test = REFUGEDataset(config.REFUGE_TEST_DIR, split="test", transform=val_tfm, image_size=image_size)
    else:
        raise ValueError(f"Unknown target dataset: {target}")

    tgt_test_loader = DataLoader(tgt_test, batch_size=1, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)

    return src_train_loader, src_val_loader, tgt_test_loader
