"""
train.py — Source Domain Training
===================================
Trains U-Net (ResNet34 encoder, ImageNet pretrained) on REFUGE training set.
Saves best checkpoint based on validation Cup Dice.

RUN:
    python train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

import config
from data.fundus_dataset import get_dataloaders
from utils.metrics import evaluate_batch, aggregate_results


# ─────────────────────────────────────────────────────────────────────────
# Loss Function
# ─────────────────────────────────────────────────────────────────────────

class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss with class weighting."""

    def __init__(self, num_classes=3, weights=None):
        super().__init__()
        w = torch.tensor(weights or [1.0]*num_classes, dtype=torch.float32)
        self.ce   = nn.CrossEntropyLoss(weight=w)
        self.dice = smp.losses.DiceLoss(mode="multiclass")

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


# ─────────────────────────────────────────────────────────────────────────
# Build Model
# ─────────────────────────────────────────────────────────────────────────

def build_model():
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=config.NUM_CLASSES,
        activation=None          # raw logits (we apply softmax separately)
    )
    return model


# ─────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────

def train():
    torch.manual_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        source="REFUGE",
        target="Drishti",
        batch_size=config.TRAIN_BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        num_workers=config.NUM_WORKERS
    )

    # Model
    model     = build_model().to(device)
    criterion = DiceCELoss(num_classes=config.NUM_CLASSES, weights=config.CLASS_WEIGHTS)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR,
                            weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    best_cup_dice = 0.0
    patience_counter = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [Train]", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)               # [B, K, H, W]
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_results = []

        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                images = images.to(device)
                import torch.nn.functional as F
                probs  = F.softmax(model(images), dim=1)
                res    = evaluate_batch(probs.cpu(), labels)
                val_results.append(res)

        agg = aggregate_results(val_results)

        # Logging
        writer.add_scalar("Loss/train",     avg_loss,                 epoch)
        writer.add_scalar("Dice/val_disc",  agg["mean_dice_disc"],    epoch)
        writer.add_scalar("Dice/val_cup",   agg["mean_dice_cup"],     epoch)
        writer.add_scalar("HD95/val_cup",   agg["mean_hd95_cup"],     epoch)

        print(f"[Epoch {epoch:03d}] Loss: {avg_loss:.4f} | "
              f"Val Disc Dice: {agg['mean_dice_disc']:.4f} | "
              f"Val Cup Dice: {agg['mean_dice_cup']:.4f} | "
              f"Val HD95 Cup: {agg['mean_hd95_cup']:.2f}")

        # Save best checkpoint (based on Cup Dice — the harder metric)
        if agg["mean_dice_cup"] > best_cup_dice:
            best_cup_dice    = agg["mean_dice_cup"]
            patience_counter = 0
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_metrics": agg
            }, ckpt_path)
            print(f"  ✓ Saved best model (Cup Dice = {best_cup_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"[Train] Early stopping at epoch {epoch}")
                break

    writer.close()
    print(f"\n[Train] Done. Best Val Cup Dice = {best_cup_dice:.4f}")
    print(f"[Train] Checkpoint saved at: {config.CHECKPOINT_DIR}/best_model.pth")


if __name__ == "__main__":
    train()
