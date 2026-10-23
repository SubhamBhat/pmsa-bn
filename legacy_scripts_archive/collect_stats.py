"""
collect_stats.py — Phase 0: Collect and Save Source Statistics
===============================================================
Run ONCE after training.
Feeds the source training set through the trained model and
saves per-class, per-channel BN statistics for each layer.

RUN:
    python collect_stats.py
"""

import torch
import os
import config
from train import build_model
from data.fundus_dataset import get_dataloaders
from adaptation.source_stats import collect_and_save


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # Load trained source model
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run train.py first."
        )

    model = build_model()
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Phase 0] Loaded source model from: {ckpt_path}")

    # Use source training set (with ground-truth labels for class conditioning)
    train_loader, _, _ = get_dataloaders(
        source="REFUGE",
        target="Drishti",
        batch_size=4,
        image_size=config.IMAGE_SIZE,
        num_workers=config.NUM_WORKERS
    )

    # Collect and save
    collect_and_save(model, train_loader, device=str(device),
                     save_dir=config.STATS_DIR)


if __name__ == "__main__":
    main()
