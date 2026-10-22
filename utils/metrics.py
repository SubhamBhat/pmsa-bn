"""
utils/metrics.py — Evaluation Metrics
======================================
Dice coefficient and Hausdorff Distance 95 (HD95) for segmentation.
Both computed per-class (disc, cup separately).
"""

import numpy as np
import torch
from scipy.ndimage import label as scipy_label
from scipy.spatial.distance import directed_hausdorff


def dice_coefficient(pred: np.ndarray, target: np.ndarray,
                     class_id: int, smooth: float = 1e-6) -> float:
    """
    Dice = 2 * |P ∩ T| / (|P| + |T|)

    Args:
        pred, target : [H, W]  integer class label arrays
        class_id     : which class to evaluate
    """
    p = (pred == class_id).astype(float)
    t = (target == class_id).astype(float)
    intersection = (p * t).sum()
    return (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray,
                           class_id: int) -> float:
    """
    95th percentile Hausdorff Distance (HD95).
    Lower is better.

    Returns 0.0 if both masks are empty.
    Returns large value if one mask is empty but not the other.
    """
    pred_bin   = (pred == class_id)
    target_bin = (target == class_id)

    if pred_bin.sum() == 0 and target_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or target_bin.sum() == 0:
        return 373.13   # diagonal of 512×512 image = worst case

    # Get boundary points
    pred_pts   = np.argwhere(pred_bin).astype(float)
    target_pts = np.argwhere(target_bin).astype(float)

    # Directed distances
    d1 = _directed_hd95(pred_pts, target_pts)
    d2 = _directed_hd95(target_pts, pred_pts)

    return max(d1, d2)


def _directed_hd95(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Compute 95th percentile of distances from each point in A to nearest in B."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pts_b)
    dists, _ = tree.query(pts_a, k=1)
    return float(np.percentile(dists, 95))


def evaluate_batch(pred_probs: torch.Tensor,
                   target_labels: torch.Tensor,
                   num_classes: int = 3) -> dict:
    """
    Compute Dice and HD95 for one image.

    Args:
        pred_probs    : [1, K, H, W]  softmax probabilities
        target_labels : [1, H, W] or [H, W]  integer labels

    Returns:
        dict with keys: "dice_rim", "dice_cup", "hd95_rim", "hd95_cup"
    """
    pred_labels = pred_probs.argmax(dim=1).squeeze().cpu().numpy()  # [H, W]
    gt          = target_labels.squeeze().cpu().numpy()              # [H, W]

    results = {}
    for cls_id, cls_name in [(1, "rim"), (2, "cup")]:
        results[f"dice_{cls_name}"] = dice_coefficient(pred_labels, gt, cls_id)
        results[f"hd95_{cls_name}"] = hausdorff_distance_95(pred_labels, gt, cls_id)

    # Combined disc Dice (cup + rim together = disc)
    pred_disc = (pred_labels >= 1).astype(float)
    gt_disc   = (gt >= 1).astype(float)
    smooth    = 1e-6
    results["dice_disc"] = (2.0 * (pred_disc * gt_disc).sum() + smooth) / \
                           (pred_disc.sum() + gt_disc.sum() + smooth)

    return results


def aggregate_results(all_results: list) -> dict:
    """
    Compute mean ± std over all images.

    Args:
        all_results: list of dicts from evaluate_batch()
    Returns:
        dict with "mean_dice_cup", "std_dice_cup", etc.
    """
    keys = all_results[0].keys()
    aggregated = {}
    for k in keys:
        vals = [r[k] for r in all_results]
        aggregated[f"mean_{k}"] = float(np.mean(vals))
        aggregated[f"std_{k}"]  = float(np.std(vals))
    return aggregated


def print_results_table(method_results: dict):
    """
    Pretty-print a comparison table.

    Args:
        method_results: {"Method Name": aggregated_dict, ...}
    """
    print("\n" + "="*80)
    print(f"{'Method':<25} {'Disc Dice':>10} {'Cup Dice':>10} {'HD95 Cup':>10}")
    print("-"*80)
    for method, res in method_results.items():
        print(f"{method:<25} "
              f"{res['mean_dice_disc']:.4f}±{res['std_dice_disc']:.4f}  "
              f"{res['mean_dice_cup']:.4f}±{res['std_dice_cup']:.4f}  "
              f"{res['mean_hd95_cup']:.2f}±{res['std_hd95_cup']:.2f}")
    print("="*80 + "\n")
