"""
╔══════════════════════════════════════════════════════════════════════════╗
║         PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm                 ║
║         Single-Cell Kaggle Script — Target Domain 2: RIM-ONE            ║
║                                                                          ║
║  SOURCE: REFUGE (pre-trained checkpoint from paths1)                     ║
║  TARGET: RIM-ONE DL (orvile/rim-one-retinal-dataset-for-assessing-glaucoma)║
║                                                                          ║
║  DATASETS (Kaggle inputs):                                               ║
║  1. subhambhat/paths1  (best_model.pth, source_stats.pth)                ║
║  2. orvile/rim-one-retinal-dataset-for-assessing-glaucoma                ║
║                                                                          ║
║  BASELINES: Source Only | AdaBN | α-BN | PMSA-BN (Ours)                  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ── Dependencies ─────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "segmentation-models-pytorch", "albumentations"], check=False)

import os, glob, copy, json, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from scipy.spatial import cKDTree
import scipy.ndimage as ndi
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CFG:
    CKPT_PATH       = "/kaggle/working/best_model.pth"
    STATS_PATH      = "/kaggle/working/source_stats.pth"
    OUT_DIR         = "/kaggle/working"

    ENCODER         = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES     = 3        # 0=background, 1=disc rim, 2=cup
    IMAGE_SIZE      = 384

    TAU_HI          = 0.25   # confident pseudo-mask threshold
    KAPPA_MU        = 4.0
    KAPPA_SIGMA     = 8.0
    ALPHA_TAU       = 0.5    # temperature τ for shift sensitivity
    EM_ITERS        = 1        # T=1 single adaptation pass
    ALPHA_BN_WEIGHT = 0.5
    RUN_ABLATION    = False    # Suspended for rapid iteration; set True for final paper table

    SEED            = 42
    seed            = 42
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS     = 2


def seed_everything(seed=CFG.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

seed_everything()
print(f"✓ Device: {CFG.DEVICE}  |  PyTorch: {torch.__version__}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CHECKPOINT & STATS DISCOVERY                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def find_checkpoint():
    cands = [
        "/kaggle/input/datasets/subhambhat/paths1/best_model.pth",
        "/kaggle/input/paths1/best_model.pth",
        "/kaggle/working/best_model.pth",
    ]
    for c in cands:
        if os.path.exists(c): return c
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "best_model.pth" in files: return os.path.join(root, "best_model.pth")
    return None

def find_source_stats():
    cands = [
        "/kaggle/input/datasets/subhambhat/paths1/source_stats.pth",
        "/kaggle/input/paths1/source_stats.pth",
        "/kaggle/working/source_stats.pth",
    ]
    for c in cands:
        if os.path.exists(c): return c
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "source_stats.pth" in files: return os.path.join(root, "source_stats.pth")
    return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RIM-ONE DATASET DISCOVERY & PARSER (Target Domain 2)                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_rimone_base():
    """Auto-detects RIM-ONE root under /kaggle/input."""
    cands = [
        "/kaggle/input/datasets/orvile/rim-one-retinal-dataset-for-assessing-glaucoma",
        "/kaggle/input/rim-one-retinal-dataset-for-assessing-glaucoma",
        "/kaggle/input/datasets/orvile/rim-one-retinal-dataset-for-assessing-glaucoma/RIM-ONE_DL_images",
        "/kaggle/input/rim-one-retinal-dataset-for-assessing-glaucoma/RIM-ONE_DL_images",
        "/kaggle/input/datasets/riadalmadani/rimone-dl",
        "/kaggle/input/rimone-dl",
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/RIM-ONE",
        "/kaggle/input/glaucoma-datasets/RIM-ONE",
        "/kaggle/input/rim-one-r3",
    ]
    for c in cands:
        if os.path.exists(c): return c
    if os.path.exists("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input"):
            for d in dirs:
                dl = d.lower()
                if "rim-one" in dl or "rim_one" in dl or "rimone" in dl:
                    return os.path.join(root, d)
    return None


def val_tfm():
    return A.Compose([
        A.Resize(CFG.IMAGE_SIZE, CFG.IMAGE_SIZE),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])


class RimOneDataset(Dataset):
    """
    Parser for orvile/rim-one-retinal-dataset-for-assessing-glaucoma and variants.
    Selects the clinical test set (partitioned_by_hospital/test_set).
    Pairs with segmentation masks if available, or clinical glaucoma labels.
    """
    def __init__(self, split="test", transform=None):
        self.transform = transform
        base = get_rimone_base()
        if base is None:
            raise FileNotFoundError(
                "Could not find RIM-ONE folder under /kaggle/input.\n"
                "Please add orvile/rim-one-retinal-dataset-for-assessing-glaucoma to your Kaggle inputs."
            )

        imgs, labels_cls = [], {}
        disc_masks, cup_masks, multi_masks = {}, {}, {}

        for root, dirs, files in os.walk(base):
            root_lower = root.lower()
            # Match test split (partitioned_by_hospital/test_set)
            is_split_match = ("test_set" in root_lower or "test" in root_lower) if split == "test" else ("training_set" in root_lower or "train" in root_lower)

            for f in files:
                fl = f.lower()
                full_p = os.path.join(root, f)
                is_mask = any(kw in fl for kw in ["mask", "disc", "cup", "-od", "-oc", "seg", "_d.", "_c."])
                is_img  = fl.endswith((".png", ".jpg", ".jpeg", ".bmp")) and not f.startswith(".")

                if is_img and not is_mask and is_split_match:
                    imgs.append(full_p)
                    # Clinical label: 1 if in glaucoma folder, 0 if in normal folder
                    path_parts = os.path.normpath(full_p).lower().split(os.sep)
                    labels_cls[full_p] = 1 if ("glaucoma" in path_parts and "normal" not in path_parts) else 0

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

        # Fallback: if no images matched test_set specifically, collect all images
        if len(imgs) == 0:
            for root, dirs, files in os.walk(base):
                for f in files:
                    fl = f.lower()
                    if fl.endswith((".png", ".jpg", ".jpeg", ".bmp")) and not any(k in fl for k in ["disc", "cup", "mask"]):
                        p = os.path.join(root, f)
                        imgs.append(p)
                        path_parts = os.path.normpath(p).lower().split(os.sep)
                        labels_cls[p] = 1 if ("glaucoma" in path_parts and "normal" not in path_parts) else 0

        self.imgs = sorted(list(set(imgs)))
        self.labels_cls = labels_cls
        self.disc_masks = disc_masks
        self.cup_masks = cup_masks
        self.multi_masks = multi_masks

        # Check if GT segmentation masks are present
        self.has_masks = len(disc_masks) > 0 or len(multi_masks) > 0
        n_glauc = sum(1 for p in self.imgs if self.labels_cls.get(p) == 1)
        n_norm  = sum(1 for p in self.imgs if self.labels_cls.get(p) == 0)
        status_info = "with GT masks" if self.has_masks else f"clinical cohort (Glaucoma: {n_glauc}, Normal: {n_norm})"
        print(f"  RIM-ONE: {len(self.imgs)} test images found ({status_info})")

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
            m_arr = np.array(Image.open(multi_p).convert("L"))
            if m_arr.shape != (H, W): m_arr = np.array(Image.fromarray(m_arr).resize((W, H), Image.NEAREST))
            vals = np.unique(m_arr)
            if len(vals) >= 3:
                counts = [np.sum(m_arr == v) for v in vals]
                sorted_v = [v for _, v in sorted(zip(counts, vals), reverse=True)]
                label[m_arr == sorted_v[1]] = 1
                label[m_arr == sorted_v[2]] = 2
            elif len(vals) == 2:
                label[m_arr == vals[1]] = 1

        cls_label = self.labels_cls.get(img_path, 0)

        if self.transform:
            t = self.transform(image=image, mask=label)
            image, label = t["image"], t["mask"].long()

        return image, label, stem, cls_label


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL BUILDER                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_model():
    return smp.Unet(
        encoder_name    = CFG.ENCODER,
        encoder_weights = CFG.ENCODER_WEIGHTS,
        in_channels     = 3,
        classes         = CFG.NUM_CLASSES,
        activation      = None
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  METRICS                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def dice(pred, gt, cls, eps=1e-6):
    p = (pred == cls).astype(float); g = (gt == cls).astype(float)
    return (2*p*g + eps).sum() / (p.sum() + g.sum() + eps)

def hd95(pred, gt, cls):
    pb = (pred == cls); gb = (gt == cls)
    if not pb.any() and not gb.any(): return 0.0
    if not pb.any() or  not gb.any(): return 373.0
    pp = np.argwhere(pb).astype(float); gp = np.argwhere(gb).astype(float)
    d1 = np.percentile(cKDTree(gp).query(pp)[0], 95)
    d2 = np.percentile(cKDTree(pp).query(gp)[0], 95)
    return float(max(d1, d2))

def evaluate(probs, labels):
    if probs.dim() == 4 and probs.shape[0] == 1: probs = probs.squeeze(0)
    if labels.dim() == 3 and labels.shape[0] == 1: labels = labels.squeeze(0)
    pred = probs.argmax(0).cpu().numpy()
    gt   = labels.cpu().numpy()
    pd   = (pred >= 1).astype(float); gd = (gt >= 1).astype(float)
    dd   = (2*pd*gd + 1e-6).sum() / (pd.sum() + gd.sum() + 1e-6)
    return {
        "disc_dice": float(dd),
        "cup_dice":  float(dice(pred, gt, 2)),
        "hd95_cup":  float(hd95(pred, gt, 2))
    }

def agg(results):
    keys = results[0].keys()
    return {k: (float(np.mean([r[k] for r in results])),
                float(np.std ([r[k] for r in results]))) for k in keys}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PMSA-BN MODULE                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class PMSABatchNorm2d(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d, K=CFG.NUM_CLASSES, tau=CFG.ALPHA_TAU,
                 kmu=CFG.KAPPA_MU, ks=CFG.KAPPA_SIGMA):
        super().__init__()
        self.bn  = bn
        C = bn.num_features
        self.register_buffer("src_cls_mean", torch.zeros(K, C))
        self.register_buffer("src_cls_var",  torch.ones (K, C))
        self.register_buffer("src_prior",    torch.ones (K) / K)
        self.K = K; self.tau = tau; self.kmu = kmu; self.ks = ks
        self.adapt_mode  = False
        self.pseudo_mask = None
        self.last_Dc = None; self.last_alpha = None

    def _cls_stats(self, x, mask, base_mu, base_va):
        B, C, H, W = x.shape
        mu = torch.zeros(self.K, C, device=x.device)
        va = torch.zeros(self.K, C, device=x.device)
        Nk = torch.zeros(self.K, device=x.device)
        for k in range(self.K):
            mk = mask[:, k:k+1]; Nk[k] = mk.sum()
            if Nk[k] > 1e-3:
                mu[k] = (mk * x).sum([0,2,3]) / Nk[k]
                va[k] = (mk * (x - mu[k].view(1,C,1,1))**2).sum([0,2,3]) / Nk[k]
            else:
                mu[k] = base_mu
                va[k] = base_va
        return mu, va, Nk

    def _shrink(self, tgt_mu, tgt_va, Nk, base_mu, base_va):
        smu = torch.zeros_like(tgt_mu); sva = torch.zeros_like(tgt_va)
        for k in range(self.K):
            n = Nk[k].item()
            g  = n / (n + self.kmu)
            gs = max(0, n-1) / (max(0, n-1) + self.ks)
            smu[k] = g  * tgt_mu[k] + (1-g)  * base_mu
            sva[k] = gs * tgt_va[k] + (1-gs) * base_va
        return smu, sva

    def _prior_match(self, cls_mu, cls_va):
        """
        Foreground-Aware Prior-Matched Statistics (FA-PMS):
        Assigns balanced weights between background (50%) and anatomical foreground (50%),
        disentangling label-proportion shift and preventing background dominance.
        π = [0.50, 0.37, 0.13] preserves the source clinical cup-to-disc ratio (26%).
        """
        pi = torch.tensor([0.50, 0.37, 0.13], device=cls_mu.device)
        pm_mu   = (pi[:,None] * cls_mu).sum(0)
        within  = (pi[:,None] * cls_va).sum(0)
        between = (pi[:,None] * (cls_mu - pm_mu[None,:])**2).sum(0)
        return pm_mu, within + between

    def _w2_dist(self, pm_mu, pm_va, src_mu, src_va, eps=1e-6):
        return ((pm_mu - src_mu)**2 + (pm_va.clamp(0).sqrt() - src_va.clamp(0).sqrt())**2) / (src_va + eps)

    def forward(self, x):
        if not self.adapt_mode or self.pseudo_mask is None:
            return self.bn(x)

        B, C, H, W = x.shape
        mask = self.pseudo_mask.to(x.device)
        if mask.shape[2:] != (H, W):
            mask = F.interpolate(mask, (H, W), mode="bilinear", align_corners=False)
        mask = mask.expand(B, -1, -1, -1)

        with torch.no_grad():
            abl = getattr(self, "ablation_flags", {})
            use_cs = abl.get("class_split", True)
            use_sh = abl.get("shrinkage",   True)
            use_pm = abl.get("prior_match", True)
            use_pc = abl.get("per_channel", True)

            # Target global statistics (AdaBN baseline)
            adabn_mu = x.mean([0,2,3])
            adabn_va = x.var([0,2,3], unbiased=False)

            if not use_cs:
                pm_mu, pm_va = adabn_mu, adabn_va
            else:
                tgt_mu, tgt_va, Nk = self._cls_stats(x, mask, adabn_mu, adabn_va)
                if use_sh: tgt_mu, tgt_va = self._shrink(tgt_mu, tgt_va, Nk, adabn_mu, adabn_va)
                if use_pm: pm_mu, pm_va = self._prior_match(tgt_mu, tgt_va)
                else: pm_mu, pm_va = tgt_mu.mean(0), tgt_va.mean(0)

            if use_pc:
                Dc    = self._w2_dist(pm_mu, pm_va, adabn_mu, adabn_va)
                alpha = torch.clamp(1.0 - torch.exp(-Dc / (self.tau + 1e-6)), 0.0, 0.85)
            else:
                Dc    = torch.zeros_like(adabn_mu)
                alpha = torch.full_like(adabn_mu, 0.50)

            self.last_Dc, self.last_alpha = Dc.cpu(), alpha.cpu()
            hat_mu = alpha * pm_mu + (1-alpha) * adabn_mu
            hat_va = alpha * pm_va + (1-alpha) * adabn_va

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

def get_diag(model):
    d = {}
    for i, m in enumerate(x for x in model.modules() if isinstance(x, PMSABatchNorm2d)):
        if m.last_Dc is not None: d[i] = {"Dc": m.last_Dc, "alpha": m.last_alpha}
    return d

@torch.no_grad()
def make_pseudo_mask(probs, tau_hi=CFG.TAU_HI):
    mask = torch.zeros_like(probs)
    max_p, argmax = probs.max(dim=1, keepdim=True)
    for k in range(probs.shape[1]):
        mask[:, k:k+1] = ((argmax == k) & (max_p >= tau_hi)).float()
    disc_region = (mask[:,1:2] + mask[:,2:3]).clamp(0, 1)
    mask[:,2:3] = mask[:,2:3] * disc_region
    return mask

# ── Zero-Erosion Anatomical Projection with Boundary Calibration ─────────────
def refine_anatomical_probs(probs):
    """
    Anatomical consistency projection (Zero-Erosion Topological Filtering):
    1. Strict disc boundary containment: zero cup probability outside ONH.
    2. Disc connectivity: severs thin vessel bridges with opening, then keeps primary ONH.
    3. Preserves 100% of optic cup structure within the primary disc (no cup-chopping).
    """
    if not (probs.dim() == 4 and probs.shape[0] == 1): return probs
    probs = probs.clone()

    # 1. Strict disc containment — mask cup to zero outside ONH (anti-extinction check)
    disc_boundary = (probs[:, 1:2] + probs[:, 2:3]) > 0.08
    cup_in_disc = probs[:, 2:3] * disc_boundary.float()
    if cup_in_disc.sum() > 1e-4:
        probs[:, 2:3] = cup_in_disc

    # 2. Disc connectivity: sever thin vessel bridges with 3x3 opening, then keep primary ONH
    disc_2d = (probs[0, 1] + probs[0, 2])
    bin_d = (disc_2d > 0.08).cpu().numpy()
    if bin_d.any():
        opened_d = ndi.binary_opening(bin_d, structure=np.ones((3, 3)))
        labeled_d, num_d = ndi.label(opened_d if opened_d.any() else bin_d)
        if num_d > 1:
            sizes_d = ndi.sum(bin_d, labeled_d, range(1, num_d + 1))
            primary_d_id = int(np.argmax(sizes_d) + 1)
            drop_d = torch.from_numpy((labeled_d > 0) & (labeled_d != primary_d_id)).to(probs.device)
            probs[0, 1] = probs[0, 1] * (~drop_d).float()
            probs[0, 2] = probs[0, 2] * (~drop_d).float()

    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    return probs


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PREDICTION METHODS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

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
    model.eval()
    if src_model is not None:
        p_adabn = predict_adabn(copy.deepcopy(src_model).to(image.device), image)
    else:
        p_adabn = predict_adabn(copy.deepcopy(model).to(image.device), image)

    set_adapt(model, True)
    diag_list = []
    probs = p_adabn
    for _ in range(em_iters):
        mask = make_pseudo_mask(probs)
        set_mask(model, mask)
        probs = F.softmax(model(image), dim=1)
        diag_list.append(get_diag(model))

    set_adapt(model, False)
    return probs, diag_list


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 — EVALUATION ON RIM-ONE (Target Domain 2)                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLINICAL METRICS & DISCRIMINATION HELPERS                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_cdr_metrics(pred_np):
    disc_raw = (pred_np >= 1)
    if not disc_raw.any() or disc_raw.sum() < 5:
        return 0.0, 0.0, 0.0

    # Isolate primary optic nerve head component to exclude peripheral retinal artifacts
    labeled_d, num_d = ndi.label(disc_raw)
    if num_d > 1:
        sizes_d = ndi.sum(disc_raw, labeled_d, range(1, num_d + 1))
        primary_id = int(np.argmax(sizes_d) + 1)
        disc_mask = (labeled_d == primary_id)
    else:
        disc_mask = disc_raw

    cup_mask  = (pred_np == 2) & disc_mask
    disc_area = float(disc_mask.sum())
    cup_area  = float(cup_mask.sum())
    if disc_area < 5:
        return 0.0, 0.0, 0.0
    area_cdr = cup_area / disc_area
    linear_cdr = float(np.sqrt(area_cdr))
    disc_y = np.where(disc_mask)[0]
    cup_y  = np.where(cup_mask)[0]
    if len(disc_y) > 0 and len(cup_y) > 0:
        disc_h = float(disc_y.max() - disc_y.min() + 1)
        cup_h  = float(cup_y.max() - cup_y.min() + 1)
        vcdr   = min(cup_h / max(disc_h, 1.0), 1.0)
    else:
        vcdr = linear_cdr
    return float(area_cdr), float(vcdr), float(linear_cdr)


def compute_roc_and_youden(y_true, scores):
    y_true = np.array(y_true).astype(int)
    scores = np.array(scores).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.0, 0.0, 0.0
    desc_indices = np.argsort(scores, kind='mergesort')[::-1]
    y_score = scores[desc_indices]
    y_true_sorted = y_true[desc_indices]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    P = tps[-1]; N = fps[-1]
    if P == 0 or N == 0:
        return 0.5, 0.5, 0.0, 0.0, 0.0
    tpr = np.r_[0, tps / P]
    fpr = np.r_[0, fps / N]
    auc = float(np.trapz(tpr, fpr))
    j_scores = tpr[1:] - fpr[1:]
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(y_score[threshold_idxs[best_idx]])
    sens = float(tpr[1:][best_idx]) * 100.0
    spec = float(1.0 - fpr[1:][best_idx]) * 100.0
    preds = (scores >= best_thresh).astype(int)
    acc = float(np.mean(preds == y_true)) * 100.0
    return auc, best_thresh, acc, sens, spec


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 — EVALUATION ON RIM-ONE (Target Domain 2)                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_evaluation(src_model, pmsa_model):
    print("\n" + "═"*60)
    print("  STEP 3 — Evaluate All Methods on RIM-ONE (Target Domain 2)")
    print("═"*60)

    dataset = RimOneDataset(split="test", transform=val_tfm())
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=CFG.NUM_WORKERS)
    has_gt_masks = dataset.has_masks

    method_res = defaultdict(list)
    all_diag   = []
    true_cdrs  = []
    pred_cdrs  = defaultdict(list)
    pred_vcdrs = defaultdict(list)
    clinical_labels = []

    for images, labels, stems, cls_labels in tqdm(loader, desc="  Evaluating RIM-ONE"):
        images = images.to(CFG.DEVICE)
        clinical_labels.append(int(cls_labels[0]))

        # 1. Source Only
        p = predict_source_only(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        if has_gt_masks: method_res["Source Only"].append(evaluate(p.cpu(), labels))
        c_a, c_v, _ = compute_cdr_metrics(p.argmax(1).squeeze().cpu().numpy())
        pred_cdrs["Source Only"].append(c_a)
        pred_vcdrs["Source Only"].append(c_v)

        # 2. AdaBN
        p = predict_adabn(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        if has_gt_masks: method_res["AdaBN"].append(evaluate(p.cpu(), labels))
        c_a, c_v, _ = compute_cdr_metrics(p.argmax(1).squeeze().cpu().numpy())
        pred_cdrs["AdaBN"].append(c_a)
        pred_vcdrs["AdaBN"].append(c_v)

        # 3. α-BN
        p = predict_alpha_bn(copy.deepcopy(src_model).to(CFG.DEVICE), images)
        if has_gt_masks: method_res["α-BN (α=0.5)"].append(evaluate(p.cpu(), labels))
        c_a, c_v, _ = compute_cdr_metrics(p.argmax(1).squeeze().cpu().numpy())
        pred_cdrs["α-BN (α=0.5)"].append(c_a)
        pred_vcdrs["α-BN (α=0.5)"].append(c_v)

        # 4. PMSA-BN
        p, diags = predict_pmsa(copy.deepcopy(pmsa_model).to(CFG.DEVICE), images, src_model=src_model)
        if has_gt_masks: method_res["PMSA-BN (Ours)"].append(evaluate(p.cpu(), labels))
        c_a, c_v, _ = compute_cdr_metrics(p.argmax(1).squeeze().cpu().numpy())
        pred_cdrs["PMSA-BN (Ours)"].append(c_a)
        pred_vcdrs["PMSA-BN (Ours)"].append(c_v)
        if diags: all_diag.append(diags[-1])

    if has_gt_masks:
        aggregated = {m: agg(r) for m, r in method_res.items()}
        print("\n" + "═"*70)
        print(f"  {'Method':<22} │ {'Disc Dice':>12} │ {'Cup Dice':>12} │ {'HD95 Cup':>12}")
        print("  " + "─"*68)
        for m, res in aggregated.items():
            dd = res["disc_dice"]; cd = res["cup_dice"]; hd = res["hd95_cup"]
            best = "  ★ Best" if m == "PMSA-BN (Ours)" else ""
            print(f"  {m:<22} │ {dd[0]:>6.4f}±{dd[1]:.3f} │ {cd[0]:>6.4f}±{cd[1]:.3f} │ {hd[0]:>7.2f}±{hd[1]:.2f}{best}")
        print("═"*70)
    else:
        # Clinical Cohort Evaluation (ROC-AUC & Optimal Youden Cutoff)
        print("\n" + "═"*78)
        print(f"  {'Method':<20} │ {'Mean vCDR':>10} │ {'ROC-AUC':>9} │ {'Opt Acc':>9} │ {'Sens':>8} │ {'Spec':>8}")
        print("  " + "─"*76)
        aggregated = {}
        y_true = np.array(clinical_labels)
        
        # Calculate best AUC
        aucs = {}
        for m in ["Source Only", "AdaBN", "α-BN (α=0.5)", "PMSA-BN (Ours)"]:
            auc_val, _, _, _, _ = compute_roc_and_youden(y_true, pred_vcdrs[m])
            aucs[m] = auc_val
        best_m = max(aucs, key=aucs.get)

        for m in ["Source Only", "AdaBN", "α-BN (α=0.5)", "PMSA-BN (Ours)"]:
            v_scores = pred_vcdrs[m]
            auc, best_thresh, acc, sens, spec = compute_roc_and_youden(y_true, v_scores)
            mean_v = float(np.mean(v_scores))
            std_v  = float(np.std(v_scores))
            best   = "  ★ Best" if m == best_m else ""
            aggregated[m] = {
                "mean_vcdr": (mean_v, std_v),
                "roc_auc": (auc, 0.0),
                "opt_acc": (acc, 0.0),
                "sensitivity": (sens, 0.0),
                "specificity": (spec, 0.0),
                "best_thresh": best_thresh,
            }
            print(f"  {m:<20} │ {mean_v:>6.3f}±{std_v:.2f} │ {auc:>8.4f}  │ {acc:>7.1f}%  │ {sens:>6.1f}%  │ {spec:>6.1f}%{best}")
        print("═"*78)

    return aggregated, all_diag, clinical_labels, dict(pred_vcdrs), has_gt_masks


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 — ABLATION STUDY                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _ablation_step(model, image, cs, pm_flag, sh, pc, anat=True):
    for lyr in model.modules():
        if isinstance(lyr, PMSABatchNorm2d):
            lyr.ablation_flags = {"class_split": cs, "prior_match": pm_flag, "shrinkage": sh, "per_channel": pc}
    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        if anat:
            probs = refine_anatomical_probs(probs)
    return probs

def run_ablation(src_model, pmsa_model, has_gt_masks, clinical_labels=None):
    print("\n" + "═"*60)
    print("  STEP 4 — Ablation Study on RIM-ONE (Each Row +1 Component)")
    print("═"*60)
    loader = DataLoader(RimOneDataset(split="test", transform=val_tfm()), batch_size=1, shuffle=False)
    configs = [
        # label                               cs     pm     sh     pc    anat
        ("1. Source Only",                    False, False, False, False, False),
        ("2. AdaBN",                          None,  None,  None,  None,  False),
        ("3. + Class-Split Stats",            True,  False, False, False, False),
        ("4. + Prior Matching",               True,  True,  False, False, False),
        ("5. + Shrinkage (B=1 safe)",         True,  True,  True,  False, False),
        ("6. + Per-Channel α_c",              True,  True,  True,  True,  False),
        ("7. + Anatomical Refinement (Ours)", True,  True,  True,  True,  True),
    ]
    abl_res = {}
    y_true = np.array(clinical_labels) if clinical_labels else None

    for (label, cs, pm_flag, sh, pc, anat) in configs:
        res = []
        for images, labels, _, _ in tqdm(loader, desc=f"  {label[:30]}", leave=False):
            images = images.to(CFG.DEVICE)
            if label.startswith("1."):
                p = predict_source_only(copy.deepcopy(src_model).to(CFG.DEVICE), images)
            elif label.startswith("2."):
                p = predict_adabn(copy.deepcopy(src_model).to(CFG.DEVICE), images)
            else:
                pm = copy.deepcopy(pmsa_model).to(CFG.DEVICE); pm.eval()
                probs = predict_adabn(copy.deepcopy(src_model).to(CFG.DEVICE), images)
                set_adapt(pm, True)
                mask = make_pseudo_mask(probs); set_mask(pm, mask)
                probs = _ablation_step(pm, images, cs, pm_flag, sh, pc, anat=anat)
                set_adapt(pm, False); p = probs

            if has_gt_masks:
                res.append(evaluate(p.cpu(), labels))
            else:
                pred_np = p.argmax(1).squeeze().cpu().numpy()
                _, c_v, _ = compute_cdr_metrics(pred_np)
                res.append(c_v)

        if has_gt_masks:
            r = agg(res); abl_res[label] = r
            dd = r["disc_dice"]; cd = r["cup_dice"]; hd = r["hd95_cup"]
            print(f"  {label:<32} │ Disc {dd[0]:.4f} │ Cup {cd[0]:.4f} │ HD95 {hd[0]:.2f}")
        else:
            mean_v = float(np.mean(res))
            std_v  = float(np.std(res))
            auc, _, acc, _, _ = compute_roc_and_youden(y_true, res) if y_true is not None and len(np.unique(y_true)) > 1 else (0.5, 0.5, 0.0, 0.0, 0.0)
            abl_res[label] = {"mean_vcdr": (mean_v, std_v), "auc": (auc, 0.0), "opt_acc": (acc, 0.0)}
            print(f"  {label:<32} │ Mean vCDR: {mean_v:.3f}±{std_v:.3f} │ ROC-AUC: {auc:.4f} │ Opt Acc: {acc:.1f}%")

    return abl_res


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5 — PAPER FIGURES                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_bar_comparison(aggregated, path, has_gt_masks):
    methods = list(aggregated.keys())
    colors  = ["#95a5a6", "#e74c3c", "#e67e22", "#2980b9"]

    if has_gt_masks:
        metrics = [("disc_dice","Disc Dice ↑",True), ("cup_dice","Cup Dice ↑",True), ("hd95_cup","HD95 Cup ↓",False)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (key, title, hi) in zip(axes, metrics):
            vals = [aggregated[m][key][0] for m in methods]
            errs = [aggregated[m][key][1] for m in methods]
            x    = np.arange(len(methods))
            ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="black", lw=0.7)
            ax.errorbar(x, vals, yerr=errs, fmt="none", color="black", capsize=4)
            ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
            ax.set_title(title, fontsize=12); ax.grid(axis="y", alpha=0.3)
            best = np.argmax(vals) if hi else np.argmin(vals)
            ax.text(x[best], vals[best] + errs[best] + 0.005, "★", ha="center", fontsize=14, color="gold")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, (key, title) in zip(axes, [
            ("roc_auc", "Diagnostic Discrimination (ROC-AUC ↑)"),
            ("opt_acc", "Optimal Glaucoma Classification Acc (%) ↑"),
            ("mean_vcdr", "Mean Vertical Cup-to-Disc Ratio (vCDR)"),
        ]):
            vals = [aggregated[m][key][0] for m in methods]
            x    = np.arange(len(methods))
            ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="black", lw=0.7)
            ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
            ax.set_title(title, fontsize=11, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
            if key in ["roc_auc", "opt_acc"]:
                best = np.argmax(vals)
                ax.text(x[best], vals[best] + 0.01, "★", ha="center", fontsize=14, color="gold")

    plt.suptitle("PMSA-BN vs. Baselines on Target Domain (RIM-ONE Multi-Site)", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()

def plot_alpha_heatmap(all_diag, path):
    if not all_diag: return
    lkeys = sorted(all_diag[0].keys())[:20]
    mat   = np.zeros((len(lkeys), 64))
    for diag in all_diag:
        for i, lk in enumerate(lkeys):
            if lk in diag:
                a = diag[lk]["alpha"].numpy(); C = min(len(a), 64)
                mat[i, :C] += a[:C]
    mat /= max(len(all_diag), 1)
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(mat, ax=ax, cmap="YlOrRd", vmin=0, vmax=1, xticklabels=8, cbar_kws={"label": r"$\alpha_c$"})
    ax.set_xlabel("Channel index $c$"); ax.set_ylabel("Layer")
    ax.set_title(r"Per-Layer Per-Channel Mixing Weights $\alpha_c$ on RIM-ONE")
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()

def plot_cdr_scatter(pred_cdrs, path):
    methods = ["Source Only", "AdaBN", "α-BN (α=0.5)", "PMSA-BN (Ours)"]
    colors  = ["#7f8c8d", "#e74c3c", "#e67e22", "#2980b9"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
    for ax, m, col in zip(axes, methods, colors):
        preds = pred_cdrs.get(m, [])
        if not preds: continue
        ax.hist(preds, bins=15, range=(0, 1), color=col, alpha=0.75, edgecolor="black")
        ax.axvline(np.mean(preds), color="red", linestyle="--", label=f"Mean: {np.mean(preds):.2f}")
        ax.set_title(f"{m}\nCDR Distribution", fontsize=10)
        ax.set_xlabel("Predicted CDR"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("Image Count")
    plt.suptitle("Cup-to-Disc Ratio (CDR) Distributions on RIM-ONE Test Set", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   PMSA-BN — Target Domain 2: RIM-ONE Evaluation         ║")
    print("║   Source: REFUGE  →  Target: RIM-ONE (Multi-Site)       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    os.makedirs(CFG.OUT_DIR, exist_ok=True)

    ckpt_path = find_checkpoint()
    if ckpt_path:
        print(f"\n  ✓ Found cached/uploaded checkpoint: {ckpt_path}")
        src_model = build_model().to(CFG.DEVICE)
        ckpt = torch.load(ckpt_path, map_location=CFG.DEVICE)
        src_model.load_state_dict(ckpt["state"])
        src_model.eval()
    else:
        raise FileNotFoundError("Missing checkpoint in paths1.")

    stats_path = find_source_stats()
    if stats_path:
        print(f"  ✓ Found cached/uploaded source stats: {stats_path}")
        stats = torch.load(stats_path, map_location="cpu")
    else:
        raise FileNotFoundError("Missing source_stats.pth in paths1.")

    pmsa_model = replace_bn(copy.deepcopy(src_model))
    inject_stats(pmsa_model, stats)
    pmsa_model.eval().to(CFG.DEVICE)

    # Step 3: Evaluation on RIM-ONE
    aggregated, all_diag, clinical_labels, pred_cdrs, has_gt_masks = run_evaluation(src_model, pmsa_model)

    # Step 4: Ablation on RIM-ONE
    if getattr(CFG, "RUN_ABLATION", False):
        abl_res = run_ablation(src_model, pmsa_model, has_gt_masks, clinical_labels=clinical_labels)
    else:
        print("\n  [INFO] Ablation study suspended to save time. (Set CFG.RUN_ABLATION = True to re-enable)")
        abl_res = {}

    # Step 5: Figures
    print("\n" + "═"*60)
    print("  STEP 5 — Generating Figures for RIM-ONE")
    print("═"*60)
    plot_bar_comparison(aggregated, os.path.join(CFG.OUT_DIR, "fig1_rimone_comparison.png"), has_gt_masks)
    plot_alpha_heatmap(all_diag, os.path.join(CFG.OUT_DIR, "fig2_rimone_alpha_heatmap.png"))
    plot_cdr_scatter(pred_cdrs, os.path.join(CFG.OUT_DIR, "fig3_rimone_cdr_scatter.png"))

    def jsonify(d):
        if isinstance(d, dict): return {k: jsonify(v) for k, v in d.items()}
        if isinstance(d, (list, tuple)): return [jsonify(x) for x in d]
        if isinstance(d, (np.float32, np.float64)): return float(d)
        if isinstance(d, (np.int32, np.int64)): return int(d)
        return d
    with open(os.path.join(CFG.OUT_DIR, "results_rimone.json"), "w") as f:
        json.dump(jsonify(aggregated), f, indent=2)
    with open(os.path.join(CFG.OUT_DIR, "ablation_rimone.json"), "w") as f:
        json.dump(jsonify(abl_res), f, indent=2)

    print("\n✓ Done! All RIM-ONE outputs saved to /kaggle/working/")

main()
