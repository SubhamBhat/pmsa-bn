# Script to upgrade kaggle_unified_benchmark.py with full multi-modal BraTS support
import sys, re

file_path = r"c:\Users\Subham Bhat\Downloads\tds\pmsa_bn_poc\kaggle_unified_benchmark.py"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update USER_CFG DOMAIN validation comment and options
old_desc = '    #   "derm"   : ISIC2018 -> [HAM10000, Derm7pt, PH2] & PH2 -> [ISIC2018] (4 pairs)\n    #   "all"    : Sequentially runs all 4 domains / 17 pairs\n    DOMAIN = "derm",'
new_desc = '    #   "derm"   : ISIC2018 -> [HAM10000, Derm7pt, PH2] & PH2 -> [ISIC2018] (4 pairs)\n    #   "brain"  : BraTS2019 -> [BraTS2020] (4-class Neuro-Oncology MRI)\n    #   "all"    : Sequentially runs all 5 domains / 19 pairs\n    DOMAIN = "brain",'
assert old_desc in text, "old_desc not found"
text = text.replace(old_desc, new_desc, 1)

# 2. Update get_benchmark_runs validation string
old_valid = "Valid options: 'fundus', 'polyp', 'cxr', 'derm', or 'all'."
new_valid = "Valid options: 'fundus', 'polyp', 'cxr', 'derm', 'brain', or 'all'."
assert old_valid in text, "old_valid not found"
text = text.replace(old_valid, new_valid, 1)

# 3. Update pip install to include nibabel and h5py
old_pip = 'subprocess.run([sys.executable, "-m", "pip", "install", "-q",\n                "segmentation-models-pytorch", "albumentations"], check=False)'
new_pip = 'subprocess.run([sys.executable, "-m", "pip", "install", "-q",\n                "segmentation-models-pytorch", "albumentations", "nibabel", "h5py"], check=False)'
assert old_pip in text, "old_pip not found"
text = text.replace(old_pip, new_pip, 1)

# 4. Add nibabel and h5py imports
old_imp = 'import albumentations as A\nfrom albumentations.pytorch import ToTensorV2\nimport segmentation_models_pytorch as smp'
new_imp = '''import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
try:
    import nibabel as nib
except ImportError:
    nib = None
try:
    import h5py
except ImportError:
    h5py = None'''
assert old_imp in text, "old_imp not found"
text = text.replace(old_imp, new_imp, 1)

# 5. Replace _find_brain_pairs with robust 2D/3D NIfTI and HDF5 discovery
old_brain_func_marker = 'def _find_brain_pairs(domain):'
idx_brain_start = text.find(old_brain_func_marker)
assert idx_brain_start != -1, "idx_brain_start not found"
idx_get_pairs = text.find('def get_pairs(domain):', idx_brain_start)
assert idx_get_pairs != -1, "idx_get_pairs not found"

new_brain_funcs = '''def _normalize_mri_slice(slice_arr):
    pmin = np.percentile(slice_arr, 1)
    pmax = np.percentile(slice_arr, 99)
    if pmax - pmin > 1e-4:
        norm = np.clip((slice_arr - pmin) / (pmax - pmin), 0.0, 1.0)
    else:
        norm = np.zeros_like(slice_arr)
    return norm.astype(np.float32)

def _find_brain_pairs(domain):
    """
    Locate multi-parametric BraTS 2019 / 2020 volumes or pre-extracted H5 slices.
    Returns list of items: either filepath strings or dictionary descriptors.
    """
    # 1. Check for pre-extracted H5 dataset (Awsaf49 BraTS 2020)
    h5_candidates = [
        "/kaggle/input/datasets/awsaf49/brats2020-training-data/BraTS2020_training_data/content/data",
        "/kaggle/input/datasets/awsaf49/brats2020-training-data",
        "/kaggle/input/brats2020-training-data/content/data",
        "/kaggle/input/brats2020-training-data",
        "/kaggle/input/brain-tumor-segmentation-brats2020",
        "./brats2020_target",
    ]
    h5_dir = None
    for c in h5_candidates:
        if os.path.exists(c):
            for root, _, files in os.walk(c):
                if any(f.endswith(".h5") for f in files):
                    h5_dir = root
                    break
            if h5_dir: break

    # 2. Check for BraTS 2019 NIfTI volume folders
    nii_candidates = [
        "/kaggle/input/datasets/walidrehamnia/brain-tumor-segmentation-brats-2019-train-valid/MICCAI_BraTS_2019_Data_Training",
        "/kaggle/input/datasets/walidrehamnia/brain-tumor-segmentation-brats-2019-train-valid",
        "/kaggle/input/brain-tumor-segmentation-brats-2019-train-valid",
        "/kaggle/input/brain-tumor-segmentation-brats-2019",
        "/kaggle/input/miccai-brats-2019-data-training",
        "./brats2019_source",
    ]
    nii_dir = None
    for c in nii_candidates:
        if os.path.exists(c):
            for root, dirs, files in os.walk(c):
                dirs[:] = [d for d in dirs if d != "data" and "2020" not in d.lower()]
                if any("flair" in f.lower() and f.endswith((".nii", ".nii.gz")) for f in files) or any(d.upper() in ["HGG", "LGG"] for d in dirs):
                    nii_dir = root
                    break
            if nii_dir: break

    # If domain is BraTS 2019 source and NIfTI exists, parse 2D slices
    if domain in ("BraTS2019", "BraTS") and nii_dir is not None:
        patient_dirs = []
        for root, dirs, files in os.walk(nii_dir):
            if any("flair" in f.lower() and f.endswith((".nii", ".nii.gz")) for f in files):
                patient_dirs.append(root)
        pairs = []
        for pdir in sorted(patient_dirs):
            flair = glob.glob(os.path.join(pdir, "*flair.nii*"))
            t1ce  = glob.glob(os.path.join(pdir, "*t1ce.nii*"))
            t2    = glob.glob(os.path.join(pdir, "*t2.nii*"))
            seg   = glob.glob(os.path.join(pdir, "*seg.nii*"))
            if flair and t1ce and t2 and seg:
                pairs.append({"type": "nii", "flair": flair[0], "t1ce": t1ce[0], "t2": t2[0], "seg": seg[0]})
        if pairs:
            return pairs

    # If H5 dataset is available, use patient-partitioned slices
    if h5_dir is not None:
        all_h5 = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
        if all_h5:
            # Deterministic partition: first 70% source, remaining 30% target
            pat_map = defaultdict(list)
            for f in all_h5:
                # filename format: volume_<id>_slice_<num>.h5
                bname = os.path.basename(f)
                pat_id = bname.split("_slice_")[0] if "_slice_" in bname else bname[:10]
                pat_map[pat_id].append(f)
            unique_pats = sorted(list(pat_map.keys()))
            split_idx = int(0.70 * len(unique_pats))
            if domain in ("BraTS2019", "BraTS"):
                sel_pats = unique_pats[:split_idx]
            else:
                sel_pats = unique_pats[split_idx:]
            h5_pairs = []
            for pid in sel_pats:
                for fpath in pat_map[pid]:
                    h5_pairs.append({"type": "h5", "path": fpath})
            return h5_pairs

    # Fallback: scan 2D image formats
    kw_map = {
        "BraTS2019": ["brats2019", "brats_2019", "brats"],
        "BraTS2020": ["brats2020", "brats_2020", "brats"],
        "TCGA-GBM":  ["tcga", "gbm", "tcga-gbm"],
    }
    kws = kw_map.get(domain, [domain.lower()])
    base = _auto_find(kws)
    if base is None: return []

    imgs, masks = [], {}
    for r, _, files in os.walk(base):
        for f in files:
            fl = f.lower()
            fp = os.path.join(r, f)
            if fl.endswith((".png", ".jpg", ".jpeg", ".npy", ".npz")):
                if "seg" in fl or "mask" in fl:
                    stem = _stem(f).replace("_seg", "").replace("_mask", "").lower()
                    masks[stem] = fp
                elif not any(k in fl for k in ["mask", "seg"]):
                    imgs.append(fp)
    pairs = []
    for im in sorted(imgs):
        s = _stem(im).lower()
        if s in masks:
            pairs.append((im, masks[s]))
    return pairs

'''

text = text[:idx_brain_start] + new_brain_funcs + text[idx_get_pairs:]

# 6. Update UnifiedDataset __getitem__ to handle BraTS dict format
old_getitem_marker = '    def __getitem__(self, idx):\n        img_path, msk_path = self.pairs[idx]'
assert old_getitem_marker in text, "old_getitem_marker not found"

new_getitem_code = '''    def __getitem__(self, idx):
        item = self.pairs[idx]
        if isinstance(item, dict):
            # BraTS MRI multi-modal item
            itype = item.get("type")
            if itype == "h5":
                with h5py.File(item["path"], "r") as hf:
                    image = hf["image"][:]   # [H, W, 3] or [H, W, 4]
                    mask  = hf["mask"][:]    # [H, W] or [H, W, 3]
                if image.ndim == 3 and image.shape[-1] >= 3:
                    image = image[..., :3]
                elif image.ndim == 2:
                    image = np.repeat(image[..., None], 3, axis=-1)
                if mask.ndim == 3:
                    mask = mask.argmax(axis=-1)
                label = mask.astype(np.int64)
                aug = self.tfm(image=image.astype(np.float32), mask=label)
                return aug["image"], aug["mask"].long(), item["path"]
            elif itype == "nii":
                fl_vol = nib.load(item["flair"]).get_fdata().astype(np.float32)
                t1_vol = nib.load(item["t1ce"]).get_fdata().astype(np.float32)
                t2_vol = nib.load(item["t2"]).get_fdata().astype(np.float32)
                sg_vol = nib.load(item["seg"]).get_fdata().astype(np.int64)
                D = fl_vol.shape[2]
                sl_idx = D // 2
                fl_sl = _normalize_mri_slice(fl_vol[:, :, sl_idx])
                t1_sl = _normalize_mri_slice(t1_vol[:, :, sl_idx])
                t2_sl = _normalize_mri_slice(t2_vol[:, :, sl_idx])
                image = np.stack([fl_sl, t1_sl, t2_sl], axis=-1)
                label = sg_vol[:, :, sl_idx]
                label[label == 4] = 3
                aug = self.tfm(image=image, mask=label)
                return aug["image"], aug["mask"].long(), item["flair"]

        img_path, msk_path = self.pairs[idx]'''

text = text.replace(old_getitem_marker, new_getitem_code, 1)

# 7. Update compute_metrics to calculate clinical BraTS metrics
old_comp_marker = '    if MODALITY == "fundus":\n        return {"disc_dice": dice(pred>=1, gt>=1), "cup_dice": dice(pred==2, gt==2)}\n    else:\n        return {"dice": dice(pred==1, gt==1), "jaccard": jac(pred==1, gt==1)}'
assert old_comp_marker in text, "old_comp_marker not found"

new_comp_code = '''    if MODALITY == "fundus":
        return {"disc_dice": dice(pred>=1, gt>=1), "cup_dice": dice(pred==2, gt==2)}
    elif MODALITY == "brain":
        wt_p, wt_g = (pred > 0), (gt > 0)
        tc_p, tc_g = np.isin(pred, [1, 3]), np.isin(gt, [1, 3])
        et_p, et_g = (pred == 3), (gt == 3)
        d_wt = dice(wt_p, wt_g)
        d_tc = dice(tc_p, tc_g)
        d_et = dice(et_p, et_g)
        d_mean = (d_wt + d_tc + d_et) / 3.0
        return {"wt_dice": d_wt, "tc_dice": d_tc, "et_dice": d_et, "mean_dice": d_mean}
    else:
        return {"dice": dice(pred==1, gt==1), "jaccard": jac(pred==1, gt==1)}'''

text = text.replace(old_comp_marker, new_comp_code, 1)

# 8. Update print_table to display BraTS clinical table format
old_print_tbl = '''    if MODALITY == "fundus":
        print(f"  {'Method':<22} │ Disc Dice         │ Cup Dice")
        print(f"  {'─'*22}─┼───────────────────┼────────────────")
        best_disc = max(vals.get("disc_dice", (0, 0))[0] for vals in agg.values())
        best_cup  = max(vals.get("cup_dice", (0, 0))[0] for vals in agg.values())
        for name, vals in agg.items():
            dd = vals.get("disc_dice", (0,0))
            cd = vals.get("cup_dice",  (0,0))
            d_star = " ★" if dd[0] == best_disc else "  "
            c_star = " ★" if cd[0] == best_cup else "  "
            print(f"  {name:<22} │ {dd[0]:.4f}±{dd[1]:.3f}{d_star} │ {cd[0]:.4f}±{cd[1]:.3f}{c_star}")
    else:
        print(f"  {'Method':<22} │ Dice              │ Jaccard")
        print(f"  {'─'*22}─┼───────────────────┼────────────────")
        best_d = max(vals.get("dice", (0, 0))[0] for vals in agg.values())
        best_j = max(vals.get("jaccard", (0, 0))[0] for vals in agg.values())
        for name, vals in agg.items():
            d = vals.get("dice",    (0,0))
            j = vals.get("jaccard", (0,0))
            d_star = " ★" if d[0] == best_d else "  "
            j_star = " ★" if j[0] == best_j else "  "
            print(f"  {name:<22} │ {d[0]:.4f}±{d[1]:.3f}{d_star} │ {j[0]:.4f}±{j[1]:.3f}{j_star}")'''

new_print_tbl = '''    if MODALITY == "fundus":
        print(f"  {'Method':<22} │ Disc Dice         │ Cup Dice")
        print(f"  {'─'*22}─┼───────────────────┼────────────────")
        best_disc = max(vals.get("disc_dice", (0, 0))[0] for vals in agg.values())
        best_cup  = max(vals.get("cup_dice", (0, 0))[0] for vals in agg.values())
        for name, vals in agg.items():
            dd = vals.get("disc_dice", (0,0))
            cd = vals.get("cup_dice",  (0,0))
            d_star = " ★" if dd[0] == best_disc else "  "
            c_star = " ★" if cd[0] == best_cup else "  "
            print(f"  {name:<22} │ {dd[0]:.4f}±{dd[1]:.3f}{d_star} │ {cd[0]:.4f}±{cd[1]:.3f}{c_star}")
    elif MODALITY == "brain":
        print(f"  {'Method':<22} │ Whole Tumor (WT)  │ Tumor Core (TC)   │ Enhancing (ET)    │ Mean Dice")
        print(f"  {'─'*22}─┼───────────────────┼───────────────────┼───────────────────┼──────────")
        best_mean = max(vals.get("mean_dice", (0, 0))[0] for vals in agg.values())
        for name, vals in agg.items():
            wt = vals.get("wt_dice", (0,0))[0] * 100
            tc = vals.get("tc_dice", (0,0))[0] * 100
            et = vals.get("et_dice", (0,0))[0] * 100
            md = vals.get("mean_dice", (0,0))[0] * 100
            star = " [BEST OVERALL]" if vals.get("mean_dice", (0,0))[0] == best_mean else ""
            print(f"  {name:<22} │ {wt:6.2f}%{' '*9} │ {tc:6.2f}%{' '*9} │ {et:6.2f}%{' '*9} │ {md:6.2f}%{star}")
    else:
        print(f"  {'Method':<22} │ Dice              │ Jaccard")
        print(f"  {'─'*22}─┼───────────────────┼────────────────")
        best_d = max(vals.get("dice", (0, 0))[0] for vals in agg.values())
        best_j = max(vals.get("jaccard", (0, 0))[0] for vals in agg.values())
        for name, vals in agg.items():
            d = vals.get("dice",    (0,0))
            j = vals.get("jaccard", (0,0))
            d_star = " ★" if d[0] == best_d else "  "
            j_star = " ★" if j[0] == best_j else "  "
            print(f"  {name:<22} │ {d[0]:.4f}±{d[1]:.3f}{d_star} │ {j[0]:.4f}±{j[1]:.3f}{j_star}")'''

assert old_print_tbl in text, "old_print_tbl not found"
text = text.replace(old_print_tbl, new_print_tbl, 1)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("SUCCESS: kaggle_unified_benchmark.py upgraded with BraTS multi-modal support!")
