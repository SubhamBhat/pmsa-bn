# PMSA-BN Benchmark Domain & Source-Target Pairs Registry

This registry defines the official benchmark domains, source datasets, target evaluation cohorts, and experimental specifications matching **Table 1** of the PMSA-BN manuscript.

Selecting `DOMAIN` at the top of [`kaggle_unified_benchmark.py`](file:///c:/Users/Subham%20Bhat/Downloads/tds/pmsa_bn_poc/kaggle_unified_benchmark.py) automatically activates all pairs in that domain.

---

## 1. Domain Registry Overview

| Domain Key | Clinical Modality | Task & Target Structure | Resolution | Classes | Primary Evaluation Metric | Total Pairs |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: |
| **`"fundus"`** | Retinal Fundus Photography | Optic Disc & Cup Segmentation | $384 \times 384$ | 3 (Bg, Disc, Cup) | Disc Dice / Cup Dice | **5 pairs** |
| **`"polyp"`** | Gastrointestinal Endoscopy | Colorectal Polyp Segmentation | $352 \times 352$ | 2 (Bg, Polyp) | Polyp Dice & Jaccard | **4 pairs** |
| **`"cxr"`** | Chest Radiography | Bilateral Lung Segmentation | $256 \times 256$ | 2 (Bg, Lungs) | Lung Dice & Jaccard | **4 pairs** |
| **`"derm"`** | Cutaneous Dermoscopy | Melanocytic Lesion Segmentation | $320 \times 320$ | 2 (Bg, Lesion) | Lesion Dice & Jaccard | **4 pairs** |
| **`"all"`** | All Modalities | Multi-Domain Benchmark Suite | Modality-specific | Modality-specific | Modality-specific | **17 pairs** |

---

## 2. Detailed Per-Domain Source-Target Specifications

### 2.1 Retinal Fundus Photography (`DOMAIN = "fundus"`)
* **Clinical Task**: Glaucoma screening via Cup-to-Disc Ratio (CDR) measurement.
* **Input Resolution**: $384 \times 384 \times 3$
* **Output Classes**: $3$ (`0`: Background, `1`: Optic Disc Rim, `2`: Optic Cup)
* **Alpha Schedule**: Layer-shift gated tanh schedule (`"sigmoid_tanh"`) with between-class variance.
* **Source Dataset**:
  * **REFUGE**: Retinal Fundus Glaucoma Challenge ($400$ training images; Canon CR-2 AF / Zeiss Visucam)

| Target Domain | Target Cohort Details | Sample Count ($n$) | Domain Shift Characteristics |
| :--- | :--- | :---: | :--- |
| **`ORIGA`** | Singapore Malay Eye Study (Singapore) | 325 (or 650 full) | Geographic cohort, illumination, device geometry |
| **`Drishti-GS`** | Aravind Eye Hospital (India) | 50 (test split) | Varied optic disc tilt, high cup depth variations |
| **`RIM-ONE`** | Hospital Universitario de Canarias (Spain) | 159 (RIM-ONE r3) | Specialized High-Res ONH cropped field |
| **`G1020`** | Large Multi-Center Glaucoma Cohort | 200 (or 1020 full) | Diverse ethnicity, variable cup boundaries |
| **`PAPILA`** | Clinical Fundus Cohort (Spain) | 99 (or 488 full) | Wide range of glaucoma severities |

---

### 2.2 Gastrointestinal Endoscopy (`DOMAIN = "polyp"`)
* **Clinical Task**: Real-time colonoscopic video surveillance for premalignant polyps.
* **Input Resolution**: $352 \times 352 \times 3$
* **Output Classes**: $2$ (`0`: Background mucosa, `1`: Polyp)
* **Alpha Schedule**: Shift-adaptive exponential (`"exp"`, $\tau=1.0, \alpha_{\max}=0.45$) with within-class variance.

#### Sequence 1: CVC-ClinicDB Source
* **Source**: **ClinicDB** ($612$ frames, Pentax colonoscope, Hospital Clínic de Barcelona, Spain)
* **Targets**:
  1. **`Kvasir`**: Kvasir-SEG ($1{,}000$ or $200$ test frames, Olympus colonoscopes, Vestre Viken, Norway)
  2. **`ETIS`**: ETIS-Larib ($196$ frames, early colon cancer detection, Lariboisière Hospital, France)

#### Sequence 2: Kvasir-SEG Source
* **Source**: **Kvasir** ($800$ training frames, Olympus colonoscopes, Norway)
* **Targets**:
  1. **`ClinicDB`**: CVC-ClinicDB ($612$ test frames, Pentax colonoscopes, Spain)
  2. **`ETIS`**: ETIS-Larib ($196$ frames, France)

---

### 2.3 Chest Radiography (`DOMAIN = "cxr"`)
* **Clinical Task**: Emergency & triage bilateral lung field segmentation.
* **Input Resolution**: $256 \times 256 \times 3$
* **Output Classes**: $2$ (`0`: Background, `1`: Bilateral Lung Fields)
* **Alpha Schedule**: Shift-adaptive hyperbolic tangent (`"tanh"`, $\tau=0.5, \alpha_{\max}=0.70$).

#### Sequence 1: Montgomery County Source
* **Source**: **Montgomery** ($138$ CXRs, Eureka / Del Medical flat-panel, Maryland, USA)
* **Targets**:
  1. **`Shenzhen`**: Shenzhen No. 3 Hospital ($336$ test CXRs, Philips Digital, China)
  2. **`JSRT`**: Japanese Society of Radiological Technology ($247$ CXRs, Japan)
  3. **`COVID-QU-Ex`**: Severe acute pulmonary pathology ($300$ or $400$ CXRs, Qatar)

#### Sequence 2: Shenzhen Hospital Source (Reverse Transfer)
* **Source**: **Shenzhen** ($566$ training CXRs, Philips Digital, China)
* **Targets**:
  1. **`Montgomery`**: Montgomery County ($138$ clinical CXRs, USA)

---

### 2.4 Cutaneous Dermoscopy (`DOMAIN = "derm"`)
* **Clinical Task**: Pigmented lesion segmentation for melanoma diagnosis.
* **Input Resolution**: $320 \times 320 \times 3$
* **Output Classes**: $2$ (`0`: Background skin, `1`: Cutaneous Lesion)
* **Alpha Schedule**: Shift-adaptive exponential (`"exp"`, $\tau=1.0, \alpha_{\max}=0.20$).

#### Sequence 1: ISIC 2018 Source (Forward Transfer)
* **Source**: **ISIC2018** Task 1 ($2{,}594$ training dermoscopy images)
* **Targets**:
  1. **`HAM10000`**: Multi-center dermatoscopy ($500$ evaluated cases, Austria & Australia)
  2. **`Derm7pt`**: 7-point criteria clinical photography ($500$ evaluated cases)
  3. **`PH2`**: Pedro Hispano Hospital ($200$ cases, Tuebinger Mole Analyzer, Portugal)

#### Sequence 2: PH² Source (Reverse Transfer)
* **Source**: **PH2** ($200$ training cases, Portugal)
* **Targets**:
  1. **`ISIC2018`**: International Skin Imaging Collaboration benchmark ($600$ test cases)

---

## 3. How to Use in `kaggle_unified_benchmark.py`

In [`kaggle_unified_benchmark.py`](file:///c:/Users/Subham%20Bhat/Downloads/tds/pmsa_bn_poc/kaggle_unified_benchmark.py), simply set:

```python
USER_CFG = dict(
    # Set to "fundus", "polyp", "cxr", "derm", or "all"
    DOMAIN = "fundus",
    BACKBONE = "mobilenet_v2",
    AUTO_TRAIN_IF_MISSING = True,
    ...
)
```
The script will resolve all source-target sequences automatically and execute the benchmark.
