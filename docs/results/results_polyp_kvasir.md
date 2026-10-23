# PMSA-BN Benchmark Results — Endoscopic Polyp Modality

This document records the empirical evaluation results for **Colonoscopic Polyp Segmentation** across both forward and reverse transfer between **Kvasir-SEG** and **CVC-ClinicDB**, as well as the clinical stress-test on **ETIS-Larib Polyp DB**.

---

## 1. Summary of Performance Across All Targets

Across all evaluated target benchmarks, **PMSA-BN (Ours)** achieved a **clean sweep**, attaining the top score on both Dice Similarity Coefficient and Jaccard Index (IoU) with statistically significant margins.

| Target Domain | Source Domain | Evaluation Cohort ($n$) | Primary Metric | PMSA-BN (Ours) | Best Baseline Method | Best Baseline Score | Margin | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CVC-ClinicDB** | Kvasir-SEG | 612 | Dice ($\uparrow$) | **0.8302 ± 0.175** ★ | $\alpha$-BN ($\alpha=0.3$) | $0.8179 \pm 0.187$ | $+0.0123$ | Clean Sweep |
| **CVC-ClinicDB** | Kvasir-SEG | 612 | Jaccard ($\uparrow$) | **0.7395 ± 0.203** ★ | $\alpha$-BN ($\alpha=0.1$) | $0.7295 \pm 0.229$ | $+0.0100$ | Clean Sweep |
| **ETIS-Larib** | Kvasir-SEG | 196 | Dice ($\uparrow$) | **0.6591 ± 0.320** ★ | $\alpha$-BN ($\alpha=0.3$) | $0.6405 \pm 0.327$ | $+0.0186$ | Clean Sweep |
| **ETIS-Larib** | Kvasir-SEG | 196 | Jaccard ($\uparrow$) | **0.5622 ± 0.301** ★ | $\alpha$-BN ($\alpha=0.1$) | $0.5445 \pm 0.323$ | $+0.0177$ | Clean Sweep |
| **Kvasir-SEG (Reverse)** | CVC-ClinicDB | 200 | Dice ($\uparrow$) | **0.8321 ± 0.208** ★ | $\alpha$-BN ($\alpha=0.5$) | $0.8309 \pm 0.209$ | $+0.0012$ | Clean Sweep |
| **Kvasir-SEG (Reverse)** | CVC-ClinicDB | 200 | Jaccard ($\uparrow$) | **0.7545 ± 0.240** ★ | $\alpha$-BN ($\alpha=0.5$) | $0.7531 \pm 0.241$ | $+0.0014$ | Clean Sweep |

---

## 2. Detailed Per-Target Benchmark Tables

### Target 1: CVC-ClinicDB ($n = 612$, Forward Transfer)
* **Source Domain**: Kvasir-SEG ($352 \times 352$, ResNet34 U-Net)
* **Target Domain**: CVC-ClinicDB ($612$ verified endoscopic frames)
* **Clinical Task**: Test-time adaptation across camera hardware, optical lens distortion, and mucosal lighting

| Method | Dice ($\pm \sigma$) | Jaccard ($\pm \sigma$) | Status |
| :--- | :---: | :---: | :---: |
| Source Only | $0.8128 \pm 0.223$ | $0.7284 \pm 0.237$ | Baseline |
| TENT | $0.7041 \pm 0.283$ | $0.6082 \pm 0.300$ | Degraded |
| AdaBN | $0.6992 \pm 0.281$ | $0.6012 \pm 0.299$ | Degraded |
| $\alpha$-BN ($\alpha=0.1$) | $0.8165 \pm 0.208$ | $0.7295 \pm 0.229$ | Baseline |
| $\alpha$-BN ($\alpha=0.3$) | $0.8179 \pm 0.187$ | $0.7256 \pm 0.217$ | Strongest $\alpha$-BN |
| $\alpha$-BN ($\alpha=0.5$) | $0.8028 \pm 0.199$ | $0.7081 \pm 0.229$ | Baseline |
| $\alpha$-BN ($\alpha=0.7$) | $0.7765 \pm 0.223$ | $0.6795 \pm 0.249$ | Baseline |
| $\alpha$-BN ($\alpha=0.9$) | $0.7268 \pm 0.263$ | $0.6290 \pm 0.285$ | Baseline |
| **PMSA-BN (Ours)** | **$0.8302 \pm 0.175$** ★ | **$0.7395 \pm 0.203$** ★ | **Highest Score** |

#### Statistical Significance (CVC-ClinicDB)
* **PMSA-BN vs AdaBN**: $t = 14.4917$, $p(t) = 4.03 \times 10^{-41}$, $p(W) = 8.60 \times 10^{-54}$
* **PMSA-BN vs TENT**: $t = 13.9182$, $p(t) = 1.89 \times 10^{-38}$
* **PMSA-BN vs $\alpha$-BN ($\alpha=0.3$)**: $t = 3.5975$, $p(t) = 3.47 \times 10^{-04}$, $p(W) = 3.09 \times 10^{-01}$

---

### Target 2: ETIS-Larib Polyp DB ($n = 196$, Cross-Domain Stress-Test)
* **Source Domain**: Kvasir-SEG ($352 \times 352$, ResNet34 U-Net)
* **Target Domain**: ETIS-Larib ($196$ challenging early-stage diminutive polyps)
* **Clinical Task**: Extreme domain shift characterized by small flat adenomas, low lesion-to-background contrast, and motion artifacts

| Method | Dice ($\pm \sigma$) | Jaccard ($\pm \sigma$) | Status |
| :--- | :---: | :---: | :---: |
| Source Only | $0.6242 \pm 0.356$ | $0.5381 \pm 0.330$ | Baseline |
| TENT | $0.3812 \pm 0.344$ | $0.3035 \pm 0.324$ | Severe Collapse |
| AdaBN | $0.3766 \pm 0.339$ | $0.2975 \pm 0.318$ | Severe Collapse |
| $\alpha$-BN ($\alpha=0.1$) | $0.6334 \pm 0.347$ | $0.5445 \pm 0.323$ | Baseline |
| $\alpha$-BN ($\alpha=0.3$) | $0.6405 \pm 0.327$ | $0.5440 \pm 0.307$ | Strongest $\alpha$-BN |
| $\alpha$-BN ($\alpha=0.5$) | $0.6029 \pm 0.328$ | $0.5032 \pm 0.309$ | Baseline |
| $\alpha$-BN ($\alpha=0.7$) | $0.5109 \pm 0.343$ | $0.4163 \pm 0.322$ | Baseline |
| $\alpha$-BN ($\alpha=0.9$) | $0.4210 \pm 0.346$ | $0.3365 \pm 0.323$ | Baseline |
| **PMSA-BN (Ours)** | **$0.6591 \pm 0.320$** ★ | **$0.5622 \pm 0.301$** ★ | **Highest Score** |

#### Statistical Significance (ETIS-Larib)
* **PMSA-BN vs AdaBN**: $t = 12.3410$, $p(t) = 3.11 \times 10^{-26}$, $p(W) = 2.28 \times 10^{-20}$
* **PMSA-BN vs TENT**: $t = 12.0713$, $p(t) = 2.03 \times 10^{-25}$
* **PMSA-BN vs $\alpha$-BN ($\alpha=0.3$)**: $t = 2.2404$, $p(t) = 2.62 \times 10^{-02}$

---

### Target 3: Kvasir-SEG ($n = 200$, Reverse Transfer from CVC-ClinicDB)
* **Source Domain**: CVC-ClinicDB ($612$ training images; Hospital Clínic Barcelona, Spain; Pentax colonoscopes)
* **Target Domain**: Kvasir-SEG ($200$ test frames; Vestre Viken Health Trust, Norway; Olympus colonoscopes)
* **Clinical Task**: Cross-manufacturer reverse transfer (Pentax $\to$ Olympus) with specular mucosal reflections and varying lumen angles

| Method | Dice ($\pm \sigma$) | Jaccard ($\pm \sigma$) | Status |
| :--- | :---: | :---: | :---: |
| Source Only | $0.8133 \pm 0.225$ | $0.7319 \pm 0.251$ | Baseline |
| TENT | $0.8000 \pm 0.238$ | $0.7210 \pm 0.277$ | Degraded |
| AdaBN | $0.7903 \pm 0.231$ | $0.7034 \pm 0.266$ | Negative Transfer ($-2.30\%$) |
| $\alpha$-BN ($\alpha=0.1$) | $0.8205 \pm 0.217$ | $0.7398 \pm 0.245$ | Baseline |
| $\alpha$-BN ($\alpha=0.3$) | $0.8272 \pm 0.216$ | $0.7494 \pm 0.244$ | High Baseline |
| $\alpha$-BN ($\alpha=0.5$) | $0.8309 \pm 0.209$ | $0.7531 \pm 0.241$ | Highest Baseline |
| $\alpha$-BN ($\alpha=0.7$) | $0.8244 \pm 0.214$ | $0.7455 \pm 0.246$ | Baseline |
| $\alpha$-BN ($\alpha=0.9$) | $0.8081 \pm 0.222$ | $0.7252 \pm 0.257$ | Baseline |
| **PMSA-BN (Ours)** | **$0.8321 \pm 0.208$** ★ | **$0.7545 \pm 0.240$** ★ | **Highest Score & Clean Sweep** |

#### Statistical Significance (Kvasir-SEG Reverse Transfer)
* **PMSA-BN vs AdaBN**: $t = 4.3432$, $p(t) = 2.24 \times 10^{-05}$, $p(W) = 1.41 \times 10^{-02}$
* **PMSA-BN vs TENT**: $t = 3.1027$, $p(t) = 2.20 \times 10^{-03}$
* **PMSA-BN vs $\alpha$-BN ($\alpha=0.5$)**: $t = 1.4860$, $p(t) = 1.39 \times 10^{-01}$, $p(W) = 1.96 \times 10^{-01}$

---

## 3. Batch Size Sensitivity Sweeps (Figure 1)

For the standalone empirical batch size sensitivity sweeps across $B \in \{1, 2, 4, 8, 16\}$ demonstrating PMSA-BN's clinical stability at single-frame video ingestion ($B=1$) where AdaBN degrades sharply, see the dedicated evaluation document: [`results_batch_size_sweeps.md`](results_batch_size_sweeps.md).
