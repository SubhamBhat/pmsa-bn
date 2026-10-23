# PMSA-BN Benchmark Results — Retinal Fundus Modality (REFUGE Source)

This document records the empirical evaluation results for **Optic Disc and Optic Cup Segmentation** using **REFUGE** as the source training domain, evaluated across five multi-center target domains: **ORIGA**, **Drishti-GS**, **RIM-ONE**, **G1020**, and **PAPILA**.

---

## 1. Summary of Performance Across All Targets

| Target Domain | Evaluation Cohort ($n$) | Primary Metrics | PMSA-BN (Ours) | Best Competing Baseline | Leading Baseline Method | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ORIGA** | 325 | Disc / Cup | **0.8920 / 0.8248** ★ | $0.8875 / 0.8190$ | $\alpha$-BN$_{0.5}$ / TENT | **Clean Sweep** |
| **Drishti-GS** | 50 | Disc / Cup | **0.9651 / 0.8104** ★ | $0.9608 / 0.7862$ | $\alpha$-BN$_{0.5}$ / $\alpha$-BN$_{0.1}$ | **Clean Sweep** |
| **RIM-ONE** | 159 | Disc / Cup | **0.6077 / 0.4494** ★ | $0.5823 / 0.4477$ | $\alpha$-BN$_{0.7}$ / $\alpha$-BN$_{0.7}$ | **Clean Sweep** |
| **G1020** | 200 | Disc / Cup | **0.8194 / 0.6173** ★ | $0.8140 / 0.6158$ | TENT / $\alpha$-BN$_{0.1}$ | **Clean Sweep** |
| **PAPILA** | 99 | Disc / Cup | **0.8869 / 0.6173** ★ | $0.8832 / 0.6161$ | TENT / TENT | **Clean Sweep** |

---

## 2. Detailed Per-Target Benchmark Tables

### Target 1: ORIGA ($n = 325$)
* **Clinical Task**: Glaucoma screening on Singapore Malay Eye Study (severe demographic shift)
* **Metrics**: Optic Disc Dice and Optic Cup Dice (biomarker for vertical cup-to-disc ratio CDR)

| Method | Disc Dice ($\pm \sigma$) | Cup Dice ($\pm \sigma$) | Benchmark Status |
| :--- | :---: | :---: | :---: |
| Source Only | $0.8763 \pm 0.094$ | $0.7938 \pm 0.147$ | Unadapted Baseline |
| TENT | $0.8815 \pm 0.055$ | $0.8190 \pm 0.113$ | Adaptive Baseline |
| AdaBN | $0.8748 \pm 0.062$ | $0.8059 \pm 0.125$ | Adaptive Baseline |
| $\alpha$-BN ($\alpha=0.1$) | $0.8810 \pm 0.094$ | $0.7976 \pm 0.145$ | Baseline |
| $\alpha$-BN ($\alpha=0.3$) | $0.8854 \pm 0.087$ | $0.8028 \pm 0.140$ | Baseline |
| $\alpha$-BN ($\alpha=0.5$) | $0.8875 \pm 0.069$ | $0.8096 \pm 0.128$ | Highest Disc Baseline |
| $\alpha$-BN ($\alpha=0.7$) | $0.8841 \pm 0.069$ | $0.8108 \pm 0.126$ | Baseline |
| $\alpha$-BN ($\alpha=0.9$) | $0.8789 \pm 0.062$ | $0.8088 \pm 0.126$ | Baseline |
| **PMSA-BN (Ours)** | **$0.8920 \pm 0.061$** ★ | **$0.8248 \pm 0.118$** ★ | **Highest Score & Clean Sweep** |

#### Statistical Significance (ORIGA)
* **PMSA-BN vs AdaBN**: $t = -2.5263, p(t) = 1.20 \times 10^{-02}, p(W) = 2.99 \times 10^{-08}$
* **PMSA-BN vs TENT**: $t = -10.2743, p(t) = 1.27 \times 10^{-21}$
* **PMSA-BN vs $\alpha$-BN ($\alpha=0.5$)**: $t = -9.0331, p(t) = 1.54 \times 10^{-17}, p(W) = 1.02 \times 10^{-35}$

---

### Target 2: Drishti-GS ($n = 50$)
* **Clinical Task**: Consensus glaucoma validation on Indian population cohort (Aravind Eye Hospital)
* **Metrics**: Optic Disc Dice and Optic Cup Dice

| Method | Disc Dice ($\pm \sigma$) | Cup Dice ($\pm \sigma$) | Benchmark Status |
| :--- | :---: | :---: | :---: |
| Source Only | $0.9563 \pm 0.051$ | $0.7861 \pm 0.097$ | Unadapted Baseline |
| TENT | $0.9564 \pm 0.043$ | $0.7492 \pm 0.084$ | Negative Transfer on Cup |
| AdaBN | $0.9555 \pm 0.041$ | $0.7553 \pm 0.079$ | Negative Transfer on Cup |
| $\alpha$-BN ($\alpha=0.1$) | $0.9577 \pm 0.047$ | $0.7862 \pm 0.094$ | Highest Cup Baseline |
| $\alpha$-BN ($\alpha=0.3$) | $0.9599 \pm 0.041$ | $0.7859 \pm 0.090$ | Baseline |
| $\alpha$-BN ($\alpha=0.5$) | $0.9608 \pm 0.036$ | $0.7830 \pm 0.086$ | Highest Disc Baseline |
| $\alpha$-BN ($\alpha=0.7$) | $0.9602 \pm 0.035$ | $0.7773 \pm 0.083$ | Baseline |
| $\alpha$-BN ($\alpha=0.9$) | $0.9577 \pm 0.038$ | $0.7658 \pm 0.081$ | Baseline |
| **PMSA-BN (Ours)** | **$0.9651 \pm 0.038$** ★ | **$0.8104 \pm 0.080$** ★ | **Highest Score & Clean Sweep** |

#### Statistical Significance (Drishti-GS)
* **PMSA-BN vs AdaBN**: $t = -0.2190, p(t) = 8.28 \times 10^{-01}, p(W) = 1.52 \times 10^{-01}$
* **PMSA-BN vs TENT**: $t = -0.4072, p(t) = 6.86 \times 10^{-01}$
* **PMSA-BN vs $\alpha$-BN ($\alpha=0.5$)**: $t = -1.3038, p(t) = 1.98 \times 10^{-01}, p(W) = 1.22 \times 10^{-02}$

---

### Target 3: RIM-ONE ($n = 159$)
* **Clinical Task**: Open-angle glaucoma evaluation on stereoscopic optic nerve head photographs
* **Metrics**: Optic Disc Dice and Optic Cup Dice

| Method | Disc Dice ($\pm \sigma$) | Cup Dice ($\pm \sigma$) | Benchmark Status |
| :--- | :---: | :---: | :---: |
| Source Only | $0.4513 \pm 0.371$ | $0.3422 \pm 0.345$ | Baseline |
| TENT | $0.5462 \pm 0.278$ | $0.4094 \pm 0.312$ | Adaptive Baseline |
| AdaBN | $0.5389 \pm 0.271$ | $0.4100 \pm 0.307$ | Adaptive Baseline |
| $\alpha$-BN ($\alpha=0.1$) | $0.4688 \pm 0.367$ | $0.3607 \pm 0.348$ | Baseline |
| $\alpha$-BN ($\alpha=0.3$) | $0.5090 \pm 0.352$ | $0.3930 \pm 0.348$ | Baseline |
| $\alpha$-BN ($\alpha=0.5$) | $0.5514 \pm 0.320$ | $0.4273 \pm 0.336$ | Baseline |
| $\alpha$-BN ($\alpha=0.7$) | $0.5823 \pm 0.291$ | $0.4477 \pm 0.324$ | Highest Baseline |
| $\alpha$-BN ($\alpha=0.9$) | $0.5779 \pm 0.270$ | $0.4373 \pm 0.312$ | Baseline |
| **PMSA-BN (Ours)** | **$0.6077 \pm 0.272$** ★ | **$0.4494 \pm 0.311$** ★ | **Highest Score & Clean Sweep** |

#### Statistical Significance (RIM-ONE)
* **PMSA-BN vs AdaBN**: $t = 11.6017, p(t) = 6.69 \times 10^{-23}, p(W) = 4.27 \times 10^{-21}$
* **PMSA-BN vs TENT**: $t = 7.8162, p(t) = 7.19 \times 10^{-13}$
* **PMSA-BN vs $\alpha$-BN ($\alpha=0.7$)**: $t = 4.4259, p(t) = 1.78 \times 10^{-05}, p(W) = 5.05 \times 10^{-05}$

---

### Target 4: G1020 ($n = 200$)
* **Clinical Task**: Large-scale uncurated clinical fundus cohort with diverse optic disc pathologies
* **Metrics**: Optic Disc Dice and Optic Cup Dice

| Method | Disc Dice ($\pm \sigma$) | Cup Dice ($\pm \sigma$) | Benchmark Status |
| :--- | :---: | :---: | :---: |
| Source Only | $0.7309 \pm 0.302$ | $0.6157 \pm 0.336$ | High Cup Baseline |
| TENT | $0.8140 \pm 0.202$ | $0.5925 \pm 0.332$ | Adaptive Baseline |
| AdaBN | $0.7967 \pm 0.205$ | $0.5826 \pm 0.328$ | Degraded Cup |
| $\alpha$-BN ($\alpha=0.1$) | $0.7321 \pm 0.311$ | $0.6158 \pm 0.340$ | Highest Cup Baseline |
| $\alpha$-BN ($\alpha=0.3$) | $0.7471 \pm 0.303$ | $0.6095 \pm 0.343$ | Baseline |
| $\alpha$-BN ($\alpha=0.5$) | $0.7667 \pm 0.284$ | $0.6068 \pm 0.339$ | Baseline |
| $\alpha$-BN ($\alpha=0.7$) | $0.7847 \pm 0.252$ | $0.5991 \pm 0.333$ | Baseline |
| $\alpha$-BN ($\alpha=0.9$) | $0.8015 \pm 0.212$ | $0.5900 \pm 0.329$ | Baseline |
| **PMSA-BN (Ours)** | **$0.8194 \pm 0.196$** ★ | **$0.6173 \pm 0.328$** ★ | **Highest Score & Clean Sweep** |

#### Statistical Significance (G1020)
* **PMSA-BN vs AdaBN**: $t = 0.6724, p(t) = 5.02 \times 10^{-01}, p(W) = 6.78 \times 10^{-01}$
* **PMSA-BN vs TENT**: $t = -3.3840, p(t) = 8.60 \times 10^{-04}$
* **PMSA-BN vs $\alpha$-BN ($\alpha=0.9$)**: $t = -0.4731, p(t) = 6.37 \times 10^{-01}, p(W) = 1.01 \times 10^{-02}$

---

### Target 5: PAPILA ($n = 99$)
* **Clinical Task**: Clinical Spanish fundus dataset with expert contour annotations
* **Metrics**: Optic Disc Dice and Optic Cup Dice

| Method | Disc Dice ($\pm \sigma$) | Cup Dice ($\pm \sigma$) | Benchmark Status |
| :--- | :---: | :---: | :---: |
| Source Only | $0.5459 \pm 0.379$ | $0.4569 \pm 0.352$ | Unadapted Baseline |
| TENT | $0.8832 \pm 0.121$ | $0.6161 \pm 0.244$ | Adaptive Baseline |
| AdaBN | $0.8806 \pm 0.128$ | $0.6095 \pm 0.246$ | Adaptive Baseline |
| $\alpha$-BN ($\alpha=0.1$) | $0.5876 \pm 0.369$ | $0.4845 \pm 0.341$ | Baseline |
| $\alpha$-BN ($\alpha=0.3$) | $0.6646 \pm 0.348$ | $0.5278 \pm 0.324$ | Baseline |
| $\alpha$-BN ($\alpha=0.5$) | $0.7289 \pm 0.323$ | $0.5629 \pm 0.307$ | Baseline |
| $\alpha$-BN ($\alpha=0.7$) | $0.8034 \pm 0.241$ | $0.5924 \pm 0.285$ | Baseline |
| $\alpha$-BN ($\alpha=0.9$) | $0.8654 \pm 0.158$ | $0.6085 \pm 0.254$ | Baseline |
| **PMSA-BN (Ours)** | **$0.8869 \pm 0.118$** ★ | **$0.6173 \pm 0.246$** ★ | **Highest Score & Clean Sweep** |

#### Statistical Significance (PAPILA)
* **PMSA-BN vs AdaBN**: $t = 2.5094, p(t) = 1.37 \times 10^{-02}, p(W) = 1.43 \times 10^{-01}$
* **PMSA-BN vs TENT**: $t = 1.6872, p(t) = 9.47 \times 10^{-02}$
* **PMSA-BN vs $\alpha$-BN ($\alpha=0.9$)**: $t = 3.9219, p(t) = 1.63 \times 10^{-04}, p(W) = 2.54 \times 10^{-03}$
