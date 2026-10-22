# PMSA-BN Benchmark Results — CXR Modality (Montgomery Source)

This document records the empirical evaluation results for **Chest X-Ray (CXR) Bilateral Lung Segmentation** using **Montgomery** as the source training domain, evaluated across three target domains: **Shenzhen**, **JSRT**, and **COVID-QU-Ex**.

---

## 1. Summary of Performance

Across all three test domains, **PMSA-BN (Ours)** achieved the highest performance in both Dice Similarity Coefficient and Jaccard Index (IoU), with statistically significant margins verified via paired two-tailed $t$-tests and Wilcoxon signed-rank tests.

| Target Domain | Test Cases ($n$) | Top Method | Dice ($\uparrow$) | Jaccard ($\uparrow$) | Status |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Shenzhen** | 336 | **PMSA-BN (Ours)** | **0.9469 ± 0.037** ★ | **0.9013 ± 0.061** ★ | Clean Sweep |
| **JSRT** | 247 | **PMSA-BN (Ours)** | **0.9663 ± 0.010** ★ | **0.9351 ± 0.019** ★ | Clean Sweep |
| **COVID-QU-Ex** | 300 | **PMSA-BN (Ours)** | **0.9360 ± 0.043** ★ | **0.8826 ± 0.072** ★ | Clean Sweep |

---

## 2. Detailed Benchmark Tables

### Target 1: Shenzhen ($n = 336$)

* **Source Domain**: Montgomery (MCUCXR)
* **Target Domain**: Shenzhen (CHNCXR)

| Method | Dice ($\pm \sigma$) | Jaccard ($\pm \sigma$) | Margin vs PMSA-BN |
| :--- | :---: | :---: | :---: |
| Source Only | $0.9440 \pm 0.041$ | $0.8965 \pm 0.066$ | $+0.0029$ |
| TENT | $0.9391 \pm 0.040$ | $0.8877 \pm 0.065$ | $+0.0078$ |
| AdaBN | $0.9386 \pm 0.039$ | $0.8866 \pm 0.064$ | $+0.0083$ |
| $\alpha$-BN ($\alpha=0.1$) | $0.9442 \pm 0.040$ | $0.8968 \pm 0.065$ | $+0.0027$ |
| $\alpha$-BN ($\alpha=0.3$) | $0.9443 \pm 0.040$ | $0.8969 \pm 0.064$ | $+0.0026$ |
| $\alpha$-BN ($\alpha=0.5$) | $0.9440 \pm 0.039$ | $0.8963 \pm 0.063$ | $+0.0029$ |
| $\alpha$-BN ($\alpha=0.7$) | $0.9430 \pm 0.038$ | $0.8944 \pm 0.062$ | $+0.0039$ |
| $\alpha$-BN ($\alpha=0.9$) | $0.9408 \pm 0.038$ | $0.8904 \pm 0.062$ | $+0.0061$ |
| **PMSA-BN (Ours)** | **$0.9469 \pm 0.037$** ★ | **$0.9013 \pm 0.061$** ★ | **Highest Score** |

#### Statistical Significance (Shenzhen)
* **PMSA-BN vs AdaBN**: $t = 12.6861$, $p(t) = 2.20 \times 10^{-30}$, $p(W) = 4.36 \times 10^{-38}$
* **PMSA-BN vs TENT**: $t = 10.8264$, $p(t) = 1.27 \times 10^{-23}$
* **PMSA-BN vs $\alpha$-BN ($\alpha=0.3$)**: $t = 6.7974$, $p(t) = 4.89 \times 10^{-11}$, $p(W) = 7.08 \times 10^{-25}$

---

### Target 2: JSRT ($n = 247$)

* **Source Domain**: Montgomery (MCUCXR)
* **Target Domain**: JSRT (Japanese Society of Radiological Technology)

| Method | Dice ($\pm \sigma$) | Jaccard ($\pm \sigma$) | Margin vs PMSA-BN |
| :--- | :---: | :---: | :---: |
| Source Only | $0.9656 \pm 0.010$ | $0.9337 \pm 0.018$ | $+0.0007$ |
| TENT | $0.9573 \pm 0.016$ | $0.9186 \pm 0.029$ | $+0.0090$ |
| AdaBN | $0.9558 \pm 0.016$ | $0.9158 \pm 0.028$ | $+0.0105$ |
| $\alpha$-BN ($\alpha=0.1$) | $0.9656 \pm 0.010$ | $0.9337 \pm 0.019$ | $+0.0007$ |
| $\alpha$-BN ($\alpha=0.3$) | $0.9652 \pm 0.011$ | $0.9330 \pm 0.019$ | $+0.0011$ |
| $\alpha$-BN ($\alpha=0.5$) | $0.9644 \pm 0.011$ | $0.9315 \pm 0.020$ | $+0.0019$ |
| $\alpha$-BN ($\alpha=0.7$) | $0.9630 \pm 0.012$ | $0.9288 \pm 0.022$ | $+0.0033$ |
| $\alpha$-BN ($\alpha=0.9$) | $0.9598 \pm 0.013$ | $0.9230 \pm 0.024$ | $+0.0065$ |
| **PMSA-BN (Ours)** | **$0.9663 \pm 0.010$** ★ | **$0.9351 \pm 0.019$** ★ | **Highest Score** |

#### Statistical Significance (JSRT)
* **PMSA-BN vs AdaBN**: $t = 16.3746$, $p(t) = 2.95 \times 10^{-41}$, $p(W) = 8.64 \times 10^{-41}$
* **PMSA-BN vs TENT**: $t = 13.5295$, $p(t) = 1.50 \times 10^{-31}$
* **PMSA-BN vs $\alpha$-BN ($\alpha=0.1$)**: $t = 5.0797$, $p(t) = 7.48 \times 10^{-07}$, $p(W) = 4.59 \times 10^{-11}$

---

### Target 3: COVID-QU-Ex ($n = 300$)

* **Source Domain**: Montgomery (MCUCXR)
* **Target Domain**: COVID-QU-Ex (Severe Pathological & Radiographic Shift)

| Method | Dice ($\pm \sigma$) | Jaccard ($\pm \sigma$) | Margin vs PMSA-BN |
| :--- | :---: | :---: | :---: |
| Source Only | $0.9262 \pm 0.054$ | $0.8669 \pm 0.088$ | $+0.0098$ |
| TENT | $0.9249 \pm 0.050$ | $0.8640 \pm 0.081$ | $+0.0111$ |
| AdaBN | $0.9230 \pm 0.048$ | $0.8606 \pm 0.079$ | $+0.0130$ |
| $\alpha$-BN ($\alpha=0.1$) | $0.9266 \pm 0.054$ | $0.8675 \pm 0.087$ | $+0.0094$ |
| $\alpha$-BN ($\alpha=0.3$) | $0.9268 \pm 0.053$ | $0.8678 \pm 0.085$ | $+0.0092$ |
| $\alpha$-BN ($\alpha=0.5$) | $0.9269 \pm 0.051$ | $0.8677 \pm 0.083$ | $+0.0091$ |
| $\alpha$-BN ($\alpha=0.7$) | $0.9265 \pm 0.050$ | $0.8670 \pm 0.082$ | $+0.0095$ |
| $\alpha$-BN ($\alpha=0.9$) | $0.9252 \pm 0.049$ | $0.8644 \pm 0.080$ | $+0.0108$ |
| **PMSA-BN (Ours)** | **$0.9360 \pm 0.043$** ★ | **$0.8826 \pm 0.072$** ★ | **Highest Score** |

#### Statistical Significance (COVID-QU-Ex)
* **PMSA-BN vs AdaBN**: $t = 10.4182$, $p(t) = 6.92 \times 10^{-22}$, $p(W) = 4.44 \times 10^{-32}$
* **PMSA-BN vs TENT**: $t = 8.1243$, $p(t) = 1.20 \times 10^{-14}$
* **PMSA-BN vs $\alpha$-BN ($\alpha=0.5$)**: $t = 10.5457$, $p(t) = 2.58 \times 10^{-22}$, $p(W) = 6.84 \times 10^{-49}$

---

## 3. Batch Size Sensitivity Sweep (Figure 1 on Shenzhen)

Evaluated across test batch sizes $B \in \{1, 2, 4, 8, 16\}$:

| Batch Size ($B$) | Source Only | AdaBN | $\alpha$-BN ($\alpha=0.5$) | PMSA-BN (Ours) | Top Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$B = 1$** | $0.9437$ | $0.9372$ | $0.9434$ | **$0.9449$** ★ | **PMSA-BN** |
| **$B = 2$** | $0.9437$ | $0.9398$ | $0.9438$ | **$0.9447$** ★ | **PMSA-BN** |
| **$B = 4$** | $0.9437$ | $0.9404$ | $0.9441$ | **$0.9447$** ★ | **PMSA-BN** |
| **$B = 8$** | $0.9437$ | $0.9407$ | $0.9441$ | **$0.9446$** ★ | **PMSA-BN** |
| **$B = 16$** | $0.9437$ | $0.9410$ | $0.9442$ | **$0.9446$** ★ | **PMSA-BN** |

### Key Observations:
1. **Batch Size Invariance**: While AdaBN degrades significantly at $B = 1$ ($0.9372$) due to high batch statistic estimation variance, PMSA-BN remains flat and achieves its highest performance at $B = 1$ ($0.9449$), confirming true test-time suitability.
2. **Consistent Superiority**: PMSA-BN outperforms all baseline configurations across every single batch size setting.
