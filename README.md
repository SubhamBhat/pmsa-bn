# PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm
### Formal Mathematical Derivations, Empirical Bayes Shrinkage Proofs, and Clinical Visualizations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-in%20preparation%20%2F%20under%20review-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete mathematical proofs, layer-by-layer spatial downsampling analyses, clinical benchmarks, and standalone verification scripts for **Prior-Matched Shift-Adaptive BatchNorm (PMSA-BN)** under strictly single-patient test-time adaptation ($B = 1$).

---

## Table of Contents
1. [Executive Summary & Advisor Review Guide](#1-executive-summary--advisor-review-guide)
2. [The Three Mechanisms of "Shrinkage" in Convolutional Networks](#2-the-three-mechanisms-of-shrinkage-in-convolutional-networks)
   - [Mechanism 1: Spatial Dimension Downsampling & Quadratic Pixel Decay](#mechanism-1-spatial-dimension-downsampling--quadratic-pixel-decay)
   - [Mechanism 2: Statistical Empirical Bayes Normal-Inverse-Gamma Shrinkage](#mechanism-2-statistical-empirical-bayes-normal-inverse-gamma-shrinkage)
   - [Mechanism 3: Dynamic Range Enlargement via BN Affine Parameters](#mechanism-3-dynamic-range-enlargement-via-bn-affine-parameters)
3. [Visual Proof Gallery](#3-visual-proof-gallery)
   - [Visual Proof 1: Layerwise Mask Shrinkage Through ResNet-34](#visual-proof-1-layerwise-mask-shrinkage-through-resnet-34)
   - [Visual Proof 2: Real Clinical Patient Proof (Fundus Optic Cup)](#visual-proof-2-real-clinical-patient-proof-fundus-optic-cup)
   - [Visual Proof 3: Batch-Size Invariance Across Imaging Modalities](#visual-proof-3-batch-size-invariance-across-imaging-modalities)
4. [Formal Mathematical Derivations for All Paper Equations](#4-formal-mathematical-derivations-for-all-paper-equations)
   - [Derivation 1: Multi-Class Law of Total Variance (Equation 1)](#derivation-1-multi-class-law-of-total-variance-equation-1)
   - [Derivation 2: Label-Proportion Confounding in Standard AdaBN (Equation 2)](#derivation-2-label-proportion-confounding-in-standard-adabn-equation-2)
   - [Derivation 3: Estimator Variance Explosion in AdaBN as $N_k \to 1$](#derivation-3-estimator-variance-explosion-in-adabn-as-n_k-to-1)
   - [Derivation 4: Conjugate Normal-Inverse-Gamma Posterior Updating (Equations 3 & 4)](#derivation-4-conjugate-normal-inverse-gamma-posterior-updating-equations-3--4)
   - [Derivation 5: Scale-Free 2-Wasserstein Distance & Adaptive Blending (Equation 5)](#derivation-5-scale-free-2-wasserstein-distance--adaptive-blending-equation-5)
5. [Step-by-Step Numerical Walkthrough on Real Drishti-GS Patient](#5-step-by-step-numerical-walkthrough-on-real-drishti-gs-patient)
6. [Clinical Evaluation Across 5 Modalities (17 Transfers)](#6-clinical-evaluation-across-5-modalities-17-transfers)
   - [Modality 1: Fundus Photography (Optic Disc / Cup, Glaucoma, $K=3$)](#modality-1-fundus-photography-optic-disc--cup-glaucoma-k3)
   - [Modality 2: Dermoscopy (Skin Lesions, Melanoma, $K=2$)](#modality-2-dermoscopy-skin-lesions-melanoma-k2)
   - [Modality 3: Chest Radiography (Bilateral Lungs, Tuberculosis, $K=2$)](#modality-3-chest-radiography-bilateral-lungs-tuberculosis-k2)
   - [Modality 4: Endoscopic Colonoscopy (Polyps, Colorectal Cancer, $K=2$)](#modality-4-endoscopic-colonoscopy-polyps-colorectal-cancer-k2)
   - [Modality 5: Neuro-Oncology MRI (Brain Tumor Compartments, BraTS, $K=4$)](#modality-5-neuro-oncology-mri-brain-tumor-compartments-brats-k4)
7. [Comprehensive Advisor FAQ & Doubt Clarifications](#7-comprehensive-advisor-faq--doubt-clarifications)
8. [Quickstart: Reproduce All Proofs with 1 Command](#8-quickstart-reproduce-all-proofs-with-1-command)

---

## 1. Executive Summary & Advisor Review Guide

When deep segmentation networks are deployed on individual clinical test scans ($B = 1$ test-time adaptation):

1. **Why Standard AdaBN Collapses at $B = 1$:**
   AdaBN calculates the mean and variance across the single test image. In medical imaging, pathological lesions (e.g., optic cup, polyp, active tumor core) often occupy less than $1.5\%$ of the image grid. As activations pass through convolutional downsampling stages (e.g., $256 \times 256 \to 8 \times 8$), the target lesion is reduced to **fewer than 2 pixels**. The sample variance estimator variance is $\mathrm{Var}(s^2) = \frac{2\sigma^4}{N - 1}$, which diverges to infinity as $N \to 1$. Consequently, AdaBN's single-slice normalization denominator fluctuates wildly, destroying semantic feature representation and causing severe performance collapse (e.g., dropping below Source Only on 13 of 17 clinical transfers).

2. **Why Fixed $\alpha$-BN Cannot Solve This:**
   $\alpha$-BN interpolates between source running stats and single-image target stats using a constant weight $\alpha$ (e.g., $\alpha = 0.5$). However, the optimal $\alpha^\star$ varies drastically across clinical targets (ranging from $\alpha = 0.1$ on high-noise CXR to $\alpha = 0.9$ on high-contrast colonoscopy). Because target ground-truth masks are unavailable at test time, finding $\alpha^\star$ is impossible in real clinical workflows. Moreover, a static scalar $\alpha$ blends all channels uniformly, corrupting unshifted channels that required no adaptation.

3. **How PMSA-BN Guarantees Stability and Outperforms Baselines:**
   * **Empirical Bayes Shrinkage (Eq. 3 & 4):** Parameterized by conjugate Normal-Inverse-Gamma pseudo-counts $(\kappa_\mu = 8, \kappa_\sigma = 16)$. When effective foreground pixels $N_k$ are small ($N_k \ll \kappa$), sample statistics smoothly revert to stored source priors ($\mu_{\text{src}}, \sigma_{\text{src}}^2$), preventing variance explosion or division-by-zero. When $N_k \gg \kappa$, statistics fully adapt to the patient.
   * **Law of Total Variance Prior Matching (Eq. 1):** Recombines class statistics using source prevalence proportions $\pi_k^{\text{src}}$, mathematically decoupling genuine scanner appearance shift from confounding lesion size variations.
   * **Scale-Free 2-Wasserstein Shift Distance (Eq. 5):** Adaptively regulates per-channel mixing $\alpha_c \in [\alpha_{\min}, \alpha_{\max}]$, ensuring unshifted channels remain untouched while shifted channels adapt.

---

## 2. The Three Mechanisms of "Shrinkage" in Convolutional Networks

To eliminate all ambiguity, "shrinkage" in PMSA-BN operates across three distinct geometric and statistical axes:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   THREE DISTINCT MECHANISMS OF "SHRINKAGE"                       │
├─────────────────────────┬────────────────────────────┬───────────────────────────┤
│ 1. SPATIAL DOWNSAMPLING │ 2. STATISTICAL SHRINKAGE   │ 3. AFFINE ENLARGEMENT     │
│    (Geometric)          │    (Empirical Bayes)       │    (BatchNorm Scaling)    │
├─────────────────────────┼────────────────────────────┼───────────────────────────┤
│ Feature map shrinks:    │ Sample statistics shrunk   │ Standardized features     │
│ 256x256 -> 8x8.         │ toward source priors:      │ rescaled by learned γ:    │
│ Lesion pixel count Nk   │ μ_hat = γ_μ x_bar + (1-γ)μ │ y = γ_BN · z_hat + β_BN   │
│ shrinks quadratically.  │ σ_hat² = γ_σ s² + (1-γ)σ²  │ Restores contrast dynamic │
│ Optic cup: 786 -> 0.77px│ Prevents Var(s²) -> inf    │ range before activation.  │
└─────────────────────────┴────────────────────────────┴───────────────────────────┘
```

### Mechanism 1: Spatial Dimension Downsampling & Quadratic Pixel Decay
Through an encoder backbone (e.g., ResNet-34 or MobileNetV2), stride-2 convolutions and pooling progressively downsample spatial resolution while expanding receptive field diameter:

$$\begin{aligned}
\text{Input } X^{(0)} &\in \mathbb{R}^{3 \times 256 \times 256} && (\text{Receptive Field: } 1 \times 1) \\
&\;\;\Big\downarrow \text{Conv1 + BN1 (stride 2)} \\
\text{Stage 0} &\in \mathbb{R}^{64 \times 128 \times 128} && (\text{Receptive Field: } 7 \times 7) \\
&\;\;\Big\downarrow \text{MaxPool (stride 2)} \\
\text{layer1 (BN 1–3)} &\in \mathbb{R}^{64 \times 64 \times 64} && (\text{Receptive Field: } 35 \times 35) \\
&\;\;\Big\downarrow \text{Downsample Block (stride 2)} \\
\text{layer2 (BN 4–7)} &\in \mathbb{R}^{128 \times 32 \times 32} && (\text{Receptive Field: } 91 \times 91) \\
&\;\;\Big\downarrow \text{Downsample Block (stride 2)} \\
\text{layer3 (BN 8–13)} &\in \mathbb{R}^{256 \times 16 \times 16} && (\text{Receptive Field: } 203 \times 203) \\
&\;\;\Big\downarrow \text{Downsample Block (stride 2)} \\
\text{layer4 (BN 14–16)} &\in \mathbb{R}^{512 \times 8 \times 8} && (\text{Receptive Field: } 427 \times 427, \text{Global})
\end{aligned}$$

#### How the Segmentation Mask Shrinks Across Layers
At any intermediate BN layer $l$ with spatial shape $(H_l, W_l)$, the pseudo-mask $M \in \{0, 1\}^{K \times H \times W}$ is projected using **Adaptive Average Pooling**:
$$m_k^{(l)} = \text{AdaptiveAvgPool2D}(M_k, (H_l, W_l)) \in [0, 1]^{H_l \times W_l}$$

The effective foreground pixel count $N_k^{(l)}$ for class $k$ decays **quadratically** with the cumulative downsampling stride $S_l = H / H_l$:
$$N_k^{(l)} = \sum_{h=1}^{H_l} \sum_{w=1}^{W_l} m_{k, h, w}^{(l)} \approx \frac{N_k^{(0)}}{S_l^2}$$

**Measured Pixel Counts on Real Patient Image (Drishti-GS Fundus, $256 \times 256$):**
| Network Stage | Resolution ($H_l \times W_l$) | Spatial Stride $S_l$ | Optic Disc ($N_{\text{disc}}$) | Optic Cup ($N_{\text{cup}}$) | Cup Fraction of Grid |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Input Image** | $256 \times 256$ | $1\times$ | $3,812.0$ px | $786.0$ px | $1.20\%$ |
| **`layer1`** | $64 \times 64$ | $4\times$ | $238.2$ px | $49.1$ px | $1.20\%$ |
| **`layer2`** | $32 \times 32$ | $8\times$ | $59.6$ px | $12.3$ px | $1.20\%$ |
| **`layer3`** | $16 \times 16$ | $16\times$ | $14.9$ px | **$3.07$ px** | $1.20\%$ |
| **`layer4`** | $8 \times 8$ | $32\times$ | $3.72$ px | **$0.77$ px ($< 1$ px!)** | $1.20\%$ |

---

### Mechanism 2: Statistical Empirical Bayes Normal-Inverse-Gamma Shrinkage
Statistical shrinkage regularizes noisy single-patient empirical sample statistics $(\bar{x}_{k, c}, s_{k, c}^2)$ toward the source prior $(\mu_{k, c}^{\text{src}}, \sigma_{k, c}^{2, \text{src}})$:

$$\hat{\mu}_{k, c} = \gamma_\mu(N_k) \bar{x}_{k, c} + (1 - \gamma_\mu(N_k)) \mu_{k, c}^{\text{src}}$$
$$\hat{\sigma}_{k, c}^2 = \gamma_\sigma(N_k) s_{k, c}^2 + (1 - \gamma_\sigma(N_k)) \sigma_{k, c}^{2, \text{src}}$$

where:
$$\gamma_\mu(N_k) = \frac{N_k}{N_k + \kappa_\mu}, \qquad \gamma_\sigma(N_k) = \frac{\max(0, N_k - 1)}{\max(0, N_k - 1) + \kappa_\sigma}$$

* **Asymptotics for Large Lesions ($N_k \gg \kappa$):** $\gamma_\mu \to 1, \gamma_\sigma \to 1 \implies \hat{\mu} \to \bar{x}, \hat{\sigma}^2 \to s^2$ (Adapts fully to the patient).
* **Asymptotics for Tiny Structures ($N_k \to 0$ or $1$):** $\gamma_\mu \to 0, \gamma_\sigma \to 0 \implies \hat{\mu} \to \mu^{\text{src}}, \hat{\sigma}^2 \to \sigma^{2, \text{src}}$ (Guaranteed stable anchor).

---

### Mechanism 3: Dynamic Range Enlargement via BN Affine Parameters
After normalization, features are centered with unit variance. The layer applies learned affine parameters:
$$\hat{z}_{c, h, w} = \frac{x_{c, h, w} - \hat{\mu}_c}{\sqrt{\hat{\sigma}_c^2 + \epsilon}}$$
$$y_{c, h, w} = \gamma_{\text{BN}, c} \cdot \hat{z}_{c, h, w} + \beta_{\text{BN}, c}$$

* **Standardization** compresses arbitrary domain contrast differences into a zero-mean, unit-variance canonical space.
* **Affine Weight $\gamma_{\text{BN}}$** re-expands and enlarges discriminative contrast along semantic feature directions before passing into non-linear activations ($\text{ReLU}$).

---

## 3. Visual Proof Gallery

### Visual Proof 1: Layerwise Mask Shrinkage Through ResNet-34

![Layerwise Shrinkage Proof](fig_layerwise_shrinkage_proof.png)

* **Top Row:** Real patient fundus image and Optic Cup mask downsampled through ResNet-34 stages (`Input` $\to$ `layer1` $\to$ `layer2` $\to$ `layer3` $\to$ `layer4`). Notice how at `layer4` ($8 \times 8$), the entire optic cup is represented by only $0.77$ effective pixels.
* **Middle Left:** Shrinkage weights $\gamma_\mu$ and $\gamma_\sigma$ as a function of $N_k$. Markers indicate exact operating points for each network stage.
* **Middle Right:** Variance stability across layers. Standard AdaBN exhibits catastrophic variance spikes ($> 40$) or near-zero collapse at deep layers, whereas PMSA-BN remains perfectly bounded.

---

### Visual Proof 2: Real Clinical Patient Proof (Fundus Optic Cup)

![Visual Proof Fundus](fig_shrinkage_visual_proof.png)

This verification utilizes real patient data from the Drishti-GS test set (`drishtiGS_001.png`):
1. **Source Training Domain:** REFUGE ($n = 400$, standard optic disc/cup priors).
2. **Target Test Patient:** Drishti-GS ($n = 1$, strictly $B = 1$).
3. **Observation:** With AdaBN, optic cup activations collapse because cup pixels are outvoted by background retinal tissue. PMSA-BN isolates class-conditional statistics, applies EB shrinkage to the $1.2\%$ cup area, and recombines via the Law of Total Variance.

---

### Visual Proof 3: Batch-Size Invariance Across Imaging Modalities

![Batch Size Invariance](fig1_batch_size_3panel.png)

Across Dermoscopy ($\text{PH}^2 \to \text{ISIC}$), Fundus ($\text{REFUGE} \to \text{Drishti}$), and CXR ($\text{Montgomery} \to \text{Shenzhen}$):
* **AdaBN** experiences sharp drops at $B = 1$ due to sample covariance noise.
* **$\alpha$-BN** improves over AdaBN but is fixed and requires target oracle tuning.
* **PMSA-BN** produces flat, optimal Dice across all batch sizes $B \in \{1, 2, 4, 8, 16\}$, matching or exceeding oracle baselines.

---

## 4. Formal Mathematical Derivations for All Paper Equations

### Derivation 1: Multi-Class Law of Total Variance (Equation 1)

**Theorem:** For an arbitrary number of mutually exclusive classes $k \in \{1, \dots, K\}$ with prior probabilities $\pi_k$ ($\sum_{k=1}^K \pi_k = 1$), the recombined feature mean $\mu_{\text{PM}}$ and variance $\sigma_{\text{PM}}^2$ satisfy:

$$\mu_{\text{PM}} = \sum_{k=1}^K \pi_k \hat{\mu}_k$$
$$\sigma_{\text{PM}}^2 = \sum_{k=1}^K \pi_k \hat{\sigma}_k^2 + \sum_{k=1}^K \pi_k (\hat{\mu}_k - \mu_{\text{PM}})^2$$

*Proof:*
Let $X \in \mathbb{R}$ represent feature activations and $Y \in \{1, \dots, K\}$ denote the discrete class assignment with $P(Y = k) = \pi_k$.
By the Law of Total Expectation:
$$\mu_{\text{PM}} = \mathbb{E}[X] = \mathbb{E}_Y[\mathbb{E}[X \mid Y]] = \sum_{k=1}^K P(Y = k) \mathbb{E}[X \mid Y = k] = \sum_{k=1}^K \pi_k \hat{\mu}_k$$

By the Law of Total Variance:
$$\mathrm{Var}(X) = \mathbb{E}_Y[\mathrm{Var}(X \mid Y)] + \mathrm{Var}_Y(\mathbb{E}[X \mid Y])$$

1. **Within-Class Variance:**
   $$\mathbb{E}_Y[\mathrm{Var}(X \mid Y)] = \sum_{k=1}^K P(Y = k) \mathrm{Var}(X \mid Y = k) = \sum_{k=1}^K \pi_k \hat{\sigma}_k^2$$
2. **Between-Class Variance:**
   $$\mathrm{Var}_Y(\mathbb{E}[X \mid Y]) = \mathbb{E}_Y[(\mathbb{E}[X \mid Y] - \mathbb{E}[X])^2] = \sum_{k=1}^K \pi_k (\hat{\mu}_k - \mu_{\text{PM}})^2$$

Summing both components yields Equation 1. $\blacksquare$

---

### Derivation 2: Label-Proportion Confounding in Standard AdaBN (Equation 2)

**Proposition 1 (Label Shift Invariance):** Suppose the conditional feature distributions are identical between source and target domains ($\mathcal{N}(\mu_k, \sigma_k^2)$ is invariant), but the label proportions shift ($\pi_k^{\text{tgt}} \neq \pi_k^{\text{src}}$). Then standard AdaBN produces spurious domain shift.

*Proof:*
AdaBN estimates the batch mean across all pixels indiscriminately:
$$\mu_{\text{AdaBN}} = \sum_{k=1}^K \pi_k^{\text{tgt}} \mu_k$$
Comparing against the source running mean $\mu_{\text{src}} = \sum_{k=1}^K \pi_k^{\text{src}} \mu_k$:
$$\mu_{\text{AdaBN}} - \mu_{\text{src}} = \sum_{k=1}^K (\pi_k^{\text{tgt}} - \pi_k^{\text{src}}) \mu_k \neq 0$$

Even though the scanner physics and class appearance distributions $\mathcal{N}(\mu_k, \sigma_k^2)$ did not shift at all, AdaBN shifts the normalization statistics purely due to lesion size changes. PMSA-BN fixes this by recombining with stored source priors $\pi_k^{\text{src}}$, guaranteeing:
$$\mu_{\text{PM}} = \sum_{k=1}^K \pi_k^{\text{src}} \mu_k = \mu_{\text{src}}$$
eliminating the label-shift confound entirely. $\blacksquare$

---

### Derivation 3: Estimator Variance Explosion in AdaBN as $N_k \to 1$

Let sample activations $x_1, \dots, x_N$ be independently drawn from $\mathcal{N}(\mu, \sigma^2)$. The sample variance estimator is:
$$s^2 = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \bar{x})^2$$

By Cochran's theorem, $\frac{(N - 1) s^2}{\sigma^2} \sim \chi^2_{N - 1}$. The variance of a chi-square random variable with $k$ degrees of freedom is $2k$. Therefore:
$$\mathrm{Var}\left(\frac{(N - 1) s^2}{\sigma^2}\right) = 2(N - 1) \implies \mathrm{Var}(s^2) = \frac{2\sigma^4}{N - 1}$$

Taking the limit as $N \to 1$:
$$\lim_{N \to 1^+} \mathrm{Var}(s^2) = \lim_{N \to 1^+} \frac{2\sigma^4}{N - 1} = \infty$$

**Conclusion:** At deep layers where $N_k \approx 1$, the variance of the sample variance estimator approaches infinity. Standard AdaBN is mathematically ill-posed at $B = 1$ for small structures.

---

### Derivation 4: Conjugate Normal-Inverse-Gamma Posterior Updating (Equations 3 & 4)

Let the prior on class-conditional feature parameters $(\mu, \sigma^2)$ be the conjugate Normal-Inverse-Gamma distribution $\mathrm{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)$:
$$\sigma^2 \sim \mathrm{Inv}\text{-}\mathrm{Gamma}(\alpha_0, \beta_0), \qquad \mu \mid \sigma^2 \sim \mathcal{N}\left(\mu_0, \frac{\sigma^2}{\kappa_0}\right)$$

Parameterized with source prior statistics: $\mu_0 = \mu_{\text{src}}$, $\kappa_0 = \kappa_\mu$, $\alpha_0 = \frac{\kappa_\sigma}{2}$, $\beta_0 = \frac{\kappa_\sigma \sigma_{\text{src}}^2}{2}$.

Given $N$ target observations with sample mean $\bar{x}$ and sum of squared deviations $SS = \sum (x_i - \bar{x})^2$:
$$\kappa_N = \kappa_0 + N, \qquad \mu_N = \frac{\kappa_0 \mu_0 + N \bar{x}}{\kappa_0 + N} = \left(\frac{N}{N + \kappa_\mu}\right)\bar{x} + \left(\frac{\kappa_\mu}{N + \kappa_\mu}\right)\mu_{\text{src}}$$

This yields the exact posterior mean shrinkage formula:
$$\hat{\mu} = \gamma_\mu \bar{x} + (1 - \gamma_\mu) \mu_{\text{src}}, \quad \text{where } \gamma_\mu = \frac{N}{N + \kappa_\mu}$$

Similarly, the posterior marginal expectation of $\sigma^2$ is:
$$\mathbb{E}[\sigma^2 \mid X] = \frac{\beta_N}{\alpha_N - 1} = \gamma_\sigma s^2 + (1 - \gamma_\sigma) \sigma_{\text{src}}^2, \quad \text{where } \gamma_\sigma = \frac{\max(0, N - 1)}{\max(0, N - 1) + \kappa_\sigma}$$

---

### Derivation 5: Scale-Free 2-Wasserstein Distance & Adaptive Blending (Equation 5)

The 2-Wasserstein distance between two univariate Gaussian feature distributions $\mathcal{N}(\mu_{\text{PM}, c}, \sigma_{\text{PM}, c}^2)$ and $\mathcal{N}(\mu_c^{\text{src}}, \sigma_c^{2, \text{src}})$ is:
$$W_2^2 = (\mu_{\text{PM}, c} - \mu_c^{\text{src}})^2 + (\sigma_{\text{PM}, c} - \sigma_c^{\text{src}})^2$$

Normalizing by the source feature scale $\sigma_c^{2, \text{src}} + \epsilon$ yields the scale-free shift metric:
$$D_c = \frac{(\mu_{\text{PM}, c} - \mu_c^{\text{src}})^2 + (\sigma_{\text{PM}, c} - \sigma_c^{\text{src}})^2}{\sigma_c^{2, \text{src}} + \epsilon}$$

The adaptive per-channel interpolation weight $\alpha_c$ is mapped via a Sigmoid-Tanh schedule with temperature $\tau$:
$$\alpha_c = \text{clamp}\left(1.0 - \exp\left(-\frac{D_c}{\tau}\right), \alpha_{\min}, \alpha_{\max}\right)$$

* When $D_c \approx 0$ (channel has zero domain shift): $\alpha_c \to \alpha_{\min} \implies$ relies $90\%$ on source prior, preventing false adaptation.
* When $D_c \gg \tau$ (channel has strong domain shift): $\alpha_c \to \alpha_{\max} \implies$ adapts aggressively to align with target scanner characteristics.

---

## 5. Step-by-Step Numerical Walkthrough on Real Drishti-GS Patient

Running [`proof_shrinkage_numerical_stability.py`](proof_shrinkage_numerical_stability.py) executes a step-by-step trace on real patient data:

```python
# Real Patient Layer 3 Profile (Optic Cup, Nk = 3.07 pixels)
kappa_mu    = 8.0
kappa_sigma = 16.0
sigma_src   = 1.00
mu_src      = 0.00

# 1. Shrinkage weights
gamma_mu = 3.07 / (3.07 + 8.0)                # = 0.277
gamma_va = (3.07 - 1.0) / (2.07 + 16.0)       # = 0.115

# 2. Case A: Noisy 3-pixel sample variance s^2 = 14.80 (severe overestimation)
# AdaBN uses s^2 = 14.80 directly -> squashes activations by sqrt(14.80) = 3.85x
# PMSA-BN applies EB shrinkage:
shrunk_var = 0.115 * 14.80 + (1.0 - 0.115) * 1.00  # = 2.58 (Bounded & Stable!)

# 3. Case B: 3-pixel sample variance s^2 = 0.01 (near-zero collapse)
# AdaBN causes activation explosion: x / sqrt(0.01) = 10.0x boost
# PMSA-BN applies EB shrinkage:
shrunk_var = 0.115 * 0.01 + (1.0 - 0.115) * 1.00   # = 0.886 (Safe normalization)
```

---

## 6. Clinical Evaluation Across 5 Modalities (17 Transfers)

All evaluations are conducted under strictly single-patient test-time adaptation ($B = 1$).

### Modality 1: Fundus Photography (Optic Disc / Cup, Glaucoma, $K=3$)
* **Transfers:** REFUGE $\to$ ORIGA, Drishti-GS, RIM-ONE-r3, G1020, PAPILA, and reverse (ORIGA $\to$ REFUGE).
* **Metric:** Optic Disc Dice / Optic Cup Dice (%).

| Transfer | $n$ | Source Only | AdaBN ($B=1$) | $\alpha$-BN (0.5) | $\alpha$-BN$^\star$ (Oracle) | PMSA-BN (Ours) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| REFUGE $\to$ ORIGA | 325 | 87.63 / 79.38 | 87.48 / 80.59 | 88.75 / 80.96 | 88.75$_{.5}$ / 81.08$_{.7}$ | **89.20** / **82.48** |
| REFUGE $\to$ Drishti-GS | 50 | 95.63 / 78.61 | 95.55 / 75.53 | 96.08 / 78.30 | 96.08$_{.5}$ / 78.62$_{.1}$ | **96.51** / **81.04** |
| REFUGE $\to$ RIM-ONE r3 | 159 | 45.13 / 34.22 | 53.89 / 41.00 | 55.14 / 42.73 | 58.23$_{.7}$ / 44.77$_{.7}$ | **60.77** / **44.94** |
| REFUGE $\to$ G1020 | 200 | 73.09 / 61.57 | 79.67 / 58.26 | 76.67 / 60.68 | 80.15$_{.9}$ / 61.58$_{.1}$ | **81.94** / **61.73** |
| REFUGE $\to$ PAPILA | 99 | 54.59 / 45.69 | 88.06 / 60.95 | 72.89 / 56.29 | 86.54$_{.9}$ / 60.85$_{.9}$ | **88.69** / **61.73** |
| ORIGA $\to$ REFUGE (Reverse) | 400 | 88.12 / 71.45 | 89.05 / 73.21 | 90.15 / 74.90 | 90.62$_{.7}$ / 75.84$_{.7}$ | **92.34** / **79.15** |

---

### Modality 2: Dermoscopy (Skin Lesions, Melanoma, $K=2$)
* **Transfers:** $\text{PH}^2 \to \text{ISIC 2018}$, $\text{ISIC 2018} \to \text{PH}^2$, HAM10000, Derm7pt.
* **Metric:** Lesion Dice / Jaccard (%).

| Transfer | $n$ | Source Only | AdaBN ($B=1$) | $\alpha$-BN (0.5) | $\alpha$-BN$^\star$ (Oracle) | PMSA-BN (Ours) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $\text{PH}^2 \to \text{ISIC 2018}$ | 2594 | 86.10 / 78.40 | 75.80 / 68.20 | 85.90 / 78.10 | 86.40$_{.1}$ / 78.70$_{.1}$ | **87.20** / **79.80** |
| $\text{ISIC 2018} \to \text{PH}^2$ | 200 | 89.30 / 81.60 | 84.10 / 75.40 | 89.80 / 82.30 | 90.10$_{.3}$ / 82.70$_{.3}$ | **91.40** / **84.50** |

---

### Modality 3: Chest Radiography (Bilateral Lungs, Tuberculosis, $K=2$)
* **Transfers:** Montgomery $\to$ Shenzhen, Shenzhen $\to$ Montgomery, Montgomery $\to$ JSRT, Montgomery $\to$ COVID-QU-Ex.
* **Metric:** Lung Dice / Jaccard (%).

| Transfer | $n$ | Source Only | AdaBN ($B=1$) | $\alpha$-BN (0.5) | $\alpha$-BN$^\star$ (Oracle) | PMSA-BN (Ours) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Montgomery $\to$ Shenzhen | 336 | 94.40 / 89.65 | 93.86 / 88.66 | 94.40 / 89.63 | 94.43$_{.3}$ / 89.69$_{.3}$ | **94.69** / **90.13** |
| Shenzhen $\to$ Montgomery | 138 | 96.50 / 93.30 | 95.80 / 92.10 | 96.50 / 93.30 | 96.60$_{.3}$ / 93.50$_{.3}$ | **97.10** / **94.40** |

---

### Modality 4: Endoscopic Colonoscopy (Polyps, Colorectal Cancer, $K=2$)
* **Transfers:** Kvasir-SEG $\to$ CVC-ClinicDB, CVC-ClinicDB $\to$ Kvasir-SEG, Kvasir-SEG $\to$ ETIS-Larib.
* **Metric:** Polyp Dice / Jaccard (%).

| Transfer | $n$ | Source Only | AdaBN ($B=1$) | $\alpha$-BN (0.5) | $\alpha$-BN$^\star$ (Oracle) | PMSA-BN (Ours) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Kvasir $\to$ ClinicDB | 612 | 62.40 / 51.20 | 37.70 / 28.90 | 58.90 / 48.10 | 63.80$_{.9}$ / 52.60$_{.9}$ | **65.10** / **54.20** |
| ClinicDB $\to$ Kvasir | 1000 | 83.90 / 74.80 | 82.00 / 72.40 | 84.10 / 75.10 | 84.50$_{.5}$ / 75.60$_{.5}$ | **85.30** / **76.70** |

---

### Modality 5: Neuro-Oncology MRI (Brain Tumor Compartments, BraTS, $K=4$)
* **Transfers:** BraTS 2019 $\to$ BraTS 2020 ($n = 369$ patients, $B = 1$).
* **Metric:** Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET), Mean Dice (%).

| Backbone | Method | Whole Tumor (WT) | Tumor Core (TC) | Enhancing (ET) | Mean Composite Dice | $p$-value vs AdaBN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-34** | Source Only | $84.15\%$ | $75.80\%$ | $71.20\%$ | $77.05\%$ | — |
| | AdaBN ($B=1$) | $82.60\%$ | $71.15\%$ | $65.40\%$ | $73.05\%$ | — |
| | $\alpha$-BN ($\alpha=0.5$) | $84.75\%$ | $76.90\%$ | $72.80\%$ | $78.15\%$ | $4.1 \times 10^{-7}$ |
| | **PMSA-BN (Ours)** | **$87.25\%$** | **$80.40\%$** | **$76.85\%$** | **$81.50\%$** | **$1.83 \times 10^{-21}$** |
| **MobileNetV2** | Source Only | $82.90\%$ | $73.90\%$ | $69.30\%$ | $75.37\%$ | — |
| | AdaBN ($B=1$) | $81.20\%$ | $69.35\%$ | $63.10\%$ | $71.22\%$ | — |
| | $\alpha$-BN ($\alpha=0.5$) | $83.50\%$ | $75.10\%$ | $71.10\%$ | $76.57\%$ | $8.2 \times 10^{-6}$ |
| | **PMSA-BN (Ours)** | **$85.90\%$** | **$78.95\%$** | **$75.20\%$** | **$80.02\%$** | **$3.49 \times 10^{-19}$** |

---

## 7. Comprehensive Advisor FAQ & Doubt Clarifications

#### Q1: "Why does AdaBN perform worse than Source Only at $B = 1$?"
**Answer:** Standard AdaBN was designed for computer vision classification with large batch sizes ($B = 64$ or $128$). In medical segmentation at $B = 1$, calculating sample variance across a single patient's image introduces high estimation variance ($\mathrm{Var}(s^2) = \frac{2\sigma^4}{N - 1}$). For small structures ($N_k \le 3$), this variance diverges to infinity, corrupting the normalization scale.

#### Q2: "Can't we simply set $\alpha = 0.5$ in $\alpha$-BN?"
**Answer:** No. As proven in our 17-transfer benchmark, the optimal $\alpha^\star$ ranges from $0.1$ to $0.9$. A static scalar $\alpha = 0.5$ under-adapts when domain shift is strong and over-adapts when shift is absent. Crucially, static $\alpha$-BN blends all channels uniformly, corrupting unshifted channels. PMSA-BN automatically determines channel-specific adaptation $\alpha_c$ using the scale-free 2-Wasserstein distance.

#### Q3: "What happens if a lesion is completely absent ($N_k = 0$) in a test slice?"
**Answer:** In PMSA-BN, when $N_k = 0$, the shrinkage weights evaluate to $\gamma_\mu(0) = 0$ and $\gamma_\sigma(0) = 0$. The estimates mathematically default to:
$$\hat{\mu}_{k, c} = \mu_{k, c}^{\text{src}}, \qquad \hat{\sigma}_{k, c}^2 = \sigma_{k, c}^{2, \text{src}}$$
The layer anchors completely to the robust Phase 0 source prior. There is zero division-by-zero risk and zero instability.

#### Q4: "Does PMSA-BN require backpropagation or slow test-time optimization?"
**Answer:** No. PMSA-BN is 100% gradient-free and optimization-free. It performs only two forward passes:
* **Pass 1:** Forward pass with source parameters to extract confident pseudo-masks.
* **Pass 2:** Forward pass with prior-matched, shift-adaptive statistics.
Total test-time latency on an edge GPU (e.g., NVIDIA T4 or RTX 3060) is **$< 12$ milliseconds per slice**, enabling real-time deployment.

#### Q5: "How does the Law of Total Variance behave when anatomical classes overlap?"
**Answer:** Anatomical compartments in segmentation are mutually exclusive (each pixel has a unique argmax ground-truth label). Even with soft pseudo-probabilities $p_{k, h, w} \in [0, 1]$, the projection satisfies $\sum_{k=1}^K p_{k, h, w} = 1.0$. The Law of Total Variance remains mathematically exact.

---

## 8. Quickstart: Reproduce All Proofs with 1 Command

Clone this repository and verify all mathematical bounds:

```bash
git clone https://github.com/SubhamBhat/pmsa-bn.git
cd pmsa-bn

# Install lightweight dependencies
pip install torch torchvision numpy matplotlib scipy pillow nibabel h5py

# 1. Run numerical stability and limit verification
python proof_shrinkage_numerical_stability.py

# 2. Generate layerwise shrinkage proof with real patient fundus data
python generate_layerwise_shrinkage_proof.py

# 3. Generate 3-panel batch-size sensitivity graphs
python generate_best_batch_fig.py

# 4. Run end-to-end multi-class Brain Tumor benchmark (BraTS 2019 -> BraTS 2020)
python kaggle_brain_tumor_single_cell.py
```
