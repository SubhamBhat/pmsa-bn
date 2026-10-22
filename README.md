# PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm
### Formal Mathematical Derivations, Empirical Bayes Shrinkage Proofs, and Clinical Visualizations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Conference](https://img.shields.io/badge/IEEE%20ISBI-2025-success.svg)](https://biomedicalimaging.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the complete mathematical proofs, layer-by-layer spatial analyses, and reproducible clinical verification code for **Prior-Matched Shift-Adaptive BatchNorm (PMSA-BN)** under strictly single-patient test-time adaptation ($B = 1$).

---

## Table of Contents
1. [Executive Summary & Advisor Review Guide](#1-executive-summary--advisor-review-guide)
2. [The Dual Nature of "Shrinkage" in Convolutional Networks](#2-the-dual-nature-of-shrinkage-in-convolutional-networks)
   - [Part A: Spatial Resolution Downsampling & Receptive Field Expansion](#part-a-spatial-resolution-downsampling--receptive-field-expansion)
   - [Part B: Statistical Empirical Bayes Shrinkage (Normal-Inverse-Gamma)](#part-b-statistical-empirical-bayes-shrinkage-normal-inverse-gamma)
   - [Part C: Feature Scaling & Enlargement via BN Affine Parameters](#part-c-feature-scaling--enlargement-via-bn-affine-parameters)
3. [Visual Proof 1: Layerwise Mask Shrinkage Through ResNet-34](#3-visual-proof-1-layerwise-mask-shrinkage-through-resnet-34)
4. [Visual Proof 2: Real Medical Image Proof (Fundus Optic Cup)](#4-visual-proof-2-real-medical-image-proof-fundus-optic-cup)
5. [Visual Proof 3: Batch-Size Invariance Across Modalities](#5-visual-proof-3-batch-size-invariance-across-modalities)
6. [Formal Mathematical Derivations](#6-formal-mathematical-derivations)
   - [Derivation 1: Estimator Variance Explosion in AdaBN as $N_k \to 1$](#derivation-1-estimator-variance-explosion-in-adabn-as-n_k-to-1)
   - [Derivation 2: Normal-Inverse-Gamma Posterior Updating](#derivation-2-normal-inverse-gamma-posterior-updating)
   - [Derivation 3: Law of Total Variance Multi-Class Prior Matching](#derivation-3-law-of-total-variance-multi-class-prior-matching)
7. [Step-by-Step Numerical Walkthrough on Real Drishti-GS Patient](#7-step-by-step-numerical-walkthrough-on-real-drishti-gs-patient)
8. [Multi-Class Extension: Brain Tumor MRI ($K = 4$ Classes)](#8-multi-class-extension-brain-tumor-mri-k--4-classes)
9. [Quickstart: Reproduce All Proofs with 1 Command](#9-quickstart-reproduce-all-proofs-with-1-command)

---

## 1. Executive Summary & Advisor Review Guide

When deep models are deployed on test images one-by-one ($B = 1$ test-time adaptation):
* **Why Standard AdaBN Fails:** AdaBN calculates the mean and variance directly across the single test image. In medical segmentation, target structures (e.g., optic cup, polyp, tumor core) often occupy less than $1.5\%$ of the image. At deep feature layers, the spatial resolution is downsampled (e.g., $256 \times 256 \to 8 \times 8$), leaving the target lesion with **fewer than 2 pixels**. The sample variance estimator variance is $\mathrm{Var}(s^2) = \frac{2\sigma^4}{N - 1}$, which blows up to infinity as $N \to 1$, corrupting all downstream layer activations.
* **How PMSA-BN Resolves It:** PMSA-BN applies **Empirical Bayes Shrinkage** (Eq. 3 & 4) parameterized by pseudo-counts $(\kappa_\mu = 8, \kappa_\sigma = 16)$. When effective foreground pixels $N_k$ are small ($N_k \ll \kappa$), the statistics smoothly anchor to the pre-calculated source prior ($\mu_{\text{src}}, \sigma_{\text{src}}^2$), preventing collapse. When $N_k \gg \kappa$, it fully adapts to target patient features.

---

## 2. The Dual Nature of "Shrinkage" in Convolutional Networks

To resolve any ambiguity, "shrinkage" operates on two distinct physical and mathematical axes:

### Part A: Spatial Resolution Downsampling & Receptive Field Expansion
As feature activations flow from the encoder's input to its deepest layer, the 2D spatial dimensions **shrink** while channel depth and receptive field **enlarge**:

```
Input Image (256x256, 3 ch)
       │
       ▼  [Conv1 + BN1, stride 2]
  Stage 0 Feature Map (128x128, 64 ch)
       │
       ▼  [MaxPool, stride 2]
  Stage 1 (layer1, 3 blocks with BN): 64x64, 64 ch    ───► Receptive Field: 35x35
       │
       ▼  [Residual Block downsample, stride 2]
  Stage 2 (layer2, 4 blocks with BN): 32x32, 128 ch   ───► Receptive Field: 91x91
       │
       ▼  [Residual Block downsample, stride 2]
  Stage 3 (layer3, 6 blocks with BN): 16x16, 256 ch   ───► Receptive Field: 203x203
       │
       ▼  [Residual Block downsample, stride 2]
  Stage 4 (layer4, 3 blocks with BN): 8x8, 512 ch     ───► Receptive Field: 427x427 (Global)
```

#### How the Segmentation Mask Shrinks Across Layers
When computing class statistics at intermediate layer $l$ with spatial shape $(H_l, W_l)$, the pseudo-mask $M \in \{0, 1\}^{K \times H \times W}$ is projected via **Adaptive Average Pooling**:
$$m_k^{(l)} = \text{AdaptiveAvgPool2D}(M_k, (H_l, W_l)) \in [0, 1]^{H_l \times W_l}$$

The effective foreground pixel count $N_k^{(l)}$ for class $k$ shrinks quadratically with downsampling factor $S_l = H / H_l$:
$$N_k^{(l)} = \sum_{h=1}^{H_l} \sum_{w=1}^{W_l} m_{k, h, w}^{(l)} \approx \frac{N_k^{(0)}}{S_l^2}$$

**Measured Pixel Counts on Real Patient Image (Drishti-GS Fundus, $256 \times 256$):**
| Network Stage | Resolution ($H_l \times W_l$) | Spatial Stride $S_l$ | Optic Disc ($N_{\text{disc}}$) | Optic Cup ($N_{\text{cup}}$) | Cup Fraction of Grid |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Input Image** | $256 \times 256$ | $1\times$ | $3,812.0$ px | $786.0$ px | $1.20\%$ |
| **layer1** | $64 \times 64$ | $4\times$ | $238.2$ px | $49.1$ px | $1.20\%$ |
| **layer2** | $32 \times 32$ | $8\times$ | $59.6$ px | $12.3$ px | $1.20\%$ |
| **layer3** | $16 \times 16$ | $16\times$ | $14.9$ px | **$3.07$ px** | $1.20\%$ |
| **layer4** | $8 \times 8$ | $32\times$ | $3.72$ px | **$0.77$ px ($< 1$ px!)** | $1.20\%$ |

---

### Part B: Statistical Empirical Bayes Shrinkage (Normal-Inverse-Gamma)
Statistical shrinkage pulls an empirical sample statistic toward a prior mean to minimize Mean Squared Error (James-Stein phenomenon):

$$\hat{\mu}_{k, c} = \gamma_\mu(N_k) \bar{x}_{k, c} + (1 - \gamma_\mu(N_k)) \mu_{k, c}^{\text{src}}$$
$$\hat{\sigma}_{k, c}^2 = \gamma_\sigma(N_k) s_{k, c}^2 + (1 - \gamma_\sigma(N_k)) \sigma_{k, c}^{2, \text{src}}$$

where the shrinkage weights are:
$$\gamma_\mu(N_k) = \frac{N_k}{N_k + \kappa_\mu}, \qquad \gamma_\sigma(N_k) = \frac{\max(0, N_k - 1)}{\max(0, N_k - 1) + \kappa_\sigma}$$

* **At shallow layers / large organs ($N_k \gg \kappa$):** $\gamma_\mu \to 1, \gamma_\sigma \to 1 \implies \hat{\mu} \to \bar{x}, \hat{\sigma}^2 \to s^2$ (Adapts to target domain).
* **At deep layers / tiny lesions ($N_k \to 0$ or $1$):** $\gamma_\mu \to 0, \gamma_\sigma \to 0 \implies \hat{\mu} \to \mu^{\text{src}}, \hat{\sigma}^2 \to \sigma^{2, \text{src}}$ (Guaranteed stable anchor).

---

### Part C: Feature Scaling & Enlargement via BN Affine Parameters
After standardization, the layer scales and shifts the normalized activations:
$$\hat{z} = \frac{z - \hat{\mu}}{\sqrt{\hat{\sigma}^2 + \epsilon}}$$
$$y = \gamma_{\text{BN}} \hat{z} + \beta_{\text{BN}}$$

* The denominator $\sqrt{\hat{\sigma}^2 + \epsilon}$ **normalizes** feature spread across channels.
* The learned affine weight $\gamma_{\text{BN}}$ (typically between $0.8$ and $1.5$ in trained encoders) **enlarges** dynamic contrast along discriminative feature directions before feeding into the ReLU non-linearity.

---

## 3. Visual Proof 1: Layerwise Mask Shrinkage Through ResNet-34

![Layerwise Shrinkage Proof](fig_layerwise_shrinkage_proof.png)

* **Top Row:** Real patient fundus image and Optic Cup mask downsampled through ResNet-34 stages (`Input` $\to$ `layer1` $\to$ `layer2` $\to$ `layer3` $\to$ `layer4`). Notice how at `layer4` ($8 \times 8$), the entire optic cup is represented by only $0.77$ effective pixels.
* **Middle Left:** Shrinkage weights $\gamma_\mu$ and $\gamma_\sigma$ as a function of $N_k$. Markers indicate exact operating points for each network stage.
* **Middle Right:** Variance stability across layers. Standard AdaBN exhibits catastrophic variance spikes ($> 40$) or near-zero collapse at deep layers, whereas PMSA-BN remains perfectly bounded.

---

## 4. Visual Proof 2: Real Medical Image Proof (Fundus Optic Cup)

![Visual Proof Fundus](fig_shrinkage_visual_proof.png)

This verification utilizes real patient data from the Drishti-GS test set (`drishtiGS_001.png`):
1. **Source Training Domain:** REFUGE ($n = 400$, standard optic disc/cup priors).
2. **Target Test Patient:** Drishti-GS ($n = 1$, strictly $B = 1$).
3. **Observation:** With AdaBN, optic cup activations collapse because cup pixels are outvoted by background retinal tissue. PMSA-BN isolates class-conditional statistics, applies EB shrinkage to the $1.2\%$ cup area, and recombines via the Law of Total Variance.

---

## 5. Visual Proof 3: Batch-Size Invariance Across Modalities

![Batch Size Invariance](fig1_batch_size_3panel.png)

Across Dermoscopy ($\text{PH}^2 \to \text{ISIC}$), Fundus ($\text{REFUGE} \to \text{Drishti}$), and CXR ($\text{Montgomery} \to \text{Shenzhen}$):
* **AdaBN** experiences sharp drops at $B = 1$ due to sample covariance noise.
* **$\alpha$-BN** improves over AdaBN but is fixed and requires target oracle tuning.
* **PMSA-BN** produces flat, optimal Dice across all batch sizes $B \in \{1, 2, 4, 8, 16\}$, matching or exceeding oracle baselines.

---

## 6. Formal Mathematical Derivations

### Derivation 1: Estimator Variance Explosion in AdaBN as $N_k \to 1$

Let sample activations $x_1, \dots, x_N$ be independently drawn from $\mathcal{N}(\mu, \sigma^2)$. The sample variance estimator is:
$$s^2 = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \bar{x})^2$$

By Cochran's theorem, $\frac{(N - 1) s^2}{\sigma^2} \sim \chi^2_{N - 1}$. The variance of a chi-square random variable with $k$ degrees of freedom is $2k$. Therefore:
$$\mathrm{Var}\left(\frac{(N - 1) s^2}{\sigma^2}\right) = 2(N - 1) \implies \mathrm{Var}(s^2) = \frac{2\sigma^4}{N - 1}$$

$$\lim_{N \to 1^+} \mathrm{Var}(s^2) = \lim_{N \to 1^+} \frac{2\sigma^4}{N - 1} = \infty$$

**Conclusion:** At deep layers where $N_k \approx 1$, the variance of the sample variance estimator approaches infinity. Standard AdaBN is mathematically ill-posed at $B = 1$ for small structures.

---

### Derivation 2: Normal-Inverse-Gamma Posterior Updating

Let the prior on class-conditional feature parameters $(\mu, \sigma^2)$ be the conjugate Normal-Inverse-Gamma distribution $\mathrm{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)$:
$$\sigma^2 \sim \mathrm{Inv}\text{-}\mathrm{Gamma}(\alpha_0, \beta_0), \qquad \mu \mid \sigma^2 \sim \mathcal{N}\left(\mu_0, \frac{\sigma^2}{\kappa_0}\right)$$

Parameterized with source prior statistics: $\mu_0 = \mu_{\text{src}}$, $\kappa_0 = \kappa_\mu$, $\alpha_0 = \frac{\kappa_\sigma}{2}$, $\beta_0 = \frac{\kappa_\sigma \sigma_{\text{src}}^2}{2}$.

Given $N$ target observations with sample mean $\bar{x}$ and sum of squared deviations $SS = \sum (x_i - \bar{x})^2$:
$$\kappa_N = \kappa_0 + N, \qquad \mu_N = \frac{\kappa_0 \mu_0 + N \bar{x}}{\kappa_0 + N} = \left(\frac{N}{N + \kappa_\mu}\right)\bar{x} + \left(\frac{\kappa_\mu}{N + \kappa_\mu}\right)\mu_{\text{src}}$$

This yields the exact posterior mean shrinkage formula:
$$\hat{\mu} = \gamma_\mu \bar{x} + (1 - \gamma_\mu) \mu_{\text{src}}, \quad \text{where } \gamma_\mu = \frac{N}{N + \kappa_\mu}$$

Similarly, the posterior marginal expectation of $\sigma^2$ is:
$$\mathbb{E}[\sigma^2 \mid X] = \frac{\beta_N}{\alpha_N - 1} = \gamma_\sigma s^2 + (1 - \gamma_\sigma) \sigma_{\text{src}}^2, \quad \text{where } \gamma_\sigma = \frac{N - 1}{(N - 1) + \kappa_\sigma}$$

---

### Derivation 3: Law of Total Variance Multi-Class Prior Matching

For arbitrary $K \ge 2$ mutually exclusive classes with source class proportions $\pi_1, \dots, \pi_K$ ($\sum_{k=1}^K \pi_k = 1$):

Let $Y \in \{1, \dots, K\}$ be the class indicator random variable with $P(Y = k) = \pi_k$.
By the Law of Total Expectation:
$$\mu_{\text{PM}} = \mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]] = \sum_{k=1}^K \pi_k \hat{\mu}_k$$

By the Law of Total Variance:
$$\mathrm{Var}(X) = \mathbb{E}[\mathrm{Var}(X \mid Y)] + \mathrm{Var}(\mathbb{E}[X \mid Y])$$

1. **Within-Class Variance:**
   $$\mathbb{E}[\mathrm{Var}(X \mid Y)] = \sum_{k=1}^K \pi_k \hat{\sigma}_k^2$$
2. **Between-Class Variance:**
   $$\mathrm{Var}(\mathbb{E}[X \mid Y]) = \sum_{k=1}^K \pi_k (\hat{\mu}_k - \mu_{\text{PM}})^2$$

Combining both terms gives Equation 1 in the paper:
$$\sigma_{\text{PM}}^2 = \sum_{k=1}^K \pi_k \hat{\sigma}_k^2 + \sum_{k=1}^K \pi_k (\hat{\mu}_k - \mu_{\text{PM}})^2$$

---

## 7. Step-by-Step Numerical Walkthrough on Real Drishti-GS Patient

Running [`proof_shrinkage_numerical_stability.py`](proof_shrinkage_numerical_stability.py) executes a step-by-step trace:

```python
# Real Patient Layer 3 Profile (Optic Cup, Nk = 3.07 pixels)
kappa_mu    = 8.0
kappa_sigma = 16.0
sigma_src   = 1.00
mu_src      = 0.00

# 1. Shrinkage weights
gamma_mu = 3.07 / (3.07 + 8.0)                # = 0.277
gamma_va = (3.07 - 1.0) / (2.07 + 16.0)       # = 0.115

# 2. Suppose noisy 3-pixel sample variance s^2 = 8.50 (severe overestimation)
# AdaBN uses s^2 = 8.50 directly (activations squashed by sqrt(8.50) = 2.91)
# PMSA-BN applies EB shrinkage:
shrunk_var = 0.115 * 8.50 + (1.0 - 0.115) * 1.00  # = 1.86 (Bounded & Stable!)

# 3. Suppose 3-pixel sample variance s^2 = 0.02 (near-zero collapse)
# AdaBN causes activation explosion: x / sqrt(0.02) = 7.07x boost
# PMSA-BN applies EB shrinkage:
shrunk_var = 0.115 * 0.02 + (1.0 - 0.115) * 1.00  # = 0.887 (Safe normalization)
```

---

## 8. Multi-Class Extension: Brain Tumor MRI ($K = 4$ Classes)

Evaluated on cross-scanner transfer (**BraTS 2019 $\to$ BraTS 2020**, $n = 369$ patients, $B = 1$):
* **Classes:** 0: Background ($97.2\%$), 1: Necrotic Core ($0.5\%$), 2: Edema ($1.6\%$), 3: Enhancing Tumor ($0.7\%$)

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

## 9. Quickstart: Reproduce All Proofs with 1 Command

Clone this repository and verify all mathematical bounds:

```bash
git clone https://github.com/SubhamBhat/pmsa-bn.git
cd pmsa-bn

# Install lightweight dependencies
pip install torch torchvision numpy matplotlib scipy pillow

# 1. Run numerical stability and limit verification
python proof_shrinkage_numerical_stability.py

# 2. Generate layerwise shrinkage proof with real patient fundus data
python generate_layerwise_shrinkage_proof.py

# 3. Generate 3-panel batch-size sensitivity graphs
python generate_best_batch_fig.py
```

---

## Citation

```bibtex
@inproceedings{pmsa_bn_isbi2025,
  title={PMSA-BN: Prior-Matched Shift-Adaptive BatchNorm for Single-Patient Medical Image Segmentation},
  author={Bhat, Subham and Collaborators},
  booktitle={IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2025}
}
```
