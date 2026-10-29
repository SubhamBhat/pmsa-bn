# PMSA-BN: Complete Resolution of Advisor & Reviewer In-Text Comments

> **Target Manuscript:** ISBI 4-Page Paper (*Prior-Matched, Shift-Adaptive Batch Normalization for Single-Image Test-Time Adaptation in Medical Image Segmentation*)  
> **Purpose:** Detailed mathematical, architectural, and clinical justifications for all 14 in-text advisor marks (`\textbf{...}` and `(??)`), formatted with standard KaTeX-compliant markdown math (`\mathrm{...}`) for flawless rendering across all markdown viewers, Overleaf, and GitHub.

---

## Quick Reference Summary Table

| # | Section & Line | Marked Text / Query | Core Resolution & Takeaway |
|:---:|:---|:---|:---|
| **1** | Sec 1 (L66) | `work at batch size one \textbf{(??)}` | Sequential clinical PACS workflow requires online, single-patient inference ($B=1$). |
| **2** | Sec 1 (L76) | `\textbf{(only at training time!)}` | Training batches average across diverse patients; single-image test-time ($B=1$) cannot average across patients. |
| **3** | Sec 2.1 (L124) | `once \textbf{(re-training?)}` | Absolutely **zero** re-training or backprop; purely a one-time forward calibration pass with frozen weights. |
| **4** | Sec 2.1 (L125) | `\textbf{(mask map to image resolution?)}` | The high-resolution mask is downsampled to match the deeper feature map grid $(H_l, W_l)$ via average pooling. |
| **5** | Sec 2.1 (L126) | `\textbf{(How do you know that?)}` | Mathematical definition of area-weighted average pooling equals receptive-field class coverage fraction. |
| **6** | Sec 2.1 (L130) | `\textbf{(clear it)}` | Strict source-free compliance: raw scans are discarded, preserving only $<150$ KB of scalar statistics. |
| **7** | Sec 2.2 (L135) | `\textbf{(you mean one test image is passing through twice?)}` | Exactly two forward passes (pilot pass for pseudo-mask, adapted pass for output), 0 backprop, $\approx 25$ ms. |
| **8** | Sec 2.2 (L143) | `M_k \textbf{(M of what?)}` | $M_k \in \{0, 1\}^{H \times W}$ is the high-confidence binary pseudo-mask for semantic class $k$. |
| **9** | Sec 2.2 (L143) | `N_k=\sum_{i,j}w_k(i,j) \textbf{(w?)}` | $w_k(i,j) \in [0,1]$ is downsampled soft spatial weight; $N_k$ is the effective spatial footprint (pixel count). |
| **10** | Sec 2.2 (L161) | `\textbf{(how to decide that?)}` | Conjugate Bayesian Normal-Inverse-Gamma prior counts ($\kappa_\mu=8, \kappa_\sigma=16$) tuned on source validation. |
| **11** | Sec 2.2 (L163) | `\textbf{(What do you mean by large and small classes?)}` | Spatial coverage relative to pseudo-count ($N_k \gg \kappa$ adapts to target; $N_k \ll \kappa$ shrinks to source prior). |
| **12** | Sec 2.2 (L172) | `simple guarantee \textbf{(of what?)}` | Theoretical guarantee of **Label-Proportion Invariance** (Proposition 1). |
| **13** | Sec 2.2 (L181) | `\textbf{(So what to do with that?)}` | Isolated appearance shift is quantified by 2-Wasserstein distance to gate per-channel adaptation $\alpha_c$. |
| **14** | Sec 2.2 (L196) | `\textbf{(tau used earlier for thresholding)}` | Disambiguated notation: confidence thresholds renamed to $\theta_{\mathrm{hi}}, \theta_{\mathrm{lo}}$; $\tau$ reserved for temperature. |

---

## Detailed Point-by-Point Clarifications & Text Revisions

### Comment 1: Why Batch Size One ($B=1$)?
* **Location:** Section 1 (Introduction, Line 66)
* **Sir's In-Text Mark:**
  > `"In clinical deployment, test images arrive one at a time, so TTA should work at batch size one \textbf{(??)}, without backpropagation..."`
* **Underlying Doubt:** Why is $B=1$ a practical clinical necessity rather than an artificial difficulty?
* **Scientific & Clinical Justification:**
  In real clinical environments (PACS workstations, emergency triage, radiology readouts, or intra-operative endoscopy), diagnostic images arrive **sequentially per patient visit**. 
  - An emergency stroke CT or an outpatient retinal fundus photo requires immediate segmentation.
  - A hospital cannot delay diagnosing a patient for hours or days simply to accumulate a batch of 16 or 32 scans.
  - Hence, clinical Test-Time Adaptation (TTA) must function **episodically on a single scan ($B=1$)**, adapting instantly without cross-patient data pooling.
* **Suggested Drop-In LaTeX Replacement:**
  ```latex
  In clinical workflows, diagnostic images arrive sequentially per patient examination rather than in synchronized batches. Therefore, practical TTA must operate on individual images ($B=1$) without backpropagation, leaving the model unchanged when no domain shift is present.
  ```

---

### Comment 2: Training-Time vs. Test-Time Batch Averaging
* **Location:** Section 1 (Introduction, Line 76)
* **Sir's In-Text Mark:**
  > `"...and re-estimating them treats it as a domain shift that, at batch size one \textbf{(only at training time!)}, cannot be averaged out."`
* **Underlying Doubt:** Sir was confused whether batch averaging is something that only happens during training.
* **Scientific & Clinical Justification:**
  - **During Training:** Standard Batch Normalization uses mini-batches of 16 to 64 images. Inter-patient anatomical differences (one patient has a large tumor, another has a tiny lesion or none) are averaged out across the diverse batch.
  - **At Test Time with $B=1$:** Traditional AdaBN recomputes mean and variance over the spatial pixels of *only one image*. If that single patient has a small lesion (e.g., optic cup covering $<1\%$ of the retina), the background completely overwhelms the statistics. Because there are no other patient scans in the batch, this anatomical imbalance cannot be averaged out, and AdaBN mistakes patient anatomy for scanner shift.
* **Suggested Drop-In LaTeX Replacement:**
  ```latex
  Unlike training where mini-batches of 16--32 images average out anatomical variance across subjects, test-time adaptation on a single image ($B=1$) cannot average across patients. Consequently, spatial label variations directly corrupt the estimated statistics.
  ```

---

### Comment 3: Is Phase 0 "Re-training"?
* **Location:** Section 2.1 (Phase 0: Source class statistics, Line 124)
* **Sir's In-Text Mark:**
  > `"After training, we pass the source training images through the network once \textbf{(re-training?)}, without augmentation."`
* **Underlying Doubt:** Did we re-train, fine-tune, or adjust weights in Phase 0?
* **Scientific & Clinical Justification:**
  - **No re-training, no loss computation, and zero backpropagation.**
  - All network weights $\theta$ are completely frozen.
  - Phase 0 is purely an offline, **one-time forward calibration pass** (`model.eval()`). It simply records the class-conditional activation means $\mu_{c,k}^{\mathrm{src}}$ and variances $\sigma_{c,k}^{2,\mathrm{src}}$, along with class proportions $\pi_k^{\mathrm{src}}$. This takes less than two minutes once training is finished.
* **Suggested Drop-In LaTeX Replacement:**
  ```latex
  Once training is complete, the network weights are permanently frozen. We perform a single, offline calibration forward pass over the source data without backpropagation or data augmentation.
  ```

---

### Comment 4: Mapping Mask to Feature Resolution
* **Location:** Section 2.1 (Phase 0: Source class statistics, Line 125)
* **Sir's In-Text Mark:**
  > `"We average-pool the ground-truth masks (expert source annotations, used only here) to each BN layer's resolution \textbf{(mask map to image resolution?)}."`
* **Underlying Doubt:** Which way is the mapping done: feature map to image, or mask to feature map?
* **Scientific & Clinical Justification:**
  - The ground-truth mask is defined at the **full input image resolution** ($H \times W$, e.g., $256 \times 256$).
  - However, intermediate feature maps $Z_l$ at layer $l$ have lower spatial resolutions (e.g., $128 \times 128, 64 \times 64, \dots, 16 \times 16$).
  - To calculate class-specific activation statistics for feature map $Z_l \in \mathbb{R}^{C \times H_l \times W_l}$, we downsample the full-resolution ground-truth mask to match $(H_l, W_l)$ using 2D average pooling.
* **Suggested Drop-In LaTeX Replacement:**
  ```latex
  Because intermediate BN feature maps $Z_l \in \mathbb{R}^{C \times H_l \times W_l}$ have lower spatial resolution than the input image, we downsample the full-resolution ground-truth masks to $(H_l, W_l)$ via average pooling.
  ```

---

### Comment 5: How Do We Know the Feature Coverage Fraction?
* **Location:** Section 2.1 (Phase 0: Source class statistics, Line 126)
* **Sir's In-Text Mark:**
  > `"This gives soft weights $w_k(i,j)\in[0,1]$, the fraction of each feature position \textbf{(How do you know that?)} covered by class $k$."`
* **Underlying Doubt:** How is this fraction known with mathematical certainty?
* **Scientific & Clinical Justification:**
  It is a direct property of **area-weighted average pooling**:
  - Suppose a feature cell at $(i, j)$ in layer $l$ corresponds to an $8 \times 8$ pixel patch in the original binary mask $Y \in \{0, 1\}^{H \times W}$.
  - If 16 of those 64 pixels belong to class $k=1$ and 48 belong to class $k=0$, average pooling with a kernel of 8 and stride of 8 computes:
    $$\frac{1}{64} \sum_{u=1}^{8}\sum_{v=1}^{8} Y_k(u, v) = \frac{16 \times 1 + 48 \times 0}{64} = 0.25$$
  - Thus, $w_k(i, j) = 0.25$ indicates that exactly 25% of that feature cell's receptive field footprint is covered by class $k$.
* **Suggested Drop-In LaTeX Replacement:**
  ```latex
  This yields continuous spatial weights $w_k(i,j) \in [0,1]$, where each value directly represents the fraction of pixels within feature position $(i,j)$'s receptive field belonging to class $k$.
  ```

---

### Comment 6: Clarifying "Source-Free" Nature
* **Location:** Section 2.1 (Phase 0: Source class statistics, Line 130)
* **Sir's In-Text Mark:**
  > `"No source images are used at test time \textbf{(clear it)}, so the method is source-free..."`
* **Underlying Doubt:** Exactly what leaves the source hospital, and why is this strictly privacy-preserving?
* **Scientific & Clinical Justification:**
  - All source patient scans, pixel arrays, and annotations are **discarded**.
  - The only data retained are compact, low-dimensional scalar summary statistics:
    - Channel class-conditional means $\mu_{c,k}^{\mathrm{src}}$ and variances $\sigma_{c,k}^{2,\mathrm{src}}$ ($2K$ numbers per channel).
    - Layer class priors $\pi_k^{\mathrm{src}}$ ($K$ numbers per layer).
  - For a standard 4-class segmentation network, these summaries total **less than 150 KB of memory**. No patient images ever leave the training institution, satisfying strict HIPAA and GDPR medical privacy mandates.
* **Suggested Drop-In LaTeX Replacement:**
  ```latex
  At deployment, all source images and masks are discarded; only the scalar summary statistics ($2K$ values per channel and $K$ priors per layer, totalling $<150$\,KB) are packaged with the frozen weights, preserving patient privacy and ensuring strict source-free operation.
  ```

---

### Comment 7: Does One Test Image Pass Through Twice?
* **Location:** Section 2.2 (Phase 1: Test-time adaptation, Line 135)
* **Sir's In-Text Mark:**
  > `"A first forward pass \textbf{(you mean one test image is passing through twice?)} with the source statistics gives class probabilities $P^{(0)}$..."`
* **Underlying Doubt:** Does processing the image twice create excessive computational latency?
* **Scientific & Clinical Justification:**
  - **Yes, exactly two forward passes (and 0 backward passes):**
    1. **Pilot Forward Pass:** The test image passes through the frozen network using default source statistics to compute preliminary class probability map $P^{(0)}$. High-confidence regions form the pseudo-masks.
    2. **Adapted Forward Pass:** The test image passes through once more, now normalized with PMSA-BN's calibrated statistics, producing the final adapted segmentation.
  - **Latency:** Because both passes are feedforward with no autograd, backward graph construction, or optimizer steps, the total runtime is **$\approx 25$ ms per image** on a single GPU. This is real-time and more than **$10\times$ faster** than backprop-based TTA methods like TENT (284 ms).
* **Suggested Drop-In LaTeX Replacement:**
  ```latex
  Adaptation proceeds in two lightweight forward passes without backpropagation: an initial pilot pass with stored source statistics computes class probability map $P^{(0)}$, followed by an adapted forward pass using our calibrated statistics.
  ```

---

### Comment 8: Clarifying $M_k$ ("M of what?")
* **Location:** Section 2.2 (Phase 1: Test-time adaptation, Line 143)
* **Sir's In-Text Mark:**
  > `"We average-pool $M_k$ \textbf{(M of what?)} to the layer's resolution to get $w_k$..."`
* **Underlying Doubt:** What object does $M_k$ denote?
* **Scientific & Clinical Justification:**
  - $M_k \in \{0, 1\}^{H \times W}$ is the **high-confidence binary pseudo-mask for class $k$** derived from the pilot forward pass:
    $$M_k(u,v) = \mathbf{1}\left[\arg\max_{c} P_c^{(0)}(u,v) = k \quad \text{and} \quad P_k^{(0)}(u,v) \ge \theta_{\mathrm{hi}}\right]$$
  - It marks all spatial coordinates in the input test image where the source model confidently predicts class $k$.
* **Suggested Drop-In LaTeX Replacement:**
  ```latex
  We average-pool the confident binary pseudo-mask $M_k \in \{0,1\}^{H \times W}$ for class $k$ to the layer's spatial grid $(H_l, W_l)$ to obtain soft weights $w_k \in [0,1]^{H_l \times W_l}$...
  ```

---

### Comment 9: Defining $w$ and $N_k$
* **Location:** Section 2.2 (Phase 1: Test-time adaptation, Line 143)
* **Sir's In-Text Mark:**
  > `"...and set $N_k=\sum_{i,j}w_k(i,j)$ \textbf{(w?)}..."`
* **Underlying Doubt:** What is $w$, and what does the sum represent?
* **Scientific & Clinical Justification:**
  - $w_k(i, j) \in [0, 1]$ is the downsampled soft spatial weight of class $k$ at feature cell $(i, j)$.
  - Summing $w_k(i, j)$ across all spatial positions $(i, j)$ in the feature map gives $N_k$, which is the **effective spatial sample size (effective feature count)** occupied by class $k$ at that layer.
* **Suggested Drop-In LaTeX Replacement:**
  ```latex
  ...and define $N_k = \sum_{i,j} w_k(i,j)$ as the effective spatial footprint (in feature units) of class $k$.
  ```

---

### Comment 10: How Are Pseudo-Counts $\kappa_\mu, \kappa_\sigma$ Decided?
* **Location:** Section 2.2 (Phase 1: Test-time adaptation, Line 161)
* **Sir's In-Text Mark:**
  > `"...they state how many target positions the source estimate is worth \textbf{(how to decide that?)}."`
* **Underlying Doubt:** How were $\kappa_\mu$ and $\kappa_\sigma$ chosen? Were they tuned on target data (violating TTA protocol)?
* **Scientific & Clinical Justification:**
  - $\kappa_\mu$ and $\kappa_\sigma$ originate from the classical **Bayesian conjugate Normal-Inverse-Gamma prior**.
  - They represent the number of pseudo-observations attributed to the source prior.
  - **Crucially, they were tuned once on source validation data only** ($\kappa_\mu = 8, \kappa_\sigma = 16$).
  - In our 27-setting hyperparameter sensitivity sweep across diverse modalities, the test Dice variation remained $\le 0.008$. The method is stable and does not require per-target tuning.
* **Suggested Drop-In LaTeX Replacement:**
  ```latex
  The pseudo-counts $\kappa_\mu$ and $\kappa_\sigma$ act as conjugate Bayesian prior counts, set once on source validation data ($\kappa_\mu=8, \kappa_\sigma=16$) and held constant across all 17 target benchmarks without per-dataset tuning.
  ```

---

### Comment 11: Meaning of "Large" and "Small" Classes
* **Location:** Section 2.2 (Phase 1: Test-time adaptation, Line 163)
* **Sir's In-Text Mark:**
  > `"A large class uses its own target statistics, and a small class moves toward its source statistics \textbf{(What do you mean by large and small classes?)}."`
* **Underlying Doubt:** Does "large/small" refer to organ anatomy or sample size in the feature map?
* **Scientific & Clinical Justification:**
  It refers to the **spatial feature count $N_k$ relative to the shrinkage pseudo-count $\kappa$**:
  - **Large Class ($N_k \gg \kappa$):** Dominant classes such as background or large organs ($N_k \approx 10{,}000$ to $60{,}000$ positions). For these, $\frac{N_k}{N_k + \kappa} \to 1$, so the layer adapts fully to target statistics.
  - **Small Class ($N_k \ll \kappa$):** Fine structures (e.g., optic cup at bottleneck layers where $N_{\mathrm{cup}} \le 2$). Without shrinkage, sample variance would diverge; the shrinkage formula automatically pulls the estimate back to the stable source prior: $\frac{\kappa}{N_k + \kappa} \to 1$.
* **Suggested Drop-In LaTeX Replacement:**
  ```latex
  When a class occupies a large spatial footprint ($N_k \gg \kappa$, e.g., background), the estimate relies on target activations; when a class has few or zero positions ($N_k \ll \kappa$, e.g., small lesions at bottleneck layers), the shrinkage formula automatically pulls the estimate back to the stable source prior.
  ```

---

### Comment 12: Guarantee of What?
* **Location:** Section 2.2 (Phase 1: Test-time adaptation, Line 172)
* **Sir's In-Text Mark:**
  > `"This gives a simple guarantee \textbf{(of what?)}."`
* **Underlying Doubt:** What specific theoretical guarantee is being claimed?
* **Scientific & Clinical Justification:**
  It is the theoretical guarantee of **Label-Proportion Invariance (Proposition 1)**:
  - If a patient has a larger or smaller lesion, but the scanner calibration and tissue contrast are identical to the source domain (pure label proportion shift: $\pi_k^{\mathrm{tgt}} \ne \pi_k^{\mathrm{src}}$, but $\mu_k^{\mathrm{tgt}} = \mu_k^{\mathrm{src}}$ and $\sigma_k^{2,\mathrm{tgt}} = \sigma_k^{2,\mathrm{src}}$), our prior-matched statistics $(\mu_{\mathrm{PM}}, \sigma^2_{\mathrm{PM}})$ remain **strictly equal** to the source BN statistics $(\mu^{\mathrm{src}}, \sigma^{2,\mathrm{src}})$.
  - Conventional AdaBN fails this test because it pools across all pixels and conflates tumor size with scanner shift.
* **Suggested Drop-In LaTeX Replacement:**
  ```latex
  This formulation yields a theoretical guarantee of label-proportion invariance (Proposition~\ref{prop:inv}):
  ```

---

### Comment 13: "So What to Do With That?"
* **Location:** Section 2.2 (Phase 1: Test-time adaptation, Line 181)
* **Sir's In-Text Mark:**
  > `"Any remaining difference between $(\mu_{\PM},\sigma^2_{\PM})$ and the source statistics is therefore appearance shift \textbf{(So what to do with that?)}."`
* **Underlying Doubt:** Once pure appearance shift is isolated, what is the exact operational next step?
* **Scientific & Clinical Justification:**
  - We measure this pure appearance shift along every channel using the **2-Wasserstein distance $D_c$**:
    $$D_c = \sqrt{\left(\mu_{c,\mathrm{PM}} - \mu_c^{\mathrm{src}}\right)^2 + \left(\sigma_{c,\mathrm{PM}} - \sigma_c^{\mathrm{src}}\right)^2}$$
  - We use $D_c$ to compute a per-channel adaptive interpolation weight:
    $$\alpha_c = 1 - \exp\left(-\frac{D_c}{\tau}\right)$$
  - **Outcome:** Channels experiencing severe appearance shift ($D_c \gg \tau \implies \alpha_c \to 1$) adapt fully to the target domain, while channels with negligible shift ($D_c \approx 0 \implies \alpha_c \to 0$) retain their pristine source parameters.
* **Suggested Drop-In LaTeX Replacement:**
  ```latex
  Any remaining difference between $(\mu_{\mathrm{PM}},\sigma^2_{\mathrm{PM}})$ and the source statistics reflects true appearance shift. We quantify this per channel via the Wasserstein distance $D_c$ to compute an adaptive interpolation weight $\alpha_c$, adapting only the corrupted channels while preserving unshifted features.
  ```

---

### Comment 14: Disambiguating $\tau$ Notation Collision
* **Location:** Section 2.2 (Phase 1: Test-time adaptation, Line 196)
* **Sir's In-Text Mark:**
  > `"One temperature $\tau$ \textbf{(tau used earlier for thresholding)} controls the transition for all channels."`
* **Underlying Doubt:** Notation ambiguity: $\tau_{\mathrm{hi}}, \tau_{\mathrm{lo}}$ were used for probability thresholding, and $\tau$ was used for the Wasserstein temperature.
* **Scientific & Clinical Justification:**
  Your professor's observation is correct. To avoid any ambiguity for reviewers:
  - Probability confidence thresholds are renamed to **$\theta_{\mathrm{hi}}$ and $\theta_{\mathrm{lo}}$**.
  - The temperature parameter in the exponential Wasserstein gate remains **$\tau$**.
* **Suggested Drop-In LaTeX Replacement:**
  1. Under the pseudo-masking definition in Section 2.2:
     ```latex
     % Replace \tau_{hi}, \tau_{lo} with \theta_{hi}, \theta_{lo}
     P_k^{(0)}(u,v) \ge \theta_{\mathrm{hi}}, \qquad P_k^{(0)}(u,v) \le \theta_{\mathrm{lo}}
     ```
  2. At Line 196:
     ```latex
     A single temperature parameter $\tau$ governs the sigmoid-like transition rate across all channels.
     ```

---

## Complete Drop-In Section 2 Replacement Block for Overleaf

To update the Overleaf manuscript directly, use this revised snippet for Sections 2.1 and 2.2:

```latex
\subsection{Source class statistics}
\label{ssec:phase0}

The pooled statistics of a BN channel decompose over the classes by the law of total variance:
\begin{equation}
\mu=\sum_k\pi_k\mu_k,\qquad
\sigma^2=\sum_k\pi_k\sigma^2_k+\sum_k\pi_k(\mu_k-\mu)^2 .
\label{eq:decomp}
\end{equation}
Here $\pi_k$ is the fraction of positions of class $k$, and $(\mu_k,\sigma^2_k)$ are its class statistics.

Once training is complete, the network weights are permanently frozen. We perform a single, offline calibration forward pass over the source data without backpropagation or data augmentation. Because intermediate BN feature maps $Z_l \in \mathbb{R}^{C \times H_l \times W_l}$ have lower spatial resolution than the input image, we downsample the full-resolution ground-truth masks to $(H_l, W_l)$ via average pooling. This yields continuous spatial weights $w_k(i,j) \in [0,1]$, where each value directly represents the fraction of pixels within feature position $(i,j)$'s receptive field belonging to class $k$. Average pooling, rather than nearest-neighbor sampling, handles positions that straddle a boundary in deep layers. From these weights, we compute the class statistics $\mu^{\mathrm{src}}_k$ and $\sigma^{2,\mathrm{src}}_k$ for every channel, and the proportions $\pi^{\mathrm{src}}_k$ for every layer. As a check, $\sum_k\pi^{\mathrm{src}}_k\mu^{\mathrm{src}}_k$ matches the BN running mean, up to the small gap caused by augmentation during training. At deployment, all source images and masks are discarded; only the scalar summary statistics ($2K$ values per channel and $K$ priors per layer, totalling $<150$\,KB) are packaged with the frozen weights, preserving patient privacy and ensuring strict source-free operation.

\subsection{Test-time adaptation}
\label{ssec:tta}

Adaptation proceeds in two lightweight forward passes without backpropagation: an initial pilot pass with stored source statistics computes class probability map $P^{(0)}$, followed by an adapted forward pass using our calibrated statistics. From $P^{(0)}$, we retain only confident predictions:
\begin{equation}
M_k(u,v) = \mathbf{1}\left[\arg\max_c P_c^{(0)}(u,v) = k \;\;\text{and}\;\; P_k^{(0)}(u,v) \ge \theta_{\mathrm{hi}}\right]
\label{eq:pmask}
\end{equation}
with $\theta_{\mathrm{hi}}=0.90$. We average-pool the confident binary pseudo-mask $M_k \in \{0,1\}^{H \times W}$ for class $k$ to the layer's spatial grid $(H_l, W_l)$ to obtain soft weights $w_k \in [0,1]^{H_l \times W_l}$ and define $N_k = \sum_{i,j} w_k(i,j)$ as the effective spatial footprint (in feature units) of class $k$. The sample moments on target image activations $Z$ are:
\begin{equation}
\hat{\mu}_k = \frac{1}{N_k}\sum_{i,j} w_k(i,j) Z(i,j), \quad
\hat{\sigma}^2_k = \frac{1}{N_k}\sum_{i,j} w_k(i,j)\left(Z(i,j)-\hat{\mu}_k\right)^2 .
\label{eq:raw_tgt}
\end{equation}
To protect against small sample sizes, we shrink toward the source class statistics using conjugate Bayesian pseudo-counts:
\begin{equation}
\tilde{\mu}_k = \frac{N_k\hat{\mu}_k + \kappa_\mu\mu^{\mathrm{src}}_k}{N_k+\kappa_\mu}, \quad
\tilde{\sigma}^2_k = \frac{N_k\hat{\sigma}^2_k + \kappa_\sigma\sigma^{2,\mathrm{src}}_k}{N_k+\kappa_\sigma} .
\label{eq:shrink}
\end{equation}
The pseudo-counts act as conjugate Bayesian prior counts, set once on source validation data ($\kappa_\mu=8, \kappa_\sigma=16$) and held constant across all 17 target benchmarks without per-dataset tuning. When a class occupies a large spatial footprint ($N_k \gg \kappa$, e.g., background), the estimate relies on target activations; when a class has few or zero positions ($N_k \ll \kappa$, e.g., small lesions at bottleneck layers), the shrinkage formula automatically pulls the estimate back to the stable source prior.

We then recombine these shrunk class statistics using the \emph{source} class proportions $\pi^{\mathrm{src}}_k$:
\begin{equation}
\mu_{\mathrm{PM}} = \sum_k \pi^{\mathrm{src}}_k \tilde{\mu}_k, \quad
\sigma^2_{\mathrm{PM}} = \sum_k \pi^{\mathrm{src}}_k \tilde{\sigma}^2_k + \sum_k \pi^{\mathrm{src}}_k (\tilde{\mu}_k - \mu_{\mathrm{PM}})^2 .
\label{eq:pm}
\end{equation}
This formulation yields a theoretical guarantee of label-proportion invariance (Proposition~\ref{prop:inv}):
\begin{proposition}[Label-Proportion Invariance]
\label{prop:inv}
If class-conditional feature distributions are identical between source and target ($\mu_k^{\mathrm{tgt}}=\mu_k^{\mathrm{src}}$, $\sigma^{2,\mathrm{tgt}}_k=\sigma^{2,\mathrm{src}}_k$ for all $k$) and pseudo-masks are accurate, then $(\mu_{\mathrm{PM}},\sigma^2_{\mathrm{PM}})=(\mu^{\mathrm{src}},\sigma^{2,\mathrm{src}})$ regardless of target class proportions $\pi^{\mathrm{tgt}}$.
\end{proposition}

Any remaining difference between $(\mu_{\mathrm{PM}},\sigma^2_{\mathrm{PM}})$ and the source statistics reflects true appearance shift. We quantify this per channel via the Wasserstein distance $D_c$ to compute an adaptive interpolation weight $\alpha_c$, adapting only the corrupted channels while preserving unshifted features:
\begin{equation}
D_c = \sqrt{(\mu_{c,\mathrm{PM}}-\mu^{\mathrm{src}}_c)^2 + (\sigma_{c,\mathrm{PM}}-\sigma^{\mathrm{src}}_c)^2}, \quad
\alpha_c = 1 - \exp\left(-\frac{D_c}{\tau}\right) .
\label{eq:adapt}
\end{equation}
A single temperature parameter $\tau$ governs the sigmoid-like transition rate across all channels.
```
