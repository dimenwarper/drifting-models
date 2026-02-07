# Drifting Models: A New Paradigm for Generative Modeling

**Paper:** Generative Modeling via Drifting
**Authors:** Mingyang Deng, He Li, Tianhong Li, Yilun Du, Kaiming He
**arXiv:** 2602.04770 (February 2026)
**License:** CC BY 4.0

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concept: Pushforward at Training Time](#core-concept-pushforward-at-training-time)
3. [The Drifting Field](#the-drifting-field)
4. [Training Objective](#training-objective)
5. [Kernel Design](#kernel-design)
6. [Feature Space Drifting](#feature-space-drifting)
7. [Classifier-Free Guidance](#classifier-free-guidance)
8. [Implementation Details](#implementation-details)
9. [Results](#results)
10. [Limitations and Open Questions](#limitations-and-open-questions)

---

## Overview

Drifting Models introduce a fundamentally new approach to generative modeling that differs from diffusion and flow-based models in a key way:

| Aspect | Diffusion/Flow Models | Drifting Models |
|--------|----------------------|-----------------|
| **Iteration happens** | At inference time | At training time |
| **Inference steps** | Many (10-1000 NFE) | One (1 NFE) |
| **Core mechanism** | Learn to denoise step-by-step | Learn equilibrium distribution directly |
| **Distribution evolution** | Sample-level at inference | Network-level at training |

The key insight is that neural network training is inherently iterative (SGD). Instead of learning an iterative denoising process, Drifting Models leverage the training loop itself to evolve the generated distribution toward the data distribution.

---

## Core Concept: Pushforward at Training Time

### Pushforward Distribution

A generator `f: ℝᶜ → ℝᴰ` maps noise `ε ~ p_ε` to outputs `x = f(ε)`. The distribution of outputs is the **pushforward distribution**:

```
q = f#p_ε
```

The goal of generative modeling is to find `f` such that `f#p_ε ≈ p_data`.

### Training-Time Evolution

Since training is iterative, we get a sequence of models `{f_i}` at each iteration `i`. This corresponds to a sequence of pushforward distributions `{q_i}`:

```
q_i = [f_i]#p_ε
```

When the network is updated, samples implicitly "drift":

```
x_{i+1} = x_i + Δx_i
```

where `Δx_i := f_{i+1}(ε) - f_i(ε)` arises from parameter updates.

**Key Realization:** The parameter update determines the sample drift. We can design a training objective that controls this drift to push `q` toward `p_data`.

---

## The Drifting Field

### Definition

A drifting field `V_{p,q}(·): ℝᵈ → ℝᵈ` governs sample movement:

```
x_{i+1} = x_i + V_{p,q_i}(x_i)
```

The field depends on:
- `p`: the target distribution (data)
- `q`: the current generated distribution

### Anti-Symmetry Property

**Proposition:** If the drifting field is anti-symmetric:

```
V_{p,q}(x) = -V_{q,p}(x),  ∀x
```

Then: `q = p ⟹ V_{p,q}(x) = 0, ∀x`

**Proof:** If `q = p`, then `V_{p,q} = V_{q,p} = -V_{p,q}`, which implies `V_{p,q} = 0`.

This means when the distributions match, samples stop drifting — **equilibrium is reached**.

### Attraction and Repulsion

The drifting field is decomposed into two components:

```
V_{p,q}(x) = V⁺_p(x) - V⁻_q(x)
```

Where:
- **V⁺_p (Attraction):** Pulls samples toward real data
- **V⁻_q (Repulsion):** Pushes samples away from other generated samples

Each component uses a mean-shift formulation:

```
V⁺_p(x) = (1/Z_p) 𝔼_{y⁺~p}[k(x, y⁺)(y⁺ - x)]
V⁻_q(x) = (1/Z_q) 𝔼_{y⁻~q}[k(x, y⁻)(y⁻ - x)]
```

Where `Z_p` and `Z_q` are normalization factors and `k(·,·)` is a kernel function.

### Combined Form

Substituting into the full expression:

```
V_{p,q}(x) = (1/(Z_p·Z_q)) 𝔼_{p,q}[k(x, y⁺)·k(x, y⁻)·(y⁺ - y⁻)]
```

This form is clearly anti-symmetric: swapping `p` and `q` flips the sign.

### Physical Intuition

Think of it like charged particles:
- Generated samples are attracted to real data points
- Generated samples repel each other
- At equilibrium, generated samples spread to cover the data distribution
- This naturally prevents mode collapse

---

## Training Objective

### Fixed-Point Formulation

At equilibrium where `V = 0`, we have:

```
f_θ̂(ε) = f_θ̂(ε) + V_{p,q_θ̂}(f_θ̂(ε))
```

This motivates a fixed-point iteration during training.

### Loss Function

```python
x = f_θ(ε)                              # prediction
target = stopgrad(x + V_{p,q_θ}(x))     # frozen target
loss = 𝔼_ε[||x - target||²]             # MSE loss
```

The loss value equals `𝔼_ε[||V(f(ε))||²]` — the squared norm of the drifting field.

**Important:** The stop-gradient prevents backpropagation through `V` (which depends on the distribution `q_θ`). Instead, we minimize the objective indirectly by moving predictions toward their drifted versions.

### Training Algorithm (Pseudocode)

```python
def training_step(f, y_pos):
    """
    f: generator network
    y_pos: batch of real data samples (positives)
    """
    # Generate samples
    e = randn([N, C])           # noise
    x = f(e)                    # generated samples
    y_neg = x                   # reuse as negatives

    # Compute drifting field
    V = compute_V(x, y_pos, y_neg)

    # Frozen target
    x_drifted = stopgrad(x + V)

    # MSE loss
    loss = mse_loss(x, x_drifted)

    return loss
```

---

## Kernel Design

### Kernel Function

The kernel measures similarity between samples:

```
k(x, y) = exp(-||x - y|| / τ)
```

Where:
- `τ` is a temperature parameter
- `||·||` is the ℓ₂ distance

### Softmax Normalization

In practice, the normalized kernel is implemented via softmax:

```
k̃(x, y) = softmax(-||x - y|| / τ)
```

The softmax is taken over `y` (the positive/negative samples). An additional softmax over `{x}` within the batch slightly improves performance.

### Multiple Temperatures

To improve robustness, multiple temperature values are used:

```
τ ∈ {0.02, 0.05, 0.2}
```

The drifts are computed for each temperature and aggregated:

```
Ṽ_j = Σ_τ Ṽ_{j,τ}
```

| Temperature | FID |
|-------------|-----|
| τ = 0.02 | 10.62 |
| τ = 0.05 | 8.67 |
| τ = 0.2 | 8.96 |
| {0.02, 0.05, 0.2} | **8.46** |

---

## Feature Space Drifting

### Why Feature Space?

Computing drifting loss directly in pixel/latent space doesn't work well on ImageNet. The kernel `k(·,·)` needs semantically meaningful distances — samples that are semantically similar should be close.

This aligns with self-supervised learning objectives (MoCo, SimCLR, MAE).

### Feature-Space Loss

```
ℒ = 𝔼[||φ(x) - stopgrad(φ(x) + V(φ(x)))||²]
```

Where:
- `x = f_θ(ε)` is the generated sample
- `φ` is a pretrained feature encoder
- Positives/negatives are also encoded: `φ(y⁺)`, `φ(y⁻)`

**Note:** Feature encoding is only used during training, not inference.

### Multi-Scale Features

For ResNet-style encoders, features are extracted at multiple scales and locations:

| Feature Type | Description |
|--------------|-------------|
| Per-location | H×W vectors from each spatial position |
| Global stats | 1 mean + 1 std per feature map |
| 2×2 pooled | Means and stds over 2×2 patches |
| 4×4 pooled | Means and stds over 4×4 patches |

Each feature gets its own drifting loss; all losses are summed.

| Features Used | FID |
|---------------|-----|
| (a, b) only | 9.58 |
| (a, b, c) | 9.10 |
| (a, b, c, d) | **8.46** |

### Relation to Perceptual Loss

| Perceptual Loss | Drifting Loss |
|-----------------|---------------|
| `||φ(x) - φ(x_target)||²` | `||φ(x) - (φ(x) + V(φ(x)))||²` |
| Requires paired samples | No pairing needed |
| Regresses to target features | Regresses to drifted features |

---

## Classifier-Free Guidance

### Training-Time CFG

Unlike diffusion models where CFG is applied at inference, Drifting Models implement CFG during training.

The negative sample distribution is modified:

```
q̃(·|c) = (1-γ)·q_θ(·|c) + γ·p_data(·|∅)
```

Where:
- `γ ∈ [0, 1)` is a mixing rate
- `p_data(·|∅)` is the unconditional data distribution
- `c` is the class label

### CFG Scale

The CFG scale `α` relates to `γ`:

```
α = 1/(1-γ) ≥ 1
```

This means:

```
q_θ(·|c) = α·p_data(·|c) - (α-1)·p_data(·|∅)
```

### Implementation

At training time:
1. Add `N_unc` unconditional samples (random classes) as extra negatives
2. Weight them by factor `w` in the kernel
3. Sample `α` randomly and condition the network on it

At inference time:
- Specify `α` as input
- Generation remains 1-NFE

---

## Implementation Details

### Generator Architecture

| Component | Specification |
|-----------|---------------|
| **Architecture** | DiT-style Transformer |
| **Normalization** | RMSNorm, QK-Norm |
| **Activation** | SwiGLU |
| **Position encoding** | RoPE |
| **Input** | `f(ε, c, α)` — noise, class, CFG scale |
| **Conditioning** | adaLN + 16 in-context tokens |

**Latent space (default):**
- Input/output: 32×32×4 (SD-VAE latent)
- Patch size: 2×2 → 256 tokens

**Pixel space:**
- Input/output: 256×256×3
- Patch size: 16×16 → 256 tokens

### Random Style Embeddings

Inspired by StyleGAN, additional randomness beyond Gaussian noise:

- 32 "style tokens"
- Each is a random index into a 64-entry learnable codebook
- Summed and added to conditioning vector

| Setting | FID |
|---------|-----|
| Without style | 8.86 |
| With style | **8.46** |

### Custom Latent-MAE Feature Encoder

A ResNet-style MAE trained directly in latent space:

**Encoder:**
- Classical ResNet with GroupNorm
- 4 stages: [3, 4, 6, 3] basic blocks
- Multi-scale outputs: 32², 16², 8², 4² resolutions
- Channel widths: C, 2C, 4C, 8C

**Decoder:**
- U-Net style with skip connections
- Bilinear upsampling + concatenation

**Training:**
- 50% random masking of 2×2 patches (zeroed)
- ℓ₂ reconstruction loss on masked regions
- Optional classification fine-tuning

### Feature Encoder Comparison

| Encoder | SSL Method | Width | Epochs | FID |
|---------|------------|-------|--------|-----|
| ResNet-50 | SimCLR | 256 | 800 | 11.05 |
| ResNet-50 | MoCo-v2 | 256 | 800 | 8.41 |
| ResNet | latent-MAE | 256 | 192 | 8.46 |
| ResNet | latent-MAE | 640 | 1280 | 4.28 |
| ResNet | latent-MAE + cls ft | 640 | 1280 | **3.36** |

### Normalization (Critical)

**Feature Normalization:**

```
φ̃ = φ / S
```

Where `S` is set so the average pairwise distance equals `√C`:

```
S = (1/√C) · 𝔼_{x,y}[||φ(x) - φ(y)||]
```

**Drift Normalization:**

```
Ṽ = V / λ
```

Where `λ` normalizes the expected squared drift:

```
λ = √(𝔼[||V||² / C])
```

These normalizations make the method robust across different feature encoders and scales.

### Batching

- `N_c`: Number of class labels per batch
- `N_pos`: Positive samples per class
- `N_neg`: Negative samples (generated) per class
- `N_unc`: Unconditional samples for CFG

Effective batch size: `B = N_c × N_neg`

| N_c | N_pos | N_neg | B | FID |
|-----|-------|-------|------|-----|
| 64 | 1 | 64 | 4096 | 20.43 |
| 64 | 64 | 64 | 4096 | **8.46** |

### Sample Queue (MoCo-style)

- Per-class queue: size 128, push 64 new samples per step
- Global unconditional queue: size 1000
- Sampling without replacement

### Training Loop

```python
for step in training:
    # 1. Sample class labels
    classes = sample_classes(N_c)

    # 2. Sample CFG scale α for each class
    alphas = sample_cfg_scales(N_c)

    # 3. Generate samples
    noise = randn(N_neg, C)
    x = generator(noise, classes, alphas)

    # 4. Get positive and unconditional samples from queue
    y_pos = sample_positives(classes, N_pos)
    y_unc = sample_unconditional(N_unc)

    # 5. Extract multi-scale features
    features_x = encoder(x)
    features_pos = encoder(y_pos)
    features_unc = encoder(y_unc)

    # 6. Compute drifting loss (sum over all scales/locations)
    loss = sum(
        drifting_loss(f_x, f_pos, f_neg, f_unc, alpha)
        for f_x, f_pos, f_neg, f_unc in zip_features(...)
    )

    # 7. Backprop and update
    loss.backward()
    optimizer.step()
```

---

## Results

### ImageNet 256×256 — Latent Space

| Method | Params | NFE | FID ↓ | IS ↑ |
|--------|--------|-----|-------|------|
| **Multi-step methods** |
| DiT-XL/2 | 675M+49M | 250×2 | 2.27 | 278.2 |
| SiT-XL/2 | 675M+49M | 250×2 | 2.06 | 270.3 |
| SiT-XL/2+REPA | 675M+49M | 250×2 | 1.42 | 305.7 |
| **Single-step methods** |
| iCT-XL/2 | 675M | 1 | 34.24 | – |
| MeanFlow-XL/2 | 676M | 1 | 3.43 | – |
| iMeanFlow-XL/2 | 610M | 1 | 1.72 | 282.0 |
| **Drifting Model, B/2** | **133M** | **1** | **1.75** | 263.2 |
| **Drifting Model, L/2** | **463M** | **1** | **1.54** | 258.9 |

**Key observations:**
- Best 1-NFE FID: **1.54** (new state-of-the-art for single-step)
- Base-size model (133M) competes with XL-size models (675M)
- Competitive with multi-step methods despite 1-step inference

### ImageNet 256×256 — Pixel Space

| Method | Params | NFE | FID ↓ | IS ↑ |
|--------|--------|-----|-------|------|
| **Multi-step methods** |
| ADM-G | 554M | 250×2 | 4.59 | 186.7 |
| VDM++, UViT/2 | 2.5B | 256×2 | 2.12 | 267.7 |
| PixelDiT/16 | 797M | 200×2 | 1.61 | 292.7 |
| **GANs** |
| StyleGAN-XL | 166M | 1 | 2.30 | 265.1 |
| **Drifting Model, B/16** | **134M** | **1** | **1.76** | 299.7 |
| **Drifting Model, L/16** | **464M** | **1** | **1.61** | 307.5 |

**Key observations:**
- Pixel-space FID **1.61** matches or beats multi-step methods
- Outperforms StyleGAN-XL (2.30) with fewer FLOPs (87G vs 1574G)

### Robotics Control

Replacing Diffusion Policy's 100-step generator with 1-step Drifting Model:

| Task | Diffusion Policy (100 NFE) | Drifting Policy (1 NFE) |
|------|---------------------------|------------------------|
| Lift (State) | 0.98 | **1.00** |
| Lift (Visual) | 1.00 | 1.00 |
| Can (State) | 0.96 | **0.98** |
| ToolHang (State) | 0.30 | **0.38** |
| BlockPush Phase 2 | 0.11 | **0.16** |

The method generalizes beyond image generation.

### Ablation: Anti-Symmetry is Critical

| Drifting Field | FID |
|----------------|-----|
| V⁺ - V⁻ (anti-symmetric, default) | **8.46** |
| 1.5·V⁺ - V⁻ | 41.05 |
| V⁺ - 1.5·V⁻ | 46.28 |
| 2·V⁺ - V⁻ | 86.16 |
| V⁺ only | 177.14 |

Breaking anti-symmetry causes catastrophic failure.

---

## Limitations and Open Questions

### Theoretical Gaps

1. **Converse implication not proven:** The paper shows `q = p ⟹ V = 0`, but `V = 0 ⟹ q = p` is not guaranteed in theory. It works empirically, but the theoretical justification is incomplete.

2. **Identifiability:** They provide a heuristic argument that zero-drift imposes enough constraints to force distribution matching, but this isn't rigorous.

### Practical Limitations

1. **Requires pretrained feature encoder:** The method doesn't work on raw ImageNet pixels without SSL features. The kernel needs semantically meaningful distances.

2. **Feature encoder quality matters:** Generation quality correlates strongly with feature encoder quality. Poor encoders lead to "flat" kernels where all samples appear far apart.

### Open Design Questions

- Optimal drifting field design beyond attraction-repulsion?
- Better kernel functions?
- Can the feature encoder be learned jointly?
- Optimal generator architecture?

---

## Key Takeaways

1. **Paradigm shift:** Move iteration from inference-time to training-time
2. **Anti-symmetry:** The mathematical property that guarantees equilibrium
3. **Attraction-repulsion:** Physical intuition for how samples evolve
4. **Feature space:** Essential for high-dimensional generation
5. **Multi-scale, multi-temperature:** Robust practical design choices
6. **1-NFE generation:** Enables fast, high-quality inference

---

## Citation

```bibtex
@article{deng2026drifting,
  title={Generative Modeling via Drifting},
  author={Deng, Mingyang and Li, He and Li, Tianhong and Du, Yilun and He, Kaiming},
  journal={arXiv preprint arXiv:2602.04770},
  year={2026}
}
```
