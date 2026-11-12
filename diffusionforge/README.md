# DiffusionForge — Diffusion Models From Scratch (DDPM + DDIM + Classifier-Free Guidance)

**Author:** Ashutosh Sahoo — M.Tech (Artificial Intelligence), NIT Silchar

A complete, self-contained implementation of modern diffusion models in pure
PyTorch — no `diffusers`, no pretrained weights, no dataset downloads. The same
mathematics that powers Stable Diffusion and DALL·E-class systems, implemented
line-by-line so every mechanism is visible: the forward noising process, the
noise-prediction objective, ancestral DDPM sampling, fast DDIM sampling, and
classifier-free guidance.

> **Honesty note:** for a zero-dependency, CPU-runnable demo, the model is
> trained on a *synthetic* 16×16 shape dataset generated in code (circles,
> squares, triangles, crosses with position/scale jitter). The point of this
> project is a correct, verified implementation of the diffusion machinery —
> the scale-up path to MNIST/CIFAR-10 on a free Kaggle GPU is documented below
> and requires no code changes beyond the dataloader.

---

## Results (measured, from `results.json`)

| Metric | Value |
|---|---|
| Model parameters | 179,281 |
| Diffusion timesteps (T) | 300 (cosine schedule) |
| Training epochs / time | 90 / 1040.8s on CPU |
| Final train loss (noise MSE) | 0.0613 |
| Held-out noise-prediction MSE | 0.0666 |
| Fidelity classifier accuracy on real test data | 99.75% |
| **DDPM (full 300 steps) — class fidelity** | **98.44%** |
| **DDIM (50 steps) — class fidelity** | **85.94%** |
| Sampling speed, 64 images — DDPM vs DDIM-50 | 87.4s vs 13.4s |
| Diversity (mean pairwise L2) — DDPM / DDIM / real data | 10.76 / 10.30 / 11.60 |
| CFG ablation fidelity — w=0 / w=1.5 / w=3.0 | 28.12% / 85.94% / 54.69% |

*Class fidelity* = a small CNN classifier is trained **only on real data**;
fidelity is the % of generated samples it assigns to the class the diffusion
model was conditioned on. Guidance (w) trades diversity for class fidelity —
w=0 is unconditional generation, so its "fidelity" sits near chance by design.

Sample grids: `samples_ddpm.png`, `samples_ddim50.png`, `samples_cfg_w0.0.png`,
`samples_cfg_w1.5.png`, `samples_cfg_w3.0.png` (rows cycle circle / square /
triangle / cross).

---

## The theory in brief

**Forward process.** A fixed Markov chain gradually destroys an image with
Gaussian noise over T steps:

```
q(x_t | x_{t-1}) = N( sqrt(1-β_t) x_{t-1},  β_t I )
q(x_t | x_0)     = N( sqrt(ᾱ_t) x_0,  (1-ᾱ_t) I )      ← closed form, one-shot noising
```

**Reverse process.** A U-Net `ε_θ(x_t, t, class)` learns to predict the noise
that was added. Training minimises the simplified ELBO objective, which reduces
to plain MSE on the noise:

```
L = E_{x_0, ε, t} || ε − ε_θ( sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε,  t ) ||²
```

**DDPM sampling** walks the learned reverse chain for all T steps (ancestral
sampling). **DDIM** reparameterises the reverse process as non-Markovian and
deterministic (η=0), so it can skip most steps — 50 steps here instead of
300 — at nearly the same quality (measured above).

**Classifier-free guidance** trains one network both conditionally and
unconditionally (the class label is dropped to a learned NULL embedding 10% of
the time). At sampling, the two predictions are extrapolated:

```
ε_guided = ε_uncond + w · (ε_cond − ε_uncond)      w=0: unconditional · w=1: conditional · w>1: amplified
```

No separate classifier is needed — the guidance comes from the model itself.

## Architecture

```
x_t (1×16×16) ─► conv_in ─► ResBlock(16) ────────────────┐ skip
                              │ downsample                │
                              ResBlock(32) @8×8 ──────┐   │
                              │ downsample            │   │
                              ResBlock(32) @4×4 (mid) │   │
                              upsample ─► ResBlock(concat skip) @8×8
                              upsample ─► ResBlock(concat skip) @16×16 ─► ε̂ (1×16×16)

t ─► sinusoidal embedding (128-d) ─► 2-layer MLP ─┐
class ─► learned embedding (incl. NULL) ─────────┴─► added inside every ResBlock
```

- Residual blocks: GroupNorm → SiLU → Conv, conditioning injection, Conv, skip.
- Cosine noise schedule (Nichol & Dhariwal) — smoother signal destruction than
  the original linear schedule; both are implemented in `diffusion.py`.
- EMA of weights (decay 0.99) used for sampling — a small trick with an
  outsized quality effect. The decay must match the run length: at 0.999 over
  a ~600-step run, the "EMA model" is still mostly the random init (see
  Engineering notes).

## Engineering notes — two real bugs, found by measurement

1. **EMA decay vs run length.** The first sampling run produced pure noise
   despite a healthy training loss (0.096). Cause: EMA decay 0.999 over only
   ~280 optimizer steps leaves the EMA weights ≈75% random initialisation.
   Fix: decay 0.99 for this ~600-step run (a 0.999 decay belongs to ~10k-step
   runs). Verified by sampling raw vs EMA weights side by side.
2. **DDPM drift without x₀ clamping.** With the textbook ε-parameterised
   mean, DDPM fidelity measured 28.1% (chance) while DDIM — which clamps the
   predicted x₀ at every step — measured 85.9%. Rewriting the DDPM step in
   posterior-mean form `q(x_{t-1}|x_t, x̂_0)` with x̂₀ clamped to [-1, 1]
   raised DDPM fidelity to **98.44%**. Both numbers are from the same
   checkpoint and classifier; only the sampler changed.
3. **Over-guidance is real.** The CFG sweep (DDIM) peaks at w=1.5 (85.94%);
   w=3.0 *drops* to 54.69% — extrapolated ε-predictions push samples
   off-manifold with a deterministic sampler. Guidance is a trade-off knob,
   not a "more is better" knob.

## SOTA lineage (context)

| Year | Method | Contribution |
|---|---|---|
| 2020 | DDPM (Ho et al.) | Noise-prediction objective, ancestral sampling |
| 2020 | DDIM (Song et al.) | Deterministic, 10–50× fewer sampling steps |
| 2021 | Improved DDPM (Nichol & Dhariwal) | Cosine schedule, learned variance |
| 2022 | Classifier-Free Guidance (Ho & Salimans) | Guidance without a classifier — the default in text-to-image |
| 2022 | Latent Diffusion (Rombach et al.) | Diffusion in a VAE latent space → Stable Diffusion |

This repo implements the first four at educational scale, from scratch.

## Run it

```bash
pip install -r requirements.txt
python data_gen.py     # generates + inspects the synthetic dataset
python train.py        # trains the DDPM (CPU), writes checkpoint.pt
python sample.py       # DDPM / DDIM / CFG sample grids (PNG)
python evaluate.py     # all metrics above -> results.json
```

## Scale-up path (real data, free GPU)

1. Kaggle notebook, GPU on. Swap `data_gen.py`'s loader for MNIST (28×28,
   pad to 32×32, base channels 32→64) or CIFAR-10 (32×32, 3 channels —
   `conv_in`/`out` take the channel count).
2. T=1000, batch 128, ~50k steps with EMA — a single free T4 session.
3. Next rung: latent diffusion — train a small VAE, run this exact
   `Diffusion` class in its latent space, add cross-attention text
   conditioning in place of the class embedding.

## Skills demonstrated

Diffusion probabilistic models · ELBO / variational objectives · U-Net
architecture design · sinusoidal & learned embeddings · EMA training ·
conditional generation & guidance · sampling-algorithm trade-offs
(quality vs steps) · evaluation design (classifier-based fidelity, diversity)
· PyTorch from-scratch implementation discipline.

## Limitations

- Synthetic 16×16 shapes, not photographs — chosen so the full pipeline runs
  and verifies on CPU with no downloads; fidelity/diversity numbers describe
  this dataset, not ImageNet-scale generation.
- Pixel-space diffusion at toy resolution; production systems diffuse in
  latent space with text cross-attention and billions of parameters.
- The fidelity classifier is itself a small CNN — a proxy metric (the
  large-scale analogue is FID/CLIP-score, which need pretrained backbones).

## Files

| File | Contents |
|---|---|
| `diffusion.py` | Schedules, forward process, DDPM + DDIM samplers, CFG |
| `model.py` | Conditional U-Net (time + class embeddings, residual blocks) |
| `data_gen.py` | Synthetic shape dataset (seeded, no downloads) |
| `train.py` | Noise-prediction training with CFG dropout + EMA |
| `sample.py` | Sample grids: DDPM vs DDIM vs guidance sweep |
| `evaluate.py` | Fidelity / diversity / speed / ablation metrics → `results.json` |
