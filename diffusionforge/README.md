# DiffusionForge — Diffusion Models From Scratch (DDPM + DDIM + Classifier-Free Guidance)

**Author:** Ashutosh Sahoo — M.Tech (Artificial Intelligence), NIT Silchar

A complete, self-contained implementation of modern diffusion models in pure
PyTorch — no `diffusers`, no pretrained weights, no dataset downloads. The same
mathematics that powers Stable Diffusion and DALL·E-class systems, implemented
line-by-line so every mechanism is visible: the forward noising process, the
noise-prediction objective, ancestral DDPM sampling, fast DDIM sampling, and
classifier-free guidance.

> **Honesty note:** the original run of this project was trained on a
> *synthetic* 16×16 shape dataset generated in code (circles, squares,
> triangles, crosses with position/scale jitter) — those results are kept
> below, demoted to **preliminary**. The headline results are now the
> **real-benchmark run on MNIST** in the next section, which reuses the
> identical architecture and evaluation method with only the data changed.

---

## Real benchmark results (MNIST-16) — measured, from `results_real.json`

> **⚠ 16×16-downscale caveat (read first):** these results are **not** on
> full-resolution MNIST. Every 28×28 MNIST image was downscaled to **16×16**
> (bilinear interpolation with antialiasing, then scaled to [−1, 1]) so the
> existing 16×16 U-Net applies unchanged. Fidelity/diversity numbers describe
> generation at 16×16 and are **not comparable** to published 28×28 MNIST
> diffusion results (which usually report FID, a different metric — no FID
> is claimed here).

**Dataset.** MNIST (Yann LeCun, Corinna Cortes, Christopher J.C. Burges),
downloaded from the TensorFlow/Keras mirror
`https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz`
(SHA-256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`).
Verified on load: **60,000 train / 10,000 test** images, labels 0–9 with
sane per-class counts (≈5.4k–6.7k train per digit). Diffusion training used
a documented 20,000-image subset of the train split; a separate 2,000-image
validation set was carved from the train split (seed-2024 permutation) and
**never used to train any diffusion model** — it drove all Phase B
checkpoint/config selection (validation noise-MSE + a DDIM-50 fidelity
proxy). The test split was evaluated exactly twice: the baseline config
once, and the final selected config once.

**Evaluator quality gate (checked first).** Fidelity is only as meaningful
as the classifier judging it. The fidelity classifier (the original
TinyClassifier, output layer widened 4→10) was trained on the full real
MNIST-16 train split (60,000 images, 5 epochs) and scores **98.13% accuracy
on the held-out real MNIST-16 test split** — **PASS** against the ≥98% gate,
so the fidelity numbers below are trustworthy to within ~2 points.

### Baseline real result (original project config; only the data changed)

20,000-image subset · 12 epochs · 1,182s CPU · EMA 0.99 · cosine schedule ·
batch 256 · lr 2e-3 · 180,049 params · final train loss 0.0613 · held-out
test noise-MSE 0.0616 (raw) / 0.0617 (EMA) · 70 generated samples per
sampler (7/class × 10 classes; the synthetic protocol's 64 = 16/class × 4):

| Metric | Baseline (MNIST-16) |
|---|---|
| **DDPM (300 steps, w=1.5) — class fidelity** | **98.57%** |
| **DDIM (50 steps, w=1.5) — class fidelity** | **77.14%** |
| Diversity (pairwise L2) — DDPM / DDIM / real data | 9.333 / 9.453 / 8.977 |
| Sampling time for 70 — DDPM vs DDIM-50 | 51.9s vs 11.2s (shared-CPU wall time, varies) |
| CFG ablation (DDIM) fidelity — w=0 / w=1.5 / w=3.0 | 8.57% / 77.14% / 34.29% |

w=0 is unconditional generation, so its fidelity sits at chance (~10%) by
design; w=3.0 over-guides and collapses fidelity, as on synthetic data.

### Improvement phase (Phase B) — every attempt screened on validation only

Selection rule (pre-committed): highest validation DDIM-50 fidelity proxy
at w=1.5 (100 generated samples), tie-break lower validation noise-MSE
(EMA). Failures are logged, not hidden:

| Attempt | Change | Val proxy fidelity | Val noise-MSE | Verdict |
|---|---|---|---|---|
| baseline | — (reference) | 81.0% | 0.06214 | reference |
| B1 | EMA decay 0.99 → 0.999 | 11.0% | 0.14755 | **drop** — at only ~950 steps the 0.999 EMA still lags the live weights (the brief's "~1–2k steps to track" is beyond this run) |
| B2 | +12 epochs **with** ±2px translation augmentation | 55.0% | 0.06452 | **drop** — augmentation confounded; see B6 |
| B3 | +6 epochs on expanded 40k data (still augmented) | 58.0% | 0.06457 | **drop** — more data did not rescue the augmented runs |
| B4 | linear beta schedule instead of cosine (fresh run) | 76.0% | 0.06256 | **drop** — close on MSE, below on fidelity; cosine kept |
| B6 | +12 epochs, **no augmentation** (isolates B2's longer-training lever) | **87.0%** | **0.05811** | **keep — selected as final** |

Guidance was then re-swept **on the validation proxy** for the selected
checkpoint (w = 0 / 1 / 1.5 / 2 / 3 → 10.0 / 76.0 / **87.0** / 67.0 / 31.0),
selecting **w = 1.5** — the same optimum as the baseline and the synthetic
run. The pattern across B2/B3 vs B6 points at the translation augmentation
(wrap-around `torch.roll` shifts) as the harmful ingredient at this scale,
not longer training or more data per se; that attribution is a hypothesis
from the ablation pattern, not an isolated measurement.

### FINAL best real result (selected on validation, evaluated on test once)

B6 = baseline + 12 further epochs (24 total, second half at lr 1e-3),
20,000-image subset, EMA 0.99, cosine schedule, no augmentation.
Cumulative training time ≈ 1,672s CPU. Test noise-MSE **0.0576** (raw) /
0.0577 (EMA).

| Metric | FINAL (MNIST-16) |
|---|---|
| **DDPM (300 steps, w=1.5) — class fidelity** | **100.0%** (70/70 samples) |
| **DDIM (50 steps, w=1.5) — class fidelity** | **84.29%** |
| Diversity (pairwise L2) — DDPM / DDIM / real data | 9.364 / 9.525 / 8.977 |
| Sampling time for 70 — DDPM vs DDIM-50 | 17.1s vs 1.9s (shared-CPU wall time; the baseline evaluation of a comparable config measured 51.9s / 11.2s under heavier contention — timings here are indicative, fidelity/diversity are the stable numbers) |

**Memorisation check (because 100% invites the question):** nearest-neighbour
L2 distance from the final DDPM samples to the 20,000 training images —
mean **3.317**, minimum **1.092**, **0%** of samples closer than 1.0. Real
test images sit at mean 3.233 with 5.6% closer than 1.0. The generated
samples are at normal MNIST inter-image distances — novel digits, not
training-set copies; the 100% is classifier agreement on an easy task
(judged by a 98.13%-accurate classifier), on a 70-sample set.

Sample grids (columns = digit class 0–9): `samples_real_final_ddpm.png`,
`samples_real_final_ddim50.png`, `samples_real_baseline_ddpm.png`,
`samples_real_baseline_ddim50.png`. Reproduce with
`python run_all_real.py` (full pipeline incl. Phase B) and
`python run_b6.py` (the B6 attempt + conditional final re-selection);
`data_real.py` / `classifier_real.py` / `train_real.py` / `evaluate_real.py`
are the pieces. Raw numbers: `results_real.json`.

---

## Preliminary results — synthetic shapes dataset (measured, from `results.json`)

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

Real-benchmark (MNIST-16) run:

```bash
curl -O https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
python data_real.py    # verifies MNIST (60k/10k), builds the 16x16 cache + splits
python run_all_real.py # classifier gate -> baseline -> test eval -> Phase B attempts -> final test eval -> results_real.json
python run_b6.py       # Phase B attempt B6 (clean longer training) + conditional final re-selection
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

- Two datasets, two honesty labels: the **synthetic** 16×16 shapes results
  are preliminary (kept for the implementation-verification story); the
  **MNIST-16** results are real benchmark data but at 16×16 downscale and
  toy scale — fidelity/diversity numbers describe these datasets, not
  ImageNet-scale generation, and are not FID-comparable to 28×28 literature.
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
| `data_real.py` | MNIST loader: verification, 28×28 → 16×16 downscale, train/val/extra splits |
| `classifier_real.py` | Fidelity classifier (10-class) + real-test accuracy gate |
| `train_real.py` | Diffusion training on MNIST-16 with per-epoch validation noise-MSE |
| `evaluate_real.py` | Validation proxy screening + the test-side protocol for MNIST-16 |
| `run_all_real.py` / `run_b6.py` | Full real-data pipeline / Phase B attempt B6 |
| `results_real.json` | All real-benchmark numbers, attempt log, selection record |
