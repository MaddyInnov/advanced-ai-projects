# DenseHAR — Transition-Aware Dense Human Activity Recognition

**One label for every timestep, not one label per window — and honest measurement of what happens at activity transitions.**

DenseHAR extends the author's **InceptionKAN** work (Sahoo & Ramanujam E, M.Tech, NIT Silchar — multi-scale 1D convolutions k=1,3,5,11 + SE attention, ~238K params, 97.25% UCI-HAR / 98.45% PAMAP2 / 95.54% WISDM v2, windowed classification) to the problem that work left open: **real activity streams are not single-activity windows.** People stand up, start walking, climb stairs. A fixed-window classifier must assign one label to a window that straddles a change — so it is structurally wrong around every transition, exactly where continuous monitoring (fall detection, rehab, elder care) cares most.

> **Honesty box — read first.** DenseHAR has now been run on **real data** (HAPT, § below — those are the headline numbers). The synthetic results further down are **real, measured runs on synthetic data** (`results.json`, reproducible: `python data_gen.py && python train.py`) and are kept as the preliminary pipeline proof they always were — they are **not** real-benchmark results and **no SOTA claim is made on UCI-HAR / HAPT / PAMAP2**. What *is* state-of-the-art here is the **methodology**: subject-disjoint tuning + test splits, a hyperparameter sweep, an ablation, MS-TCN-style segmental metrics (F1@10/25/50, edit score) alongside frame metrics, and a windowed baseline evaluated densely so the transition collapse is quantified, not asserted.

## Real-data results (HAPT) — the headline

Real smartphone IMU data (HAPT, UCI id 341; 30 subjects, acc+gyro @ 50 Hz, per-timestep labels incl. 6 postural-transition classes). **Fixed subject-disjoint split:** train 21 / val 4 / **test 5 unseen subjects** (2, 4, 9, 10, 12); all selection on val; test evaluated once per final candidate. Full report: [`REAL_DATA_RESULTS.md`](REAL_DATA_RESULTS.md) · all 22 logged experiments: `results_real.json`.

| Metric (test, unseen subjects) | **DenseHAR final system**¹ | DenseHAR + refinement² | Phase-1 config A | Window-128 baseline (broadcast) |
|---|---|---|---|---|
| Per-timestep accuracy | **90.38%** | 89.90% | 78.86% | 78.91% |
| Macro F1 | 0.9398 | **0.9415** | 0.7253 | 0.7258 |
| **Transition-window accuracy (±10 steps)** | **0.6093** | **0.6246** | 0.5955 | 0.4787 |
| Accuracy on transition-class timesteps | 0.8744 | **0.8946** | 0.8571 | 0.7730 |
| Boundary F1 (±10) | 0.409 | **0.505** | 0.135 | 0.087 |
| Segmental F1@10 / @25 / @50 | 0.836 / 0.822 / 0.788 | **0.858 / 0.840 / 0.819** | 0.410 / 0.388 / 0.355 | 0.775 / 0.749 / 0.693 |
| Edit score | 84.88 | **86.23** | 63.42 | 75.52 |

¹ Physics augmentation + 12-class supervision + 3-seed ensemble + val-tuned post-processing (val-selected headline, val F1 0.9675). ² Co-leading candidate: single seed + MS-TCN refinement stage + post-processing — best test F1/transition/segmental scores, ranked 4th on val.

**The honest arc:** Phase 1 (config A as-is) only *tied* the baseline on real data (78.9% vs 78.9%) while over-segmenting badly (boundary precision 0.078). Phase 2 — every lever screened on validation only, guided by [`IMPROVEMENT_RESEARCH.md`](IMPROVEMENT_RESEARCH.md) — found the fixes in order of impact: **physics augmentation** (joint acc+gyro rotation, magnitude/time warping; val F1 0.919→0.958), **12-class transition supervision** (→0.965), **post-processing** (segmental F1@25 0.64→0.87 at zero retraining), **refinement stage**, **seed ensembling**. Negative results kept: median-frequency class weights *hurt* (val F1 0.897); boundary-weighted CE gave nothing over plain CE on real data. Remaining weakness, unchanged in kind: sitting vs standing (F1 ≈ 0.82–0.85).

## Preliminary pipeline proof (synthetic, unseen test subjects)

| Metric (test, subjects 12–15, unseen) | **DenseHAR (final)** | Window-128 baseline (broadcast) |
|---|---|---|
| Per-timestep accuracy | **95.35%** | 85.71% |
| Macro F1 | **0.9545** | 0.8573 |
| **Transition-window accuracy (±10 steps)** | **90.63%** | **48.03%** |
| Boundary F1 (±10 tolerance) | **0.525** | 0.174 |
| Boundary precision / recall | 0.360 / 0.974 | 0.196 / 0.156 |
| Segmental F1@10 / @25 / @50 | 0.621 / 0.607 / 0.572 | **0.899 / 0.892 / 0.855** |
| Edit score | 74.50 | **85.92** |
| Parameters | 68,898 | 40,674 |

**The paper story, quantified:** the windowed baseline looks respectable overall (85.71%) but **collapses to 48.03% within ±10 timesteps of a true transition** (−37.7 points). DenseHAR drops only 95.35% → 90.63% (−4.7 points).

**The honest counter-result:** the baseline *wins* segmental F1 and edit score. That is a known property of these metrics, not a contradiction: segmental F1/edit reward segment-count parsimony and label-order correctness. The broadcast baseline can emit at most 4 segments per sequence and usually gets the label *order* right, while DenseHAR over-segments on unseen subjects (boundary recall 0.974 but precision 0.360 — it finds nearly every true boundary, plus spurious ones). Reporting only frame accuracy would hide this; reporting only segmental scores would hide the transition collapse. We report all of them — that is the point of the SOTA metric suite.

Demo (one test sequence, unseen subject 13):

```
TRUE: [0-171] walking_upstairs -> [172-319] standing -> [320-511] laying
PRED: [0-173] walking_upstairs -> [174-319] standing -> [320-511] laying   (99.6% per-timestep)
```

## Method

```
6ch IMU (acc xyz + gyro xyz), 512 steps @ 50 Hz
        │
 Stem Conv1d k=5, stride 2  ── half-rate encoder (2× receptive field, ½ compute)
        │
 InceptionBlock ×2: parallel Conv1d k ∈ {1,3,5,11} (from InceptionKAN) + SE attention + residual, GroupNorm
        │
 Dilated temporal stack: Conv1d k=3, dilations 1,2,4,8 (MS-TCN-style context)
        │
 Linear upsample ×2 → per-timestep head Conv1d 1×1 → (batch, 512, 6)
```

Design choices, each deliberate:
- **GroupNorm, not BatchNorm** — batch statistics are subject statistics; GroupNorm generalises better to unseen wearers (this change plus the wider receptive field took validation accuracy from ~53% in an early prototype to 99.3%).
- **Half-rate encoder** — stride-2 stem, upsampled head: boundary quantisation is 2 samples, far inside the ±10 evaluation tolerance.
- **Transition-aware training (the ablation-tested package):**
  - *Boundary-weighted cross-entropy:* weight `1 + 3·exp(−dist_to_boundary/10)` per timestep.
  - *MS-TCN-style smoothing loss* (λ=0.15): truncated MSE between adjacent log-probabilities, penalising over-segmentation.

## SOTA protocol & sweep (validation = held-out subjects 10–11)

| Config | Width | Transition-aware | Val acc | Val macro F1 | Val seg F1@25 | Val edit | Val boundary F1 |
|---|---|---|---|---|---|---|---|
| **A (selected)** | 48 | ✅ bw + smoothing | **99.32%** | **0.9928** | **0.882** | **85.2** | **0.849** |
| B | 48 | ❌ plain CE | 99.13% | 0.9907 | 0.857 | 81.6 | 0.814 |
| C | 32 | ✅ bw + smoothing | 99.16% | 0.9910 | 0.840 | 77.7 | 0.782 |

Frame accuracy saturates and barely separates the configs — **segmental metrics do**, which is why the field uses them. The best config was retrained on all 300 training sequences (10 epochs, cosine LR; training loss plateaued 0.058→0.055 over the last epochs).

**Ablation on test** (tuning models, identical 6-epoch sub-train budget): boundary+smoothing vs plain — accuracy 94.21% vs 93.99%, transition accuracy **86.52% vs 84.48%**, seg F1@25 0.533 vs 0.529, edit 65.7 vs 64.6. Small, consistent, and in the expected direction; the full-data final model improves every one of these (above).

## SOTA context (published reference — for context only, NOT comparable)

| Work | Task | Number |
|---|---|---|
| InceptionKAN (author's paper) | Windowed classification, UCI-HAR | 97.25% acc |
| InceptionKAN | Windowed, PAMAP2 | 98.45% acc |
| InceptionKAN | Windowed, WISDM v2 | 95.54% acc |
| MS-TCN (Farha & Gall, CVPR 2019) | Dense segmentation, 50 Salads etc. | segmental F1/edit protocol adopted here |
| **DenseHAR (this repo)** | **Dense, real HAPT, unseen subjects** | **90.38% frame acc / 0.9398 macro F1 — subject-disjoint, this protocol** |
| **DenseHAR (this repo)** | **Dense, synthetic, unseen subjects** | **95.35% frame acc — synthetic only, not a benchmark claim** |

Dense/transition results are **not directly comparable** to windowed accuracy: a windowed model is scored only where it was designed to work. Scored densely (the deployment reality), the windowed baseline here loses 37.7 points at transitions.

## Datasets

**Real (headline): HAPT** — see the real-data section above and `REAL_DATA_RESULTS.md`. `load_hapt.py` builds `data_hapt.npz` from the UCI HAPT zip: 512-step windows (stride 256) over per-timestep labelled streams, transition classes kept both as 6-class target-mapped labels (`y6`) and as an evaluation mask (`y_trans`), fixed subject-disjoint split 21/4/5, normalisation from train streams only. 862 train / 145 val / 177 test sequences.

**Synthetic (preliminary, clearly labelled):** `data_gen.py` (fixed seed): 6 activities (walking, walking_upstairs, walking_downstairs, sitting, standing, laying) × 6 channels with activity-specific frequency / amplitude / gravity-orientation / noise profiles; per-subject amplitude, frequency, noise and offset profiles; sequences of 512 steps containing 2–4 segments with smoothed boundaries. **Split in the leave-one-subject-out spirit:** train subjects 0–11 (300 seqs; 10–11 reserved as validation during tuning), test subjects 12–15 (100 seqs) — **no test subject appears in training**.

## Run it

```bash
# Real data (headline)
python load_hapt.py    # HAPT zip -> data_hapt.npz
python train_real.py phase1   # config A + baseline on HAPT -> results_real.json
# Synthetic (preliminary pipeline proof)
python data_gen.py     # -> data/train.npz, data/test.npz
python train.py        # tuning sweep + final + baseline -> results.json, densehar.pt
python demo.py         # true vs predicted timeline for one test sequence
python export_onnx.py  # ONNX for mobile (needs `onnx` package; writes a note if unavailable)
```

Pure PyTorch + NumPy, CPU-only, no scikit-learn (macro F1, segmental F1, edit distance implemented in `metrics.py` and unit-checked: perfect prediction → F1 1.0 / edit 100).

## Limitations

- **Real-data run (HAPT):** small subject counts (4 val / 5 test) and large seed variance (val F1 0.932–0.965 across seeds of the winning config) — numbers are for the documented fixed split. The val subjects proved much easier than the test subjects for every model, so fine distinctions between top candidates are provisional. HAPT windows contain at most 2 basic activities (single-bout recordings); long multi-activity streams remain untested on real data. Sitting vs standing (F1 ≈ 0.82–0.85) is the residual error core. Full discussion in `REAL_DATA_RESULTS.md` §8.
- **Synthetic data.** Profiles are stylised; real IMU noise, device placement and activity confusability are harder — confirmed by the real-data run (Phase-1 config A: 78.9% on HAPT vs 95.35% synthetic). Unseen-subject generalisation is the open problem, as in the InceptionKAN LOSO results; physics augmentation is the current best mitigation.
- **Over-segmentation on unseen subjects** was the main synthetic failure mode (boundary precision 0.360) and was far worse in the real Phase-1 run (0.078); the Phase-2 package (augmentation + refinement + post-processing) raised test boundary F1 to 0.41–0.50.
- ONNX export in the reference environment lacked the `onnx` package — `export_onnx.py` catches this and writes `ONNX_EXPORT_NOTE.md`; see `MOBILE.md` for the full deployment path.

## Next steps

1. ~~**Real data:** HAPT~~ — **done** (headline section above; `REAL_DATA_RESULTS.md`). UCI-HAR dense labels can still be reconstructed from the raw inertial signals as a second real benchmark. Original Kaggle plan in [`REAL_DATA_KAGGLE.md`](REAL_DATA_KAGGLE.md).
2. Multi-stage refinement (MS-TCN++ style) to attack over-segmentation — first stage done in Phase 2 (single refinement stage); per-subject head-only personalisation on-device (see `MOBILE.md`).
3. More sensors (magnetometer, barometer) and variable-length streaming inference; a sitting/standing-specific attack (gravity-direction features helped val but did not beat augmentation — worth one more A/B at larger scale).

## Files

`data_gen.py` · `model.py` (DenseHARNet, 68,898 params @ width 48) · `metrics.py` · `train.py` (stages: `tuning` / `final`, checkpointed, resumable) · `demo.py` · `export_onnx.py` · `MOBILE.md` · `REAL_DATA_KAGGLE.md` · `results.json` (synthetic numbers) · `tuning.json` · **`load_hapt.py` · `train_real.py` (real-data experiments: phase1 / exp / postproc / confirm / ensemble, all resumable) · `results_real.json` (all 22 real-data experiments) · `REAL_DATA_RESULTS.md` · `IMPROVEMENT_RESEARCH.md` (literature brief behind Phase 2)**
