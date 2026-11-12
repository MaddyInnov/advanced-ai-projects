# DenseHAR Improvement Research — Literature Deep-Dive

**Purpose.** DenseHAR (multi-scale inception convs k=1,3,5,11 + SE attention + dilated temporal head, ~69K params, pure PyTorch, CPU-trained) is now being run on **real HAPT data**. Its known failure modes, measured on the synthetic run and expected to persist or worsen on real data:

1. **Over-segmentation** — boundary recall 0.97 but precision 0.36; segmental F1 far below a windowed baseline. The model finds almost every true boundary but fires ~3× too many.
2. **Subject-disjoint generalization gap** — HAPT evaluation is subject-disjoint; per-timestep accuracy on unseen subjects is the hard part (the windowed 12-class HAPT reference band from prior honest work is ~90% accuracy / ~0.79 macro-F1, and dense per-timestep scoring is stricter than window-majority scoring).
3. **Sitting/standing confusion** — the classic static-pair problem; on waist-worn HAPT data the two differ mainly in gravity-vector orientation and micro-sway.
4. **Transitions are short and rare** — HAPT's 6 postural-transition classes occupy a small fraction of timesteps, so plain per-timestep CE under-weights exactly the timesteps the project cares about.

Every technique below is ranked by **expected gain on these failure modes × implementation ease** for a small CPU-feasible PyTorch model. All quoted numbers come from the cited sources, with their protocols stated. **Strong caveat that applies to half the list:** the best-developed over-segmentation literature (MS-TCN, ASRF) is from *video* temporal action segmentation (I3D features, minutes-long activities, 15–30 fps). The mechanisms transfer to 50 Hz IMU streams, but the magnitudes will not transfer 1:1 — IMU segments are shorter, noisier, and boundaries are less visually crisp. Treat video numbers as directional evidence, and re-measure everything on HAPT.

---

## Top 3 highest-value moves

### Move 1 — Inference-time post-processing pack (no retraining; do this first, today)
Gaussian smoothing of the per-timestep **probabilities**, followed by **minimum-duration relabeling** (merge any predicted segment shorter than θt into its neighbours), plus an optional median filter on labels. In ASRF's controlled comparison on 50Salads, starting from an already-decent segmenter (F1@{10,20,50} = 76.1/74.5/66.7, Edit 68.5, Acc 82.6): Gaussian probability smoothing (K=15) raised Edit to **73.6** and F1@50 to **70.9** at *identical* accuracy; relabeling short segments (θt=15 frames) raised Edit to **76.2** and F1@50 to **72.5**. This attacks our single worst failure mode (boundary precision 0.36 ⇒ most predicted segments are spurious fragments, exactly what min-duration merging deletes) for ~30 lines of inference code and zero training cost. Tune θt on validation subjects only, and keep θt **below the shortest true HAPT transition** (≈1 s ≈ 50 samples; start θt ∈ {10, 15, 25} samples = 0.2–0.5 s).

### Move 2 — One MS-TCN-style refinement stage + truncated-MSE smoothing loss
Keep the current encoder as the "prediction generation" stage, then add **one refinement stage that takes the stage-1 class probabilities as input** (not features — Farha & Gall found refinement on probabilities works better) and re-predicts per-timestep classes through a small stack of dilated residual convs. Train **all** stages with CE + λ·(truncated MSE on log-probability differences), λ=0.15, τ=4. Evidence (50Salads, 5-fold CV): going from a single-stage TCN to 4 stages moved F1@50 from **21.5 → 64.5** and Edit from **20.5 → 67.9** at nearly unchanged frame accuracy (78.2 → 80.7) — i.e., refinement stages buy segmental quality, not frame accuracy, which is precisely our deficit. The smoothing loss alone (CE → CE+T-MSE, same 4-stage net) moved F1@50 **60.7 → 64.5** and Edit **64.2 → 67.9**. A single refinement stage at width 32–48 adds roughly 60–150K params — still trivially CPU-trainable at HAPT scale.

### Move 3 — Physics-respecting augmentation suite in the data loader
Random 3D rotation applied **jointly** to the acc and gyro triads (simulates device orientation differences across subjects), magnitude warping, time warping, jitter, and per-channel scaling — applied on-the-fly to training sequences only. This is the best-evidenced lever for the subject-disjoint gap: Um et al. (2017) improved CNN accuracy on wearable data from **77.54% → 86.88%** with exactly this family of augmentations (small clinical dataset, windowed classification — directional evidence); a 2025 physics-simulation study (PPDA) found augmentation improved macro-F1 by **+3.7 pp on average (max +13 pp)** across three HAR datasets and matched full-data performance with up to **60% fewer training subjects** — the cross-subject data-efficiency result is the part that matters for us. ~80 lines in the collate/dataset code, no model change. **Do not use permutation augmentation** for dense labels (it destroys the temporal structure the segmental metrics measure), and apply rotation to acc+gyro with the *same* matrix or the cross-modal physics breaks.

---

## Ranked techniques

### 1. Post-processing pack: probability smoothing + min-duration relabeling + median filter
- **What:** At inference: (a) smooth per-timestep class probabilities with a 1D Gaussian (σ ≈ 5–8 samples at 50 Hz, kernel 15–25); (b) argmax; (c) run-length encode; (d) iteratively merge segments shorter than θt into the longer neighbour; optionally (e) median-filter labels (width 11–25) before step (c).
- **Why it helps us:** Our boundary precision/recall profile (0.36 / 0.97) is the textbook signature of spurious micro-segments. Merging directly converts those false boundaries into correct interior timesteps: segmental F1 and edit score jump, frame accuracy barely moves (it can even rise slightly, since micro-segments are usually wrong). Median filtering of predictions is also long-standing practice in deployed wearable HAR (e.g., an on-device smartwatch system uses median filtering + hysteresis specifically to suppress transient prediction spikes).
- **Implementation sketch (~30 lines, pure NumPy/PyTorch):**
  ```python
  probs = gaussian_filter1d(probs, sigma=6, axis=0)          # (T, C)
  labels = probs.argmax(-1)
  labels = median_filter(labels, size=15)                    # optional
  # run-length merge loop:
  while True:
      segs = run_length_encode(labels)
      short = [s for s in segs if s.len < theta_t]           # theta_t ~ 10-25
      if not short: break
      s = min(short, key=lambda s: s.len)
      labels[s.start:s.end] = neighbour_label(s)             # longer neighbour wins
  ```
- **Evidence:** ASRF (Ishikawa et al., WACV 2021) Table 3, 50Salads, same base predictions for all rows — no post-processing: F1@{10,20,50} 76.1/74.5/66.7, Edit 68.5, Acc 82.6 → Gaussian smoothing K=15: 80.8/79.0/70.9, Edit 73.6, Acc 82.6 → Relabeling θt=15: 82.4/80.6/72.5, Edit 76.2, Acc 82.7. Median-filter/hysteresis use in on-device HAR: *Electronics* 2026 smartwatch study (qualitative, no isolated gain reported). Sources: https://arxiv.org/pdf/2007.06866 · https://www.mdpi.com/2079-9292/15/7/1351
- **Caveats:** Video evidence (15 fps, so K=15 ≈ 1 s; scale kernels by fps ratio — at 50 Hz use ~3× the frame counts, then tune). Relabeling with too-large θt deletes *real* short segments — HAPT transitions can be ~1 s, so θt must stay ≤ ~0.5 s and be validated per-class. Smoothing probabilities cannot fix a systematically wrong class (sitting↔standing), only fragmentation.

### 2. Refinement stage(s) on class probabilities (MS-TCN / MS-TCN++)
- **What:** Multi-stage architecture: stage 1 (our current DenseHAR encoder + head) outputs per-timestep class probabilities; each refinement stage consumes the previous stage's **probabilities** and outputs refined probabilities. Each stage = 1×1 conv → L dilated residual conv layers (dilation 2^l, kernel 3) → 1×1 conv to C classes + softmax. Loss is applied at **every** stage output.
- **Why it helps us:** Refinement stages see the whole predicted label sequence at a large receptive field, so they can learn "a 6-sample island of *standing* inside a long *sitting* segment is implausible" — a sequence-level correction our single-pass encoder cannot make. This is the canonical, best-ablated cure for over-segmentation in dense temporal prediction.
- **Implementation sketch (~120 lines):**
  ```python
  class RefinementStage(nn.Module):
      def __init__(self, n_classes, width=48, layers=8):
          self.inp = nn.Conv1d(n_classes, width, 1)
          self.blocks = nn.ModuleList([
              DilatedResidualConv(width, dilation=2**l) for l in range(layers)])
          self.out = nn.Conv1d(width, n_classes, 1)
      def forward(self, probs):            # (B, C, T)
          h = self.inp(probs)
          for b in self.blocks: h = b(h)
          return self.out(h)               # logits
  # training: loss = sum over stages of CE(stage_logits, y) + lam * TMSE(stage_logprobs)
  ```
  MS-TCN++ variant: use **dual dilated layers** in the *generation* stage (parallel dilation 2^l and 2^(L-1-l) branches, concatenated) and share weights across refinement stages for a compact model. On 50Salads MS-TCN++ reached F1@{10,25,50} **80.7/78.5/70.1**, Edit **74.3**, Acc **83.7** vs MS-TCN's 76.3/74.0/64.5, 67.9, 80.7.
- **Evidence:** Farha & Gall, CVPR 2019 (MS-TCN), 50Salads ablation — SS-TCN (1 stage): F1@{10,25,50} 27.0/25.3/21.5, Edit 20.5, Acc 78.2; 2 stages: 55.5/52.9/47.3, 47.9, 79.8; 4 stages: 76.3/74.0/64.5, 67.9, 80.7. A 48-layer *single*-stage TCN only reached F1@50 40.2 / Edit 40.7 — depth is not a substitute for stages. Li et al., TPAMI 2020 (MS-TCN++), numbers above. Sources: https://arxiv.org/pdf/1903.01945 · https://arxiv.org/pdf/2006.09220
- **Caveats:** All video-domain (I3D features). Returns saturate/overfit past ~3–4 stages on small data (MS-TCN 5 stages ≈ 4 stages; ASRF found 4 BRB stages worse than 3 on 50Salads). For HAPT's ~110K labelled segments, start with **one** refinement stage and add a second only if validation segmental F1 still climbs. Refinement stages add sequential compute; at our width this is still CPU-fine.

### 3. Truncated-MSE smoothing loss (T-MSE), optionally Gaussian-similarity-weighted (GS-TMSE)
- **What:** Auxiliary loss penalizing abrupt changes in predicted log-probabilities between adjacent timesteps, with the penalty **truncated** so genuine transitions are not crushed:
  `Δ_{t,c} = |log y_{t,c} − log y_{t−1,c}|`, `Δ̃ = min(Δ, τ)`, `L_smooth = mean(Δ̃²)`, τ=4, total loss `CE + λ·L_smooth`, λ=0.15. GS-TMSE additionally multiplies each timestep's penalty by `exp(−||x_t − x_{t−1}||² / 2σ²)` computed on the input (or encoder features), so smoothing is relaxed exactly where the *signal itself* changes — i.e., at real transitions.
- **Why it helps us:** Directly optimizes against the flicker that produces boundary precision 0.36, during training rather than only at inference. GS-TMSE is the version that protects our transition timesteps: plain TMSE also penalizes the (correct) sharp change at a sit→stand boundary; the Gaussian gate removes most of that penalty where the IMU signal actually jumps.
- **Implementation sketch (~15 lines):**
  ```python
  def tmse(log_probs, tau=4.0):                      # (B, C, T)
      d = (log_probs[:, :, 1:] - log_probs[:, :, :-1]).abs()
      return torch.clamp(d, max=tau).pow(2).mean()
  # GS variant: w_t = exp(-((x[:, :, 1:] - x[:, :, :-1])**2).sum(1) / (2*sigma**2))
  ```
- **Evidence:** MS-TCN loss ablation (50Salads, 4-stage): CE only → F1@{10,25,50} 71.3/69.7/60.7, Edit 64.2, Acc 79.9; CE+T-MSE (λ=0.15, τ=4) → **76.3/74.0/64.5, Edit 67.9, Acc 80.7**. λ is sensitive: 0.05 under-smooths (F1@50 62.4), 0.25 costs accuracy (78.9). ASRF loss ablation: CE+GS-TMSE beat CE+TMSE (F1@50 72.7 vs 69.1 with plain CE; see Technique 5 for the full table). Sources: https://arxiv.org/pdf/1903.01945 · https://arxiv.org/pdf/2007.06866
- **Caveats:** Video-domain evidence; on 50 Hz IMU the log-prob differences are noisier — consider computing Δ on probabilities smoothed over 3 samples, and re-tune λ ∈ {0.05, 0.15, 0.3} on validation subjects. **Relation to our own prior null result:** our team's earlier *boundary-aware losses* (boundary-weighted CE variants) did not help **windowed** HAPT classification. T-MSE is a different object — it penalizes prediction *flicker* in a dense output, not window labels near boundaries — but the prior null is a reason to A/B it on validation subjects and keep it only if segmental F1 actually moves, not to adopt it on faith.

### 4. Augmentation suite (rotation / magnitude-warp / time-warp / jitter / scaling)
- **What:** On-the-fly training-sequence transforms: (i) **rotation** — sample a random small-to-moderate rotation matrix R (e.g., axis random, angle ~U(−30°, 30°), occasionally full random orientation) and apply the *same* R to acc xyz and gyro xyz at every timestep; (ii) **magnitude warping** — multiply each channel by a smooth random curve (cubic spline through ~4 knots, σ≈0.2); (iii) **time warping** — smooth monotonic warp of the time axis (spline, σ≈0.2), labels warped identically; (iv) **jitter** — Gaussian noise σ≈0.03–0.05 (in g / rad·s⁻¹ units after normalization, tune); (v) **scaling** — per-sequence factor ~N(1, 0.1).
- **Why it helps us:** Subject-disjoint error is dominated by exactly the nuisances these simulate: different device orientation in the waistband/pocket (rotation), different gait amplitude and speed across people (magnitude/time warp), sensor noise (jitter). It is the cheapest known way to buy cross-subject robustness without new data or target-subject adaptation.
- **Implementation sketch (~80 lines):** a `augment(x, y)` function in the Dataset/`collate_fn`, applied with p≈0.5–0.8 per transform per sample; rotation via `scipy.spatial.transform.Rotation.random()` or a hand-rolled axis-angle matrix; warps via `np.interp` on a warped index grid (labels use nearest-neighbour interp).
- **Evidence:** Um et al., ICMI 2017 — CNN on wearable Parkinson's data, accuracy **77.54% → 86.88%** with appropriate augmentation; rotation, rotation+permutation and time-warping among the most effective single transforms. Oishi et al. 2025 (PPDA, physics-simulated augmentation, REALDISP/REALWORLD/MM-Fit): macro-F1 **+3.7 pp average, up to +13 pp**, and parity with up to **60% fewer training subjects**. A 2025 WEAR-challenge solution likewise built its robustness entirely on train-time jitter/scaling/rotation/channel-dropout matched to test perturbations. Sources: https://arxiv.org/pdf/1706.00527 · https://arxiv.org/pdf/2508.13284 · https://arxiv.org/pdf/2510.21282
- **Caveats:** Um et al. is windowed classification on a small clinical set — expect single-digit, not 9-point, gains. **Permutation is excluded deliberately**: shuffling sub-segments teaches the model that label sequences are unordered, poison for a dense segmenter. Over-strong time warping can smear the short transition classes into their neighbours; cap warp strength and verify transition-class F1 on validation. Rotation range is a trade-off: full random orientation also destroys the gravity-direction cue that separates sitting/standing/laying — use moderate angles (±30–45°) for the static-class problem we have, not uniform SO(3).

### 5. ASRF-style boundary branch + boundary-pooled majority refinement
- **What:** Add a second, **class-agnostic boundary head** on the shared encoder features: a small TCN (can reuse the refinement-stage block) outputting a per-timestep boundary probability p_t ∈ [0,1]. Train with binary logistic loss with positive weight = reciprocal of boundary frequency (boundaries are ~1–2% of timesteps), optionally against Gaussian-smoothed boundary targets. **Inference refinement:** take local maxima of p_t above threshold θp (≈0.5) as segment cuts; within each resulting segment, relabel every timestep by **majority vote of the main head's predictions** (or mean-probability argmax) in that segment.
- **Why it helps us:** This is the strongest published over-segmentation fix after plain multi-stage refinement, and it is *boundary-aware* in the way our failure mode needs: our problem is not finding boundaries (recall 0.97) but that the class head flickers *within* segments. Boundary-pooled majority voting enforces "one segment = one class" using cuts decided by a head trained only for cut placement. Notably, ASRF's boundary head itself had **low precision (~37%) and recall ~63%** (at a strict ±5-frame tolerance) and refinement *still* produced large gains — an over-firing boundary head is tolerable because spurious cuts inside a homogeneous region are harmless after majority voting (both sides vote the same class). That asymmetry matches our 0.36-precision regime well.
- **Implementation sketch (~150 lines):** boundary head = 1×1 conv → 6–8 dilated residual convs (width 32) → 1×1 conv → sigmoid (~40–60K params, or share the encoder and make it ~15K). Loss: `L = L_asb + λ_b · L_boundary`, λ_b = 0.1–0.2 (ASRF used 0.1 for 50Salads/Breakfast, 0.2 for GTEA). Inference: peak-pick p_t (local max, > θp, min separation ~10 samples), split, majority-vote per segment.
- **Evidence:** Ishikawa et al., WACV 2021 (ASRF). 50Salads vs its own unrefined output (Table 3): boundary refinement F1@{10,20,50} **84.9/83.5/77.3**, Edit **79.3**, Acc **84.5** vs no post-processing 76.1/74.5/66.7, 68.5, 82.6 — beating both Gaussian smoothing and fixed-θt relabeling on every metric. Vs MS-TCN on Breakfast: F1@10 **74.3 vs 58.2**, Edit **72.4 vs 61.4**, Acc 67.6 vs 65.1. Paper-reported improvements "up to 16.1% segmental F1, 13.7% edit, 2.6% frame accuracy" over prior SOTA. Source: https://arxiv.org/pdf/2007.06866
- **Caveats:** Largest implementation item on this list, and the refinement is only as good as the base head — ASRF note it "cannot reassign completely incorrect segments" (their Figure 2 failure case: majority vote can also *lock in* a wrong class for a whole segment). Their boundary-similarity variant (cuts at input-similarity minima) **failed badly** (F1@50 18.1) — use the learned boundary head, not signal-similarity cuts. Video-domain evidence; HAPT boundaries are genuinely ambiguous at ±5 samples (0.1 s), so evaluate our boundary head with a ±25-sample (0.5 s) tolerance.

### 6. Class-weighted cross-entropy (median-frequency balancing or effective-number weights) — and a warning about focal loss
- **What:** Weight per-timestep CE by class: median-frequency balancing `w_c = median(freq) / freq_c` (ASRF's choice), or Cui et al.'s effective-number weight `w_c ∝ (1−β)/(1−β^{n_c})`, β≈0.999–0.9999, which down-weights less aggressively than raw inverse frequency. Apply to the dense CE (per-timestep class counts), all stages.
- **Why it helps us:** In 12-class HAPT the transition classes are a few percent of timesteps; in 6-class mode the static pair still dominates gradient mass. Class weighting is the one loss change with *positive* controlled evidence in the segmentation literature, and it directly targets transition recall and minority static-class F1.
- **Implementation sketch (~5 lines):** precompute `w` from training-subject timestep counts; `nn.CrossEntropyLoss(weight=w)`.
- **Evidence:** ASRF loss ablation (50Salads, F1@{10,20,50} / Edit / Acc): CE+TMSE = 79.7/77.7/69.1 / 73.4 / 79.8 → **class-weighted CE + TMSE = 83.3/81.8/75.1 / 77.0 / 81.8** → class-weighted CE + GS-TMSE = **84.9/83.5/77.3 / 79.3 / 84.5** (best). Cui et al., CVPR 2019 (vision, long-tailed CIFAR-10, imbalance factor 200): class-balanced loss cut error **34.32% → 31.11%** vs plain softmax. Sources: https://arxiv.org/pdf/2007.06866 · https://arxiv.org/abs/1901.05555?context=cs
- **⚠️ Negative evidence — focal loss:** In the same ASRF ablation, swapping CE for **focal loss made everything worse**: focal+TMSE = 77.3/75.8/67.1 / 71.3 / 78.0 (vs 79.7/77.7/69.1 / 73.4 / 79.8 for CE+TMSE). Focal's hard-example emphasis amplifies exactly the noisy boundary timesteps a dense HAR model should be *less* confident about. Combined with our team's own prior null for boundary-emphasis losses on windowed HAPT, the recommendation is: **class-weighted CE yes; focal loss and boundary-weighted CE no** (unless a validation A/B says otherwise on real HAPT). Focal loss is still used in some windowed HAR systems (e.g., a federated CNN-BiGRU-Transformer uses class-balanced focal, γ=2, for PAMAP2's rare classes), but there is no dense-HAR evidence for it and direct dense-segmentation evidence against it.

### 7. Normalization: train-only global stats in, gravity preserved, GroupNorm inside
- **What:** (a) **Input:** z-score each channel with mean/std computed on **training subjects only** (leakage-free standard). (b) **Do NOT per-sequence z-score the raw stream** as the only representation: per-sequence standardization deletes the per-sequence channel means — i.e., the gravity direction — which is the primary cue separating sitting/standing/laying on a waist-worn phone. If per-sequence normalization is desired for the dynamic component, **append the removed per-sequence channel means (and stds) as auxiliary constant channels/features** so the posture cue survives. (c) **Inside the network:** prefer **GroupNorm over BatchNorm** in the encoder/refinement blocks (batch statistics shift across subjects; GN is batch-independent and is the norm used in most small TCN segmenters' residual blocks in practice).
- **Why it helps us:** Cross-subject shift in HAR is largely an amplitude/offset shift per channel. Per-subject normalization is common LOSO practice (e.g., a LOSO VO₂max study normalizes "independently for each IMU channel on a per-subject basis, preventing information leakage across subjects"), and in a controlled cross-subject study (EEG, not IMU) z-score normalization produced both the smallest source→target generalization gap and the best unseen-subject accuracy (~70% of timesteps correct) among compared schemes. The gravity caveat is HAPT-specific and important: it is the flip side of the same coin.
- **Implementation sketch:** preprocessing change only; optionally add 6 extra input channels = low-pass-filtered acc/gyro (gravity estimate, 0.3 Hz Butterworth or simple moving average over ~1 s) so orientation is explicit even after normalization — a cheap, classic HAR feature that also feeds Technique 8.
- **Evidence:** Ramírez et al. 2019, cross-subject normalization study (EEG workload; z-score best generalization gap + accuracy). Per-subject IMU normalization as LOSO methodology: *Electronics* 2025 VO₂max study. Sources: https://arxiv.org/pdf/1906.08823 · https://www.mdpi.com/2079-9292/15/13/2843/pdf
- **Caveats:** The quantitative normalization evidence is from EEG and from methodology papers, not a controlled IMU-HAR ablation — treat (a) as mandatory hygiene, (b) as a correctness warning, (c) as low-risk default rather than a proven gain. Never compute normalization stats on test subjects' *labels*; per-subject stats from the test subject's own unlabelled stream are legitimate only in a transductive/deployment setting — say which protocol a number came from.

### 8. Gravity/orientation features + longer effective context for the static pair (sitting vs standing)
- **What:** Two cheap additions aimed squarely at sitting/standing: (i) append **signal-magnitude channels** (|acc|, |gyro|) and a **gravity-vector estimate** (low-pass acc, 3 channels) to the 6 raw channels → 11 channels; magnitude is orientation-invariant (helps cross-subject), gravity direction is the posture cue (helps static classes). (ii) Ensure the receptive field at the head covers ≥4–5 s (dilations to 128–256 at 50 Hz) — static classes are separated by micro-sway statistics that only exist over multi-second spans, and MS-TCN++'s dual-dilation result (below) shows small+large receptive fields in parallel beat large-only.
- **Why it helps us:** Sitting vs standing confusion in waist-worn data is an *information* problem before it is a capacity problem: instantaneous acc/gyro are near-identical; the discriminative signal is thigh/trunk orientation (gravity direction) plus sway spectrum. Giving the network the gravity vector explicitly, instead of forcing it to re-derive it through normalization layers, is the standard classical-HAR solution (feature sets in the Bulling et al. tutorial tradition include orientation/magnitude features for exactly this reason).
- **Implementation sketch:** preprocessing: `grav = lowpass(acc, 0.3 Hz)`; `dyn = acc − grav`; input = [dyn(3), grav(3), gyro(3), |acc|, |gyro|]. Model change: none, except first conv in_channels 6→11. Optionally add MS-TCN++ dual-dilated layers in stage 1: two parallel dilated convs with dilations 2^l and 2^(L−1−l), concatenated + 1×1.
- **Evidence:** MS-TCN++ ablation (50Salads): adding dual dilated layers to MS-TCN improved F1@50 **64.5 → 67.3** and Edit **67.9 → 69.8** (full MS-TCN++: F1@50 70.1, Edit 74.3). Classical feature rationale: Bulling, Blanke & Schiele, ACM Computing Surveys 2014 tutorial on the activity recognition chain. Sources: https://arxiv.org/pdf/2006.09220 · https://www.collaborative-ai.org/publications/bulling14_csur.pdf
- **Caveats:** No controlled HAPT ablation of gravity channels was found in this search — this is a well-founded engineering recommendation, not a quoted gain; A/B it. Note the interaction with Technique 4: if gravity channels are inputs, keep augmentation rotations moderate, or the model learns to ignore the very cue this adds.

### 9. Viterbi / HMM-style structured decoding with learned transition matrix
- **What:** Replace (or follow) argmax decoding with Viterbi decoding over the model's per-timestep log-probabilities plus a first-order transition model: score = Σ_t log p(y_t|x) + β · Σ_t log A[y_{t−1}, y_t], where A is the class-transition matrix counted from **training subjects' label sequences** (with self-transition probabilities naturally ≈0.99+ at 50 Hz), and β (~0.5–2) trades emission confidence against sequence plausibility. A cheaper special case: forbid transitions that never occur in training (e.g., laying→walking without a transition class in 12-class mode) by setting those A entries to ~0.
- **Why it helps us:** It is Technique 1 with a learned, class-aware prior instead of a fixed duration threshold: isolated 3-sample islands are removed not because they are short per se, but because *enter→island→exit* costs two unlikely transitions. In 12-class HAPT mode it can also enforce the legal transition grammar (you cannot go stand→sit except via stand-to-sit). HMM/CRF sequence decoding is the classical HAR machinery for exactly this (Bulling et al. tutorial, §3.4: HMMs and CRFs are the standard temporal probabilistic models in the activity recognition chain).
- **Implementation sketch (~60 lines):** standard Viterbi DP in NumPy over (T × C) log-emissions — O(T·C²) = trivial for C=6–12; estimate A with add-ε smoothing from train label streams; tune β on validation subjects. A linear-chain CRF layer (e.g., via `torchcrf`-style implementation) trained end-to-end is the heavier alternative — defer unless Viterbi shows promise.
- **Evidence:** Qualitative/methodological (Bulling et al. 2014). ASRF's design note is indirectly supportive: they chose a class-agnostic boundary regressor over an HMM over classes partly because full class-transition modelling "requires far more data" — with HAPT's constrained grammar and 30 subjects we are closer to the regime where a counted transition matrix is reliable, but no published dense-HAR Viterbi gain on HAPT was found in this search. Source: https://www.collaborative-ai.org/publications/bulling14_csur.pdf · https://arxiv.org/pdf/2007.06866
- **Caveats:** No hard gain numbers found for dense IMU HAR — rank it below the post-processing pack (Technique 1), which captures most of the benefit with fewer assumptions. Viterbi with an over-strong β delays true boundary placement (boundaries shift into the following segment); check boundary F1, not just segmental F1, when tuning β. Offline (whole-sequence) decoding only; for streaming, use fixed-lag Viterbi.

### 10. Architecture alternative if refinement stalls: U-Net-style dense encoder–decoder (optionally depthwise-separable)
- **What:** Replace/augment the flat dilated encoder with a 1D U-Net: 3–4 downsampling stages (stride-2 convs), skip connections to a symmetric decoder, per-timestep head. Depthwise-separable convs keep it light.
- **Why it might help us:** Skip connections restore fine temporal detail lost in downsampling — the published dense-HAR family (FCN → U-Net → Seq2Dense U-Net → DS U-Net) exists precisely because encoder–decoders localize short activities/transitions well, and they are the *only* dense architectures with published HAPT numbers (see benchmark box below).
- **Evidence:** DS U-Net (Kim & Park, *Applied Sciences* 2025): **subject-independent splits** on MHEALTH/PAMAP2/WISDM; MHEALTH accuracy **0.9259**, macro-F1 **0.922** (best among compared models; +7.89% accuracy vs comparison models on the out-of-lab MHEALTH set per the paper's abstract), with **57–89% fewer parameters** than standard U-Net baselines. Seq2Dense U-Net (Meena & Sarawadekar, *IEEE Sensors Journal* 2023) reports UCI-HAPT accuracy **0.9474** / F1 **0.9379** and UCI-HAR **0.9547** / 0.9406 — **but** the retrieved text does not state a subject-disjoint protocol for those tables, so treat them as non-comparable upper references, not targets. Sources: https://www.mdpi.com/2076-3417/15/16/9134 · https://idr-sdlib.iitbhu.ac.in/server/api/core/bitstreams/5246346a-a631-4eda-868c-39c8aba27a29/content
- **Caveats:** A rewrite, not an increment — only worth it if Techniques 1–3 plateau. Downsampling risks the opposite failure (under-segmentation of ~1 s transitions); keep the finest decoder level at full 50 Hz resolution and do not exceed 3 downsampling stages. Seq2Dense protocol caveat above is important for honest reporting: its 0.9474 HAPT number must never be quoted next to our subject-disjoint numbers without the protocol difference attached.

### 11. Seed ensembling + probability averaging (cheap, boring, reliable)
- **What:** Train 3–5 seeds of the final configuration, average per-timestep probabilities at inference, then run Technique 1's post-processing on the averaged probabilities.
- **Why it helps us:** Variance across seeds on small subject-disjoint HAR sets is large; averaging probabilities specifically reduces the random flicker component of over-segmentation before any structural fix. CPU training of a 69K-param model makes 3–5 seeds affordable, and it also gives an honest ± spread to report.
- **Caveats:** No single canonical citation/gain quoted here — standard practice; expect small (sub-point to ~1 pt) accuracy gains and a somewhat larger segmental-F1 gain via flicker reduction. Do this last, after the architecture settles, or every ablation doubles in cost.

---

## What NOT to do (negative evidence)

- **Focal loss as the main fix** — directly ablated *worse* than CE in dense segmentation (ASRF Table 7, quoted in Technique 6). Its hard-example focus amplifies label noise at boundaries.
- **Boundary-weighted / boundary-emphasis CE as the main fix** — our own team's controlled windowed-HAPT experiments already found boundary-aware losses did **not** beat plain CE. The segmentation literature's wins come from *smoothing losses, refinement structure, and inference-time boundary pooling* — different mechanisms. If boundary weighting is retried in dense mode, it must win a validation A/B first.
- **Signal-similarity boundary cuts** (ASRF's "Similarity" baseline): catastrophic in their ablation (F1@50 18.1 vs 66.7 unrefined). Learned boundaries or duration priors only.
- **Permutation augmentation** for dense training (destroys sequence structure; see Technique 4).
- **Per-sequence z-scoring that discards channel means** without re-injecting them (deletes the gravity/posture cue; see Technique 7).
- **Quoting Seq2Dense U-Net's 0.9474 HAPT accuracy as a subject-disjoint target** — protocol not established as subject-disjoint in the source text retrieved.

## Reference points: published dense HAR numbers (with protocols)

| Model | Dataset | Number | Protocol (as stated in source) |
|---|---|---|---|
| Seq2Dense U-Net (IEEE Sensors J. 2023) | UCI-HAPT | Acc 0.9474, F1 0.9379 | Split not stated as subject-disjoint in retrieved text — **not comparable** to LOSO/subject-disjoint |
| Seq2Dense U-Net | UCI-HAR | Acc 0.9547, F1 0.9406 | Same caveat |
| DS U-Net (Appl. Sci. 2025) | MHEALTH | Acc 0.9259, macro-F1 0.922 | **Subject-independent** split; best vs compared models |
| Windowed reference (team's prior honest work) | HAPT, 12-class | ~90% acc, ~0.79 macro-F1 | Windowed majority-label, subject-disjoint — different (easier) scoring than dense per-timestep |

No published **dense, per-timestep, subject-disjoint HAPT** benchmark with segmental metrics was found in this search — which means a clean DenseHAR HAPT result with segmental F1 / edit score reported is itself a contribution worth stating carefully in the paper/report.

## Suggested experiment order (each step A/B'd on validation subjects, segmental F1@25 + edit + boundary P/R reported)

1. Technique 1 post-processing on the *current* checkpoint (θt, σ grid) — establishes how much of the gap is pure decoding.
2. Technique 7(a) normalization hygiene + Technique 8 gravity/magnitude channels (one retrain).
3. Technique 4 augmentation (one retrain) — watch transition-class F1 for warp damage.
4. Technique 3 T-MSE, then GS-TMSE (λ grid) — keep only if it beats step-1 post-processing alone.
5. Technique 2 one refinement stage (width 48, 8 layers) with per-stage CE+T-MSE.
6. Technique 6 class-weighted CE swapped in (median-frequency weights).
7. Technique 5 ASRF boundary branch + majority refinement, replacing/augmenting Technique 1.
8. Technique 9 Viterbi on top of the best configuration; Technique 11 seed ensemble last.

## Sources

- Farha & Gall, *MS-TCN*, CVPR 2019 — https://arxiv.org/pdf/1903.01945
- Li et al., *MS-TCN++*, TPAMI 2020 — https://arxiv.org/pdf/2006.09220
- Ishikawa et al., *ASRF: Alleviating Over-segmentation Errors by Detecting Action Boundaries*, WACV 2021 — https://arxiv.org/pdf/2007.06866
- Um et al., *Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using CNNs*, ICMI 2017 — https://arxiv.org/pdf/1706.00527
- Oishi et al., *Physically Plausible Data Augmentations for Wearable IMU-based HAR*, 2025 — https://arxiv.org/pdf/2508.13284
- PatchTST ensembles with test-matched augmentation (WEAR challenge), 2025 — https://arxiv.org/pdf/2510.21282
- Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019 — https://arxiv.org/abs/1901.05555?context=cs
- Meena & Sarawadekar, *Seq2Dense U-Net*, IEEE Sensors Journal 2023 (thesis chapter mirror) — https://idr-sdlib.iitbhu.ac.in/server/api/core/bitstreams/5246346a-a631-4eda-868c-39c8aba27a29/content
- Kim & Park, *Depthwise-Separable U-Net for Wearable Sensor-Based HAR*, Applied Sciences 2025 — https://www.mdpi.com/2076-3417/15/16/9134
- Ramírez et al., *Cross-Subject Statistical Shift Estimation…* (normalization study), 2019 — https://arxiv.org/pdf/1906.08823
- *Activity-Independent Estimation of VO2max…* (per-subject IMU normalization under LOSO), Electronics 2025 — https://www.mdpi.com/2079-9292/15/13/2843/pdf
- Bulling, Blanke & Schiele, *A Tutorial on Human Activity Recognition Using Body-Worn Inertial Sensors*, ACM Computing Surveys 2014 — https://www.collaborative-ai.org/publications/bulling14_csur.pdf
- *On-Device Motion Activity Intensity Recognition Using Smartwatch Accelerometer* (median filter + hysteresis post-processing in deployed HAR), Electronics 2026 — https://www.mdpi.com/2079-9292/15/7/1351
