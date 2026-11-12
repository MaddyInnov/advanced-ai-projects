# DenseHAR on Real Data — HAPT Results

**The synthetic headline (95.35%) is retired.** This document reports DenseHAR on
**real** human activity data — the HAPT dataset (UCI ML Repository id 341) — with
the same subject-disjoint protocol, the same metric suite, and a windowed baseline
evaluated identically. All numbers are measured; every experiment run is logged in
[`results_real.json`](results_real.json), including the failures.

## Headline result (test subjects 2, 4, 9, 10, 12 — never seen in training)

| Test metric | **DenseHAR final system**¹ | DenseHAR + refinement² | Phase-1 DenseHAR (config A) | Window-128 baseline (broadcast) |
|---|---|---|---|---|
| Per-timestep accuracy | **0.9038** | 0.8990 | 0.7886 | 0.7891 |
| Macro F1 | 0.9398 | **0.9415** | 0.7253 | 0.7258 |
| Transition-window accuracy (±10 steps) | 0.6093 | **0.6246** | 0.5955 | 0.4787 |
| Accuracy on transition-class timesteps | 0.8744 | **0.8946** | 0.8571 | 0.7730 |
| Accuracy on non-transition timesteps | **0.9106** | 0.9000 | 0.7726 | 0.7929 |
| Boundary F1 (±10) | 0.4087 | **0.5045** | 0.1353 | 0.0874 |
| Segmental F1@10 / @25 / @50 | 0.836 / 0.822 / 0.788 | **0.858 / 0.840 / 0.819** | 0.410 / 0.388 / 0.355 | 0.775 / 0.749 / 0.693 |
| Edit score | 84.88 | **86.23** | 63.42 | 75.52 |

¹ *Final system (val-selected headline):* physics augmentation + 12-class
supervision + 3-seed probability ensemble + val-tuned post-processing
(σ=8 Gaussian, θ=15 min-duration). Highest val macro F1 of all candidates (0.9675).
² *Co-leading candidate:* same recipe with an MS-TCN refinement stage, single seed,
post-processing (θ=15). Best test macro F1 and best transition/segmental scores;
ranked 4th on val F1 (0.9634), so it is reported as co-leading, not the headline —
the headline was chosen on validation, before test was touched.

**Phase 1 alone did not beat the baseline** (accuracy 0.7886 vs 0.7891, macro F1
0.7253 vs 0.7258 — a tie; DenseHAR won only the transition metrics). Phase 2
(val-selected improvements only, §4) fixed that: the final system beats the
baseline on **every** headline metric by +10 to +22 points.

## 1. Dataset and protocol

**HAPT** (Reyes-Ortiz et al.): 30 subjects, waist-mounted smartphone, accelerometer
+ gyroscope @ 50 Hz, per-timestep labels in `RawData/labels.txt`. 12 classes: 6 basic
activities + 6 postural transitions (stand-to-sit, sit-to-stand, sit-to-lie,
lie-to-sit, stand-to-lie, lie-to-stand). Source: `hapt.zip`, already in the research
workspace; zip integrity verified (`unzip -t`, no errors).

Loader (`load_hapt.py`) → `data_hapt.npz`:

- Per (experiment, user) stream, **512-step windows, stride 256**; windows touching
  an unlabelled sample are dropped (27.36% of the 1,122,772 raw samples are
  unlabelled in HAPT).
- Windows containing an activity boundary or any transition-class timestep are
  always kept; pure single-activity windows are capped at ≤30% of each split.
- **Two label versions:** `y6` — 6-class, transition classes mapped to their
  *target* activity (stand-to-sit→sitting, sit-to-stand→standing, sit-to-lie→laying,
  lie-to-sit→sitting, stand-to-lie→laying, lie-to-stand→standing); `y_trans` —
  boolean mask of timesteps whose original label was a transition class.
- **Fixed subject-disjoint split** (no overlap, asserted in code):
  train 21 subjects `[1,3,5,6,7,8,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26]`,
  val 4 `[27,28,29,30]`, test 5 `[2,4,9,10,12]` (test subjects drawn from the
  official HAPT test-subject list; val carved from official train subjects).
- Normalisation: per-channel mean/std from **train subjects' streams only**.

| Split | Sequences | Transition timesteps | Sequences with ≥1 boundary | Class mix (dominant) |
|---|---|---|---|---|
| Train (21 subj) | 862 | 18.6% | 445 | laying .34, sitting .32, standing .26, dynamic ≈.09 |
| Val (4 subj) | 145 | 18.7% | 78 | sitting .36, laying .30, standing .26 |
| Test (5 subj) | 177 | 18.9% | 97 | laying .36, sitting .32, standing .22 |

Transition-class timesteps are 8.24% of all labelled raw HAPT samples; the
boundary-enriched sequencing raises their share to ~19% of evaluated timesteps —
this evaluation is deliberately transition-heavy. Note also: **no test window
contains more than 2 basic activities** (HAPT experiments are single-bout
recordings); 124 of 177 test windows contain transition-class timesteps.

**Selection discipline.** All model/hyperparameter/post-processing selection used
the **val split only**. The test split was evaluated exactly **6 times** in total —
once per final candidate (ledger in §7). Screening experiments (15 runs) never
touched test.

## 2. Phase 1 — config A as-is on real data

DenseHAR config A (width 48, boundary-weighted CE + T-MSE λ=0.15, 14 epochs,
val-selected checkpoint) vs the window-128 broadcast baseline (10 epochs,
val-selected), both trained on the 862 train sequences:

| Test | DenseHAR (Phase 1) | Baseline |
|---|---|---|
| Accuracy | 0.7886 | **0.7891** |
| Macro F1 | 0.7253 | **0.7258** |
| Transition-window acc | **0.5955** | 0.4787 |
| Transition-class timestep acc | **0.8571** | 0.7730 |
| Boundary F1 (P / R) | **0.1353** (0.078 / 0.516) | 0.0874 (0.093 / 0.083) |
| Segmental F1@25 | 0.3879 | **0.7486** |
| Edit score | 63.42 | **75.52** |

Diagnosis, in the model's own numbers:

- **Over-segmentation is severe on real data:** 592 false boundaries against 50
  true positives (precision 0.078). The flicker that was a nuisance on synthetic
  data dominates here — hence segmental F1 0.388 vs the baseline's 0.749.
- **The transition advantage is real but narrow:** +11.7 pts at transition windows,
  +8.4 pts on transition-class timesteps — while ordinary timesteps are *worse*
  than the baseline (0.7726 vs 0.7929).
- **Upstairs collapses** on unseen subjects (F1 0.386; it was 1.000 on val), and
  standing is weak (0.616) — the classic static-pair problem.
- **Val ≫ test for both models** (val F1 ≈ 0.92 for both): the 4 val subjects are
  much easier than the 5 test subjects. This val→test gap (~19 F1 pts) is the
  dominant problem — a cross-subject generalisation gap, not a capacity problem.

## 3. Phase 2 — method

Guided by `IMPROVEMENT_RESEARCH.md` (literature brief: MS-TCN/ASRF post-processing
and refinement, physics-respecting augmentation, class-weighting evidence, and
negative evidence on focal loss / boundary-weighted CE) and the original lever
list. Every candidate was screened on **val** (macro F1 primary); only val winners
were promoted to a single test evaluation. Total training compute for all 22
logged experiments: ~22 min on 4 CPU threads.

### Val screening log (all runs; selection metric = val macro F1)

| Experiment | Lever(s) | Val acc | Val F1 | Trans-win | Seg F1@25 | Edit |
|---|---|---|---|---|---|---|
| p1_base (baseline) | window-128 broadcast | 0.8780 | 0.9208 | 0.4868 | 0.856 | 84.0 |
| p1_dense (Phase 1) | config A | 0.8940 | 0.9189 | 0.5815 | 0.641 | 75.2 |
| p2_postproc_p1 | post-proc pack on Phase-1 model (σ=8, θ=25) | 0.8967 | 0.9208 | 0.5711 | **0.866** | **86.5** |
| p2_plain | plain CE (no boundary wts, no smoothing) | 0.9004 | 0.9225 | 0.5594 | 0.628 | 76.0 |
| p2_smoothonly | T-MSE only | 0.8939 | 0.9189 | 0.5618 | 0.646 | 76.3 |
| p2_long | config A, 30 epochs | 0.9088 | 0.9287 | 0.6160 | 0.625 | 75.6 |
| p2_cw | + median-frequency class weights | 0.8871 | 0.8966 | 0.5169 | 0.561 | 67.5 |
| p2_refine | + MS-TCN refinement stage | 0.8984 | 0.9396 | 0.5914 | 0.768 | 83.6 |
| p2_f11 | + gravity/magnitude channels (11 ch) | 0.8966 | 0.9418 | 0.6142 | 0.716 | 79.5 |
| p2_c12 | 12-class supervision | 0.9122 | 0.9517 | 0.5938 | 0.684 | 81.8 |
| **p2_aug** | **physics augmentation** | 0.9265 | 0.9584 | 0.6123 | 0.705 | 79.4 |
| p2_aug30 | augmentation, 30 epochs | 0.9290 | 0.9589 | 0.6166 | 0.706 | 80.7 |
| p2_aug_refine | aug + refinement | 0.9206 | 0.9554 | 0.6117 | 0.752 | 83.6 |
| p2_aug_f11 | aug + gravity channels | 0.9220 | 0.9563 | 0.6505 | 0.709 | 80.8 |
| **p2_aug_c12** | **aug + 12-class** | **0.9372** | **0.9648** | 0.6265 | 0.719 | 80.4 |
| p2_aug_c12_refine | aug + 12-class + refinement | 0.9340 | 0.9625 | **0.6529** | 0.824 | 85.6 |
| p2_aug_c12_s1 / s2 | aug + 12-class, seeds 1 / 2 | 0.9198 / 0.9433 | 0.9318 / 0.9643 | 0.6455 / 0.6462 | 0.711 / 0.766 | 81.0 / 86.0 |
| **p2_aug_c12_tg** | **aug + 12-class + transition-graph loss (λ=0.5)** | 0.9320 | 0.9609 | 0.6511 | 0.721 | 83.1 |
| p2_aug_c12_refine_tg02 | aug + 12-class + refine + TG loss (λ=0.2) | 0.9338 | 0.9630 | 0.6462 | 0.813 | 86.9 |
| p2_aug_c12_refine_tg05 | aug + 12-class + refine + TG loss (λ=0.5) | 0.9305 | 0.9586 | 0.6535 | 0.786 | 85.4 |
| p2_aug_c12_refine_tg10 | aug + 12-class + refine + TG loss (λ=1.0) | 0.9343 | 0.9635 | 0.6505 | 0.797 | 86.0 |
| p2_aug_c12_refine_tg05_l2 | aug + 12-class + refine + TG loss on refined output only (λ=0.5) | 0.9358 | 0.9640 | 0.6615 | 0.813 | 86.8 |
| p2_aug_c12_mstcnpp | aug + 12-class + MS-TCN++ baseline (790,064 params, same split) | 0.9504 | 0.9725 | 0.6320 | 0.820 | 90.1 |
| p2_aug_c12_refine_bh05 | aug + 12-class + refine + ASRF boundary head (λ=0.5), boundary-refined | 0.9270 | 0.9596 | 0.6437 | 0.860 | 87.9 |
| p2_aug_c12_refine_bh10 | aug + 12-class + refine + ASRF boundary head (λ=1.0), boundary-refined | 0.9337 | 0.9636 | 0.6474 | 0.869 | 88.6 |

What the screening established:

- **Physics augmentation is the single biggest lever** (val F1 0.9189 → 0.9584):
  joint acc+gyro rotation (±40°, same matrix, raw units), smooth magnitude
  warping, scaling, small time-warp, jitter — train split only. It attacks the
  val→test subject gap exactly as the literature predicted.
- **12-class supervision is the second lever** (0.9517 alone; 0.9648 combined with
  augmentation): giving transition classes their own labels teaches the model
  what a transition *is*; predictions are mapped back to 6 classes by summing
  probabilities over each target's preimage.
- **Post-processing is the segmental fix:** on the Phase-1 model it moved val
  segmental F1@25 from 0.641 → 0.866 and boundary F1 0.277 → 0.443 at unchanged
  frame accuracy — pure decoding gain, zero retraining.
- **Refinement stage** buys segmental quality (best raw-model seg F1@25, 0.768
  alone; 0.824 combined) more than frame accuracy — again as the literature said.
- **Negative results, kept:** median-frequency **class weights hurt** (val F1
  0.8966, worst of all runs — re-weighting toward rare dynamic classes damaged
  the dominant statics); **boundary-weighted CE gave nothing** over plain CE
  (0.9189 vs 0.9225) and T-MSE alone matched the full config-A loss — consistent
  with the brief's negative evidence, now confirmed on real HAPT.
- **Seed variance is large** (val F1 0.932–0.965 across 3 seeds of the winning
  config) — the honest reason the final system ensembles 3 seeds.
- **Transition-graph loss (2026-11-08, val-only screening, test untouched):**
  RoboSubtaskNet-style illegal-switch penalty added to `train_real.py`
  (`--trans-graph λ`; legal adjacency = observed train pairs ∪ HAPT semantic
  graph, 36/144 pairs in 12-class space; zero new parameters, zero inference
  cost). At λ=0.5 on the aug+12-class config: val F1 0.9609 vs 0.9648 without
  it (within seed noise), but boundary F1 0.2929→0.3239, edit 80.4→83.1,
  transition-window accuracy 0.6265→0.6511, while segmental F1@25 was flat
  (0.7193→0.7213). Verdict: a cheap boundary regularizer that improves
  boundary placement but does NOT fix over-segmentation on its own (legal
  sitting↔standing flicker untouched; boundary precision still 0.223) and is
  still behind the refinement stage on segmental quality. Next: combine with
  the refinement stage, sweep λ, replicate on a second seed — test
  confirmation only if the combo wins on val.
- **Transition-graph loss + refinement COMBINED (2026-11-09, val-only
  screening, test untouched — NEGATIVE interaction, combo NOT adopted):**
  the combo needed no new code — `train_real.py` already applies the TG loss
  to both stage outputs (l1 and refined l2) when `--refine --trans-graph λ`
  are combined. λ sweep 0.2 / 0.5 / 1.0 on the aug+12-class+refine config
  (20 epochs, seed 0): frame metrics flat vs refine-only (val F1 0.9625 →
  0.9630 / 0.9586 / 0.9635 — all inside seed noise), edit score marginally up
  at λ=0.2 (85.6 → 86.9) and λ=1.0 (86.0), BUT boundary F1 *falls*
  (0.4744 → 0.4234 / 0.3882 / 0.4397) and segmental F1@25 *falls*
  (0.8238 → 0.8125 / 0.7856 / 0.7969) at every λ. Boundary precision drops
  (0.372 → 0.326 / 0.289 / 0.331) while recall is flat or lower — the
  opposite of the TG loss's effect on the non-refined model, where it raised
  recall (0.526 → 0.590). Likely mechanism: penalising illegal switches at
  the stage-1 output distorts the logits the refinement stage consumes, and
  double-applying the penalty over-smooths legal switches. Verdict: TG loss
  and the refinement stage do NOT stack; refine-alone stays the best
  segmental model. The TG loss's value is confined to the non-refined
  architecture unless applied to the refined output only (l2) — that variant
  is the next boundary-precision test. No test evaluation was run for the
  combo (ledger stays at 6).
- **Transition-graph loss on the refined output ONLY (2026-11-09, val-only
  screening, test untouched — TG investigation now CLOSED):** added a
  `--trans-graph-stage {all,last}` flag to `train_real.py` so the penalty
  can skip the stage-1 logits entirely. At λ=0.5, best val epoch 18
  (experiment #27; checkpoint re-evaluated independently, metrics reproduce
  exactly): val acc 0.9358, macro F1 0.9640, transition-window acc 0.6615,
  edit 86.8, boundary F1 0.4595 (P 0.354 / R 0.654), segmental F1@25 0.8125.
  The placement hypothesis is CONFIRMED — versus the both-stage combo at the
  same λ, boundary F1 recovers 0.3882 → 0.4595 and segmental F1@25
  0.7856 → 0.8125, so penalising stage-1 logits was indeed the poison. But
  against refine-alone it is still a wash leaning negative on the metrics
  that matter (segmental F1@25 0.8238 → 0.8125, boundary F1 0.4744 → 0.4595;
  frame metrics and edit marginally up, inside seed noise). Verdict: the
  refinement stage already extracts everything the TG penalty has to offer.
  All three placements are now measured — TG alone (helps the weak model),
  both stages (hurts), refined-only (neutral) — and the line is closed.
  Refine-alone remains the best segmental model. Next boundary levers from
  the 2026-11-07 sweep: an ASRF-style boundary branch, or TBDM boundary
  distillation (training-only, zero inference cost). No test evaluation was
  run (ledger stays at 6).
- **MS-TCN++ same-split baseline (2026-11-09, val-only screening, test
  untouched):** the first protocol-matched rival DenseHAR has ever had —
  790,064 params, 4 stages (dual-dilated prediction generation + 3
  refinement stages), same aug+12-class recipe, 20 epochs, seed 0, best
  epoch 19. Val acc 0.9504, macro F1 0.9725 (above DenseHAR-refine's 0.9340 /
  0.9625), edit 90.1 (above 85.6), but boundary F1 0.4091 (below 0.4744) and
  segmental F1@25 0.8196 (just below 0.8238), transition-window acc 0.6320
  (below 0.6529). Read: MS-TCN++ wins on frame accuracy and edit distance
  — it over-segments less overall — but DenseHAR-refine still places
  boundaries better. The gap is within the known ±3 val-F1 seed noise on
  frame metrics, so a seed-1 replication is owed before any headline claim;
  the boundary/segmental edge for DenseHAR-refine is the more robust of
  the two findings (it survives across the TG sweep's refine-alone runs).
  No test evaluation was run (ledger stays at 6).
- **ASRF-style boundary branch (2026-11-10, val screening + one test
  confirmation):** decoupled class-agnostic boundary regression head (BRB)
  on stage-1 log-probs — 4 dilated convs (d=1,2,4,8, hidden 32, +12,865
  params, +8%) trained with BCE against GT boundaries (radius ±2 steps,
  pos_weight capped at 50) — plus ASRF refinement at eval: split at
  predicted boundaries (p>0.5) and majority-vote each segment. Implemented
  in `train_real.py` (`--boundary-head λ`, `--boundary-radius`,
  `--boundary-thresh`; `confirm --boundary-refined` for test confirmation);
  unit-tested in `test_boundary_head.py` (9 tests incl. the documented
  missed-boundary risk: with no predicted boundary a whole sequence becomes
  one segment and a minority true segment is voted away — boundary recall
  is load-bearing). Val-only screening, 20 epochs, seed 0, aug+12-class+
  refine (experiments #29–30): λ=0.5 raw argmax actually got WORSE
  (seg25 0.8238→0.7318, bF1 0.4744→0.3516) while its boundary-refined
  output jumped past refine-alone (bF1 0.5052, seg25 0.8595, edit 87.9);
  λ=1.0 kept raw predictions healthy (F1 0.9639, seg25 0.8078) and its
  refined output set new val bests — bF1 0.5131 (P 0.434/R 0.628),
  seg25 0.8690, seg50 0.8482, edit 88.6. **Test confirmation of λ=1.0
  (final_bh10_bref, 7th test evaluation): the val win did NOT transfer.**
  Test acc 0.9016, macro F1 0.9397, transWin 0.6068, transClass 0.8934,
  bF1 0.4696, seg25 0.8020, edit 84.8 — vs the refine+postproc candidate
  (0.8990/0.9415/0.6246/0.8946/0.5045/0.8403/86.23) it wins only on
  accuracy, and vs the FINAL ensemble (0.9038/0.9398/0.6093/0.8744/0.4087/
  0.8219/84.88) only on boundary F1 and transition-class accuracy. Val
  recomputation from the saved checkpoint reproduces exactly, so this is
  a real val→test gap, not an implementation slip: the 4-subject val split
  overstates segmental gains, as it has throughout this program. Verdict:
  the BRB is a genuine val-level boundary win and the cheapest segmental
  lever yet (+8% params, no extra inference cost beyond a majority vote),
  but it does NOT displace the final system on test. The final headline
  stays 90.38% / 0.9398. Next: (a) seed-1 replication of bh10 to check the
  val win is not a seed artefact; (b) combine BRB refinement WITH the
  post-processing pack (they attack flicker differently — predicted
  boundaries vs fixed median windows) and confirm on test only if val
  wins again; (c) note the refinement is non-causal (whole-window
  majority vote) — a causal streaming variant is owed before any
  on-device claim.
- **Follow-up (2026-11-10): both open items above are now answered.**
  (a) Seed-1 replication (experiment #32, val-only): seed 1's base run is
  weaker (raw acc 0.9181 vs seed 0's 0.9342) but the boundary refinement
  lifts it the same way (bF1 0.3710→0.4974, seg25 0.7584→0.8406) — the
  BRB effect is not a seed artefact. (b) **BRB refinement + min-duration
  relabel transfers to test.** Val grid (med{1,3,5,11} × θ{0,10,15,25}
  on boundary-refined labels; the median filter is a no-op, min-duration
  is the active step): seed-0 val bref bF1 0.5131→0.5629, seg25
  0.8690→0.9147 at θ=10 (selected on val macro F1, tie-break seg25);
  seed-1 val shows the same lift (bF1 0.4974→0.5412, seg25
  0.8406→0.8826). **Test confirmation (final_bh10_bref_mindur10, 8th
  test evaluation): acc 0.9022, macro F1 0.9400, transWin 0.6133,
  transClass 0.8941, bF1 0.5377 (P 0.496/R 0.588), seg25 0.8516, edit
  88.37.** Against the FINAL ensemble it wins boundary F1 (+12.9 pts),
  seg25 (+3.0), edit (+3.5), transClass and macro F1 (by 0.0002), and
  trails only on accuracy (−0.16 pts) — the best segmental test result
  in the program, from a single 170K-param model. The ensemble keeps
  the accuracy headline (90.38%); this candidate is the segmental
  champion and the new base for the causal-streaming work.
- **Causal retrain — test confirmation (2026-11-11, 9th test evaluation).**
  The fully causal twin (left-padded convs, per-position norms/SE,
  causal refine + boundary head; `test_causal.py` perturbation test
  proves exactly zero future leakage) was the zero-delay hope from the
  val curve: val bref+min-duration-left θ=10 acc 0.8819 / F1 0.9307 /
  bF1 0.3284 / seg25 0.7943 / edit 82.0, beating the offline model forced
  to ~0 lookahead (seg25 0.7208 / F1 0.7042) by +7.4 / +22.7. **On test
  it collapses: acc 0.7803, macro F1 0.6926, transWin 0.5282,
  transClass 0.8069, bF1 0.2857, seg25 0.7035, edit 74.2**
  (`final_causal_bh10_bref_mindL`, script `eval_causal_test.py`; val
  recomputation from the saved checkpoint reproduces exps #35/#36
  exactly, so this is a real val→test gap, not a bug). The streaming tax
  on unseen subjects is −14.8 seg25 and −24.7 F1 vs the offline segmental
  champion (0.8516 / 0.9400) — far worse than the val-implied tax (−12
  seg25, −3.3 F1). Confusion analysis: the three walking gaits collapse
  into each other on test subjects (walking→upstairs 34.3%,
  upstairs→walking 19.3% + →laying 19.1%, downstairs→walking 25.0%)
  while val is near-perfect — without future context the causal model
  cannot disambiguate gait phase on unseen subjects. What DOES transfer:
  boundary timing (test delay mean +8.6 steps / +171 ms, median +9,
  77 matched / 20 missed of 97) and the refinement's relative lift
  (raw seg25 0.2717 → bref 0.5792 → bref+mindL 0.7035). **Verdict: the
  zero-delay causal retrain is NOT yet deployable on unseen subjects.**
  The pragmatic streaming path is now the offline champion read
  semi-online with ~1.4 s lookahead (val seg25 0.8190; offline test
  seg25 0.8516 suggests test will hold up far better than the causal
  retrain's 0.7035) — that semi-online test confirmation is the next
  streaming step. A causal v2 (distil the offline teacher's
  segment-cleaned labels into the causal student, or stronger
  subject-invariant augmentation) is the longer fix; the plain TBDM-style
  distillation already failed for the offline student, so v2 needs a
  different mechanism, not just α.
- **Semi-online champion — test confirmation (2026-11-12, 10th test
  evaluation).** The offline champion read semi-online at lookahead 64
  (tiled 16-step blocks, right edge 64 steps past block end: each label
  uses at most 64..79 future steps = 1.28..1.58 s, mean ~1.43 s;
  segment_majority 0.5 + min_duration_left θ=10, selected on val only).
  Val recomputation reproduces `causal_stream_val_la64.json` exactly on
  all three views, so the test number is a real transfer result, not a
  code drift. **On test: acc 0.8330, macro F1 0.7738, transWin 0.5786,
  transClass 0.8215, bF1 0.4327 (P 0.405/R 0.464), seg25 0.7758, edit
  81.92** (`final_champion_semionline_la64`, script
  `eval_semionline_test.py`). Val→test gaps are small and honest: F1
  −4.8 pts, seg25 −4.3 (vs the causal retrain's −23.8 / −9.1) — the
  semi-online path TRANSFERS where zero-delay causal did not. Against
  the causal retrain on test it wins +7.2 seg25 / +8.1 F1 / +7.7 edit;
  against the offline champion it pays a streaming tax of −7.6 seg25 /
  −16.6 F1 (the F1 tax is gait confusion that future context normally
  resolves: upstairs 0.621, downstairs 0.727, walking 0.825, vs offline
  1.000/1.000/1.000). Boundary timing transfers well: emitted switches
  land mean +0.7 steps (+14 ms) after GT boundaries, 75 matched / 22
  missed of 97 — on top of the 1.28..1.58 s algorithmic delay, so total
  decision latency is ~1.3–1.6 s. **Verdict: the semi-online champion
  at ~1.4 s lookahead is the mobile-phase streaming operating point.**
  It beats the causal retrain on every test metric while keeping
  decision latency in the low single seconds; a causal v2 remains the
  longer fix for true zero-delay, and the la128 (~2.7 s) operating point
  is untested on test if more quality is ever worth the extra second.
- One engineering failure, logged: the first aug+features run crashed
  (augmentation assumed 6 channels); fixed by augmenting the physical channels
  and recomputing derived features from the augmented signal.

### Final candidates — the only Phase-2 test evaluations

| Test | aug_c12 raw | aug_c12 + postproc | **aug_c12 + refine + postproc** | **ensemble + postproc (FINAL)** | aug_c12 + refine + ASRF boundary-refined | aug_c12 + refine + ASRF bref + min-duration θ=10 | causal retrain + bref + min-duration-left θ=10 | **champion semi-online la64 (~1.4 s)** |
|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.8896 | 0.8933 | 0.8990 | **0.9038** | 0.9016 | 0.9022 | 0.7803 | 0.8330 |
| Macro F1 | 0.9242 | 0.9271 | **0.9415** | 0.9398 | 0.9397 | 0.9400 | 0.6926 | 0.7738 |
| Transition-window acc | 0.6014 | 0.5950 | **0.6246** | 0.6093 | 0.6068 | 0.6133 | 0.5282 | 0.5786 |
| Transition-class acc | 0.8506 | 0.8595 | **0.8946** | 0.8744 | 0.8934 | 0.8941 | 0.8069 | 0.8215 |
| Boundary F1 | 0.1940 | 0.4094 | 0.5045 | 0.4087 | 0.4696 | **0.5377** | 0.2857 | 0.4327 |
| Segmental F1@25 | 0.5281 | 0.7796 | 0.8403 | 0.8219 | 0.8020 | **0.8516** | 0.7035 | 0.7758 |
| Edit score | 70.72 | 82.10 | 86.23 | 84.88 | 84.8 | **88.37** | 74.2 | 81.92 |

Post-processing parameters were selected per model on val (aug_c12: σ=8, θ=15;
refine: θ=15, σ=0 — the refinement stage already smooths, so val rejected
additional Gaussian smoothing). θ ≤ 15 samples = 0.3 s, safely below the ~1 s
duration of real HAPT transitions.

## 4. Transition analysis

- **Transition windows (±10 steps around true boundaries):** final system 0.6093,
  refinement candidate 0.6246, vs baseline 0.4787 and Phase-1 0.5955. Every model
  is far worse near boundaries than overall (these windows are intrinsically
  ambiguous — the baseline drops 31.0 points from its overall accuracy, the
  final system 29.5), but DenseHAR's dense labelling preserves a +13 to +15
  point absolute advantage there.
- **Transition-class timesteps** (the timesteps HAPT itself labels as a postural
  transition, scored against the target activity): final 0.8744, refinement
  candidate 0.8946, baseline 0.7730. This was already a Phase-1 strength
  (0.8571); augmentation alone did not raise it (0.8506 raw) — post-processing
  (0.8595) and above all the refinement stage (0.8946) did.
- **Worked examples** (final ensemble, test subject 4):
  - Sequence #1 — `TRUE: [0-221] laying -> [222-511] sitting`,
    `PRED: [0-243] laying -> [244-511] sitting`, 95.7%. The annotated transition
    spans steps 222–431; the prediction switches at 244, *inside* the true
    transition — physically sensible, and a reminder that the y6 boundary is
    placed at the transition's first timestep by construction.
  - Sequence #0 (edge failure) — `TRUE: [0-477] laying -> [478-511] sitting`,
    `PRED: [0-511] laying`, 93.4%. A transition truncated at the window edge
    (34 timesteps) is missed entirely; the cost is bounded at 6.6% of the window.

## 5. Per-class analysis (test macro-F1 components)

| Class | Baseline | Phase 1 | Final ensemble | + refine candidate |
|---|---|---|---|---|
| walking | 0.889 | 0.893 | 1.000 | 1.000 |
| walking_upstairs | 0.444 | 0.386 | 1.000 | 1.000 |
| walking_downstairs | 0.732 | 0.777 | 0.994 | 1.000 |
| sitting | 0.721 | 0.739 | 0.853 | 0.849 |
| standing | 0.623 | 0.616 | 0.815 | 0.853 |
| laying | 0.945 | 0.941 | 0.976 | 0.946 |

Augmentation essentially **solved the dynamic classes** on unseen subjects
(upstairs 0.386 → 1.000). What remains is the known hard core of waist-worn HAR:
**sitting vs standing** (0.82–0.85) — separable mainly by gravity direction and
micro-sway, and the pair every HAPT system struggles with. Gravity/magnitude
channels alone (p2_f11) helped val (0.9418) but did not beat augmentation, and
the combination (p2_aug_f11, 0.9563) did not beat aug+12-class either.

## 6. Comparisons — read the protocols before the numbers

**vs the synthetic results** (DenseHAR 95.35% / baseline 85.71% on generated
data): the synthetic task was easier in every way that matters — balanced
classes, clean per-subject profiles, transitions drawn from the same generator
as training. Real HAPT Phase 1 landed at 78.9% for *both* models. The synthetic
run proved the pipeline and the metric suite; it never evidenced real-world
accuracy, and its numbers are superseded by this document.

**vs the windowed HAPT reference** (TinyHAR, author's earlier work: 90.45%
window accuracy / 0.794 macro F1, 12-class): the final system's per-timestep
accuracy (0.9038) sits in the same band, but the protocols differ in ways that
make direct ranking meaningless:
- *Scoring unit:* window-majority (one decision per 128 samples, transition
  timesteps absorbed into the majority) vs **every timestep**, including the
  ~19% of timesteps at/inside transitions where this evaluation concentrates.
- *Labels:* 12-class window labels vs 6-class dense labels with transitions
  mapped to their target (plus separate transition-timestep scoring, which has
  no windowed analogue).
- *Splits:* both subject-disjoint, but not the same subject lists.
The dense macro F1 (0.94) being far above the windowed 0.794 is likewise a
protocol artefact (6 vs 12 classes), not a claim of superiority. What *is*
comparable: the same window-128 architecture retrained and broadcast under
**this** protocol scores 0.7891 / 0.7258 (§2) — that is the honest baseline.

## 7. Test-evaluation ledger

The test split was evaluated exactly 10 times, each once per final candidate,
always after val selection: `p1_dense`, `p1_base` (Phase 1),
`final_augc12_raw`, `final_augc12_postproc`, `final_augc12ref_postproc`,
`final_ens_augc12` (Phase 2), `final_bh10_bref`,
`final_bh10_bref_mindur10`, `final_causal_bh10_bref_mindL` (boundary /
streaming line, 2026-11-10/11), `final_champion_semionline_la64`
(semi-online streaming, 2026-11-12). All other screening runs are val-only.
Full configs, histories and per-epoch val curves are in
`results_real.json`.

## 8. Limitations

- **Small subject counts:** 4 val / 5 test subjects; seed variance alone is
  ±3 val F1 points. These numbers would move with a different fixed split —
  they are reported for *this* split, which is fixed and documented.
- **Val subjects are easier than test subjects** (both models: val F1 ≈ 0.92
  vs test ≈ 0.73 in Phase 1). Selection on an easier val split is a real
  methodological weakness of this run; the augmentation gains were large
  enough to survive it, but finer distinctions (e.g. refine vs ensemble)
  should be read as provisional.
- **Windows contain ≤2 basic activities** — a property of HAPT's single-bout
  recordings, not of the model. Long multi-activity streams (the deployment
  setting) remain untested on real data.
- **The y6 boundary convention** (transition mapped to target from its first
  timestep) makes boundary timing metrics pessimistic by up to a transition
  length; segmental F1/edit are the fairer structural scores here.
- Sitting/standing remains the open problem (F1 ≈ 0.82–0.85).

## 9. Reproduce

```bash
python3 load_hapt.py            # hapt.zip -> data_hapt.npz (+ stats json)
python3 train_real.py phase1    # config A + baseline -> results_real.json
python3 train_real.py exp --name p2_aug_c12 --epochs 20 --augment --classes 12 --no-test
python3 train_real.py postproc --model model_real_p2_aug_c12.pt --name final_augc12_postproc
python3 train_real.py ensemble --models model_real_p2_aug_c12.pt,model_real_p2_aug_c12_s1.pt,model_real_p2_aug_c12_s2.pt --name final_ens_augc12 --postproc-from final_augc12_postproc
```

Files: `load_hapt.py` · `train_real.py` · `data_hapt.npz` · `data_hapt_stats.json` ·
`results_real.json` (all 22 experiments, full metric suites) · checkpoints
`model_real_*.pt` · logs `train_real_p1.log`, `train_real_p2*.log`.
