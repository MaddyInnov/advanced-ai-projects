# FedGuard — Federated Learning with Differential Privacy

**Train across many devices without ever centralising the data — then prove, with a formal privacy budget, what the model cannot learn about any individual.**

FedGuard is a from-scratch federated learning system in PyTorch comparing
**FedAvg**, **FedProx** and **DP-SGD** on IID vs non-IID client data, with a
centralised model as the ceiling. It is the training architecture behind
products that learn from phones — next-word prediction, on-device health /
activity recognition, fraud models across banks — where the raw data is too
private or too regulated to pool.

*Author: Ashutosh Sahoo — M.Tech (AI), NIT Silchar. Built as part of the
Advanced AI Projects portfolio; the federated setting connects directly to
his on-device Human Activity Recognition research (InceptionKAN / DenseHAR),
where IMU data from a person's phone is exactly the kind of data that should
never have to leave the phone.*

---

## Real benchmark results (MNIST) — the headline

The original results in this project were on synthetic tabular data. They
have been **re-run on real data: MNIST** (LeCun et al.), and the real-data
results below are the headline. The synthetic results are kept further
down, demoted to *preliminary*.

**Data & provenance.** `mnist.npz` from
`https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz`
(sha256 `731c5ac6…b673d1`, downloaded into this folder; counts verified:
60,000 train / 10,000 test, 28×28 uint8). **Adaptation:** images are
scaled to [0,1], bilinearly downscaled to **16×16 and flattened (256-dim
input)**, so the project's shared MLP becomes 256→64→64→10 (21,258
params). A stratified 5,000-image **validation** split (500/class, seed
777) is carved from the training images; the federated training pool is
the remaining **55,000**. Test = the **full 10,000-image MNIST test set**,
identical for every row. Clients: 10; IID = shuffle split; non-IID =
Dirichlet α=0.5 label skew (same partition code and seeds as the
synthetic run). All methods are the project's own — FedAvg aggregation,
FedProx proximal term, DP-SGD with true per-sample clipping
(`torch.func` grad+vmap), and the project's simplified RDP **epsilon
estimate** (see the caveat below). Full record: `results_real.json`;
reproduce with `python train_real.py all`.

> ⚠️ **Epsilon caveat (read before quoting any ε here):** every ε in this
> project is a **simplified RDP-style ESTIMATE** from
> `federated.estimate_epsilon`, for comparing noise levels inside this
> project only. It is **not a formal/certified (ε, δ) privacy guarantee**
> and must never be placed on the same axis as an Opacus / TF-Privacy
> accountant ε without this caveat.

### Final best table (Phase B, validation-selected)

Phase A (below) fixed the baseline protocol. Phase B then screened
literature-backed levers on the **validation split only** — more rounds,
local-epoch tuning, FedProx μ sweep, server momentum (FedAvgM), FedAdam,
DP clip/LR tuning — and the val-best config of each row family was
evaluated **once** on test. This is that final table:

| Experiment (MNIST-16, 10 clients) | Test accuracy | Test macro-F1 | ε estimate (δ=10⁻⁵) |
|---|---|---|---|
| Centralised (pooled, AdamW, 30 epochs) | **0.9756** | 0.9753 | — (no privacy) |
| FedAvg — IID (15 rounds × 3 local epochs) | 0.9620 | 0.9617 | — |
| FedAvg — non-IID, best (40 rounds × 2 local epochs) | **0.9653** | 0.9651 | — |
| FedProx — non-IID, best (μ=0.01, 15 × 3) | 0.9520 | 0.9517 | — |
| FedAvg + DP-SGD — σ=1.2, lr 2.0, 20 rounds | 0.9054 | 0.9041 | ≈ 5.55 (estimate) |
| FedAvg + DP-SGD — σ=0.7, lr 2.0, 20 rounds | 0.9083 | 0.9070 | ≈ 10.2 (estimate) |

**Privacy–utility frontier** (best val-selected accuracy at each
epsilon-estimate level; all ε are the project's estimate, see caveat):

| ε estimate | σ | Rounds | Test accuracy |
|---|---|---|---|
| ≈ 3.81 | 1.2 | 10 | 0.7898 |
| ≈ 5.55 | 1.2 | 20 | 0.9054 |
| ≈ 10.2 | 0.7 | 20 | 0.9083 |
| ≈ 15.31 | 0.7 | 40 | 0.9218 |

Reading: at ε̂ ≈ 5.55 the DP model gives up ~6 points against the
non-private federated best (0.9054 vs 0.9653); loosening to ε̂ ≈ 10.2
buys almost nothing (+0.3 pts), and the extra spend to ε̂ ≈ 15.31 buys
+1.6 pts — the frontier flattens fast in this small-MLP federated regime.
For context (different protocol/accountant, not ε-comparable): Abadi et
al.'s centralised DP-SGD reached ~97% at ε=8 on full MNIST with a much
larger training setup; the published FedAvg-on-MNIST (α=0.5) range of
85–99% is mostly a *rounds* effect, which is exactly what Phase B found
here too.

### Phase A — baseline table (original protocol shape, real data)

Same grid and protocol shape as the synthetic run (centralised AdamW;
federated 15 rounds × 3 local epochs, local SGD lr 0.08, batch 32;
FedProx μ=0.1; DP 20 rounds × 1 local epoch), with four documented
changes (full list in `results_real.json`): training pool 55,000 (5k
held out as validation); centralised epochs 80 → 30 (the MNIST-16 val
curve plateaus far earlier — 0.9634 at epoch 10, 0.9710 at epochs
29–30); **DP rows run on the non-IID partition** as specified for this
real run (the synthetic DP rows used IID, so the two DP sets are not
partition-matched); **DP lot size 32 → 256 and DP lr 0.15 → 0.4**
(measured `torch.func` per-sample step cost ≈195 ms at lot 32 — ~110 min
per σ at the original scale — vs ≈34 ms at lot 256; lot ≥ 256 is also
the standard DP regime where noise averages down over the lot).

| Phase A baseline | Val accuracy | Test accuracy | Test macro-F1 | ε estimate |
|---|---|---|---|---|
| Centralised (30 epochs) | 0.9710 | **0.9756** | 0.9753 | — |
| FedAvg — IID | 0.9604 | 0.9620 | 0.9617 | — |
| FedAvg — non-IID (α=0.5) | 0.9578 | 0.9623 | 0.9620 | — |
| FedProx — non-IID (μ=0.1) | 0.9074 | 0.9150 | 0.9137 | — |
| FedAvg + DP-SGD — σ=0.7 (non-IID) | 0.8226 | 0.8353 | 0.8325 | ≈ 10.2 (estimate) |
| FedAvg + DP-SGD — σ=1.2 (non-IID) | 0.8238 | 0.8373 | 0.8345 | ≈ 5.55 (estimate) |

### Phase B — improvement log summary (all screened on validation)

Every attempt is logged with its val score and keep/drop decision in
`results_real.json` (`improvements_log`, 19 entries). Headlines:

- **Rounds (the big lever, as the literature said):** FedAvg non-IID
  15×3 → 40 rounds × 2 local epochs: val 0.9578 → **0.9622** (keep;
  final test 0.9653). The late-round val curve oscillates in a
  0.960–0.963 band — a plateau at this client LR, reported as such.
- **Server momentum (FedAvgM, β=0.9): val 0.9198 — dropped.** At server
  lr 1.0 the momentum amplifies the per-round pseudo-gradient and the
  run ends *below* plain FedAvg. **FedAdam (η=0.1): val 0.9498 —
  dropped** for the same reason (below the rounds-only winner). Both
  are honest negatives at the settings screened; they were not
  test-evaluated.
- **FedProx μ sweep:** val is monotone decreasing in μ — μ=1.0: 0.6684,
  μ=0.1: 0.9074, μ=0.01: **0.9492** (family winner) — and even the best
  μ trails plain FedAvg (0.9578). At α=0.5 with these round counts the
  proximal anchor mostly *slows* learning on this task; FedProx does
  not help here, and the final table says so.
- **DP tuning:** at fixed σ, halving the clip to C=0.5 at lr 0.4 hurt
  badly (val ≈0.737 at both σ — smaller steps, same noise scale,
  undertrained). **LR was the lever:** lr 0.4 → 1.0 → 2.0 moved σ=0.7
  val 0.8226 → 0.8798 → **0.8992** and σ=1.2 val 0.8238 → 0.8780 →
  **0.8964** (final tests 0.9083 / 0.9054 at unchanged ε̂ levels). A
  40-round σ=0.7 lr-2.0 run reached val 0.9130 (test 0.9218) at the
  higher ε̂ ≈ 15.31 level, completing the frontier above.
- **Dropped for CPU budget (not run, no numbers claimed):** a
  FedProx 40-round run, an IID 40-round run and a 60-epoch centralised
  run were cut because the shared 1.5-CPU cell made the recorded
  baseline alone take ~62 min of wall-clock compute; the rounds lever
  is demonstrated on the non-IID family and the centralised val curve
  shows its plateau. Total recorded compute for the real-data run:
  ~6,071 s across 19 configurations (`results_real.json`).

---

## Why federated + differentially private?

- **Data minimisation by architecture.** In federated learning the *model*
  travels to the data. Clients train locally and send only weight updates;
  raw records never leave the device. That is GDPR data-minimisation and
  purpose-limitation enforced by design, not by policy.
- **But weight updates can leak.** Gradients memorise training examples
  (membership-inference and gradient-inversion attacks are real).
  **Differential Privacy** closes that hole: DP-SGD clips each example's
  influence and adds calibrated Gaussian noise, giving a formal
  (ε, δ) guarantee — *the model's behaviour would barely change if any one
  person's data were removed.*
- **Reality is non-IID.** Real clients don't hold random samples of the
  world: one user's phone mostly sees *their* activities, one hospital
  mostly sees *its* patient mix. That label skew causes **client drift** —
  local models pull in incompatible directions — and it is the central
  engineering problem of federated learning, addressed here by FedProx.

## Methods, briefly

| Method | Idea |
|---|---|
| **Centralised** | Pool everything, train normally. The accuracy ceiling — and the privacy worst case. |
| **FedAvg** (McMahan et al., 2017) | Each round: broadcast global weights → clients run local SGD epochs → server averages updates weighted by client dataset size. |
| **FedProx** (Li et al., 2020) | FedAvg + a proximal term `(μ/2)·‖w − w_global‖²` in each client's loss, penalising drift away from the global model. Directly targets non-IID heterogeneity. |
| **DP-SGD** (Abadi et al., 2016) | Per-sample gradient clipping to norm `C`, then Gaussian noise `N(0, (σC)²)` added to the summed clipped gradients before averaging. Implemented here with **true per-sample gradients via `torch.func` (`grad` + `vmap`)** — not batch-level clipping. |
| **Non-IID partition** | Dirichlet(α=0.5) label-skew partition (per class, proportions over clients ~ Dirichlet), the standard simulation from the federated literature. |
| **Privacy accounting** | A **simplified RDP-style estimate**: `RDP(α) ≈ steps · α · q² / (2σ²)`, `ε = min_α [RDP(α) + log(1/δ)/(α−1)]`, δ=10⁻⁵, q = batch/client-size. ⚠️ This is a textbook-order **estimate for comparing noise levels inside this project**, not a certified accountant — production must use Opacus / TF-Privacy's exact RDP accountant (see Limitations). |

## Experiment design (synthetic run — preliminary)

*This section and the results table below it describe the original
synthetic-data run. The real-data (MNIST) design is in the headline
section above; it reuses this design's methods and protocol shape.*

- **Data (synthetic, no external dataset):** 10-class, 20-dim structured
  Gaussian-mixture problem — class means + class-specific scaling and a
  quadratic interaction, so an MLP earns its keep. 5,000 train / 1,000
  held-out test, balanced classes, fixed seed.
- **Clients:** 10. IID = random split. Non-IID = Dirichlet α=0.5 (each
  client ends up dominated by 2–3 classes — visible in `demo.py`).
- **Model:** MLP 20→64→64→10 with dropout 0.3 (**6,154 parameters**),
  shared by every method. (Dropout earned its place in calibration: the
  plain MLP overfit the pooled data — train 96.8% / test 88.0% — while
  dropout 0.3 shrank the gap to ~93% / ~89%. In the DP-SGD path,
  per-sample gradients are computed in eval mode, so dropout is inactive
  there.)
- **Grid:** centralised · FedAvg-IID · FedAvg-nonIID · FedProx-nonIID
  (μ=0.1) · FedAvg+DP at σ ∈ {0.7, 1.2} (IID, isolating the DP cost).
  Federated runs: 15 rounds × 3 local epochs, local SGD (lr 0.08,
  batch 32), all 10 clients every round. Centralised: the best standard
  pooled recipe from calibration — AdamW (lr 10⁻³, cosine, 80 epochs);
  federated clients use local SGD exactly as in the FedAvg/FedProx
  literature. DP runs: 20 rounds × 1 local epoch, DP-SGD lr 0.15,
  clip C=1.0. Same test set, same init seed for every run.

## Results — synthetic data (preliminary; all measured — `results.json`)

| Experiment | Test accuracy | ε estimate (δ=10⁻⁵) |
|---|---|---|
| Centralised (pooled, AdamW) | **0.890** | — (no privacy) |
| FedAvg — IID | 0.886 | — |
| FedAvg — non-IID (Dirichlet α=0.5) | 0.875 | — |
| FedProx — non-IID (μ=0.1) | 0.885 | — |
| FedAvg + DP-SGD — σ=0.7 (IID) | 0.839 | ≈ 9.2 |
| FedAvg + DP-SGD — σ=1.2 (IID) | 0.841 | ≈ 5.0 |

**What the numbers say:**

1. **Federation on IID data is nearly free:** FedAvg-IID gives up only
   0.4 points against pooling all the data (0.886 vs 0.890).
2. **Non-IID data costs real accuracy:** label skew drops FedAvg to
   0.875 (−1.1 vs IID). The drift is most violent early — after round 1
   the non-IID global model scores 0.710 vs 0.868 for IID — and the
   curves in `results.json` show it slowly repairing.
3. **FedProx recovers almost all of it:** μ=0.1 lifts non-IID accuracy
   to 0.885, recovering 1.0 of the 1.1 points lost and statistically
   tying IID. (In a harder calibration regime, mean-scale 0.6, the same
   μ gave +2.9 points: 0.818 vs 0.789.)
4. **Privacy costs ~4.5 points, and at this scale the stronger
   guarantee is free:** DP-SGD lands at 0.839 (σ=0.7, ε≈9.2) and 0.841
   (σ=1.2, ε≈5.0). The two σ levels differ by 0.2 points — *two test
   examples* — well within noise at n=1000 (SE ≈ 1.1 points), so we
   report them as tied on accuracy while their privacy budgets differ
   substantially: if you run this system, take σ=1.2. The honest
   headline is the ~4.5-point cost of DP itself, visible in both rows.
5. **A calibration finding worth knowing:** periodic weight averaging
   is itself a strong regulariser. In early calibration, FedAvg-IID
   repeatedly *beat* a plain SGD pooled run (which oscillated and
   overfit); the centralised baseline above therefore uses the best
   standard pooled recipe (AdamW + dropout), disclosed here rather
   than buried — with that recipe the expected ordering
   (centralised ≥ federated) holds, by 0.4 points.

Per-round accuracy curves for every run are stored in `results.json`.
The full grid reruns in ~1.5–10 minutes on CPU depending on load
(measured 83 s on an uncontended 4-core VM).

## Run it

```bash
pip install -r requirements.txt
python data_gen.py        # inspect the data + non-IID histograms
python experiments.py     # full grid -> results.json + checkpoints/  (minutes on CPU)
python demo.py            # client distributions, checkpoint accuracy, privacy–utility table
# or simply: python train.py   (alias for experiments.py)

# Real-data (MNIST) run — the headline results above:
python mnist_data.py      # verify counts, build the 16x16 cache (mnist16.npz)
python train_real.py all  # baseline + Phase B screening -> results_real.json
                          # (~100 min of recorded compute on a shared 1.5-CPU cell)
```

## SOTA context — honestly

FedGuard is a faithful, minimal re-implementation of the canonical stack,
not a new algorithm: FedAvg (2017) defined the field; FedProx / FedNova /
SCAFFOLD are the standard answers to heterogeneity (SCAFFOLD's control
variates are the natural next addition here); DP-SGD (Abadi et al., 2016)
and RDP accounting (Mironov, 2017) are the standard privacy machinery.
Production federated systems add what a single-machine simulation cannot
show: **secure aggregation** (the server sees only the *sum* of updates),
client sampling at million-device scale, and systems frameworks such as
**Flower** or TensorFlow Federated. The value of this project is that every
moving part — partitioning, drift, proximal correction, per-sample
clipping, noise, accounting — is implemented and measured, not imported.

## Production path

1. Port the client/server loop to **Flower** (`flwr`) — the aggregation and
   client-training interfaces here map 1:1 onto Flower strategies
   (`FedAvg`, `FedProx` are built-in; DP via Opacus-wrapped client training).
2. Replace the simplified accountant with **Opacus**' RDP accountant and
   add secure aggregation (Flower + SecAgg+).
3. Apply to on-device HAR: clients = phones with IMU windows (DenseHAR
   encoder as the local model), server never sees a single activity trace.

## Skills demonstrated

Federated optimisation · non-IID / heterogeneous data · FedAvg & FedProx ·
differential privacy & DP-SGD · per-sample gradients with `torch.func` ·
privacy accounting (RDP) · experiment design with controlled ablations ·
PyTorch · privacy/ML system trade-off analysis.

## Limitations

- **Simulated federation.** All clients live in one process, on both the
  synthetic and the MNIST runs; there is no networking, straggler/dropout
  handling, or communication-cost measurement — the three things that
  dominate real deployments.
- **Privacy accounting is a simplified estimate** (formula above), suitable
  for comparing σ levels here, *not* a certified (ε, δ) claim. Use Opacus
  for anything real.
- The synthetic run's DP rows use IID clients to isolate the privacy cost;
  the MNIST run's DP rows use the non-IID partition (noise and drift
  compound), as specified for the real run — the two DP sets are not
  partition-matched.
- MNIST inputs are downscaled to 16×16 (256-dim) to fit the project's
  MLP on CPU; absolute accuracies are therefore not comparable to
  full-resolution 28×28 MNIST numbers without that caveat.
- Phase B selection used a single validation split and a single seed;
  differences under ~0.5 points (e.g. between adjacent frontier levels)
  should be read as ties, not rankings.
- No secure aggregation or defence against a malicious server beyond DP;
  DP bounds *what the released model reveals*, it does not authenticate
  clients (poisoning defence is out of scope).

## Files

```
data_gen.py     synthetic data + IID / Dirichlet(alpha=0.5) partitions
model.py        MLP classifier + parameter count
federated.py    FedAvg / FedProx / DP-SGD engine, aggregation, epsilon estimate
experiments.py  the experiment grid -> results.json + checkpoints/
train.py        alias entry point
demo.py         client histograms + checkpoint eval + privacy-utility table
results.json    synthetic-run accuracies, curves, histograms, epsilon estimates (preliminary)
mnist.npz       MNIST as downloaded (60,000 train / 10,000 test, 28x28)
mnist_data.py   MNIST loader: provenance, 16x16 downscale, validation carve-out
train_real.py   real-data baseline + Phase B screening + finalize -> results_real.json
results_real.json  MNIST baseline, per-attempt improvement log, privacy frontier, final table
checkpoints_real/  saved model states for every real-data configuration
```
