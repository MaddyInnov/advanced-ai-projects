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

## Experiment design

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

## Results (all measured — `results.json`)

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

- **Synthetic data, simulated federation.** All clients live in one
  process; there is no networking, straggler/dropout handling, or
  communication-cost measurement — the three things that dominate real
  deployments.
- **Privacy accounting is a simplified estimate** (formula above), suitable
  for comparing σ levels here, *not* a certified (ε, δ) claim. Use Opacus
  for anything real.
- DP runs use IID clients to isolate the privacy cost; DP × non-IID
  combined (where noise and drift compound) is the obvious next experiment.
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
results.json    measured accuracies, curves, histograms, epsilon estimates
```
