# Real-Data Path (Kaggle) — DenseHAR on HAPT / UCI-HAR

Goal: replace the synthetic generator with real dense labels and run the identical
protocol (subject-disjoint tuning/test, sweep, segmental metrics) on GPU.

## Datasets
1. **HAPT (UCI Machine Learning Repository, id 341)** — the natural benchmark for this
   project: 30 subjects, smartphone acc+gyro @50 Hz, 6 basic activities **plus 6
   postural transitions** (stand-to-sit, sit-to-stand, sit-to-lie, lie-to-sit,
   stand-to-lie, lie-to-stand) with **per-timestep labels** in `RawData/labels.txt`
   (experiment, user, activity, start, end). No label reconstruction needed.
2. **UCI-HAR (id 240)** — raw inertial signals (`Inertial Signals/` + activity labels per
   window) can be re-windowed into dense streams by concatenating a subject's windows in
   order; transitions are approximate (window boundaries), so HAPT is primary.

## Kaggle notebook plan
1. Attach the UCI HAPT mirror dataset (if the mirror is not wired, symlink it into
   `/kaggle/working/data/` before the loader runs — direct UCI downloads are unreliable
   on the Kaggle network path).
2. Build sequences: per subject/experiment, slice the labelled stream into 512-step
   sequences that contain ≥1 transition; map the 6 postural-transition classes either
   (a) to their own labels (12-class problem) or (b) to the following activity
   (6-class, comparable to this repo). Report both.
3. Split by subject: train 21 / val 4 / test 5 subjects (fixed list in the notebook).
4. Reuse this repo unchanged: `model.py`, `metrics.py`, and `train.py` stages —
   only the loader changes (`.npz` with the same `X (N,512,6)`, `y`, `subjects` schema).
5. GPU budget: the full sweep + final + baseline is <30 min on a T4 at these sizes.
6. Report the same table as the README (frame acc, macro F1, transition-window acc,
   boundary F1, segmental F1@10/25/50, edit score) for DenseHAR vs the window baseline.
   **Only numbers from that run may be quoted as real-benchmark results.**

## Expected differences vs synthetic
Real transitions are messier (HAPT annotates the transition itself as a class),
per-class confusion between upstairs/downstairs is higher, and over-segmentation
typically worsens before the smoothing-loss weight is re-tuned on the validation subjects.
