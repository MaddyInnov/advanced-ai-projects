"""Prepare HARTH in the DenseHAR trainer format (data_harth.npz).

Mirrors load_hapt.py's output schema exactly: X (N,512,6) standardized
windows, y6 == y12 (N,512) int64 class indices 0..11, y_trans (N,512)
bool, subjects (N,), split (N,) 0/1/2, plus subject lists and norm stats.

Protocol (declared before any training, fixed):
- Subject-disjoint split by sorted subject id: val = ids at sorted
  indices [1, 11, 21] -> [8, 19, 29]; test = indices [3, 8, 13, 18]
  -> [10, 16, 21, 26]; train = remaining 15 subjects.
- Subject 6 (100 Hz) downsampled x2 to 50 Hz before everything.
- Standardization: per-channel mean/std from TRAIN subjects' raw
  streams only.
- Windows: 512 steps (10.24 s at 50 Hz), stride 256, per subject;
  all windows kept (dense task, transitions included).
- y_trans: timesteps within +/-25 steps (0.5 s) of a label change,
  computed on the full stream before windowing (HARTH has no named
  transition classes; this is the boundary-adjacency analogue of
  HAPT's transition-class mask, and is labelled as such in results).
- Label map: sorted original codes [1,2,3,4,5,6,7,8,13,14,130,140]
  -> 0..11.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOAL_CODE = os.path.expanduser(
    "~/workspace/goals/transition-aware-dense-har-research/code")
sys.path.insert(0, GOAL_CODE)

from harbench.harth import load_harth, LABEL_NAMES  # noqa: E402

WINDOW, STRIDE, TRANS_HALF = 512, 256, 25


def main():
    subs = load_harth(os.path.join(GOAL_CODE, "data"))
    # Unify rate: subject 6 is 100 Hz -> take every 2nd sample.
    if subs[6]["median_dt_ms"] < 15:
        subs[6] = dict(subs[6], X=subs[6]["X"][::2], y=subs[6]["y"][::2])
        print("subject 6 downsampled x2 ->", len(subs[6]["y"]), "samples")

    ids = sorted(subs)
    val_ids = [ids[i] for i in (1, 11, 21)]
    test_ids = [ids[i] for i in (3, 8, 13, 18)]
    train_ids = [s for s in ids if s not in val_ids + test_ids]
    print("train", train_ids, "\nval", val_ids, "\ntest", test_ids)

    codes = sorted({int(c) for d in subs.values() for c in np.unique(d["y"])})
    code2idx = {c: i for i, c in enumerate(codes)}
    print("label map:", {c: (code2idx[c], LABEL_NAMES[c]) for c in codes})

    # Train-only standardization stats.
    cat = np.concatenate([subs[s]["X"] for s in train_ids], axis=0)
    mean, std = cat.mean(axis=0), cat.std(axis=0) + 1e-8
    print("norm mean", mean.round(3), "std", std.round(3))

    Xs, Ys, Ts, Ss, Ps = [], [], [], [], []
    for split, group in ((0, train_ids), (1, val_ids), (2, test_ids)):
        for sid in group:
            d = subs[sid]
            X = ((d["X"] - mean) / std).astype(np.float32)
            y = np.vectorize(code2idx.get)(d["y"]).astype(np.int64)
            change = np.zeros(len(y), dtype=bool)
            change[1:] = y[1:] != y[:-1]
            tmask = np.zeros(len(y), dtype=bool)
            for p in np.where(change)[0]:
                tmask[max(0, p - TRANS_HALF):p + TRANS_HALF + 1] = True
            for s in range(0, len(y) - WINDOW + 1, STRIDE):
                Xs.append(X[s:s + WINDOW])
                Ys.append(y[s:s + WINDOW])
                Ts.append(tmask[s:s + WINDOW])
                Ss.append(sid)
                Ps.append(split)
    X = np.stack(Xs); Y = np.stack(Ys); T = np.stack(Ts)
    S = np.array(Ss); P = np.array(Ps)
    print("windows:", X.shape, "train/val/test:",
          int((P == 0).sum()), int((P == 1).sum()), int((P == 2).sum()))
    print("transition-mask fraction: %.4f" % float(T.mean()))
    np.savez_compressed(
        os.path.join(HERE, "data_harth.npz"), X=X, y6=Y, y12=Y, y_trans=T,
        subjects=S, split=P,
        train_subjects=np.array(train_ids), val_subjects=np.array(val_ids),
        test_subjects=np.array(test_ids), norm_mean=mean, norm_std=std)
    meta = dict(dataset="HARTH", label_map={str(c): code2idx[c] for c in codes},
                label_names={str(code2idx[c]): LABEL_NAMES[c] for c in codes},
                train_subjects=train_ids, val_subjects=val_ids,
                test_subjects=test_ids, window=WINDOW, stride=STRIDE,
                trans_half_steps=TRANS_HALF,
                y_trans_definition="timesteps within +/-25 steps of a label change")
    with open(os.path.join(HERE, "data_harth_stats.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("saved data_harth.npz + data_harth_stats.json")


if __name__ == "__main__":
    main()
