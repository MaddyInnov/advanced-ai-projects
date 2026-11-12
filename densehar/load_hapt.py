"""Load REAL HAPT data into the DenseHAR npz schema.

HAPT (Reyes-Ortiz et al., UCI id 341): 30 subjects, waist-mounted smartphone,
acc + gyro @ 50 Hz, per-timestep labels in RawData/labels.txt:
(exp, user, activity 1-12, start, end) with 1-based inclusive sample indices.
Activities 1-6 are basic (walking, walking_upstairs, walking_downstairs,
sitting, standing, laying); 7-12 are postural transitions (stand-to-sit,
sit-to-stand, sit-to-lie, lie-to-sit, stand-to-lie, lie-to-stand).

Output data_hapt.npz:
  X        (N,512,6) float32, per-channel standardised with TRAIN-stream stats
  y6       (N,512)   int64, 6-class labels; transition classes mapped to their
                     TARGET activity (stand-to-sit->sitting, sit-to-stand->
                     standing, sit-to-lie->laying, lie-to-sit->sitting,
                     stand-to-lie->laying, lie-to-stand->standing)
  y12      (N,512)   int64, original 12-class labels (6 basic + 6 transitions)
  y_trans  (N,512)   bool, True where the ORIGINAL label was a transition class
  subjects (N,)      int64 HAPT user id (1-30)
  split    (N,)      int8: 0=train, 1=val, 2=test (subject-disjoint, fixed lists)
  norm_mean/norm_std (6,) float32 normalisation stats (train streams only)

Sequencing: per (experiment, user) stream, 512-step windows, stride 256.
Windows touching an unlabelled sample are dropped. Windows containing an
activity boundary (in y6) or any transition-class timestep are always kept;
pure single-activity windows are capped at <=30% of each split.
"""
import os, re, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data_hapt_raw", "RawData")
OUT = os.path.join(HERE, "data_hapt.npz")
SEED = 0
WIN, STRIDE = 512, 256
PURE_MAX_SHARE = 0.30

# Fixed subject split (subject-disjoint). Test subjects are drawn from the
# official HAPT test-subject list [2,4,9,10,12,13,18,20,24]; val subjects are
# carved from official train subjects (same carve as the author's earlier
# windowed HAPT work); the remaining 21 subjects form the train split.
TEST_SUBJECTS = [2, 4, 9, 10, 12]
VAL_SUBJECTS = [27, 28, 29, 30]
TRAIN_SUBJECTS = [s for s in range(1, 31)
                  if s not in TEST_SUBJECTS and s not in VAL_SUBJECTS]
assert len(TRAIN_SUBJECTS) == 21 and len(VAL_SUBJECTS) == 4 and len(TEST_SUBJECTS) == 5
assert not (set(TRAIN_SUBJECTS) & set(VAL_SUBJECTS))
assert not (set(TRAIN_SUBJECTS) & set(TEST_SUBJECTS))
assert not (set(VAL_SUBJECTS) & set(TEST_SUBJECTS))

ACTIVITIES_6 = ["walking", "walking_upstairs", "walking_downstairs",
                "sitting", "standing", "laying"]
# 0-based original label -> 6-class label (transitions -> target activity)
MAP12_TO_6 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
              6: 3,   # stand-to-sit  -> sitting
              7: 4,   # sit-to-stand  -> standing
              8: 5,   # sit-to-lie    -> laying
              9: 3,   # lie-to-sit    -> sitting
              10: 5,  # stand-to-lie  -> laying
              11: 4}  # lie-to-stand  -> standing
MAP12_TO_6_ARR = np.array([MAP12_TO_6[i] for i in range(12)], dtype=np.int64)


def parse_labels(raw_dir):
    segs = {}
    with open(os.path.join(raw_dir, "labels.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            exp, user, act, start, end = map(int, parts)
            segs.setdefault((exp, user), []).append((act - 1, start - 1, end - 1))
    return segs


def load_streams(raw_dir):
    """Return {(exp,user): (signal (T,6) float32, labels12 (T,) int64, -1=unlabelled)}."""
    segs = parse_labels(raw_dir)
    streams = {}
    for fname in sorted(os.listdir(raw_dir)):
        m = re.match(r"acc_exp(\d+)_user(\d+)\.txt", fname)
        if not m:
            continue
        exp, user = int(m.group(1)), int(m.group(2))
        acc = np.loadtxt(os.path.join(raw_dir, fname), dtype=np.float32)
        gyr = np.loadtxt(os.path.join(
            raw_dir, f"gyro_exp{exp:02d}_user{user:02d}.txt"), dtype=np.float32)
        assert len(acc) == len(gyr), f"acc/gyro length mismatch exp{exp} user{user}"
        labels = np.full(len(acc), -1, dtype=np.int64)
        for act, s, e in segs.get((exp, user), []):
            labels[s:e + 1] = act
        streams[(exp, user)] = (np.concatenate([acc, gyr], axis=1), labels)
    return streams


def split_of(user):
    if user in TRAIN_SUBJECTS:
        return 0
    if user in VAL_SUBJECTS:
        return 1
    return 2


def main():
    rng = np.random.default_rng(SEED)
    streams = load_streams(RAW)
    print(f"streams: {len(streams)} (exp,user) pairs")

    total_samples = sum(len(sig) for sig, _ in streams.values())
    unlabelled = sum(int((lab == -1).sum()) for _, lab in streams.values())
    print(f"total samples: {total_samples:,}; unlabelled: {unlabelled:,} "
          f"({100.0 * unlabelled / total_samples:.2f}%)")
    trans_samples = sum(int((lab >= 6).sum()) for _, lab in streams.values())
    labelled = total_samples - unlabelled
    print(f"transition-class samples: {trans_samples:,} "
          f"({100.0 * trans_samples / max(labelled, 1):.2f}% of labelled)")

    # Normalisation statistics from TRAIN subjects' streams only.
    train_sigs = [sig for (exp, user), (sig, _) in streams.items()
                  if user in TRAIN_SUBJECTS]
    cat = np.concatenate(train_sigs, axis=0)
    mean = cat.mean(axis=0).astype(np.float32)
    std = cat.std(axis=0).astype(np.float32)
    print(f"norm mean: {np.round(mean, 4).tolist()}")
    print(f"norm std : {np.round(std, 4).tolist()}")

    per_split = {0: ([], []), 1: ([], []), 2: ([], [])}  # split -> (interesting, pure)
    # each entry: (X window, y6 window, y_trans window, user, y12 window)
    for (exp, user), (sig, lab12) in sorted(streams.items()):
        sp = split_of(user)
        sign = ((sig - mean) / std).astype(np.float32)
        y6_full = np.where(lab12 >= 0, MAP12_TO_6_ARR[np.clip(lab12, 0, 11)], -1)
        ytr_full = lab12 >= 6
        for s in range(0, len(sig) - WIN + 1, STRIDE):
            wlab = lab12[s:s + WIN]
            if (wlab == -1).any():
                continue  # window touches an unlabelled region
            wy6 = y6_full[s:s + WIN]
            wytr = ytr_full[s:s + WIN]
            entry = (sign[s:s + WIN].copy(), wy6.copy(), wytr.copy(), user,
                     wlab.astype(np.int64).copy())
            if len(np.unique(wy6)) > 1 or wytr.any():
                per_split[sp][0].append(entry)
            else:
                per_split[sp][1].append(entry)

    Xs, Ys, Ts, Ss, Ps, Y12s = [], [], [], [], [], []
    stats = {}
    for sp, name in ((0, "train"), (1, "val"), (2, "test")):
        interesting, pure = per_split[sp]
        max_pure = int((PURE_MAX_SHARE / (1.0 - PURE_MAX_SHARE)) * len(interesting))
        rng.shuffle(pure)
        kept = interesting + pure[:max_pure]
        rng.shuffle(kept)
        for x, y6, ytr, user, y12 in kept:
            Xs.append(x); Ys.append(y6); Ts.append(ytr)
            Ss.append(user); Ps.append(sp); Y12s.append(y12)
        if kept:
            Ycat = np.stack([k[1] for k in kept])
            Tcat = np.stack([k[2] for k in kept])
            dist = {ACTIVITIES_6[c]: round(float((Ycat == c).mean()), 4)
                    for c in range(6)}
            stats[name] = dict(
                sequences=len(kept),
                interesting=int(len(interesting)),
                pure_kept=int(min(len(pure), max_pure)),
                subjects=sorted({k[3] for k in kept}),
                class_distribution_timestep_share=dist,
                transition_timestep_share=round(float(Tcat.mean()), 4),
                sequences_with_boundary=int(sum(len(np.unique(k[1])) > 1 for k in kept)),
            )
        print(f"{name}: {stats.get(name)}")

    X = np.stack(Xs).astype(np.float32)
    Y = np.stack(Ys).astype(np.int64)
    Y12 = np.stack(Y12s).astype(np.int64)
    T = np.stack(Ts).astype(bool)
    S = np.array(Ss, dtype=np.int64)
    P = np.array(Ps, dtype=np.int8)
    # Final leakage assertions.
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert not (set(S[P == a].tolist()) & set(S[P == b].tolist())), "subject leakage!"
    np.savez_compressed(OUT, X=X, y6=Y, y12=Y12, y_trans=T, subjects=S, split=P,
                        norm_mean=mean, norm_std=std,
                        train_subjects=np.array(TRAIN_SUBJECTS),
                        val_subjects=np.array(VAL_SUBJECTS),
                        test_subjects=np.array(TEST_SUBJECTS))
    print(f"saved {OUT}: X={X.shape} y6={Y.shape} y_trans={T.shape}")
    with open(os.path.join(HERE, "data_hapt_stats.json"), "w") as f:
        json.dump(dict(splits=stats, total_samples=int(total_samples),
                       unlabelled_samples=int(unlabelled),
                       transition_samples=int(trans_samples),
                       norm_mean=mean.tolist(), norm_std=std.tolist(),
                       train_subjects=TRAIN_SUBJECTS, val_subjects=VAL_SUBJECTS,
                       test_subjects=TEST_SUBJECTS), f, indent=2)
    print("saved data_hapt_stats.json")


if __name__ == "__main__":
    main()
