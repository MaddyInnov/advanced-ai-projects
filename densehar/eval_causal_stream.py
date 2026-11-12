"""Semi-online (causal) evaluation of the bh10 champion pipeline — VAL only.

The offline champion reads a whole 512-step window before labelling any
timestep. This script measures the streaming operating point instead:

- Trailing windows: each sequence is tiled by windows ending at
  t = 15, 31, ..., 511 (left edge-padded before the sequence start).
  Sequence position s is read from the window ending at the next tile
  boundary, at window position 496 + (s mod 16) — so the label for s
  uses sensor data up to at most s + 15 steps (fixed lag <= 15 steps
  = 0.30 s at 50 Hz) and never anything beyond that.
- The same tiling supplies boundary probabilities from the BRB head.
- Post-processing, causal form: segment-majority at predicted cuts
  (a segment's majority is final once the next cut occurs), then
  min-duration merging into the LEFT neighbour only (the offline pack
  may merge into the longer of left/right; a streamer cannot know the
  right neighbour's length when it must emit).
- Boundary-detection delay: for each GT boundary, the signed offset of
  the first emitted label switch within +-25 steps (the program's first
  boundary-delay measurement).
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_real as tr

TILE = 16


def tiled_probs(model, meta, X, lookahead=0):
    """Read each 16-step block from a window whose right edge sits
    `lookahead` steps past the block end, at window positions
    [496-lookahead, 512-lookahead] — the label for position s therefore
    uses data up to s + lookahead + (0..15) steps and nothing beyond.
    Windows past the sequence end read a right-extended copy (last
    sample held); windows before the start are left edge-padded."""
    N, L, C = X.shape
    Xe = np.concatenate(
        [X, np.repeat(X[:, -1:], lookahead, axis=1)], axis=1) \
        if lookahead > 0 else X
    ends = [lookahead + TILE - 1 + TILE * j for j in range(L // TILE)]
    W = []
    for i in range(N):
        for t in ends:
            lo = t - (L - 1)
            if lo >= 0:
                W.append(Xe[i, lo:t + 1])
            else:
                pad = np.repeat(Xe[i, :1], -lo, axis=0)
                W.append(np.concatenate([pad, Xe[i, :t + 1]], axis=0))
    W = np.stack(W)
    P = tr.model_probs(model, W, meta["num_classes"], meta["refine"])
    B = tr.model_boundary_probs(model, W)
    P = P.reshape(N, len(ends), L, 6)
    B = B.reshape(N, len(ends), L)
    Ps = np.zeros((N, L, 6)); Bs = np.zeros((N, L))
    a = L - TILE - lookahead
    for j, t in enumerate(ends):
        s0 = TILE * j
        Ps[:, s0:s0 + TILE] = P[:, j, a:a + TILE]
        Bs[:, s0:s0 + TILE] = B[:, j, a:a + TILE]
    return Ps, Bs


def min_duration_left(P, theta):
    """Merge segments shorter than theta into their LEFT neighbour only
    (first segment merges right — nothing emitted before it)."""
    if theta <= 1:
        return P
    out = P.copy()
    for b in range(len(P)):
        lab = out[b].copy()
        while True:
            segs = tr.segmental_segments(lab)
            if len(segs) <= 1:
                break
            short = [(e - s + 1, i) for i, (l, s, e) in enumerate(segs)
                     if e - s + 1 < theta]
            if not short:
                break
            _, i = min(short)
            l, s, e = segs[i]
            nl = segs[i - 1][0] if i > 0 else segs[i + 1][0]
            lab[s:e + 1] = nl
        out[b] = lab
    return out


def boundary_delays(pred, y, tol=25):
    """Signed switch offsets (pred switch pos - GT boundary pos) for GT
    boundaries with a pred switch within +-tol; plus miss count."""
    offs, miss = [], 0
    for b in range(len(y)):
        gt = np.where(np.diff(y[b]) != 0)[0] + 1
        pr = np.where(np.diff(pred[b]) != 0)[0] + 1
        for g in gt:
            near = pr[np.abs(pr - g) <= tol]
            if len(near):
                offs.append(int(near[np.argmin(np.abs(near - g))] - g))
            else:
                miss += 1
    return offs, miss


def show(tag, v):
    print(f"{tag} acc {v['accuracy']:.4f} F1 {v['macro_f1']:.4f} "
          f"transWin {v['transition_window_accuracy']:.4f} "
          f"transClass {v['transition_class_timestep_accuracy']:.4f} "
          f"bF1 {v['boundary_f1']:.4f} seg25 {v['segmental_f1@25']:.4f} "
          f"edit {v['edit_score']:.1f}", flush=True)
    return v


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookahead", type=int, default=0,
                    help="steps of future context past each block end "
                         "(the latency budget); read positions sit this "
                         "far inside the window from its right edge")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = tr.load_data()
    model, meta = tr.load_model(os.path.join(
        tr.HERE, "model_real_p2_aug_c12_refine_bh10.pt"))
    y6, ytr = data["val"]["y6"], data["val"]["y_trans"]
    print(f"running tiled (semi-online) inference on val, "
          f"lookahead={args.lookahead} ...", flush=True)
    Ps, Bs = tiled_probs(model, meta, data["val"]["X"],
                         lookahead=args.lookahead)
    raw = Ps.argmax(-1)
    res = {"lookahead": args.lookahead}
    res["stream_raw"] = show("[stream raw]        ", tr.eval_from_preds(raw, y6, ytr))
    bref = tr.segment_majority(raw, Bs, thresh=0.5)
    res["stream_bref"] = show("[stream bref]       ", tr.eval_from_preds(bref, y6, ytr))
    caus = min_duration_left(bref, 10)
    res["stream_bref_mindur_left"] = show(
        "[stream bref+mindL] ", tr.eval_from_preds(caus, y6, ytr))
    offs, miss = boundary_delays(caus, y6)
    offs = np.array(offs, dtype=float)
    res["boundary_delay"] = dict(
        n_matched=int(len(offs)), n_missed=int(miss),
        mean_steps=float(offs.mean()), median_steps=float(np.median(offs)),
        mean_ms=float(offs.mean() * 20), median_ms=float(np.median(offs) * 20),
        note="emitted switch position minus GT boundary, tol +-25 steps; "
             "excludes the algorithmic delay (lookahead + up to 15 steps), "
             "which adds to it")
    print(f"boundary delay: matched {len(offs)} missed {miss} "
          f"mean {offs.mean():+.1f} steps ({offs.mean() * 20:+.0f} ms) "
          f"median {np.median(offs):+.0f} steps "
          f"({np.median(offs) * 20:+.0f} ms), plus lookahead "
          f"{args.lookahead}..{args.lookahead + 15} steps "
          f"({(args.lookahead) * 20}..{(args.lookahead + 15) * 20} ms) "
          f"algorithmic delay", flush=True)
    out = args.out or os.path.join(
        tr.HERE, f"causal_stream_val_la{args.lookahead}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"saved {out}", flush=True)
    print("offline refs (val): raw seg25 0.8078 bF1 0.4444 | bref seg25 "
          "0.8690 bF1 0.5131 | bref+mindur seg25 0.9147 bF1 0.5629 edit 91.0",
          flush=True)


if __name__ == "__main__":
    main()
