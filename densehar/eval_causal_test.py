"""Causal-retrain test confirmation #1 (HAPT, 5 unseen subjects).

One model, one test pass, three operating-point views from the same
prediction set — same precedent as eval_harth_test.py (HARTH
test_confirmation_1). Model: p2_aug_c12_causal_bh10 (fully causal twin of
the bh10 champion; perturbation test proves exactly zero future leakage,
test_causal.py). Operating point selected on VAL only: segment_majority
(thresh 0.5) + min_duration_left (theta 10) — exp #36 val: acc 0.8819 /
F1 0.9307 / bF1 0.3284 / seg25 0.7943 / edit 82.0.

Causality note, stated plainly: the MODEL is provably causal (each output
position depends only on past inputs), so a full-window forward pass is
streaming-equivalent. The POST-PROCESSING delays emission: segment_majority
finalises a segment's label once the next boundary cut occurs, and
min_duration_left merges into the left neighbour only. The raw-argmax row
is the pure zero-delay readout (no post-processing at all). Boundary
delays are measured on the emitted (post-processed) test predictions.

Writes the full record into results_real.json as experiment
final_causal_bh10_bref_mindL (kind test_confirmation_causal) and appends
the test_evaluations ledger entry (9th).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_real as tr  # noqa: E402
from eval_causal_stream import boundary_delays, min_duration_left  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def show(tag, v):
    print(f"{tag} acc {v['accuracy']:.4f} F1 {v['macro_f1']:.4f} "
          f"transWin {v['transition_window_accuracy']:.4f} "
          f"transClass {v['transition_class_timestep_accuracy']:.4f} "
          f"bF1 {v['boundary_f1']:.4f} seg25 {v['segmental_f1@25']:.4f} "
          f"edit {v['edit_score']:.1f}", flush=True)
    return v


def views(model, meta, X, y6, ytr):
    P = tr.model_probs(model, X, meta["num_classes"], meta["refine"])
    B = tr.model_boundary_probs(model, X)
    raw = P.argmax(-1)
    bref = tr.segment_majority(raw, B, thresh=0.5)
    mindL = min_duration_left(bref, 10)
    return {
        "raw": show("raw         ", tr.eval_from_preds(raw, y6, ytr)),
        "bref": show("bref        ", tr.eval_from_preds(bref, y6, ytr)),
        "bref_mindL": show("bref+mindL   ",
                            tr.eval_from_preds(mindL, y6, ytr)),
    }, mindL


def main():
    data = tr.load_data()
    print(f"splits: train {len(data['train']['X'])} "
          f"val {len(data['val']['X'])} test {len(data['test']['X'])}",
          flush=True)
    model, meta = tr.load_model(os.path.join(
        HERE, "model_real_p2_aug_c12_causal_bh10.pt"))
    print(f"model: {meta['tag']} arch={meta.get('arch')} "
          f"boundary_head={meta.get('boundary_head')} "
          f"best_epoch={meta.get('best_epoch')}", flush=True)

    print("VAL (reproduction check vs exp #35/#36):", flush=True)
    val, _ = views(model, meta, data["val"]["X"],
                   data["val"]["y6"], data["val"]["y_trans"])
    print("TEST (single evaluation, 5 unseen subjects):", flush=True)
    test, test_pred = views(model, meta, data["test"]["X"],
                            data["test"]["y6"], data["test"]["y_trans"])
    offs, miss = boundary_delays(test_pred, data["test"]["y6"])
    offs = np.array(offs, dtype=float)
    delay = dict(
        n_matched=int(len(offs)), n_missed=int(miss),
        mean_steps=float(offs.mean()), median_steps=float(np.median(offs)),
        mean_ms=float(offs.mean() * 20), median_ms=float(np.median(offs) * 20),
        note="emitted switch position minus GT boundary, tol +-25 steps; "
             "excludes the segment-close emission delay of the refinement")
    print(f"TEST boundary delay: matched {len(offs)} missed {miss} "
          f"mean {offs.mean():+.1f} steps ({offs.mean() * 20:+.0f} ms) "
          f"median {np.median(offs):+.0f} steps "
          f"({np.median(offs) * 20:+.0f} ms)", flush=True)
    print("TEST per-class F1 (bref+mindL):",
          {k: round(v, 3) for k, v in
           test["bref_mindL"]["per_class_f1_named"].items()}, flush=True)

    res = json.load(open(os.path.join(HERE, "results_real.json")))
    exp = dict(
        name="final_causal_bh10_bref_mindL",
        kind="test_confirmation_causal",
        config=dict(
            base_model="model_real_p2_aug_c12_causal_bh10.pt",
            base_experiment="p2_aug_c12_causal_bh10",
            pipeline="causal model (full-past context, zero lookahead) -> "
                     "segment_majority(0.5) -> min_duration_left(theta=10)",
            selected_on="val only (exp #36: F1 0.9307, seg25 0.7943, "
                         "edit 82.0)",
            causality="model provably causal (test_causal.py perturbation "
                       "test, zero leakage); segment_majority finalises a "
                       "segment at its next cut and min_duration_left merges "
                       "left-only, so emission is delayed to segment close; "
                       "raw row is the pure zero-delay readout",
            boundary_delay_test=delay),
        params=170133,
        best_epoch=meta.get("best_epoch"),
        val=val["bref_mindL"],
        val_views={k: v for k, v in val.items()},
        test=test["bref_mindL"],
        test_views={k: v for k, v in test.items()},
        history=[], train_seconds=0.0)
    tr.append_result(res, exp)
    res = json.load(open(os.path.join(HERE, "results_real.json")))
    res["test_evaluations"].append(
        {"experiment": "final_causal_bh10_bref_mindL",
         "kind": "test_confirmation_causal_streaming"})
    json.dump(res, open(os.path.join(HERE, "results_real.json"), "w"),
              indent=2)
    print("test_evaluations ledger now:", len(res["test_evaluations"]),
          flush=True)


if __name__ == "__main__":
    main()
