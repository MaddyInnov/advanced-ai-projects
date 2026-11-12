"""Semi-online test confirmation of the OFFLINE champion (HAPT, 5 unseen
subjects) — lookahead 64 (~1.4 s), the re-ranked streaming path.

After the 2026-11-11 causal-test result (zero-delay causal retrain does
NOT transfer: test F1 0.6926 vs val 0.9307), the pragmatic streaming path
is the offline champion read semi-online: tiled windows whose right edge
sits 64 steps past each 16-step block end, so each label uses at most
64..79 steps (1.28..1.58 s) of future context. Same tiling + causal
post-processing as eval_causal_stream.py (segment_majority thresh 0.5,
then min_duration_left theta 10); operating point selected on VAL only
(causal_stream_val_la64.json: bref+mindL acc 0.8520 / F1 0.8221 /
bF1 0.4943 / seg25 0.8190 / edit 83.0).

One model, one test pass, three views from the same prediction set,
precedent: eval_causal_test.py (9th test evaluation). Val is recomputed
first as a determinism check against causal_stream_val_la64.json.

Writes the full record into results_real.json as experiment
final_champion_semionline_la64 (kind test_confirmation_semionline) and
appends the test_evaluations ledger entry (10th).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_real as tr  # noqa: E402
from eval_causal_stream import (  # noqa: E402
    boundary_delays, min_duration_left, tiled_probs)

HERE = os.path.dirname(os.path.abspath(__file__))
LOOKAHEAD = 64


def show(tag, v):
    print(f"{tag} acc {v['accuracy']:.4f} F1 {v['macro_f1']:.4f} "
          f"transWin {v['transition_window_accuracy']:.4f} "
          f"transClass {v['transition_class_timestep_accuracy']:.4f} "
          f"bF1 {v['boundary_f1']:.4f} seg25 {v['segmental_f1@25']:.4f} "
          f"edit {v['edit_score']:.1f}", flush=True)
    return v


def views(model, meta, X, y6, ytr):
    Ps, Bs = tiled_probs(model, meta, X, lookahead=LOOKAHEAD)
    raw = Ps.argmax(-1)
    bref = tr.segment_majority(raw, Bs, thresh=0.5)
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
        HERE, "model_real_p2_aug_c12_refine_bh10.pt"))
    print(f"model: {meta['tag']} arch={meta.get('arch')} "
          f"boundary_head={meta.get('boundary_head')} "
          f"best_epoch={meta.get('best_epoch')}", flush=True)

    print(f"VAL (reproduction check vs causal_stream_val_la64.json, "
          f"lookahead={LOOKAHEAD}):", flush=True)
    val, _ = views(model, meta, data["val"]["X"],
                   data["val"]["y6"], data["val"]["y_trans"])
    ref = json.load(open(os.path.join(
        HERE, f"causal_stream_val_la{LOOKAHEAD}.json")))
    for k in ("raw", "bref", "bref_mindL"):
        refv = ref[{"raw": "stream_raw", "bref": "stream_bref",
                    "bref_mindL": "stream_bref_mindur_left"}[k]]
        ok = (abs(val[k]["accuracy"] - refv["accuracy"]) < 1e-9
              and abs(val[k]["macro_f1"] - refv["macro_f1"]) < 1e-9
              and abs(val[k]["segmental_f1@25"]
                      - refv["segmental_f1@25"]) < 1e-9)
        print(f"  val {k}: reproduction {'EXACT' if ok else 'MISMATCH'} "
              f"(ref seg25 {refv['segmental_f1@25']:.4f})", flush=True)

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
             "excludes the algorithmic delay (lookahead 64..79 steps = "
             "1.28..1.58 s), which adds to it")
    print(f"TEST boundary delay: matched {len(offs)} missed {miss} "
          f"mean {offs.mean():+.1f} steps ({offs.mean() * 20:+.0f} ms) "
          f"median {np.median(offs):+.0f} steps "
          f"({np.median(offs) * 20:+.0f} ms), plus lookahead "
          f"{LOOKAHEAD}..{LOOKAHEAD + 15} steps "
          f"({LOOKAHEAD * 20}..{(LOOKAHEAD + 15) * 20} ms) algorithmic "
          f"delay", flush=True)
    print("TEST per-class F1 (bref+mindL):",
          {k: round(v, 3) for k, v in
           test["bref_mindL"]["per_class_f1_named"].items()}, flush=True)

    res = json.load(open(os.path.join(HERE, "results_real.json")))
    exp = dict(
        name="final_champion_semionline_la64",
        kind="test_confirmation_semionline",
        config=dict(
            base_model="model_real_p2_aug_c12_refine_bh10.pt",
            base_experiment="p2_aug_c12_refine_bh10 (offline champion)",
            lookahead_steps=LOOKAHEAD,
            lookahead_seconds="1.28..1.58 (64 + up to 15 tile steps "
                              "@ 50 Hz; mean ~1.43 s)",
            pipeline="offline champion read semi-online: tiled windows "
                     "(16-step blocks, right edge 64 steps past block "
                     "end) -> segment_majority(0.5) -> "
                     "min_duration_left(theta=10)",
            selected_on="val only (causal_stream_val_la64.json: "
                         "F1 0.8221, seg25 0.8190, edit 83.0)",
            causality="model is offline (uses future context within the "
                       "window); streaming-equivalence is by construction: "
                       "each label reads at most 64..79 future steps. "
                       "Post-processing finalises a segment at its next "
                       "cut and merges left-only, so emission is delayed "
                       "to segment close; raw row is the un-post-processed "
                       "tiled readout"),
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
        {"experiment": "final_champion_semionline_la64",
         "kind": "test_confirmation_semionline_streaming"})
    json.dump(res, open(os.path.join(HERE, "results_real.json"), "w"),
              indent=2)
    print("test_evaluations ledger now:", len(res["test_evaluations"]),
          flush=True)


if __name__ == "__main__":
    main()
