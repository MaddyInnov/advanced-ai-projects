"""Corrected TRUE 12-class evaluation for the HARTH screen.

The 2026-11-10 harth_p2_aug_c12_refine_bh10 run TRAINED correctly on
HARTH's 12 classes, but train_real.model_probs aggregated its outputs
through HAPT's MAP12_TO_6 (the HAPT dual-label design), so every val
number the run printed/recorded is scrambled (predictions folded into
6 HAPT groups vs 12 HARTH labels). This script re-evaluates the saved
checkpoints with SINGLE_LABEL_12 on and macro-F1 computed over 12
classes, for: the best-val checkpoint (selected under the scrambled
metric — selection itself was distorted) and the final epoch-20 state.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_real as tr  # noqa: E402
from metrics import macro_f1  # noqa: E402

tr.SINGLE_LABEL_12 = True
NAMES = ["walking", "running", "shuffling", "stairs_ascending",
         "stairs_descending", "standing", "sitting", "lying",
         "cycling_sit", "cycling_stand", "cycling_sit_inactive",
         "cycling_stand_inactive"]


def full_metrics(P, Y, Tmask):
    r = tr.eval_from_preds(P, Y, Tmask)
    mf1, per = macro_f1(Y.ravel(), P.ravel(), C=12)
    r["macro_f1"] = mf1
    r["per_class_f1"] = per
    r["per_class_f1_named"] = {NAMES[i]: per[i] for i in range(12)}
    return r


def show(tag, r):
    print("%s acc %.4f F1 %.4f transWin %.4f bF1 %.4f (P %.3f R %.3f) "
          "seg25 %.4f edit %.1f" % (
              tag, r["accuracy"], r["macro_f1"],
              r["transition_window_accuracy"], r["boundary_f1"],
              r["boundary_precision"], r["boundary_recall"],
              r["segmental_f1@25"], r["edit_score"]))


def main():
    data = tr.load_data("harth")
    Yv, Tv = data["val"]["y6"], data["val"]["y_trans"]
    out = {}

    model, meta = tr.load_model(os.path.join(
        HERE, "model_real_harth_p2_aug_c12_refine_bh10.pt"))
    print("best checkpoint meta:", {k: meta[k] for k in
          ("num_classes", "refine", "boundary_head", "best_epoch")})
    Pv = tr.model_probs(model, data["val"]["X"], 12, True)
    Bv = tr.model_boundary_probs(model, data["val"]["X"])
    raw = Pv.argmax(-1)
    bref = tr.segment_majority(raw, Bv, thresh=0.5)
    out["best_ckpt_raw"] = full_metrics(raw, Yv, Tv); show("BEST raw ", out["best_ckpt_raw"])
    out["best_ckpt_bref"] = full_metrics(bref, Yv, Tv); show("BEST bref", out["best_ckpt_bref"])
    for th in (10, 25, 50, 100):
        r = full_metrics(tr.min_duration_relabel(bref, th), Yv, Tv)
        out[f"best_ckpt_bref_mindur{th}"] = r
        show(f"BEST bref+mindur{th}", r)
    print("per-class F1 (best, bref):",
          {k: round(v, 3) for k, v in out["best_ckpt_bref"]["per_class_f1_named"].items()})

    state = torch.load(os.path.join(
        HERE, "ckpt_state_harth_p2_aug_c12_refine_bh10.pt"),
        map_location="cpu", weights_only=False)
    print("final state epoch:", state["epoch"], "of", state["epochs_total"])
    m2 = tr.build_model(48, 12, True, in_ch=6, arch="densehar",
                        boundary_head=True)
    m2.load_state_dict(state["model"])
    Pv2 = tr.model_probs(m2, data["val"]["X"], 12, True)
    Bv2 = tr.model_boundary_probs(m2, data["val"]["X"])
    raw2 = Pv2.argmax(-1)
    bref2 = tr.segment_majority(raw2, Bv2, thresh=0.5)
    out["final_raw"] = full_metrics(raw2, Yv, Tv); show("FINAL raw ", out["final_raw"])
    out["final_bref"] = full_metrics(bref2, Yv, Tv); show("FINAL bref", out["final_bref"])
    for th in (10, 25, 50, 100):
        r = full_metrics(tr.min_duration_relabel(bref2, th), Yv, Tv)
        out[f"final_bref_mindur{th}"] = r
        show(f"FINAL bref+mindur{th}", r)
    print("per-class F1 (final, bref):",
          {k: round(v, 3) for k, v in out["final_bref"]["per_class_f1_named"].items()})

    with open(os.path.join(HERE, "results_harth.json"), "w") as f:
        json.dump(dict(experiment="harth_p2_aug_c12_refine_bh10",
                       note="corrected true-12-class val evaluation; the "
                            "run's own recorded val metrics used HAPT's "
                            "MAP12_TO_6 aggregation and are invalid",
                       val=out), f, indent=1)
    print("saved results_harth.json")


if __name__ == "__main__":
    main()
