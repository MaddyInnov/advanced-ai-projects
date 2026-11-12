"""HARTH test confirmation #1: best checkpoint + bref + mindur25
(theta selected on val) evaluated ONCE on the test subjects
[10, 16, 21, 26]. Raw and bref rows come from the same prediction set.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_real as tr  # noqa: E402
from eval_harth import full_metrics, show  # noqa: E402

tr.SINGLE_LABEL_12 = True
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    data = tr.load_data("harth")
    Yt, Tt = data["test"]["y6"], data["test"]["y_trans"]
    model, meta = tr.load_model(os.path.join(
        HERE, "model_real_harth_p2_aug_c12_refine_bh10.pt"))
    Pt = tr.model_probs(model, data["test"]["X"], 12, True)
    Bt = tr.model_boundary_probs(model, data["test"]["X"])
    raw = Pt.argmax(-1)
    bref = tr.segment_majority(raw, Bt, thresh=0.5)
    res = {
        "raw": full_metrics(raw, Yt, Tt),
        "bref": full_metrics(bref, Yt, Tt),
        "bref_mindur25": full_metrics(tr.min_duration_relabel(bref, 25), Yt, Tt),
    }
    show("TEST raw          ", res["raw"])
    show("TEST bref         ", res["bref"])
    show("TEST bref+mindur25", res["bref_mindur25"])
    print("per-class F1 (test, bref+mindur25):",
          {k: round(v, 3) for k, v in
           res["bref_mindur25"]["per_class_f1_named"].items()})
    d = json.load(open(os.path.join(HERE, "results_harth.json")))
    d["test_confirmation_1"] = dict(
        candidate="best ckpt (ep16) + segment_majority(0.5) + min_duration(25)",
        theta_selected_on_val=True, results=res)
    json.dump(d, open(os.path.join(HERE, "results_harth.json"), "w"), indent=1)
    print("recorded in results_harth.json")


if __name__ == "__main__":
    main()
