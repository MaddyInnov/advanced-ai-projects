"""BRB refinement + post-processing pack, combined — VAL-first evaluation.

Takes a boundary-head checkpoint, computes ASRF boundary-refined labels
(segment_majority), then grids the label-domain post-processing steps
(majority filter window, min-duration relabel theta) on top of the refined
labels. Selection rule matches the existing postproc pack: val macro F1,
tie-break segmental F1@25. Test is evaluated ONLY when --test-med/--test-theta
are given (i.e. after a val win has been inspected).
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_real as tr


def ev(labels, split):
    return tr.eval_from_preds(labels, split["y6"], split["y_trans"])


def fmt(tag, v):
    print(f"{tag} acc {v['accuracy']:.4f} F1 {v['macro_f1']:.4f} "
          f"transWin {v['transition_window_accuracy']:.4f} "
          f"transClass {v['transition_class_timestep_accuracy']:.4f} "
          f"bF1 {v['boundary_f1']:.4f} (P {v['boundary_precision']:.3f} "
          f"R {v['boundary_recall']:.3f}) seg25 {v['segmental_f1@25']:.4f} "
          f"edit {v['edit_score']:.1f}", flush=True)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--test-med", type=int, default=None)
    ap.add_argument("--test-theta", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = tr.load_data()
    model, meta = tr.load_model(os.path.join(tr.HERE, args.model))
    assert getattr(model, "bhead", None) is not None, "checkpoint has no boundary head"

    Pv = tr.model_probs(model, data["val"]["X"], meta["num_classes"], meta["refine"])
    Bv = tr.model_boundary_probs(model, data["val"]["X"])
    L0 = tr.segment_majority(Pv.argmax(-1), Bv, thresh=args.thresh)
    print(f"== {args.model} thresh={args.thresh}", flush=True)
    base = fmt("[bref alone]        ", ev(L0, data["val"]))

    rows, best, best_key = [], None, None
    for med in (1, 3, 5, 11):
        for theta in (0, 10, 15, 25):
            if med == 1 and theta == 0:
                continue
            L = tr.min_duration_relabel(tr.majority_filter(L0, med), theta)
            v = ev(L, data["val"])
            rows.append(dict(median=med, theta=theta, val=v))
            fmt(f"[bref+pp med={med} th={theta}] ", v)
            key = (v["macro_f1"], v["segmental_f1@25"])
            if best_key is None or key > best_key:
                best_key, best = key, (med, theta, v)
    med, theta, bv = best
    print(f"BEST by (F1, seg25): med={med} theta={theta}", flush=True)
    fmt("[bref+pp best]      ", bv)
    print(f"bref alone: bF1 {base['boundary_f1']:.4f} seg25 "
          f"{base['segmental_f1@25']:.4f} edit {base['edit_score']:.1f} "
          f"F1 {base['macro_f1']:.4f}", flush=True)

    test = None
    if args.test_med is not None and args.test_theta is not None:
        Pt = tr.model_probs(model, data["test"]["X"], meta["num_classes"], meta["refine"])
        Bt = tr.model_boundary_probs(model, data["test"]["X"])
        Lt0 = tr.segment_majority(Pt.argmax(-1), Bt, thresh=args.thresh)
        test_base = fmt("[TEST bref alone]   ", ev(Lt0, data["test"]))
        Lt = tr.min_duration_relabel(tr.majority_filter(Lt0, args.test_med),
                                     args.test_theta)
        test = fmt(f"[TEST bref+pp med={args.test_med} th={args.test_theta}] ", ev(Lt, data["test"]))
        test = dict(median=args.test_med, theta=args.test_theta,
                    bref_alone=test_base, combined=test)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(model=args.model, thresh=args.thresh,
                           bref_alone=base, grid=rows,
                           best=dict(median=med, theta=theta, val=bv),
                           test=test), f, indent=2, default=float)
        print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
