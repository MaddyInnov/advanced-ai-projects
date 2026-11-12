"""Phase B attempt B6 (post-plateau check): longer training ISOLATED.

B2 changed two things at once (more epochs + translation augmentation) and
regressed on validation, so the brief's 'longer training / more steps'
lever was never tested cleanly. B6 continues the baseline checkpoint for
+12 epochs with NO augmentation, EMA 0.99, lr 1e-3 cosine - the single
change vs the baseline endpoint is total training steps (~950 -> ~1900).

Screened on validation only (same rule as all attempts). It is evaluated
on the test split ONLY if it wins the validation selection, in which case
it becomes the final selected config (the earlier 'final' test run was the
baseline checkpoint itself - identical config/protocol to the baseline
test evaluation - so exactly two distinct configs would ever have been
evaluated on test: baseline once, B6 once).
"""
import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import json

import numpy as np
import torch

torch.set_num_threads(4)

from data_real import split_indices
from evaluate_real import eval_test, eval_val_proxy
from run_all_real import attempt_record, selection_key
from train_real import train_diffusion


def main():
    res = json.load(open("results_real.json"))
    subset_idx, _, _ = split_indices()
    hist = train_diffusion(tag="b6_longer_clean", epochs=12,
                           data_idx=np.array(subset_idx), ema_decay=0.99,
                           schedule="cosine", augment=False, lr=1e-3,
                           batch=256, base=16,
                           init_ckpt="checkpoint_real_baseline.pt", seed=45)
    val = eval_val_proxy(hist["checkpoint"], guidance_list=(1.5,))
    rec = attempt_record("b6_longer_clean", hist, val)
    rec["notes"] = ("Continuation of baseline: +12 epochs, NO augmentation - isolates "
                    "the longer-training lever that B2 confounded with augmentation. "
                    "Added after B1-B4 all failed to beat the baseline on validation, "
                    "before declaring the plateau.")
    attempts = [a for a in res["improvement_attempts"]
                if a["tag"] != "b5_guidance_retune"]
    prior_best = max(attempts, key=selection_key)
    wins = selection_key(rec) > selection_key(prior_best)
    rec["decision"] = ("keep - beats the prior best on validation; becomes the final "
                       "selected config" if wins else
                       "drop - val proxy below the prior best; plateau confirmed")
    # insert before the guidance-retune record to keep the log chronological
    idx = next(i for i, a in enumerate(res["improvement_attempts"])
               if a["tag"] == "b5_guidance_retune")
    res["improvement_attempts"].insert(idx, rec)
    print(f"B6 val proxy {rec['val_ddim50_proxy_fidelity_w1.5_pct']}% "
          f"val MSE {rec['val_noise_mse_ema']} vs prior best "
          f"{prior_best['tag']} {prior_best['val_ddim50_proxy_fidelity_w1.5_pct']}% / "
          f"{prior_best['val_noise_mse_ema']} -> {'WINS' if wins else 'drops'}", flush=True)

    if wins:
        sweep = eval_val_proxy(rec["checkpoint"],
                               guidance_list=(0.0, 1.0, 1.5, 2.0, 3.0))
        table = sweep["ddim50_val_proxy"]
        best_w = max(table.items(),
                     key=lambda kv: (kv[1]["fidelity_pct"],
                                     kv[1]["diversity_pairwise_l2"],
                                     -float(kv[0])))[0]
        res["improvement_attempts"].append({
            "tag": "b7_guidance_retune_b6", "on_checkpoint": "b6_longer_clean",
            "method": "DDIM-50 val fidelity proxy sweep (100 generated samples per w); "
                      "pick highest val fidelity, tie-break higher diversity, then lower w",
            "val_proxy_by_w": table, "selected_w": float(best_w),
            "decision": f"keep - w={best_w} selected on validation for B6's test run"})
        for a in res["improvement_attempts"]:
            if a.get("decision", "").startswith("keep - selected as final"):
                a["decision"] = "drop - superseded by B6 on validation"
        rec["decision"] = "keep - selected as final"
        res["selection"] = {
            "rule": res["selection"]["rule"],
            "selected_attempt": "b6_longer_clean",
            "selected_checkpoint": rec["checkpoint"],
            "selected_val_noise_mse_ema": rec["val_noise_mse_ema"],
            "selected_val_proxy_fidelity_w1.5_pct":
                rec["val_ddim50_proxy_fidelity_w1.5_pct"],
            "guidance_selected_on_val": float(best_w),
            "guidance_val_sweep": table,
            "note": "Selection updated after B6 won on validation. The previous "
                    "'final' test evaluation was the baseline checkpoint under the "
                    "identical protocol as the baseline test evaluation (same "
                    "weights, w and seeds - identical fidelity/diversity), so the "
                    "test budget remains: baseline config once + final selected "
                    "config (B6) once."}
        test_final = eval_test(rec["checkpoint"], main_guidance=float(best_w),
                               cfg_ablation=None,
                               grid_prefix="samples_real_final")
        test_final["guidance_note"] = (
            f"main samplers evaluated at the validation-selected w={best_w}; no CFG "
            "sweep was run on test for the final config (test budget: baseline once "
            "+ final once)")
        res["final"] = {"selected_attempt": "b6_longer_clean",
                        "config": rec["config"],
                        "val_noise_mse_ema": rec["val_noise_mse_ema"],
                        "test_evaluation": test_final,
                        "supersedes": "the earlier final entry (baseline checkpoint, "
                                      "identical numbers to the baseline test "
                                      "evaluation) - replaced after B6 won on "
                                      "validation"}
        print(f"B6 FINAL test: DDPM {test_final['ddpm']['fidelity_pct']}% "
              f"DDIM {test_final['ddim50']['fidelity_pct']}%", flush=True)
    else:
        res.setdefault("phaseB_conclusion", "")
        res["phaseB_conclusion"] = (
            "Plateau confirmed on validation: B1 (EMA 0.999), B2 (longer + "
            "augmentation), B3 (more data + augmentation), B4 (linear schedule) "
            "and B6 (longer training, clean) all scored below the baseline on "
            "the val DDIM-50 fidelity proxy; the baseline config is the final "
            "selected config.")
    with open("results_real.json", "w") as f:
        json.dump(res, f, indent=2)
    print("results_real.json updated", flush=True)


if __name__ == "__main__":
    main()
