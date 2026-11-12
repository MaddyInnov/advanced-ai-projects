"""Orchestrator for the DiffusionForge real-data (MNIST-16) run.

Stages (each stage's numbers are flushed to results_real.json as soon as
they exist, so progress survives interruption):
  1. Dataset load + verification (MNIST, 60k/10k, 28x28 -> 16x16).
  2. Fidelity classifier trained on real train data; quality gate >= 98%
     real-test accuracy (auto-extends training until the gate passes or
     14 epochs are reached) - reported BEFORE any fidelity number.
  3. BASELINE diffusion run: original project config (EMA 0.99, cosine
     schedule, no augmentation), 20,000-image subset, 12 epochs.
     Test-side evaluation #1 (the baseline protocol, incl. CFG ablation).
  4. Phase B attempts, screened ONLY on validation (val noise-MSE of the
     EMA model + DDIM-50 val fidelity proxy at w=1.5):
       B1 EMA decay 0.999 (brief: ~0.999 for short runs, not 0.9999)
       B2 continue best-so-far +12 epochs with +-2px translation augmentation
       B3 continue best-so-far +6 epochs on expanded 40k data (val excluded)
       B4 linear-schedule comparison (fresh, baseline-sized)
       B5 guidance retune on the val proxy for the selected checkpoint
  5. FINAL test-side evaluation #2 (single selected config, guidance chosen
     on validation) + sample grids.

Selection rule (pre-committed): primary = val DDIM-50 fidelity proxy at
w=1.5; tie-break = lower val noise-MSE (EMA). Test fidelity is never used
for selection.
"""
import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import json
import time
import traceback

import numpy as np
import torch

torch.set_num_threads(4)

from classifier_real import train_classifier
from data_real import load_mnist16, split_indices
from evaluate_real import eval_test, eval_val_proxy
from train_real import train_diffusion

RESULTS_PATH = "results_real.json"
GATE_PCT = 98.0


def save_results(res):
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)


def selection_key(attempt):
    """Higher is better: (val proxy fidelity @w1.5, -val noise MSE)."""
    return (attempt["val_ddim50_proxy_fidelity_w1.5_pct"],
            -attempt["val_noise_mse_ema"])


def attempt_record(tag, hist, val):
    return {
        "tag": tag,
        "config": {"data_n": hist["data_n"], "epochs_this_run": hist["epochs_this_run"],
                   "epochs_total": hist["epochs_total"], "ema_decay": hist["ema_decay"],
                   "schedule": hist["schedule"], "augment": hist["augment"],
                   "lr": hist["lr"], "batch": hist["batch"], "base_channels": hist["base"],
                   "init_ckpt": hist.get("init_ckpt") if "init_ckpt" in hist else None,
                   "params": hist["params"]},
        "train_seconds": hist["train_seconds"],
        "final_train_loss": hist["final_train_loss"],
        "val_noise_mse_ema": val["val_noise_mse_ema"],
        "val_ddim50_proxy_fidelity_w1.5_pct": val["ddim50_val_proxy"]["1.5"]["fidelity_pct"],
        "val_ddim50_proxy_diversity_w1.5": val["ddim50_val_proxy"]["1.5"]["diversity_pairwise_l2"],
        "checkpoint": hist["checkpoint"],
    }


def main():
    t_start = time.time()
    res = {"dataset": None, "status": "running"}
    subset_idx, val_idx, extra_idx = split_indices()
    subset_idx = np.array(subset_idx); extra_idx = np.array(extra_idx)
    expanded_idx = np.concatenate([subset_idx, extra_idx])

    # ---- Stage 1+2: data + classifier gate -------------------------------
    x_train, y_train, x_test, y_test, meta = load_mnist16()
    res["dataset"] = {**meta,
                      "diffusion_baseline_subset_n": int(len(subset_idx)),
                      "validation_n": int(len(val_idx)),
                      "validation_source": "2,000 images carved from the real TRAIN "
                                           "split (seed-2024 permutation positions "
                                           "20,000-22,000); never used to train any "
                                           "diffusion model in this run",
                      "phaseB_expanded_train_n": int(len(expanded_idx)),
                      "split_seed": 2024}
    save_results(res)
    print(f"dataset verified: train={len(x_train)} test={len(x_test)}", flush=True)

    _, clf_acc, clf_secs, clf_epochs = train_classifier(
        x_train, y_train, x_test, y_test, epochs=5)
    clf_total_secs = clf_secs
    while clf_acc < GATE_PCT and clf_epochs < 14:
        print(f"classifier gate not met ({clf_acc}% < {GATE_PCT}%) - extending", flush=True)
        _, clf_acc, clf_secs, clf_epochs = train_classifier(
            x_train, y_train, x_test, y_test, epochs=3,
            init_ckpt="classifier_real.pt", seed=43)
        clf_total_secs += clf_secs
    res["fidelity_classifier"] = {
        "architecture": "TinyClassifier from evaluate.py, final layer 4 -> 10 classes",
        "trained_on": "full real MNIST-16 TRAIN split (60,000 images)",
        "epochs": clf_epochs, "train_seconds": round(clf_total_secs, 1),
        "real_test_accuracy_pct": clf_acc,
        "quality_gate": f"real-test accuracy >= {GATE_PCT}% required before "
                        "fidelity scores are trusted (brief target)",
        "quality_gate_result": "PASS" if clf_acc >= GATE_PCT else "FAIL - fidelity "
                               "numbers below must be read against this weak evaluator",
        "checkpoint": "classifier_real.pt"}
    save_results(res)
    print(f"classifier real-test accuracy: {clf_acc}% (gate "
          f"{res['fidelity_classifier']['quality_gate_result']})", flush=True)

    attempts = []          # every diffusion attempt incl. baseline, val-screened
    res["improvement_attempts"] = attempts

    def run_attempt(tag, **kw):
        hist = train_diffusion(tag=tag, **kw)
        val = eval_val_proxy(hist["checkpoint"], guidance_list=(1.5,),
                             base=kw.get("base", 16))
        rec = attempt_record(tag, hist, val)
        attempts.append(rec)
        save_results(res)
        print(f"[{tag}] val proxy fidelity {rec['val_ddim50_proxy_fidelity_w1.5_pct']}% "
              f"val MSE {rec['val_noise_mse_ema']}", flush=True)
        return rec

    # ---- Stage 3: baseline ----------------------------------------------
    baseline = run_attempt("baseline", epochs=12, data_idx=subset_idx,
                           ema_decay=0.99, schedule="cosine", augment=False,
                           lr=2e-3, batch=256, base=16, seed=42)
    baseline["role"] = "baseline (original project config; only the data changed)"
    res["baseline"] = {"config": baseline["config"],
                       "train_seconds": baseline["train_seconds"],
                       "final_train_loss": baseline["final_train_loss"],
                       "val_noise_mse_ema": baseline["val_noise_mse_ema"],
                       "checkpoint_selection": "final-epoch EMA weights (mirrors the "
                                               "original run); val metrics logged but "
                                               "not used to pick a baseline checkpoint"}
    save_results(res)
    test_base = eval_test("checkpoint_real_baseline.pt", main_guidance=1.5,
                          cfg_ablation=(0.0, 1.5, 3.0),
                          grid_prefix="samples_real_baseline")
    test_base["sample_count_note"] = ("70 samples (7/class x 10 classes); the original "
                                      "synthetic protocol used 64 (16/class x 4 classes) - "
                                      "changed so the 10 digit classes stay balanced")
    res["baseline"]["test_evaluation"] = test_base
    save_results(res)
    print(f"baseline test: DDPM {test_base['ddpm']['fidelity_pct']}% "
          f"DDIM {test_base['ddim50']['fidelity_pct']}%", flush=True)

    # ---- Stage 4: Phase B -------------------------------------------------
    try:
        b1 = run_attempt("b1_ema999", epochs=12, data_idx=subset_idx,
                         ema_decay=0.999, schedule="cosine", augment=False,
                         lr=2e-3, batch=256, base=16, seed=42)
        best = max([baseline, b1], key=selection_key)
        b1["decision"] = ("keep - becomes the continuation parent"
                          if best is b1 else
                          "drop as final candidate (EMA 0.999 at only ~950 steps "
                          "lags); baseline stays the continuation parent")
        b1["notes"] = "Single-lever change vs baseline: EMA decay 0.99 -> 0.999."
        save_results(res)

        parent = best
        b2 = run_attempt("b2_longer_augment", epochs=12, data_idx=subset_idx,
                         ema_decay=parent["config"]["ema_decay"], schedule="cosine",
                         augment=True, lr=1e-3, batch=256, base=16,
                         init_ckpt=parent["checkpoint"], seed=43)
        b2["notes"] = (f"Continuation of {parent['tag']}: +12 epochs (longer training) "
                       "with random +-2px translation augmentation (brief lever 5; "
                       "no flips/rotations - not label-preserving for digits). "
                       "EMA decay inherited from parent so the EMA horizon keeps "
                       "tracking as total steps grow.")
        b2["decision"] = "screened on val - see selection outcome"
        save_results(res)

        best = max(attempts, key=selection_key)
        b3 = run_attempt("b3_moredata40k", epochs=6, data_idx=expanded_idx,
                         ema_decay=best["config"]["ema_decay"], schedule="cosine",
                         augment=True, lr=1e-3, batch=256, base=16,
                         init_ckpt=best["checkpoint"], seed=44)
        b3["notes"] = (f"Continuation of {best['tag']}: +6 epochs on expanded 40,000-image "
                       "train pool (baseline 20k + 20k extra train-side images; the "
                       "2,000-image validation block remains excluded).")
        b3["decision"] = "screened on val - see selection outcome"
        save_results(res)

        b4 = run_attempt("b4_linear_schedule", epochs=12, data_idx=subset_idx,
                         ema_decay=0.99, schedule="linear", augment=False,
                         lr=2e-3, batch=256, base=16, seed=42)
        b4["notes"] = ("Schedule comparison, single-lever vs baseline: linear beta "
                       "schedule instead of cosine (brief lever 3 / execution order "
                       "step 3), EMA 0.99 and all other settings as baseline.")
        b4["decision"] = "screened on val - see selection outcome"
        save_results(res)
    except Exception:
        res["phaseB_error"] = traceback.format_exc()
        save_results(res)
        print("Phase B training attempt failed; continuing to selection with "
              "the attempts logged so far.", flush=True)

    # ---- Selection + guidance retune on val -------------------------------
    selected = max(attempts, key=selection_key)
    for a in attempts:
        if "decision" not in a or a["decision"].startswith("screened"):
            a["decision"] = ("keep - selected as final" if a is selected else
                             "drop - val proxy below the selected attempt")
    sweep = eval_val_proxy(selected["checkpoint"], guidance_list=(0.0, 1.0, 1.5, 2.0, 3.0),
                           base=selected["config"]["base_channels"])
    table = sweep["ddim50_val_proxy"]
    best_w = max(table.items(),
                 key=lambda kv: (kv[1]["fidelity_pct"], kv[1]["diversity_pairwise_l2"],
                                 -float(kv[0])))[0]
    attempts.append({
        "tag": "b5_guidance_retune", "on_checkpoint": selected["tag"],
        "method": "DDIM-50 val fidelity proxy sweep (100 generated samples per w, "
                  "10/class); pick highest val fidelity, tie-break higher diversity, "
                  "then lower w",
        "val_proxy_by_w": table, "selected_w": float(best_w),
        "decision": f"keep - w={best_w} selected on validation for the final test run"})
    res["selection"] = {
        "rule": "primary: val DDIM-50 fidelity proxy at w=1.5; tie-break: lower val "
                "noise-MSE (EMA). Test fidelity never used for selection.",
        "selected_attempt": selected["tag"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_val_noise_mse_ema": selected["val_noise_mse_ema"],
        "selected_val_proxy_fidelity_w1.5_pct": selected["val_ddim50_proxy_fidelity_w1.5_pct"],
        "guidance_selected_on_val": float(best_w),
        "guidance_val_sweep": table}
    save_results(res)
    print(f"selected: {selected['tag']} guidance w={best_w}", flush=True)

    # ---- Stage 5: final test evaluation (once) ----------------------------
    test_final = eval_test(selected["checkpoint"], main_guidance=float(best_w),
                           cfg_ablation=None,
                           base=selected["config"]["base_channels"],
                           grid_prefix="samples_real_final")
    test_final["guidance_note"] = ("main samplers evaluated at the validation-selected "
                                   f"w={best_w}; no CFG sweep was run on test for the "
                                   "final config (test budget: baseline once + final once)")
    res["final"] = {"selected_attempt": selected["tag"],
                    "config": selected["config"],
                    "train_seconds_total_for_attempt_chain": None,
                    "val_noise_mse_ema": selected["val_noise_mse_ema"],
                    "test_evaluation": test_final}
    try:
        syn = json.load(open("results.json"))
        res["synthetic_reference"] = syn
    except Exception:
        pass
    res["status"] = "complete"
    res["total_wall_seconds"] = round(time.time() - t_start, 1)
    save_results(res)
    print(f"FINAL test: DDPM {test_final['ddpm']['fidelity_pct']}% "
          f"DDIM {test_final['ddim50']['fidelity_pct']}%", flush=True)
    print(f"all done in {res['total_wall_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
