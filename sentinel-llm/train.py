"""
Sentinel-LLM — Local CPU training with SOTA-style experimental protocol.
Author: Ashutosh Sahoo, M.Tech AI, NIT Silchar

Protocol (mirrors how a real LLM LoRA study is run):
  Split      : stratified 70% train / 15% validation / 15% test (seed-fixed).
               Validation is used ONLY for hyperparameter selection;
               test is touched once per finished model.
  Stage A    : Brief base training — 1 epoch, all parameters.
               -> base_accuracy (test). This is the shared starting
               checkpoint for every experiment below (fair comparison).
  Stage B    : Full fine-tuning BASELINE — from the Stage-A checkpoint,
               train ALL parameters for 8 more epochs.
               -> full_ft_accuracy, 100% trainable params.
  Stage C    : LoRA hyperparameter SWEEP — from the SAME Stage-A
               checkpoint, freeze the base and train only LoRA adapters
               for 4 configs (r = 4/8/16 and a learning-rate variant),
               10 epochs each, selected on VALIDATION accuracy
               (tie-break: lower val loss, then fewer trainable params).
  Stage D    : The winning LoRA model is evaluated once on test
               -> lora_accuracy, saved as sentinel_lora.pt.

Efficiency story: LoRA is expected to match / beat full fine-tuning
while training <5% of the parameters — that parity-at-a-fraction-of-
the-cost is exactly why LoRA/QLoRA is the current SOTA practice for
adapting LLMs (Hu et al. 2022; Dettmers et al. 2023).

Label space: union of `label` values in dataset.jsonl (27 labels
across 6 tasks — see data_gen.py). Input = instruction + input text.
Tokenisation is pre-encoded once (hashlib hashing) for speed.
"""
import copy
import json
import os
import random
import time
from collections import defaultdict, Counter

import torch
import torch.nn as nn

from model import SentinelClassifier, encode, VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM
from lora import inject_lora, unfreeze_lora_only, count_parameters

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.jsonl")
RESULTS_PATH = os.path.join(BASE_DIR, "results.json")
CKPT_PATH = os.path.join(BASE_DIR, "sentinel_lora.pt")

BATCH = 32
BASE_EPOCHS = 1          # deliberately brief — the "pretrained base"
BASE_LR = 3e-3
FULL_FT_EPOCHS = 8
FULL_FT_LR = 3e-3
SWEEP_EPOCHS = 10
SWEEP_CONFIGS = [
    {"r": 4,  "alpha": 8,  "lr": 8e-3},
    {"r": 8,  "alpha": 16, "lr": 8e-3},
    {"r": 16, "alpha": 32, "lr": 8e-3},
    {"r": 8,  "alpha": 16, "lr": 3e-3},
]


# ---------------------------------------------------------------- data
def load_splits():
    rows = [json.loads(l) for l in open(DATA_PATH)]
    labels = sorted(set(r["label"] for r in rows))
    label2id = {l: i for i, l in enumerate(labels)}
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    train_rows, val_rows, test_rows = [], [], []
    rng = random.Random(123)
    for lab, items in by_label.items():
        rng.shuffle(items)
        c1 = int(len(items) * 0.70)
        c2 = int(len(items) * 0.85)
        train_rows.extend(items[:c1])
        val_rows.extend(items[c1:c2])
        test_rows.extend(items[c2:])
    for split in (train_rows, val_rows, test_rows):
        rng.shuffle(split)
    return rows, train_rows, val_rows, test_rows, labels, label2id


class Split:
    """Pre-encoded split: token ids computed once, collated per batch."""
    def __init__(self, rows, label2id):
        self.enc = [encode(r["instruction"] + " " + r["input"]) for r in rows]
        self.ys = [label2id[r["label"]] for r in rows]
        self.tasks = [r["task"] for r in rows]
        self.n = len(rows)

    def batch(self, idxs):
        batch = [self.enc[i] for i in idxs]
        maxlen = max(len(b) for b in batch)
        x = torch.zeros(len(batch), maxlen, dtype=torch.long)
        m = torch.zeros(len(batch), maxlen, dtype=torch.float)
        for i, b in enumerate(batch):
            x[i, :len(b)] = torch.tensor(b, dtype=torch.long)
            m[i, :len(b)] = 1.0
        y = torch.tensor([self.ys[i] for i in idxs], dtype=torch.long)
        return x, m, y


# ---------------------------------------------------------------- train / eval
def evaluate(model, split: Split):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    correct = 0; total_loss = 0.0
    per_task_c = Counter(); per_task_t = Counter()
    with torch.no_grad():
        for i in range(0, split.n, 64):
            idxs = list(range(i, min(i + 64, split.n)))
            x, m, y = split.batch(idxs)
            logits = model(x, m)
            total_loss += loss_fn(logits, y).item()
            for j, p in enumerate(logits.argmax(-1).tolist()):
                t = split.tasks[idxs[j]]; per_task_t[t] += 1
                if p == split.ys[idxs[j]]:
                    correct += 1; per_task_c[t] += 1
    acc = correct / split.n
    per_task = {t: round(per_task_c[t] / per_task_t[t], 4) for t in sorted(per_task_t)}
    return {"accuracy": round(acc, 4), "loss": round(total_loss / split.n, 4), "per_task": per_task}


def train_epochs(model, split: Split, epochs, lr, seed=42, verbose=True):
    torch.manual_seed(seed); random.seed(seed)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        order = list(range(split.n)); random.shuffle(order)
        total = 0.0
        for i in range(0, split.n, BATCH):
            idxs = order[i:i + BATCH]
            x, m, y = split.batch(idxs)
            loss = loss_fn(model(x, m), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idxs)
        sched.step()
        if verbose:
            print(f"    epoch {ep+1}/{epochs} loss={total/split.n:.4f}")
    return model


def fresh_from(base_state, num_labels):
    m = SentinelClassifier(num_labels=num_labels)
    m.load_state_dict(copy.deepcopy(base_state))
    return m


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    if not os.path.exists(DATA_PATH):
        raise SystemExit("dataset.jsonl missing — run: python data_gen.py")
    rows, train_rows, val_rows, test_rows, labels, label2id = load_splits()
    train_s, val_s, test_s = Split(train_rows, label2id), Split(val_rows, label2id), Split(test_rows, label2id)
    print(f"Dataset: {len(rows)} total | train {train_s.n} | val {val_s.n} | test {test_s.n} | labels {len(labels)}")
    print(f"Labels: {labels}")

    # ---- Stage A: brief base ----
    print(f"\n[Stage A] Brief BASE training ({BASE_EPOCHS} epoch, all params)...")
    torch.manual_seed(42); random.seed(42)
    base_model = SentinelClassifier(num_labels=len(labels))
    untrained = evaluate(base_model, test_s)
    print(f"  Untrained test accuracy: {untrained['accuracy']:.4f} (chance ≈ {1/len(labels):.4f})")
    t = time.time()
    train_epochs(base_model, train_s, epochs=BASE_EPOCHS, lr=BASE_LR)
    base_time = time.time() - t
    base_test = evaluate(base_model, test_s)
    base_val = evaluate(base_model, val_s)
    print(f"  Base test accuracy: {base_test['accuracy']:.4f} | val: {base_val['accuracy']:.4f}")
    base_state = copy.deepcopy(base_model.state_dict())
    base_stats = count_parameters(base_model)

    # ---- Stage B: full fine-tuning baseline ----
    print(f"\n[Stage B] FULL fine-tuning baseline ({FULL_FT_EPOCHS} epochs, 100% params)...")
    ft_model = fresh_from(base_state, len(labels))
    t = time.time()
    train_epochs(ft_model, train_s, epochs=FULL_FT_EPOCHS, lr=FULL_FT_LR)
    full_time = time.time() - t
    ft_test = evaluate(ft_model, test_s)
    ft_val = evaluate(ft_model, val_s)
    ft_stats = count_parameters(ft_model)
    print(f"  Full-FT test accuracy: {ft_test['accuracy']:.4f} | val: {ft_val['accuracy']:.4f} | trainable {ft_stats['trainable']} (100%)")

    # ---- Stage C: LoRA sweep on validation ----
    print(f"\n[Stage C] LoRA sweep ({len(SWEEP_CONFIGS)} configs x {SWEEP_EPOCHS} epochs, selected on VAL)...")
    sweep = []
    best = None  # (sort_key, config, model, stats, val_metrics, elapsed)
    for cfg in SWEEP_CONFIGS:
        torch.manual_seed(7); random.seed(7)
        m = fresh_from(base_state, len(labels))
        replaced = inject_lora(m, target_suffixes=("fc1", "fc2"), r=cfg["r"], alpha=cfg["alpha"], dropout=0.05)
        unfreeze_lora_only(m)
        stats = count_parameters(m)
        t = time.time()
        train_epochs(m, train_s, epochs=SWEEP_EPOCHS, lr=cfg["lr"], seed=7, verbose=False)
        elapsed = time.time() - t
        val_m = evaluate(m, val_s)
        entry = {"r": cfg["r"], "alpha": cfg["alpha"], "lr": cfg["lr"], "epochs": SWEEP_EPOCHS,
                 "trainable_params": stats["trainable"], "trainable_pct": stats["trainable_pct"],
                 "val_accuracy": val_m["accuracy"], "val_loss": val_m["loss"],
                 "training_time_sec": round(elapsed, 1)}
        sweep.append(entry)
        print(f"  r={cfg['r']:>2} alpha={cfg['alpha']:>2} lr={cfg['lr']:.4f}: val_acc={val_m['accuracy']:.4f} "
              f"val_loss={val_m['loss']:.4f} trainable={stats['trainable']} ({stats['trainable_pct']}%) {elapsed:.0f}s")
        sort_key = (val_m["accuracy"], -val_m["loss"], -stats["trainable"])
        if best is None or sort_key > best[0]:
            best = (sort_key, cfg, m, stats, val_m, elapsed, replaced)

    _, best_cfg, best_model, best_stats, best_val, best_time, replaced = best
    print(f"\n  >> Best config (by val): r={best_cfg['r']} alpha={best_cfg['alpha']} lr={best_cfg['lr']} "
          f"val_acc={best_val['accuracy']:.4f}")

    # ---- Stage D: single test evaluation of the winner ----
    lora_test = evaluate(best_model, test_s)
    print(f"  Best LoRA test accuracy: {lora_test['accuracy']:.4f} | per-task: {lora_test['per_task']}")
    total_time = time.time() - t0

    torch.save({
        "model_state": best_model.state_dict(),
        "labels": labels, "label2id": label2id,
        "config": {"vocab_size": VOCAB_SIZE, "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                   "lora_r": best_cfg["r"], "lora_alpha": best_cfg["alpha"], "num_labels": len(labels)},
        "base_accuracy": base_test["accuracy"], "lora_accuracy": lora_test["accuracy"],
    }, CKPT_PATH)
    print(f"\nSaved checkpoint -> {CKPT_PATH}")

    results = {
        "project": "Sentinel-LLM",
        "author": "Ashutosh Sahoo, M.Tech AI, NIT Silchar",
        "dataset_size": len(rows),
        "train_size": train_s.n, "val_size": val_s.n, "test_size": test_s.n,
        "num_labels": len(labels), "labels": labels,
        "tasks": sorted(set(r["task"] for r in rows)),
        "split": "stratified 70/15/15 train/val/test, seed 123",
        "total_params": best_stats["total"],
        "base_total_params": base_stats["total"],
        "trainable_lora_params": best_stats["trainable"],
        "trainable_pct": best_stats["trainable_pct"],
        "lora_config": {"r": best_cfg["r"], "alpha": best_cfg["alpha"], "lr": best_cfg["lr"],
                        "target_layers": replaced, "dropout": 0.05},
        "base_epochs": BASE_EPOCHS, "lora_epochs": SWEEP_EPOCHS,
        "untrained_accuracy": untrained["accuracy"],
        "base_accuracy": base_test["accuracy"],
        "base_val_accuracy": base_val["accuracy"],
        "lora_accuracy": lora_test["accuracy"],
        "lora_val_accuracy": best_val["accuracy"],
        "accuracy_gain_vs_base": round(lora_test["accuracy"] - base_test["accuracy"], 4),
        "per_task_accuracy_base": base_test["per_task"],
        "per_task_accuracy_lora": lora_test["per_task"],
        "hyperparameter_sweep": sweep,
        "sweep_selection": "best validation accuracy; tie-break lower val loss, then fewer trainable params",
        "full_finetuning_baseline": {
            "epochs": FULL_FT_EPOCHS, "lr": FULL_FT_LR,
            "trainable_params": ft_stats["trainable"], "trainable_pct": 100.0,
            "val_accuracy": ft_val["accuracy"], "test_accuracy": ft_test["accuracy"],
            "per_task_accuracy": ft_test["per_task"],
            "training_time_sec": round(full_time, 1),
        },
        "efficiency_summary": (
            f"LoRA (best config) trains {best_stats['trainable']} of {best_stats['total']} params "
            f"({best_stats['trainable_pct']}%) vs full FT {ft_stats['trainable']} (100%) — "
            f"test {lora_test['accuracy']:.4f} (LoRA) vs {ft_test['accuracy']:.4f} (full FT)."
        ),
        "training_time_sec": round(total_time, 1),
        "base_training_time_sec": round(base_time, 1),
        "lora_training_time_sec": round(best_time, 1),
        "device": "CPU",
        "notes": "Local demo: small hashed-embedding classifier proving the from-scratch LoRA pipeline with a SOTA-style sweep + full-FT baseline. Synthetic templated data is strongly separable (see README limitations). Full LLM QLoRA run: train_lora_peft.py on GPU — its accuracy is NOT claimed here because it has not been run.",
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results -> {RESULTS_PATH}")
    print(json.dumps({k: results[k] for k in ["base_accuracy", "lora_accuracy", "full_finetuning_baseline", "trainable_pct", "efficiency_summary"]}, indent=2))


if __name__ == "__main__":
    main()
