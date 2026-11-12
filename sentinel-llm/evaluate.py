"""
Sentinel-LLM — Evaluate the trained LoRA checkpoint on the held-out test split.
Author: Ashutosh Sahoo
Usage: python evaluate.py
"""
import json, os, random
from collections import Counter, defaultdict
import torch
from model import SentinelClassifier, encode_batch
from lora import inject_lora

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model():
    ckpt = torch.load(os.path.join(BASE_DIR, "sentinel_lora.pt"), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]; labels = ckpt["labels"]
    model = SentinelClassifier(num_labels=cfg["num_labels"], vocab_size=cfg["vocab_size"],
                               embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"])
    inject_lora(model, target_suffixes=("fc1", "fc2"), r=cfg["lora_r"], alpha=cfg["lora_alpha"], dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, labels, ckpt

def main():
    rows = [json.loads(l) for l in open(os.path.join(BASE_DIR, "dataset.jsonl"))]
    labels = sorted(set(r["label"] for r in rows))
    label2id = {l: i for i, l in enumerate(labels)}
    # Reproduce the exact test split from train.py (stratified 70/15/15)
    by_label = defaultdict(list)
    for r in rows: by_label[r["label"]].append(r)
    test_rows = []
    rng = random.Random(123)
    for lab, items in by_label.items():
        rng.shuffle(items)
        cut2 = int(len(items) * 0.85)
        test_rows.extend(items[cut2:])
    model, ckpt_labels, ckpt = load_model()
    assert ckpt_labels == labels, "label mismatch between checkpoint and dataset"
    correct = 0; per_task_c = Counter(); per_task_t = Counter()
    confusion = Counter()
    with torch.no_grad():
        for i in range(0, len(test_rows), 64):
            batch = test_rows[i:i+64]
            texts = [r["instruction"] + " " + r["input"] for r in batch]
            idx, mask = encode_batch(texts)
            preds = model(idx, mask).argmax(-1).tolist()
            for r, p in zip(batch, preds):
                per_task_t[r["task"]] += 1
                if labels[p] == r["label"]:
                    correct += 1; per_task_c[r["task"]] += 1
                else:
                    confusion[(r["label"], labels[p])] += 1
    acc = correct / len(test_rows)
    print(f"Test set: {len(test_rows)} examples")
    print(f"Overall accuracy (LoRA checkpoint): {acc:.4f}  ({correct}/{len(test_rows)})")
    print(f"Checkpoint base accuracy was: {ckpt['base_accuracy']:.4f} | LoRA accuracy at train time: {ckpt['lora_accuracy']:.4f}")
    print("\nPer-task accuracy:")
    for t in sorted(per_task_t):
        print(f"  {t:22s} {per_task_c[t]/per_task_t[t]:.4f}  ({per_task_c[t]}/{per_task_t[t]})")
    if confusion:
        print("\nTop confusions (true -> predicted):")
        for (true_l, pred_l), c in confusion.most_common(8):
            print(f"  {true_l} -> {pred_l}: {c}")
    else:
        print("\nNo misclassifications on the test split.")

if __name__ == "__main__":
    main()
