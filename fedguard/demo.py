"""
FedGuard demo.

- Shows each non-IID client's label distribution as text bars (the
  non-IID problem made visible).
- Loads the FedProx non-IID checkpoint (falls back to FedAvg non-IID),
  evaluates the global model on the held-out test set.
- Prints the privacy-utility tradeoff table from results.json.
Run experiments.py first.
"""
import json
import os

import torch

from data_gen import generate_data, dirichlet_partition, iid_partition, label_histogram
from federated import evaluate
from model import MLP, set_state

HERE = os.path.dirname(os.path.abspath(__file__))


def bar(hist):
    peak = max(hist) if max(hist) else 1
    return " ".join(f"{h:3d}" for h in hist)


def main():
    with open(os.path.join(HERE, "results.json")) as f:
        results = json.load(f)

    Xtr, ytr, Xte, yte = generate_data()
    noniid = dirichlet_partition(ytr)

    print("=" * 72)
    print("FedGuard demo — train on 10 clients, data never leaves the client")
    print("=" * 72)
    print("\nNon-IID client label distributions (counts per class 0-9):")
    for i, idx in enumerate(noniid):
        hist = label_histogram(ytr, idx)
        top = sorted(range(10), key=lambda c: -hist[c])[:3]
        print(f"  client {i} (n={len(idx):3d}) {bar(hist)}  dominant classes: {top}")

    ckpt = None
    for name in ("fedprox_noniid.pt", "fedavg_noniid.pt"):
        p = os.path.join(HERE, "checkpoints", name)
        if os.path.exists(p):
            ckpt = (name, p)
            break
    if ckpt:
        state = torch.load(ckpt[1], weights_only=True)
        acc = evaluate(state, Xte, yte)
        print(f"\nLoaded checkpoint {ckpt[0]}")
        print(f"Global model test accuracy (re-measured now): {acc:.4f} "
              f"on {len(Xte)} held-out examples")
        # A few example predictions
        model = MLP(); set_state(model, state); model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(Xte[:8])).argmax(dim=1).tolist()
        print(f"First 8 test examples — true: {yte[:8].tolist()}")
        print(f"First 8 test examples — pred: {preds}")
    else:
        print("\nNo checkpoint found — run experiments.py first.")

    print("\nPrivacy-utility tradeoff (from results.json, all measured):")
    print(f"  {'experiment':28s} {'accuracy':>9s} {'epsilon (est.)':>15s}")
    for name, e in results["experiments"].items():
        eps = f"{e['dp']['epsilon_estimate']}" if "dp" in e else "-"
        print(f"  {name:28s} {e['final_accuracy']:9.4f} {eps:>15s}")
    print("\nReading: centralised is the ceiling; federation loses a little,")
    print("non-IID loses more, FedProx recovers part of it, and DP-SGD trades")
    print("a few accuracy points for a formal (epsilon, delta) privacy budget.")


if __name__ == "__main__":
    main()
