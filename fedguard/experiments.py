"""
FedGuard experiment grid.

Runs (all with the SAME test set and SAME initialisation seed):
  1. Centralised            (upper bound, data pooled)
  2. FedAvg  IID
  3. FedAvg  non-IID (Dirichlet alpha=0.5)
  4. FedProx non-IID (mu tuned, see MU below)
  5. FedAvg + DP-SGD, sigma=0.7  (IID partition, to isolate DP cost)
  6. FedAvg + DP-SGD, sigma=1.2

Outputs results.json with final accuracies, per-round curves, client
histograms and DP epsilon estimates — every number measured, none assumed.
Also saves checkpoints/fedavg_noniid.pt and checkpoints/fedprox_noniid.pt
for demo.py.
"""
import json
import os
import time

import torch

from data_gen import (generate_data, iid_partition, dirichlet_partition,
                      label_histogram, NUM_CLIENTS)
from federated import run_federated, train_centralised, evaluate
from model import MLP, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
ROUNDS = 15
LOCAL_EPOCHS = 3
LR = 0.08
BATCH = 32
MU = 0.1            # FedProx proximal coefficient (tuned: mu sweep in README)
CENTRAL_EPOCHS = 80
DP_ROUNDS = 20
DP_LOCAL_EPOCHS = 1
DP_CLIENTS_PER_ROUND = 10
DP_LR = 0.15


def main():
    torch.set_num_threads(2)
    t_start = time.time()
    Xtr, ytr, Xte, yte = generate_data()
    iid = iid_partition(ytr)
    noniid = dirichlet_partition(ytr)

    results = {
        "dataset": {"n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                    "input_dim": int(Xtr.shape[1]), "num_classes": 10,
                    "num_clients": NUM_CLIENTS, "seed": 42,
                    "non_iid": "Dirichlet alpha=0.5 label-skew partition"},
        "model": {"name": "MLP 20->64->64->10, dropout 0.3", "params": count_params(MLP())},
        "config": {"rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "lr": LR,
                   "batch_size": BATCH, "fedprox_mu": MU,
                   "central_epochs": CENTRAL_EPOCHS,
                   "central_optimiser": "AdamW lr=0.001 cosine (federated clients use local SGD, per FedAvg/FedProx literature)",
                   "dp_rounds": DP_ROUNDS, "dp_local_epochs": DP_LOCAL_EPOCHS,
                   "dp_lr": DP_LR, "dp_clip": 1.0},
        "client_histograms_iid": [label_histogram(ytr, idx) for idx in iid],
        "client_histograms_noniid": [label_histogram(ytr, idx) for idx in noniid],
        "client_sizes_noniid": [len(s) for s in noniid],
        "experiments": {},
    }

    def record(name, state, curve, info=None):
        acc = evaluate(state, Xte, yte)
        entry = {"final_accuracy": round(float(acc), 4), "curve": curve}
        if info and "dp" in info:
            entry["dp"] = info["dp"]
        results["experiments"][name] = entry
        print(f"== {name}: final {acc:.4f}", flush=True)
        return state

    print("== Centralised ==", flush=True)
    state, curve = train_centralised(Xtr, ytr, Xte, yte, epochs=CENTRAL_EPOCHS, lr=0.001)
    record("centralised", state, curve)

    print("== FedAvg IID ==", flush=True)
    state, curve, _ = run_federated(Xtr, ytr, iid, Xte, yte, rounds=ROUNDS,
                                    local_epochs=LOCAL_EPOCHS, lr=LR,
                                    log_prefix="fedavg-iid")
    record("fedavg_iid", state, curve)

    print("== FedAvg non-IID ==", flush=True)
    state, curve, _ = run_federated(Xtr, ytr, noniid, Xte, yte, rounds=ROUNDS,
                                    local_epochs=LOCAL_EPOCHS, lr=LR,
                                    log_prefix="fedavg-noniid")
    record("fedavg_noniid", state, curve)
    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    torch.save(state, os.path.join(HERE, "checkpoints", "fedavg_noniid.pt"))

    print(f"== FedProx non-IID mu={MU} ==", flush=True)
    state, curve, _ = run_federated(Xtr, ytr, noniid, Xte, yte, rounds=ROUNDS,
                                    local_epochs=LOCAL_EPOCHS, lr=LR, mu=MU,
                                    log_prefix="fedprox-noniid")
    record("fedprox_noniid", state, curve)
    torch.save(state, os.path.join(HERE, "checkpoints", "fedprox_noniid.pt"))

    for sigma in (0.7, 1.2):
        key = f"fedavg_dp_sigma{sigma}"
        print(f"== FedAvg + DP sigma={sigma} (IID) ==", flush=True)
        state, curve, info = run_federated(
            Xtr, ytr, iid, Xte, yte, rounds=DP_ROUNDS,
            local_epochs=DP_LOCAL_EPOCHS, lr=DP_LR,
            clients_per_round=DP_CLIENTS_PER_ROUND,
            sigma=sigma, clip=1.0, log_prefix=key)
        record(key, state, curve, info)
        torch.save(state, os.path.join(HERE, "checkpoints", f"{key}.pt"))

    results["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {results['runtime_seconds']}s -> results.json")
    for k, v in results["experiments"].items():
        extra = f" eps~{v['dp']['epsilon_estimate']}" if "dp" in v else ""
        print(f"  {k:24s} {v['final_accuracy']:.4f}{extra}")


if __name__ == "__main__":
    main()
