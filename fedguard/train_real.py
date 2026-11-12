"""
FedGuard on REAL data: MNIST federated + DP evaluation.

Phase A (baseline): the original synthetic-run comparison table, same
protocol shape, on MNIST downscaled to 16x16 (256-dim input):
  centralised / FedAvg IID / FedAvg non-IID (Dirichlet alpha=0.5) /
  FedProx mu=0.1 non-IID / DP-SGD sigma in {0.7, 1.2}.

Phase B (improvements): literature-backed levers screened on a 5,000-image
VALIDATION split carved from the training data (see mnist_data.py). Test
is evaluated for baseline configs and, once, for the val-selected final
config of each row family (see `finalize`). Screening runs never touch
the test set.

Methods are the project's own (federated.py / model.py / data_gen.py):
FedAvg weighted aggregation, FedProx proximal term, DP-SGD with TRUE
per-sample clipping via torch.func grad+vmap, and the project's
simplified RDP epsilon ESTIMATE (federated.estimate_epsilon, reused
unchanged; always labelled an estimate, never a formal guarantee).
The training loops below are federated.py's loops with the model
instantiated as MLP(input_dim=256); Phase B adds server-side momentum
(FedAvgM) and FedAdam (FedOpt, Reddi et al. 2020) at the aggregation step.

Documented protocol changes vs the synthetic run (also in
results_real.json):
  1. Training pool 55,000 (60k MNIST train minus the 5k validation
     carve-out used for Phase B selection); test = full 10,000 MNIST test.
  2. Centralised epochs 80 -> 30: the synthetic problem needed 80; the
     MNIST-16 MLP's validation curve plateaus far earlier (visible in the
     stored per-epoch val curve) and CPU is shared.
  3. DP rows use the non-IID partition, as specified for this real run.
     (The synthetic run's DP rows used IID to isolate the DP cost; the
     synthetic DP numbers are therefore not partition-matched to these.)
  4. DP lot (batch) size 32 -> 256 and DP lr 0.15 -> 0.4: at lot 32 the
     torch.func per-sample step costs ~195 ms on this CPU (measured),
     i.e. ~110 min per sigma at the original scale; at lot 256 the step
     costs ~34 ms (~2.5 min per sigma). Lot >= 256 is also the brief's
     recommended DP regime (noise averages down over the lot). The lr
     change partially compensates for 8x fewer steps per epoch. Epsilon
     estimates are recomputed for the actual (sigma, q, steps) used.
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_gen import dirichlet_partition, iid_partition, label_histogram, NUM_CLIENTS
from federated import aggregate, estimate_epsilon
from mnist_data import INPUT_DIM, SOURCE_URL, load_mnist16
from model import MLP, count_params, get_state, set_state

torch.set_num_threads(4)

RESULTS_PATH = os.path.join(HERE, "results_real.json")
CKPT_DIR = os.path.join(HERE, "checkpoints_real")
CRITERION = nn.CrossEntropyLoss()
SEED = 42


def make_model():
    return MLP(input_dim=INPUT_DIM)


# ---------------------------------------------------------------- eval
def evaluate_full(state, X, y, batch=1024):
    """Returns (accuracy, macro_f1) of a state dict on (X, y)."""
    model = make_model()
    set_state(model, state)
    model.eval()
    preds_all = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch])
            preds_all.append(model(xb).argmax(dim=1).numpy())
    preds = np.concatenate(preds_all)
    acc = float((preds == y).mean())
    f1s = []
    for c in range(10):
        tp = int(((preds == c) & (y == c)).sum())
        fp = int(((preds == c) & (y != c)).sum())
        fn = int(((preds != c) & (y == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return acc, float(np.mean(f1s))


# ---------------------------------------------------------------- clients
def _prox_term(model, global_state, mu):
    term = 0.0
    for name, p in model.named_parameters():
        term = term + ((p - global_state[name]) ** 2).sum()
    return (mu / 2.0) * term


def train_client_standard(state, X, y, epochs=3, lr=0.08, batch_size=32,
                          mu=0.0, seed=0):
    """federated.train_client_standard, MLP(input_dim=256)."""
    torch.manual_seed(seed)
    model = make_model()
    set_state(model, state)
    global_state = get_state(model)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = CRITERION(model(Xt[idx]), yt[idx])
            if mu > 0:
                loss = loss + _prox_term(model, global_state, mu)
            loss.backward()
            opt.step()
    return get_state(model)


def train_client_dp(state, X, y, epochs=1, lr=0.4, batch_size=256,
                    sigma=1.0, clip=1.0, seed=0):
    """federated._dp_train_client, MLP(input_dim=256): true per-sample
    clipping via torch.func grad+vmap, Gaussian noise on the clipped sum."""
    torch.manual_seed(seed)
    model = make_model()
    set_state(model, state)
    model.eval()
    params = {k: v.detach().clone().requires_grad_(True)
              for k, v in model.named_parameters()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    full = {**params, **buffers}

    def loss_fn(p, x_single, y_single):
        out = functional_call(model, p, (x_single.unsqueeze(0),))
        return nn.functional.cross_entropy(out, y_single.unsqueeze(0))

    per_sample_grad = vmap(grad(loss_fn), in_dims=(None, 0, 0))
    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    n = len(X)
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            B = len(idx)
            grads = per_sample_grad(full, xb, yb)
            flat = torch.cat([g.reshape(B, -1) for g in grads.values()
                              if g is not None], dim=1)
            norms = flat.norm(dim=1).clamp(min=1e-6)
            scale = (clip / norms).clamp(max=1.0)
            new_params = {}
            for k, p in params.items():
                g = grads[k]
                shape = [B] + [1] * (g.dim() - 1)
                clipped_sum = (g * scale.view(*shape)).sum(dim=0)
                noise = torch.normal(mean=0.0, std=sigma * clip, size=p.shape)
                avg_grad = (clipped_sum + noise) / B
                new_params[k] = (p - lr * avg_grad).detach().clone().requires_grad_(True)
            params = new_params
            full = {**params, **buffers}
            steps += 1
    out_model = make_model()
    sd = out_model.state_dict()
    for k in sd:
        if k in params:
            sd[k] = params[k].detach()
    return sd, steps


# ---------------------------------------------------------------- server
def run_federated(X_train, y_train, client_idx_list, eval_sets,
                  rounds=15, local_epochs=3, lr=0.08, batch_size=32,
                  mu=0.0, sigma=None, clip=1.0, seed=SEED,
                  server="avg", server_beta=0.9, server_lr=1.0,
                  adam_eta=0.1, adam_betas=(0.9, 0.99), adam_tau=1e-3,
                  log_prefix=""):
    """Federated loop with FedAvg / FedAvgM / FedAdam server step.

    eval_sets: dict name -> (X, y); every set is evaluated each round.
    Returns (final_state, curves{name: [(acc, f1), ...]}, info).
    """
    torch.manual_seed(seed)
    state = get_state(make_model())
    curves = {name: [] for name in eval_sets}
    total_dp_steps = {}
    velocity = {k: torch.zeros_like(v) for k, v in state.items()}
    adam_m = {k: torch.zeros_like(v) for k, v in state.items()}
    adam_v = {k: torch.zeros_like(v) for k, v in state.items()}

    for r in range(rounds):
        states, weights = [], []
        for c, idx in enumerate(client_idx_list):
            Xc, yc = X_train[idx], y_train[idx]
            if sigma is not None:
                new_state, steps = train_client_dp(
                    state, Xc, yc, epochs=local_epochs, lr=lr,
                    batch_size=batch_size, sigma=sigma, clip=clip,
                    seed=seed + r * 100 + c)
                total_dp_steps[c] = total_dp_steps.get(c, 0) + steps
            else:
                new_state = train_client_standard(
                    state, Xc, yc, epochs=local_epochs, lr=lr,
                    batch_size=batch_size, mu=mu, seed=seed + r * 100 + c)
            states.append(new_state)
            weights.append(len(idx))
        agg = aggregate(states, weights)
        if server == "avg":
            state = agg
        elif server == "avgm":  # FedAvgM (Hsu et al. 2019)
            new_state = {}
            for k in state:
                delta = agg[k] - state[k]
                velocity[k] = server_beta * velocity[k] + delta
                new_state[k] = state[k] + server_lr * velocity[k]
            state = new_state
        elif server == "adam":  # FedAdam (Reddi et al. 2020)
            t = r + 1
            b1, b2 = adam_betas
            new_state = {}
            for k in state:
                g = state[k] - agg[k]  # pseudo-gradient
                adam_m[k] = b1 * adam_m[k] + (1 - b1) * g
                adam_v[k] = b2 * adam_v[k] + (1 - b2) * g * g
                mhat = adam_m[k] / (1 - b1 ** t)
                vhat = adam_v[k] / (1 - b2 ** t)
                new_state[k] = state[k] - adam_eta * mhat / (vhat.sqrt() + adam_tau)
            state = new_state
        else:
            raise ValueError(server)
        msg = []
        for name, (Xe, ye) in eval_sets.items():
            acc, f1 = evaluate_full(state, Xe, ye)
            curves[name].append((round(acc, 4), round(f1, 4)))
            msg.append(f"{name} {acc:.4f}")
        print(f"  [{log_prefix}] round {r + 1}/{rounds} " + " ".join(msg),
              flush=True)

    info = {}
    if sigma is not None:
        avg_client_n = float(np.mean([len(s) for s in client_idx_list]))
        q = batch_size / avg_client_n
        worst_steps = max(total_dp_steps.values()) if total_dp_steps else 0
        info["dp"] = {
            "sigma": sigma, "clip": clip, "lot_size": batch_size,
            "lr": lr, "rounds": rounds,
            "sampling_rate_q": round(float(q), 6),
            "worst_client_steps": int(worst_steps),
            "epsilon_estimate": round(estimate_epsilon(sigma, q, worst_steps), 2),
            "delta": 1e-5,
            "accountant": "simplified RDP estimate (project's existing method, "
                          "federated.estimate_epsilon) - an ESTIMATE, not a "
                          "formal/certified (epsilon, delta) guarantee",
        }
    return state, curves, info


def train_centralised(X_train, y_train, eval_sets, epochs=30, lr=0.001,
                      batch_size=32, seed=SEED, log_prefix="central"):
    """federated.train_centralised recipe (AdamW + cosine), 256-dim MLP."""
    torch.manual_seed(seed)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(X_train)
    yt = torch.tensor(y_train)
    curves = {name: [] for name in eval_sets}
    n = len(X_train)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = CRITERION(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
        msg = []
        state = get_state(model)
        for name, (Xe, ye) in eval_sets.items():
            acc, f1 = evaluate_full(state, Xe, ye)
            curves[name].append((round(acc, 4), round(f1, 4)))
            msg.append(f"{name} {acc:.4f}")
        print(f"  [{log_prefix}] epoch {ep + 1}/{epochs} " + " ".join(msg),
              flush=True)
    return get_state(model), curves


# ---------------------------------------------------------------- registry
# Baseline configs mirror experiments.py; Phase B configs are screened on
# validation only (eval_test=False).
BASELINE = {
    "centralised": dict(kind="central", epochs=30, lr=0.001, batch_size=32),
    "fedavg_iid": dict(kind="fed", partition="iid", rounds=15, local_epochs=3,
                       lr=0.08, batch_size=32, mu=0.0),
    "fedavg_noniid": dict(kind="fed", partition="noniid", rounds=15,
                          local_epochs=3, lr=0.08, batch_size=32, mu=0.0),
    "fedprox_noniid": dict(kind="fed", partition="noniid", rounds=15,
                           local_epochs=3, lr=0.08, batch_size=32, mu=0.1),
    "fedavg_dp_sigma0.7": dict(kind="fed", partition="noniid", rounds=20,
                               local_epochs=1, lr=0.4, batch_size=256,
                               sigma=0.7, clip=1.0),
    "fedavg_dp_sigma1.2": dict(kind="fed", partition="noniid", rounds=20,
                               local_epochs=1, lr=0.4, batch_size=256,
                               sigma=1.2, clip=1.0),
}

PHASEB = {
    # rounds lever (brief lever 1) + local-epoch tuning (lever 2)
    "pb_fedavg_noniid_long": dict(kind="fed", partition="noniid", rounds=40,
                                  local_epochs=2, lr=0.08, batch_size=32, mu=0.0),
    # server momentum (brief lever 3)
    "pb_fedavgm_noniid": dict(kind="fed", partition="noniid", rounds=30,
                              local_epochs=2, lr=0.08, batch_size=32, mu=0.0,
                              server="avgm", server_beta=0.9, server_lr=1.0),
    # FedAdam (brief lever 3, prioritised in the task directive)
    "pb_fedadam_noniid": dict(kind="fed", partition="noniid", rounds=30,
                              local_epochs=2, lr=0.08, batch_size=32, mu=0.0,
                              server="adam", adam_eta=0.1),
    # FedProx mu sweep (brief lever 4)
    "pb_fedprox_mu0.01": dict(kind="fed", partition="noniid", rounds=15,
                              local_epochs=3, lr=0.08, batch_size=32, mu=0.01),
    "pb_fedprox_mu1.0": dict(kind="fed", partition="noniid", rounds=15,
                             local_epochs=3, lr=0.08, batch_size=32, mu=1.0),
    # NOTE: a FedProx+rounds (40r) run, an IID+rounds run and a 60-epoch
    # centralised run were dropped from Phase B for CPU budget (shared
    # 1.5-CPU cell, 5 parallel project agents; centralised baseline alone
    # took 811 s). The rounds lever is screened on the non-IID FedAvg
    # family; the centralised baseline's per-epoch val curve (stored in
    # results_real.json) shows the plateau. Dropped configs are listed in
    # results_real.json phaseb_scope_notes - not silently omitted.
    # DP clipping sweep at fixed sigma (brief lever 5)
    "pb_dp_s0.7_clip0.5": dict(kind="fed", partition="noniid", rounds=20,
                               local_epochs=1, lr=0.4, batch_size=256,
                               sigma=0.7, clip=0.5),
    "pb_dp_s1.2_clip0.5": dict(kind="fed", partition="noniid", rounds=20,
                               local_epochs=1, lr=0.4, batch_size=256,
                               sigma=1.2, clip=0.5),
    # DP lr sweep at fixed sigma/clip
    "pb_dp_s0.7_lr1.0": dict(kind="fed", partition="noniid", rounds=20,
                             local_epochs=1, lr=1.0, batch_size=256,
                             sigma=0.7, clip=1.0),
    "pb_dp_s1.2_lr1.0": dict(kind="fed", partition="noniid", rounds=20,
                             local_epochs=1, lr=1.0, batch_size=256,
                             sigma=1.2, clip=1.0),
    # DP frontier extra point: fewer rounds -> fewer steps -> lower epsilon estimate
    "pb_dp_s1.2_clip0.5_lr1.0_10r": dict(kind="fed", partition="noniid", rounds=10,
                                         local_epochs=1, lr=1.0, batch_size=256,
                                         sigma=1.2, clip=0.5),
    # Extension round (added after the first finalize, because the val
    # trend lr 0.4 -> 1.0 was still rising: +5.7 pts at sigma=0.7):
    "pb_dp_s0.7_lr2.0": dict(kind="fed", partition="noniid", rounds=20,
                             local_epochs=1, lr=2.0, batch_size=256,
                             sigma=0.7, clip=1.0),
    "pb_dp_s1.2_lr2.0": dict(kind="fed", partition="noniid", rounds=20,
                             local_epochs=1, lr=2.0, batch_size=256,
                             sigma=1.2, clip=1.0),
    # Higher-epsilon frontier point: more rounds at the better lr
    # (steps double, so the epsilon estimate rises - a distinct level).
    "pb_dp_s0.7_lr2.0_40r": dict(kind="fed", partition="noniid", rounds=40,
                                 local_epochs=1, lr=2.0, batch_size=256,
                                 sigma=0.7, clip=1.0),
}

FAMILIES = {
    "centralised": ["centralised"],
    "fedavg_iid": ["fedavg_iid"],
    "fedavg_noniid": ["fedavg_noniid", "pb_fedavg_noniid_long",
                       "pb_fedavgm_noniid", "pb_fedadam_noniid"],
    "fedprox_noniid": ["fedprox_noniid", "pb_fedprox_mu0.01",
                        "pb_fedprox_mu1.0"],
    "dp_sigma0.7": ["fedavg_dp_sigma0.7", "pb_dp_s0.7_clip0.5", "pb_dp_s0.7_lr1.0",
                    "pb_dp_s0.7_lr2.0"],
    "dp_sigma1.2": ["fedavg_dp_sigma1.2", "pb_dp_s1.2_clip0.5", "pb_dp_s1.2_lr1.0",
                    "pb_dp_s1.2_lr2.0"],
    "dp_sigma1.2_low_eps_frontier": ["pb_dp_s1.2_clip0.5_lr1.0_10r"],
    "dp_sigma0.7_high_eps_frontier": ["pb_dp_s0.7_lr2.0_40r"],
}


# ---------------------------------------------------------------- results io
def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None


def save_results(res):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, RESULTS_PATH)


def setup_results(X_pool, y_pool, X_val, y_val, X_test, y_test, meta, iid, noniid):
    res = {
        "dataset": {
            "name": "MNIST (LeCun et al.)",
            "source_url": SOURCE_URL,
            "file": "mnist.npz (downloaded into this project folder)",
            "sha256_mnist_npz": meta.get("sha256_mnist_npz"),
            "verified_counts": {"train_images_in_file": 60000,
                                "test_images_in_file": 10000},
            "adaptation": "28x28 uint8 -> [0,1] -> bilinear downscale to "
                          "16x16 -> flattened 256-dim input",
            "input_dim": INPUT_DIM,
            "num_classes": 10,
        },
        "splits": {
            "train_pool": int(len(X_pool)),
            "validation": int(len(X_val)),
            "validation_carve": "stratified 500/class (5000 total) carved "
                                "from the 60000 training images, seed 777; "
                                "used ONLY for Phase B config selection",
            "test": int(len(X_test)),
            "test_note": "full 10000-image MNIST test set, identical for "
                         "every configuration",
        },
        "model": {"name": "MLP 256->64->64->10, dropout 0.3 (project's "
                          "shared MLP with input_dim=256)",
                  "params": count_params(make_model())},
        "partitions": {
            "num_clients": NUM_CLIENTS,
            "iid": "random shuffle split (data_gen.iid_partition, seed 43)",
            "noniid": "Dirichlet alpha=0.5 label-skew partition "
                      "(data_gen.dirichlet_partition, seed 44)",
            "client_sizes_iid": [len(s) for s in iid],
            "client_sizes_noniid": [len(s) for s in noniid],
            "client_histograms_noniid": [label_histogram(y_pool, idx) for idx in noniid],
        },
        "protocol_changes_vs_synthetic_run": [
            "Training pool is 55,000 not 60,000: 5,000 training images are "
            "held out as the Phase B validation split. Test set unchanged "
            "(full 10,000).",
            "Centralised epochs 80 -> 30 (baseline): MNIST-16 validation "
            "curve plateaus far earlier than the synthetic problem; CPU is "
            "shared. A 60-epoch run is screened in Phase B.",
            "DP rows use the non-IID (Dirichlet alpha=0.5) partition as "
            "specified for this real run; the synthetic run's DP rows used "
            "IID, so synthetic-vs-real DP rows are not partition-matched.",
            "DP lot size 32 -> 256 and DP lr 0.15 -> 0.4: measured "
            "torch.func per-sample step cost ~195 ms at lot 32 (~110 min "
            "per sigma at original scale) vs ~34 ms at lot 256 (~2.5 min "
            "per sigma); lot>=256 is also the recommended DP regime in "
            "the improvement brief. Epsilon estimates recomputed for the "
            "actual sigma/q/steps.",
            "Input dimension 20 -> 256 (16x16 MNIST), model params "
            "6,154 -> 21,258. All other methods (FedAvg aggregation, "
            "FedProx mu, per-sample DP clipping, epsilon estimation) are "
            "the project's own, unchanged.",
        ],
        "epsilon_caveat": "All epsilon values are the project's simplified "
                          "RDP ESTIMATE (federated.estimate_epsilon), for "
                          "comparing noise levels inside this project only. "
                          "They are NOT a formal/certified (epsilon, delta) "
                          "privacy guarantee; production accounting requires "
                          "Opacus / TF-Privacy's exact RDP accountant.",
        "baseline": {},
        "phaseb": {},
        "improvements_log": [],
        "privacy_frontier": [],
        "final": {},
    }
    save_results(res)
    return res


# ---------------------------------------------------------------- run one
def run_one(name, cfg, data, res, eval_test):
    X_pool, y_pool, X_val, y_val, X_test, y_test, iid, noniid = data
    eval_sets = {"val": (X_val, y_val)}
    if eval_test:
        eval_sets["test"] = (X_test, y_test)
    t0 = time.time()
    print(f"== {name} ({cfg}) eval_test={eval_test} ==", flush=True)
    if cfg["kind"] == "central":
        state, curves = train_centralised(
            X_pool, y_pool, eval_sets, epochs=cfg["epochs"], lr=cfg["lr"],
            batch_size=cfg["batch_size"], log_prefix=name)
        info = {}
    else:
        clients = iid if cfg["partition"] == "iid" else noniid
        state, curves, info = run_federated(
            X_pool, y_pool, clients, eval_sets,
            rounds=cfg["rounds"], local_epochs=cfg["local_epochs"],
            lr=cfg["lr"], batch_size=cfg["batch_size"],
            mu=cfg.get("mu", 0.0), sigma=cfg.get("sigma"),
            clip=cfg.get("clip", 1.0),
            server=cfg.get("server", "avg"),
            server_beta=cfg.get("server_beta", 0.9),
            server_lr=cfg.get("server_lr", 1.0),
            adam_eta=cfg.get("adam_eta", 0.1),
            log_prefix=name)
    runtime = round(time.time() - t0, 1)
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = os.path.join(CKPT_DIR, f"{name}.pt")
    torch.save(state, ckpt)
    entry = {
        "config": {k: v for k, v in cfg.items()},
        "val_curve": curves["val"],
        "val_accuracy": curves["val"][-1][0],
        "val_macro_f1": curves["val"][-1][1],
        "runtime_seconds": runtime,
        "state_file": f"checkpoints_real/{name}.pt",
        "status": "done",
    }
    if eval_test:
        entry["test_curve"] = curves["test"]
        entry["test_accuracy"] = curves["test"][-1][0]
        entry["test_macro_f1"] = curves["test"][-1][1]
    if "dp" in info:
        entry["dp"] = info["dp"]
    section = "baseline" if name in BASELINE else "phaseb"
    res[section][name] = entry
    save_results(res)
    extra = ""
    if "dp" in entry:
        extra = f" eps_estimate~{entry['dp']['epsilon_estimate']}"
    test_msg = f" test {entry['test_accuracy']:.4f}" if eval_test else ""
    print(f"== {name}: val {entry['val_accuracy']:.4f}{test_msg}{extra} "
          f"({runtime}s) -> results_real.json", flush=True)
    return entry


def get_data_and_partitions():
    X_pool, y_pool, X_val, y_val, X_test, y_test, meta = load_mnist16()
    iid = iid_partition(y_pool)
    noniid = dirichlet_partition(y_pool)
    return (X_pool, y_pool, X_val, y_val, X_test, y_test, iid, noniid), meta


def finalize(res, data):
    """Pick the val-best config per family; evaluate its test accuracy
    ONCE (baseline winners reuse their already-recorded baseline test
    numbers - no second test touch)."""
    X_pool, y_pool, X_val, y_val, X_test, y_test, iid, noniid = data
    pools = {}
    pools.update(res.get("baseline", {}))
    pools.update(res.get("phaseb", {}))
    log = []
    final = {}
    frontier = []
    for family, members in FAMILIES.items():
        present = [m for m in members if m in pools and pools[m].get("status") == "done"]
        if not present:
            continue
        ranked = sorted(present, key=lambda m: -pools[m]["val_accuracy"])
        winner = ranked[0]
        for m in present:
            log.append({
                "family": family, "config_name": m,
                "config": pools[m]["config"],
                "val_accuracy": pools[m]["val_accuracy"],
                "val_macro_f1": pools[m]["val_macro_f1"],
                "decision": "keep (family winner, val-selected)" if m == winner
                            else "drop",
            })
        w = pools[winner]
        if "test_accuracy" in w:
            test_acc, test_f1, note = w["test_accuracy"], w["test_macro_f1"], \
                "test score from the baseline evaluation of this same config"
        else:
            state = torch.load(os.path.join(HERE, w["state_file"]),
                               weights_only=True)
            test_acc, test_f1 = evaluate_full(state, X_test, y_test)
            test_acc, test_f1 = round(test_acc, 4), round(test_f1, 4)
            note = "single test evaluation of the val-selected final config"
        entry = {
            "selected_config_name": winner,
            "config": w["config"],
            "val_accuracy": w["val_accuracy"],
            "val_macro_f1": w["val_macro_f1"],
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "test_evaluation_note": note,
        }
        if "dp" in w:
            entry["dp"] = w["dp"]
            frontier.append({
                "sigma": w["dp"]["sigma"],
                "rounds": w["dp"]["rounds"],
                "epsilon_estimate": w["dp"]["epsilon_estimate"],
                "epsilon_note": "project's simplified RDP estimate - NOT a "
                                "formal guarantee",
                "selected_config_name": winner,
                "val_accuracy": w["val_accuracy"],
                "test_accuracy": test_acc,
            })
        final[family] = entry
        print(f"FINAL {family}: {winner} val {w['val_accuracy']:.4f} "
              f"test {test_acc:.4f}", flush=True)
    res["improvements_log"] = log
    res["privacy_frontier"] = frontier
    res["final"] = final
    save_results(res)
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    data, meta = get_data_and_partitions()
    res = load_results()
    if res is None:
        X_pool, y_pool, X_val, y_val, X_test, y_test, iid, noniid = data
        res = setup_results(X_pool, y_pool, X_val, y_val, X_test, y_test,
                            meta, iid, noniid)
        print("results_real.json header written", flush=True)

    if mode == "setup":
        return
    if mode == "run":
        name = sys.argv[2]
        if name in BASELINE:
            if res["baseline"].get(name, {}).get("status") == "done":
                print(f"skip {name} (already done)")
                return
            run_one(name, BASELINE[name], data, res, eval_test=True)
        else:
            if res["phaseb"].get(name, {}).get("status") == "done":
                print(f"skip {name} (already done)")
                return
            run_one(name, PHASEB[name], data, res, eval_test=False)
        return
    if mode == "finalize":
        finalize(res, data)
        return
    if mode == "all":
        for name, cfg in BASELINE.items():
            if res["baseline"].get(name, {}).get("status") != "done":
                run_one(name, cfg, data, res, eval_test=True)
        for name, cfg in PHASEB.items():
            if res["phaseb"].get(name, {}).get("status") != "done":
                run_one(name, cfg, data, res, eval_test=False)
        finalize(res, data)
        return
    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
