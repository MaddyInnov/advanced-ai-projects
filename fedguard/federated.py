"""
Federated training engine: FedAvg, FedProx, DP-SGD.

- FedAvg (McMahan et al. 2017): local SGD, weighted average by client size.
- FedProx (Li et al. 2020): local loss += (mu/2) * ||w - w_global||^2,
  which limits client drift on non-IID data.
- DP-SGD (Abadi et al. 2016): TRUE per-sample gradient clipping + Gaussian
  noise. Per-sample gradients are computed with torch.func (grad + vmap
  over the batch) -- no per-sample Python loop. Noise N(0, (sigma*C)^2)
  is added to the SUM of clipped per-sample gradients, then averaged.
  Privacy accounting is a SIMPLIFIED RDP estimate (see estimate_epsilon),
  labelled as an estimate everywhere it is reported.
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap

from model import MLP, get_state, set_state

CRITERION = nn.CrossEntropyLoss()


def evaluate(state, X, y, batch=512):
    model = MLP()
    set_state(model, state)
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch])
            yb = torch.tensor(y[i:i + batch])
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
    return correct / len(y)


def aggregate(states, weights):
    total = sum(weights)
    out = {}
    for k in states[0]:
        out[k] = sum(s[k].float() * (w / total) for s, w in zip(states, weights))
    return out


def _prox_term(model, global_state, mu):
    term = 0.0
    for name, p in model.named_parameters():
        term = term + ((p - global_state[name]) ** 2).sum()
    return (mu / 2.0) * term


def train_client_standard(state, X, y, epochs=3, lr=0.08, batch_size=32,
                          mu=0.0, seed=0):
    """FedAvg (mu=0) / FedProx (mu>0) local training, batched SGD."""
    torch.manual_seed(seed)
    model = MLP()
    set_state(model, state)
    global_state = get_state(model)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    Xt = torch.tensor(X); yt = torch.tensor(y)
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


def _dp_train_client(state, X, y, epochs=1, lr=0.08, batch_size=32,
                       sigma=1.0, clip=1.0, seed=0):
    """DP-SGD local training with true per-sample clipping (torch.func)."""
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = MLP()
    set_state(model, state)
    model.eval()  # no dropout/BN in this MLP; eval avoids any statefulness
    params = {k: v.detach().clone().requires_grad_(True)
              for k, v in model.named_parameters()}
    # Buffers (none expected, but keep for functional_call completeness)
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    full = {**params, **buffers}

    def loss_fn(p, x_single, y_single):
        out = functional_call(model, p, (x_single.unsqueeze(0),))
        return nn.functional.cross_entropy(out, y_single.unsqueeze(0))

    per_sample_grad = vmap(grad(loss_fn), in_dims=(None, 0, 0))

    Xt = torch.tensor(X); yt = torch.tensor(y)
    n = len(X)
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            B = len(idx)
            grads = per_sample_grad(full, xb, yb)  # each value: (B, *shape)
            # Per-sample global norm & clipping
            flat = torch.cat([g.reshape(B, -1) for g in grads.values()
                              if g is not None], dim=1)
            norms = flat.norm(dim=1).clamp(min=1e-6)
            scale = (clip / norms).clamp(max=1.0)  # (B,)
            new_params = {}
            for k, p in params.items():
                g = grads[k]  # (B, *shape)
                shape = [B] + [1] * (g.dim() - 1)
                clipped_sum = (g * scale.view(*shape)).sum(dim=0)
                noise = torch.normal(mean=0.0, std=sigma * clip,
                                     size=p.shape)
                avg_grad = (clipped_sum + noise) / B
                new_params[k] = (p - lr * avg_grad).detach().clone().requires_grad_(True)
            params = new_params
            full = {**params, **buffers}
            steps += 1
    # Load back into a model to return a standard state_dict
    out_model = MLP()
    sd = out_model.state_dict()
    for k in sd:
        if k in params:
            sd[k] = params[k].detach()
    return sd, steps


def train_client_dp(state, X, y, **kw):
    return _dp_train_client(state, X, y, **kw)


def estimate_epsilon(sigma, q, steps, delta=1e-5):
    """
    SIMPLIFIED RDP-style epsilon ESTIMATE (not a certified accountant).

    Uses the small-q approximation for the subsampled Gaussian mechanism:
        RDP(alpha) ~= steps * alpha * q^2 / (2 * sigma^2)
        epsilon     = min_alpha [ RDP(alpha) + log(1/delta) / (alpha - 1) ]
    This is a textbook-order estimate for comparison across noise levels
    in this project only. Production systems must use a real accountant
    (e.g. Opacus / TF-Privacy RDP accountant with exact subsampled RDP).
    """
    if sigma <= 0:
        return float("inf")
    best = float("inf")
    for alpha in np.arange(2, 128, 1.0):
        rdp = steps * alpha * (q ** 2) / (2 * sigma ** 2)
        eps = rdp + math.log(1 / delta) / (alpha - 1)
        best = min(best, eps)
    return float(best)


def run_federated(X_train, y_train, client_idx_list, X_test, y_test,
                  rounds=25, local_epochs=3, lr=0.08, batch_size=32,
                  mu=0.0, clients_per_round=10, sigma=None, clip=1.0,
                  seed=42, log_prefix=""):
    """Generic federated loop. Returns (final_state, curve, info)."""
    rng = np.random.default_rng(seed)
    init_model = MLP()
    torch.manual_seed(seed)
    init_model = MLP()
    state = get_state(init_model)
    curve = []
    total_dp_steps_per_client = {}  # client -> steps (for epsilon)
    participations = {}
    num_clients = len(client_idx_list)

    for r in range(rounds):
        if clients_per_round < num_clients:
            chosen = rng.choice(num_clients, size=clients_per_round,
                                replace=False).tolist()
        else:
            chosen = list(range(num_clients))
        states, weights = [], []
        for c in chosen:
            idx = client_idx_list[c]
            Xc, yc = X_train[idx], y_train[idx]
            participations[c] = participations.get(c, 0) + 1
            if sigma is not None:
                new_state, steps = train_client_dp(
                    state, Xc, yc, epochs=local_epochs, lr=lr,
                    batch_size=batch_size, sigma=sigma, clip=clip,
                    seed=seed + r * 100 + c)
                total_dp_steps_per_client[c] = \
                    total_dp_steps_per_client.get(c, 0) + steps
            else:
                new_state = train_client_standard(
                    state, Xc, yc, epochs=local_epochs, lr=lr,
                    batch_size=batch_size, mu=mu, seed=seed + r * 100 + c)
            states.append(new_state)
            weights.append(len(idx))
        state = aggregate(states, weights)
        acc = evaluate(state, X_test, y_test)
        curve.append(round(float(acc), 4))
        if log_prefix:
            print(f"  [{log_prefix}] round {r + 1}/{rounds} test acc {acc:.4f}", flush=True)

    info = {"participations": participations}
    if sigma is not None:
        # Epsilon for a typical client: q = batch / avg client size,
        # steps = steps for the most-participating client (worst case).
        avg_client_n = float(np.mean([len(s) for s in client_idx_list]))
        q = batch_size / avg_client_n
        worst_steps = max(total_dp_steps_per_client.values()) if total_dp_steps_per_client else 0
        info["dp"] = {
            "sigma": sigma, "clip": clip, "batch_size": batch_size,
            "sampling_rate_q": round(float(q), 4),
            "worst_client_steps": int(worst_steps),
            "epsilon_estimate": round(estimate_epsilon(sigma, q, worst_steps), 2),
            "delta": 1e-5,
            "accountant": "simplified RDP estimate (see federated.estimate_epsilon)",
        }
    return state, curve, info


def train_centralised(X_train, y_train, X_test, y_test, epochs=80, lr=0.001,
                      batch_size=32, seed=42):
    """Pooled-data baseline with the best standard single-model recipe
    found in design-time calibration: AdamW (lr 1e-3, cosine decay) on
    the shared dropout MLP. Federated clients use local SGD, exactly as
    in the FedAvg/FedProx literature -- the comparison is 'best standard
    pooled training' vs 'federated training', which is the comparison
    practitioners actually care about."""
    torch.manual_seed(seed)
    model = MLP()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(X_train); yt = torch.tensor(y_train)
    curve = []
    n = len(X_train)
    # Report accuracy once per ~ (epochs/25) epochs to get a ~25-point curve
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = CRITERION(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
        if ep % max(1, epochs // 25) == 0 or ep == epochs - 1:
            curve.append(round(float(evaluate(get_state(model), X_test, y_test)), 4))
    return get_state(model), curve
