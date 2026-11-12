"""Train the conditional DDPM on real MNIST-16 data.

Same training method as the original train.py (noise-prediction MSE,
CFG label dropout p=0.1, AdamW, cosine LR decay, EMA weights for sampling);
configuration differences are explicit arguments, logged per run:
  * 10 classes (null label index 10)
  * EMA decay, schedule, epochs, data pool and augmentation vary by attempt

Checkpoint selection protocol (pre-committed): after every epoch the EMA
model's noise-prediction MSE is measured on the fixed held-out validation
set (2,000 real train-side images, fixed timesteps/noise, seed 777) and
logged. Model/attempt selection uses ONLY these validation numbers and the
validation fidelity proxy in evaluate_real.py - never test fidelity.
"""
import copy
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_real import NULL_LABEL, load_mnist16, split_indices
from diffusion import Diffusion
from model import SmallUNet, count_params

T = 300


def make_val_protocol(diff, x_val):
    """Fixed timesteps + noise for the validation set (seed 777)."""
    g = torch.Generator().manual_seed(777)
    t = torch.randint(0, T, (len(x_val),), generator=g)
    noise = torch.randn(x_val.shape, generator=g)
    return t, noise


@torch.no_grad()
def val_noise_mse(model, diff, x_val, t_val, noise_val, y_val, batch=250):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(x_val), batch):
        x = x_val[i:i + batch]
        t = t_val[i:i + batch]
        noise = noise_val[i:i + batch]
        s_ab = diff._extract(diff.sqrt_ab, t, x.shape)
        s_1m = diff._extract(diff.sqrt_1m_ab, t, x.shape)
        x_t = s_ab * x + s_1m * noise
        total += F.mse_loss(model(x_t, t, y_val[i:i + batch]), noise,
                            reduction="sum").item()
        n += noise.numel()
    return round(total / n, 5)


def augment_shift(x, gen):
    """Random +-2 px translation of the whole batch (one shift per step,
    drawn fresh each step; digits are translation-tolerant, and wrapping
    only moves background pixels for centred MNIST digits)."""
    dx = int(torch.randint(-2, 3, (1,), generator=gen))
    dy = int(torch.randint(-2, 3, (1,), generator=gen))
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


def train_diffusion(tag, epochs, data_idx, ema_decay=0.99, schedule="cosine",
                    augment=False, lr=2e-3, batch=256, base=16,
                    init_ckpt=None, seed=42, device="cpu"):
    torch.manual_seed(seed)
    x_train_all, y_train_all, _, _, _ = load_mnist16()
    _, val_idx, _ = split_indices()
    x_val, y_val = x_train_all[val_idx], y_train_all[val_idx]
    xs, ys = x_train_all[data_idx], y_train_all[data_idx]
    loader = DataLoader(TensorDataset(xs, ys), batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    diff = Diffusion(T=T, schedule=schedule, device=device)
    model = SmallUNet(num_classes=10, base=base).to(device)
    ema = copy.deepcopy(model)
    start_epoch_total = 0
    if init_ckpt is not None:
        ck = torch.load(init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        start_epoch_total = ck.get("epochs_total", 0)
        print(f"[{tag}] continuing from {init_ckpt} (epochs_total={start_epoch_total})", flush=True)
    for p in ema.parameters():
        p.requires_grad_(False)
    params = count_params(model)
    t_val, noise_val = make_val_protocol(diff, x_val)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    aug_gen = torch.Generator().manual_seed(seed + 5)
    history, val_hist = [], []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if augment:
                x = augment_shift(x, aug_gen)
            drop = torch.rand(y.shape[0], device=device) < 0.1
            y_in = torch.where(drop, torch.full_like(y, NULL_LABEL), y)
            t = torch.randint(0, T, (x.shape[0],), device=device)
            x_t, noise = diff.q_sample(x, t)
            pred = model(x_t, t, y_in)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(ema_decay).add_(pm.detach(), alpha=1 - ema_decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            total += loss.item()
        avg = total / len(loader)
        vmse = val_noise_mse(ema, diff, x_val, t_val, noise_val, y_val)
        history.append(round(avg, 5)); val_hist.append(vmse)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "T": T,
                    "epochs_this_run": epoch + 1,
                    "epochs_total": start_epoch_total + epoch + 1,
                    "history": history, "val_history": val_hist,
                    "config": {"tag": tag, "data_n": int(len(xs)), "ema_decay": ema_decay,
                               "schedule": schedule, "augment": augment, "lr": lr,
                               "batch": batch, "base": base, "seed": seed,
                               "init_ckpt": init_ckpt}},
                   f"checkpoint_real_{tag}.pt")
        print(f"[{tag}] epoch {epoch + 1}/{epochs} train_loss {avg:.4f} "
              f"val_noise_mse_ema {vmse:.5f} ({time.time() - t0:.0f}s)", flush=True)
    secs = round(time.time() - t0, 1)
    hist = {"tag": tag, "params": params, "T": T, "epochs_this_run": epochs,
            "epochs_total": start_epoch_total + epochs, "data_n": int(len(xs)),
            "ema_decay": ema_decay, "schedule": schedule, "augment": augment,
            "lr": lr, "batch": batch, "base": base, "init_ckpt": init_ckpt,
            "loss_history": history, "val_noise_mse_ema_history": val_hist,
            "final_train_loss": history[-1], "final_val_noise_mse_ema": val_hist[-1],
            "best_val_noise_mse_ema": min(val_hist), "train_seconds": secs,
            "checkpoint": f"checkpoint_real_{tag}.pt"}
    with open(f"train_history_real_{tag}.json", "w") as f:
        json.dump(hist, f, indent=2)
    print(f"[{tag}] done in {secs:.0f}s final loss {history[-1]} "
          f"val MSE {val_hist[-1]}", flush=True)
    return hist
