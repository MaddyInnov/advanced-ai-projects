"""Train the conditional DDPM noise predictor on the synthetic shape data.

- MSE loss on predicted noise (the DDPM simplified objective)
- classifier-free-guidance training: class label dropped to NULL with p=0.1
- EMA of weights (decay 0.99, matched to the ~600-step run length) saved for
  sampling — standard practice; note a 0.999 decay would need ~10k steps to
  forget the random init, which is why the decay must match run length.
"""
import copy
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_gen import generate_dataset
from diffusion import Diffusion
from model import NULL_LABEL, SmallUNet, count_params

T = 300
EPOCHS = 90
BATCH = 256
LR = 2e-3
CFG_DROP = 0.1


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = "cpu"
    (xtr, ytr), (xte, yte) = generate_dataset()
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=BATCH, shuffle=True)
    diff = Diffusion(T=T, schedule="cosine", device=device)
    model = SmallUNet().to(device)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"params: {count_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(loader))
    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # classifier-free guidance: drop label with p=0.1
            drop = torch.rand(y.shape[0], device=device) < CFG_DROP
            y_in = torch.where(drop, torch.full_like(y, NULL_LABEL), y)
            t = torch.randint(0, T, (x.shape[0],), device=device)
            x_t, noise = diff.q_sample(x, t)
            pred = model(x_t, t, y_in)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(0.99).add_(pm.detach(), alpha=0.01)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            total += loss.item()
        avg = total / len(loader)
        history.append(round(avg, 5))
        if (epoch + 1) % 5 == 0:  # periodic checkpoint: insures against interruption
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "T": T,
                        "epochs": epoch + 1, "history": history}, "checkpoint.pt")
        if epoch % 2 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:02d}  loss {avg:.4f}  ({time.time()-t0:.0f}s)")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "T": T,
                "epochs": EPOCHS, "history": history}, "checkpoint.pt")
    with open("train_history.json", "w") as f:
        json.dump({"loss_history": history, "train_seconds": round(time.time() - t0, 1),
                   "params": count_params(model), "T": T, "epochs": EPOCHS}, f, indent=2)
    print(f"done in {time.time()-t0:.0f}s, final loss {history[-1]}")


if __name__ == "__main__":
    main()
