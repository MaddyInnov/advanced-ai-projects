"""Evaluation with REAL measured metrics (saved to results.json):

1. Noise-prediction MSE on held-out test images (random t).
2. Class fidelity: a tiny CNN classifier trained ONLY on real synthetic data;
   % of generated samples it assigns to their intended (conditioning) class.
3. Diversity: mean pairwise L2 distance, generated vs real.
4. DDIM-50 vs DDPM-full: fidelity + wall-clock speed.
5. Guidance ablation: fidelity at w = 0, 1.5, 3.0.
"""
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_gen import generate_dataset
from diffusion import Diffusion
from model import SmallUNet
from sample import load_model


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        return self.net(x)


def get_classifier(xtr, ytr, xte, yte, device="cpu"):
    try:
        clf = TinyClassifier().to(device)
        clf.load_state_dict(torch.load("classifier.pt", map_location=device, weights_only=True))
        clf.eval()
        return clf, None
    except Exception:
        pass
    clf = TinyClassifier().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=2e-3)
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=64, shuffle=True)
    for _ in range(15):
        clf.train()
        for x, y in loader:
            loss = F.cross_entropy(clf(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(clf.state_dict(), "classifier.pt")
    clf.eval()
    with torch.no_grad():
        acc = (clf(xte).argmax(1) == yte).float().mean().item()
    return clf, round(acc * 100, 2)


@torch.no_grad()
def fidelity(clf, imgs, labels, device="cpu"):
    pred = clf(imgs.to(device)).argmax(1).cpu()
    return round((pred == labels.cpu()).float().mean().item() * 100, 2)


def pairwise_diversity(imgs, max_n=96):
    x = imgs[:max_n].flatten(1).float()
    d = torch.cdist(x, x)
    mask = ~torch.eye(len(x), dtype=torch.bool)
    return round(d[mask].mean().item(), 3)


def main():
    torch.manual_seed(123)
    torch.set_num_threads(2)
    device = "cpu"
    (xtr, ytr), (xte, yte) = generate_dataset()
    clf, clf_acc = get_classifier(xtr, ytr, xte, yte, device)
    if clf_acc is None:  # loaded from cache: measure real-test accuracy now
        with torch.no_grad():
            clf_acc = round((clf(xte).argmax(1) == yte).float().mean().item() * 100, 2)
    print(f"classifier real-test accuracy: {clf_acc}%")

    model, T = load_model(device)
    diff = Diffusion(T=T, schedule="cosine", device=device)

    # 1) held-out noise-prediction MSE (raw model weights, random t)
    raw = SmallUNet().to(device)
    raw.load_state_dict(torch.load("checkpoint.pt", map_location=device, weights_only=False)["model"])
    raw.eval()
    mses = []
    with torch.no_grad():
        for i in range(0, len(xte), 100):
            x = xte[i:i + 100]
            t = torch.randint(0, T, (x.shape[0],))
            x_t, noise = diff.q_sample(x, t)
            mses.append(F.mse_loss(raw(x_t, t, yte[i:i + 100]), noise).item())
    mse = round(sum(mses) / len(mses), 4)

    # sampling sets: 16 per class = 64 samples per condition
    y = torch.tensor([c for c in range(4) for _ in range(16)])
    torch.manual_seed(99)
    t0 = time.time(); g_ddpm = diff.sample_ddpm(model, y.clone(), guidance=1.5); sec_ddpm = time.time() - t0
    torch.manual_seed(99)
    t0 = time.time(); g_ddim = diff.sample_ddim(model, y.clone(), steps=50, guidance=1.5); sec_ddim = time.time() - t0

    results = {
        "timesteps_T": T,
        "noise_pred_mse_heldout": mse,
        "classifier_real_test_accuracy_pct": clf_acc,
        "ddpm": {"steps": T, "fidelity_pct": fidelity(clf, g_ddpm, y),
                 "seconds_for_64_samples": round(sec_ddpm, 1),
                 "diversity_pairwise_l2": pairwise_diversity(g_ddpm)},
        "ddim50": {"steps": 50, "fidelity_pct": fidelity(clf, g_ddim, y),
                   "seconds_for_64_samples": round(sec_ddim, 1),
                   "diversity_pairwise_l2": pairwise_diversity(g_ddim)},
        "real_data_diversity_pairwise_l2": pairwise_diversity(xte),
        "guidance_ablation_fidelity_pct": {},
    }
    for w in (0.0, 1.5, 3.0):
        torch.manual_seed(99)
        g = diff.sample_ddim(model, y.clone(), steps=50, guidance=w)
        results["guidance_ablation_fidelity_pct"][str(w)] = fidelity(clf, g, y)
    try:
        hist = json.load(open("train_history.json"))
        results.update({"model_params": hist["params"], "train_epochs": hist["epochs"],
                        "final_train_loss": hist["loss_history"][-1],
                        "train_seconds": hist["train_seconds"]})
    except Exception:
        pass
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
