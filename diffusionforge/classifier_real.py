"""Fidelity classifier for the real-data run.

Same TinyClassifier architecture as the original evaluate.py, with the final
layer widened 4 -> 10 classes. Trained ONLY on real MNIST-16 TRAIN data.
Its accuracy on the held-out real MNIST-16 TEST split is the quality gate
for every fidelity number in results_real.json (brief target: >= 98%).
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

CKPT = "classifier_real.pt"


class TinyClassifier10(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(),
            nn.Linear(64, num_classes))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def classifier_accuracy(clf, x, y, device="cpu", batch=512):
    clf.eval()
    correct = 0
    for i in range(0, len(x), batch):
        pred = clf(x[i:i + batch].to(device)).argmax(1).cpu()
        correct += int((pred == y[i:i + batch]).sum())
    return round(correct / len(x) * 100, 2)


def train_classifier(xtr, ytr, xte, yte, epochs=6, batch=256, lr=2e-3,
                     device="cpu", init_ckpt=None, seed=42):
    """Train (or continue training) the fidelity classifier; returns
    (state_dict, test_accuracy_pct, seconds, epochs_done_total)."""
    torch.manual_seed(seed)
    clf = TinyClassifier10().to(device)
    epochs_done = 0
    if init_ckpt is not None:
        sd = torch.load(init_ckpt, map_location=device, weights_only=False)
        clf.load_state_dict(sd["state_dict"])
        epochs_done = sd.get("epochs", 0)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch, shuffle=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    t0 = time.time()
    for _ in range(epochs):
        clf.train()
        for x, y in loader:
            loss = F.cross_entropy(clf(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    secs = time.time() - t0
    acc = classifier_accuracy(clf, xte, yte, device)
    torch.save({"state_dict": clf.state_dict(), "epochs": epochs_done + epochs,
                "test_accuracy_pct": acc}, CKPT)
    return clf.state_dict(), acc, round(secs, 1), epochs_done + epochs


def load_classifier(device="cpu"):
    clf = TinyClassifier10().to(device)
    sd = torch.load(CKPT, map_location=device, weights_only=False)
    clf.load_state_dict(sd["state_dict"])
    clf.eval()
    return clf
