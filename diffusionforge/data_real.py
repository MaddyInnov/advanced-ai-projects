"""Real-data loader for DiffusionForge: MNIST downscaled to 16x16.

Dataset: MNIST (Yann LeCun, Corinna Cortes, Christopher J.C. Burges) —
60,000 train / 10,000 test handwritten-digit images, 28x28 grayscale.
Source file: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
(saved locally as mnist.npz; SHA-256 recorded in results_real.json).

The ONLY change vs the original synthetic pipeline is the data: each 28x28
image is downscaled to 16x16 (bilinear interpolation with antialiasing) so
the existing 16x16 conditional U-Net applies unchanged, then scaled from
[0, 1] to [-1, 1] to match the synthetic data range.

Deterministic train-side split (seed 2024 permutation of the 60,000 train
indices):
  * subset_idx : first 20,000  -> baseline diffusion training subset
  * val_idx    : next 2,000    -> held-out validation (noise-MSE + the
                                  labels used for the val fidelity proxy);
                                  NEVER used to train any diffusion model,
                                  including the expanded-data Phase B run
  * extra_idx  : next 20,000   -> Phase B data-expansion pool (train-side,
                                  disjoint from val)
"""
import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
NPZ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist.npz")
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_real16.pt")
NUM_CLASSES = 10
NULL_LABEL = 10  # class_emb index of the unconditional embedding for 10 classes
SPLIT_SEED = 2024


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _downscale(x_uint8: np.ndarray) -> torch.Tensor:
    """(N,28,28) uint8 -> (N,1,16,16) float in [-1,1], bilinear + antialias."""
    x = torch.from_numpy(x_uint8.astype(np.float32) / 255.0).unsqueeze(1)
    out = []
    for i in range(0, len(x), 5000):
        out.append(F.interpolate(x[i:i + 5000], size=(16, 16), mode="bilinear",
                                 align_corners=False, antialias=True))
    return torch.cat(out) * 2.0 - 1.0


def load_mnist16():
    """Returns (x_train, y_train, x_test, y_test, meta) with 16x16 tensors."""
    if os.path.exists(CACHE_PATH):
        d = torch.load(CACHE_PATH, weights_only=False)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"], d["meta"]
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(
            f"{NPZ_PATH} missing - download it from {URL} first.")
    raw = np.load(NPZ_PATH)
    x_tr, y_tr = raw["x_train"], raw["y_train"]
    x_te, y_te = raw["x_test"], raw["y_test"]
    assert x_tr.shape == (60000, 28, 28) and x_te.shape == (10000, 28, 28), \
        f"unexpected MNIST shapes: {x_tr.shape} {x_te.shape}"
    assert y_tr.min() >= 0 and y_tr.max() <= 9 and y_te.min() >= 0 and y_te.max() <= 9
    meta = {
        "name": "MNIST",
        "provenance": "MNIST handwritten digits - Yann LeCun, Corinna Cortes, "
                      "Christopher J.C. Burges (via the TensorFlow/Keras datasets mirror)",
        "source_url": URL,
        "source_file": "mnist.npz",
        "source_sha256": file_sha256(NPZ_PATH),
        "train_images_original": int(len(x_tr)),
        "test_images_original": int(len(x_te)),
        "train_label_counts": np.bincount(y_tr, minlength=10).tolist(),
        "test_label_counts": np.bincount(y_te, minlength=10).tolist(),
        "original_size": "28x28 uint8 grayscale, pixel range 0-255",
        "downscale": "28x28 -> 16x16, bilinear interpolation with antialiasing, "
                     "then scaled [0,1] -> [-1,1] to match the synthetic pipeline",
        "classes": [str(d) for d in range(10)],
    }
    x_train, x_test = _downscale(x_tr), _downscale(x_te)
    y_train = torch.from_numpy(y_tr.astype(np.int64))
    y_test = torch.from_numpy(y_te.astype(np.int64))
    torch.save({"x_train": x_train, "y_train": y_train, "x_test": x_test,
                "y_test": y_test, "meta": meta}, CACHE_PATH)
    return x_train, y_train, x_test, y_test, meta


def split_indices():
    """Deterministic train-side split: (subset 20k, val 2k, extra 20k)."""
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(60000)
    return perm[:20000], perm[20000:22000], perm[22000:42000]


if __name__ == "__main__":
    xtr, ytr, xte, yte, meta = load_mnist16()
    sub, val, extra = split_indices()
    print(f"train {xtr.shape} range [{xtr.min():.2f},{xtr.max():.2f}]  test {xte.shape}")
    print(f"split: subset={len(sub)} val={len(val)} extra={len(extra)} "
          f"val/subset overlap={len(set(val.tolist()) & set(sub.tolist()))} "
          f"val/extra overlap={len(set(val.tolist()) & set(extra.tolist()))}")
    print(meta)
