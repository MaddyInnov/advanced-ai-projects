"""
FedGuard real-data loader: MNIST (LeCun et al.), the real benchmark that
replaces the synthetic Gaussian-mixture data of the original run.

Provenance
----------
- Source file: mnist.npz downloaded from
  https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
  (the TensorFlow/Keras mirror of MNIST), stored in this folder.
- sha256(mnist.npz) = 731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1
- Verified counts on load: x_train (60000, 28, 28) uint8, y_train (60000,),
  x_test (10000, 28, 28) uint8, y_test (10000,). Labels 0-9.

Adaptation (stated everywhere it matters)
-----------------------------------------
- Images are downscaled 28x28 -> 16x16 with bilinear interpolation
  (torch.nn.functional.interpolate, align_corners=False) on [0,1]-scaled
  pixels, then flattened: input_dim = 256 for the project's shared MLP
  (256 -> 64 -> 64 -> 10, dropout 0.3; 21,258 params vs 6,154 at 20-dim).
- A stratified validation split of 5,000 images (500/class, seed 777) is
  carved out of the 60,000 training images. The remaining 55,000 form the
  federated training pool. The full 10,000-image MNIST test set is used
  for test evaluation, identically for every configuration.
- Partitions reuse data_gen.iid_partition / data_gen.dirichlet_partition
  unchanged (IID shuffle split; Dirichlet alpha=0.5 label skew, 10 clients,
  same seeds as the synthetic run: partition seed offsets +1 / +2 on 42).
"""
import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MNIST_NPZ = os.path.join(HERE, "mnist.npz")
CACHE_NPZ = os.path.join(HERE, "mnist16.npz")
SOURCE_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
VAL_PER_CLASS = 500
VAL_SEED = 777
INPUT_DIM = 256


def _downscale(x_uint8):
    """(N,28,28) uint8 -> (N,256) float32 in [0,1], bilinear 16x16."""
    t = torch.tensor(x_uint8, dtype=torch.float32).unsqueeze(1) / 255.0
    out = []
    for i in range(0, len(t), 10000):
        small = F.interpolate(t[i:i + 10000], size=(16, 16),
                              mode="bilinear", align_corners=False)
        out.append(small.reshape(small.shape[0], -1))
    return torch.cat(out).numpy().astype(np.float32)


def load_mnist16():
    """Returns (X_pool, y_pool, X_val, y_val, X_test, y_test, meta).

    X_pool is the 55,000-image federated training pool (60k train minus
    the 5k stratified validation carve-out). All X are float32 (N, 256).
    """
    if os.path.exists(CACHE_NPZ):
        z = np.load(CACHE_NPZ)
        return (z["X_pool"], z["y_pool"], z["X_val"], z["y_val"],
                z["X_test"], z["y_test"], {"cached": True})

    raw = np.load(MNIST_NPZ)
    x_train, y_train = raw["x_train"], raw["y_train"]
    x_test, y_test = raw["x_test"], raw["y_test"]
    assert x_train.shape == (60000, 28, 28), x_train.shape
    assert y_train.shape == (60000,), y_train.shape
    assert x_test.shape == (10000, 28, 28), x_test.shape
    assert y_test.shape == (10000,), y_test.shape

    X_all = _downscale(x_train)
    X_test = _downscale(x_test)
    y_all = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    # Stratified validation carve-out from the training images only.
    rng = np.random.default_rng(VAL_SEED)
    val_idx, pool_idx = [], []
    for c in range(10):
        idx = np.where(y_all == c)[0]
        rng.shuffle(idx)
        val_idx.extend(idx[:VAL_PER_CLASS].tolist())
        pool_idx.extend(idx[VAL_PER_CLASS:].tolist())
    val_idx = np.array(sorted(val_idx))
    pool_idx = np.array(sorted(pool_idx))
    rng.shuffle(pool_idx)

    X_pool, y_pool = X_all[pool_idx], y_all[pool_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    np.savez_compressed(CACHE_NPZ, X_pool=X_pool, y_pool=y_pool,
                        X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
    sha = hashlib.sha256(open(MNIST_NPZ, "rb").read()).hexdigest()
    meta = {"cached": False, "sha256_mnist_npz": sha,
            "n_train_file": 60000, "n_test_file": 10000}
    return X_pool, y_pool, X_val, y_val, X_test, y_test, meta


if __name__ == "__main__":
    Xp, yp, Xv, yv, Xt, yt, meta = load_mnist16()
    print("pool", Xp.shape, "val", Xv.shape, "test", Xt.shape)
    print("pool class counts", np.bincount(yp).tolist())
    print("val class counts", np.bincount(yv).tolist())
    print("test class counts", np.bincount(yt).tolist())
    print(meta)
