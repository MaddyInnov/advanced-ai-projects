"""Synthetic 16x16 shape dataset generated entirely in code (no downloads).

Classes: 0=circle, 1=square, 2=triangle, 3=cross/plus.
Images are single-channel float tensors in [-1, 1].
Each shape is drawn on an analytic coordinate grid with random
position / scale / intensity jitter, then light Gaussian noise is added.
"""
import numpy as np
import torch

CLASS_NAMES = ["circle", "square", "triangle", "cross"]


def _draw_batch(rng: np.random.Generator, n: int):
    yy, xx = np.mgrid[0:16, 0:16].astype(np.float32)  # (16,16)
    images = np.full((n, 1, 16, 16), -1.0, dtype=np.float32)
    labels = rng.integers(0, 4, size=n)
    for i in range(n):
        cx = 7.5 + rng.uniform(-2.0, 2.0)
        cy = 7.5 + rng.uniform(-2.0, 2.0)
        s = rng.uniform(3.6, 5.6)          # characteristic size
        amp = rng.uniform(1.6, 2.0)        # foreground intensity (-> ~1 after shift)
        cls = labels[i]
        dx, dy = xx - cx, yy - cy
        if cls == 0:   # filled circle
            mask = (dx ** 2 + dy ** 2) <= s ** 2
        elif cls == 1:  # filled square
            mask = (np.abs(dx) <= s * 0.85) & (np.abs(dy) <= s * 0.85)
        elif cls == 2:  # triangle: apex up, base at bottom
            # inside if below apex line on both sides and above base
            half_base = s
            base_y = cy + s * 0.8
            # width shrinks linearly from base to apex
            frac = (base_y - yy) / (2 * s * 0.8 + 1e-6)  # 1 at apex, 0 at base
            width = half_base * np.clip(1.0 - frac, 0.0, 1.0)
            mask = (yy <= base_y) & (yy >= cy - s * 0.8) & (np.abs(dx) <= width)
        else:           # cross / plus
            w = s * 0.38
            mask = ((np.abs(dx) <= w) & (np.abs(dy) <= s)) | ((np.abs(dy) <= w) & (np.abs(dx) <= s))
        img = np.where(mask, amp - 1.0, -1.0).astype(np.float32)
        # soft edge: slight blur via neighbour averaging once
        img = (img + np.roll(img, 1, 0) + np.roll(img, -1, 0) + np.roll(img, 1, 1) + np.roll(img, -1, 1)) / 5.0 \
            if False else img
        img = img + rng.normal(0, 0.06, size=img.shape).astype(np.float32)
        images[i, 0] = np.clip(img, -1.0, 1.0)
    return torch.from_numpy(images), torch.from_numpy(labels.astype(np.int64))


def generate_dataset(n_train: int = 1600, n_test: int = 400, seed: int = 42):
    rng = np.random.default_rng(seed)
    x_train, y_train = _draw_batch(rng, n_train)
    rng_test = np.random.default_rng(seed + 1000)
    x_test, y_test = _draw_batch(rng_test, n_test)
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (xtr, ytr), (xte, yte) = generate_dataset()
    torch.save({"x_train": xtr, "y_train": ytr, "x_test": xte, "y_test": yte}, "dataset.pt")
    for c, name in enumerate(CLASS_NAMES):
        print(f"class {c} {name}: train={int((ytr == c).sum())} test={int((yte == c).sum())}")
    print(f"saved dataset.pt  train={xtr.shape} test={xte.shape} range=[{xtr.min():.2f},{xtr.max():.2f}]")
