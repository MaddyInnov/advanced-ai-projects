"""Image-enhancement module (OpenCV): CLAHE + denoising + unsharp masking.

Designed for low-light / hazy aerial and surveillance frames. Evaluated
honestly in run_benchmarks.py: clean COCO images are degraded with a
known synthetic low-light transform (gamma + Gaussian noise), enhanced,
and scored with PSNR / SSIM against the clean originals. The degradation
is synthetic and is labelled as such in every report — the enhancement
*method* and its measured recovery are real.
"""
from __future__ import annotations

import cv2
import numpy as np


def enhance_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel (LAB), fast-NLM denoising, unsharp mask."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    out = cv2.fastNlMeansDenoisingColored(out, None, 6, 6, 7, 21)
    blur = cv2.GaussianBlur(out, (0, 0), 2.0)
    out = cv2.addWeighted(out, 1.4, blur, -0.4, 0)
    return out


def enhance_frame_fast(frame_bgr: np.ndarray) -> np.ndarray:
    """CLAHE + unsharp only (no NLM denoise) — the throughput variant used
    for the ExDark detection A/B study, where 735 images are enhanced.
    Labelled as the fast variant wherever its results are reported."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 2.0)
    return cv2.addWeighted(out, 1.4, blur, -0.4, 0)


def degrade_lowlight(img_bgr: np.ndarray, gamma: float = 2.2, noise_sigma: float = 12.0,
                      seed: int = 0) -> np.ndarray:
    """Synthetic low-light degradation: gamma darkening + Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = img_bgr.astype(np.float32) / 255.0
    x = np.power(x, gamma)
    noise = rng.normal(0.0, noise_sigma / 255.0, x.shape).astype(np.float32)
    x = np.clip(x + noise, 0.0, 1.0)
    return (x * 255).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM on grayscale (standard constants)."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mua, mub = ga.mean(), gb.mean()
    va, vb = ga.var(), gb.var()
    cov = ((ga - mua) * (gb - mub)).mean()
    return float(((2 * mua * mub + c1) * (2 * cov + c2)) /
                 ((mua ** 2 + mub ** 2 + c1) * (va + vb + c2)))
