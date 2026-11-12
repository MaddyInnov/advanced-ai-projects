"""Evaluation for the real-data (MNIST-16) run - same method as evaluate.py.

Metrics:
  * noise-prediction MSE on held-out real images (original evaluate.py used
    the RAW model weights with random t; we report raw and EMA)
  * class fidelity: % of generated samples the fidelity classifier assigns
    to the conditioned class (classifier trained only on real train data)
  * diversity: mean pairwise L2 (same pairwise_diversity as the original)
  * sampling wall time

Two modes:
  mode="val"  - validation proxy ONLY (train-side): val noise-MSE (EMA) +
                DDIM-50 fidelity proxy on generated samples. Used to screen
                Phase B attempts. Never touches the test split.
  mode="test" - the full test-side protocol. Run ONCE for the baseline
                config and ONCE for the final selected config.

Sample count: the original used 64 samples (16/class x 4 classes). With 10
digit classes a balanced set needs a multiple of 10, so the test protocol
uses 70 samples (7/class); the change is documented in results_real.json.
The val proxy uses 100 samples (10/class).
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from classifier_real import load_classifier
from data_real import NULL_LABEL, load_mnist16, split_indices
from diffusion import Diffusion
from model import SmallUNet
from train_real import make_val_protocol, val_noise_mse

TEST_N_PER_CLASS = 7    # 70 samples total (original: 64 = 16/class x 4 classes)
VAL_N_PER_CLASS = 10    # 100 samples for the val fidelity proxy


def fidelity(clf, imgs, labels, device="cpu"):
    with torch.no_grad():
        pred = clf(imgs.to(device)).argmax(1).cpu()
    return round((pred == labels.cpu()).float().mean().item() * 100, 2)


def pairwise_diversity(imgs, max_n=96):
    x = imgs[:max_n].flatten(1).float()
    d = torch.cdist(x, x)
    mask = ~torch.eye(len(x), dtype=torch.bool)
    return round(d[mask].mean().item(), 3)


def save_grid(imgs, path, cols=10):
    """imgs (N,1,16,16) in [-1,1] -> PNG grid scaled x4 (nearest)."""
    arr = ((imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()[:, 0]
    n = arr.shape[0]
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * 18, cols * 18), dtype=np.uint8)
    for i, im in enumerate(arr):
        r, c = divmod(i, cols)
        grid[r * 18: r * 18 + 16, c * 18: c * 18 + 16] = im
    Image.fromarray(grid).resize((cols * 18 * 4, rows * 18 * 4), Image.NEAREST).save(path)
    print(f"saved {path} ({n} samples)", flush=True)


def load_ema_model(ckpt_path, device="cpu", base=16):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SmallUNet(num_classes=10, base=base).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    return model, ckpt


def class_major_labels(n_per_class):
    return torch.tensor([c for c in range(10) for _ in range(n_per_class)])


def grid_order(n_per_class):
    """Permutation turning class-major samples into a grid whose columns
    are digit classes (col = class, row = sample index)."""
    return [c * n_per_class + r for r in range(n_per_class) for c in range(10)]


def eval_val_proxy(ckpt_path, guidance_list=(1.5,), base=16, device="cpu"):
    """Validation-only screening: EMA val noise-MSE + DDIM-50 fidelity proxy."""
    _, _, _, _, _ = load_mnist16()
    x_train_all, y_train_all, _, _, _ = load_mnist16()
    _, val_idx, _ = split_indices()
    x_val, y_val = x_train_all[val_idx], y_train_all[val_idx]
    clf = load_classifier(device)
    model, ckpt = load_ema_model(ckpt_path, device, base)
    diff = Diffusion(T=ckpt["T"], schedule=ckpt["config"].get("schedule", "cosine"),
                     device=device)
    t_val, noise_val = make_val_protocol(diff, x_val)
    out = {"val_noise_mse_ema": val_noise_mse(model, diff, x_val, t_val, noise_val, y_val),
           "ddim50_val_proxy": {}}
    y = class_major_labels(VAL_N_PER_CLASS)
    for w in guidance_list:
        torch.manual_seed(99)
        g = diff.sample_ddim(model, y.clone(), steps=50, guidance=w,
                             null_label=NULL_LABEL)
        out["ddim50_val_proxy"][str(w)] = {
            "fidelity_pct": fidelity(clf, g, y, device),
            "diversity_pairwise_l2": pairwise_diversity(g)}
    return out


def eval_test(ckpt_path, main_guidance=1.5, cfg_ablation=None, base=16,
              device="cpu", grid_prefix=None):
    """Full test-side protocol (original evaluate.py method, 10 classes)."""
    torch.manual_seed(123)
    x_train_all, y_train_all, x_test, y_test, _ = load_mnist16()
    clf = load_classifier(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    schedule = ckpt["config"].get("schedule", "cosine")
    diff = Diffusion(T=ckpt["T"], schedule=schedule, device=device)
    model, _ = load_ema_model(ckpt_path, device, base)

    # held-out noise-prediction MSE on real test images: raw weights (the
    # original definition) and EMA weights, random t, seed protocol as original
    raw = SmallUNet(num_classes=10, base=base).to(device)
    raw.load_state_dict(ckpt["model"]); raw.eval()
    mses_raw, mses_ema = [], []
    with torch.no_grad():
        for i in range(0, len(x_test), 100):
            x = x_test[i:i + 100]
            t = torch.randint(0, ckpt["T"], (x.shape[0],))
            x_t, noise = diff.q_sample(x, t)
            mses_raw.append(F.mse_loss(raw(x_t, t, y_test[i:i + 100]), noise).item())
            mses_ema.append(F.mse_loss(model(x_t, t, y_test[i:i + 100]), noise).item())
    results = {
        "timesteps_T": ckpt["T"],
        "noise_pred_mse_heldout_test_raw": round(sum(mses_raw) / len(mses_raw), 4),
        "noise_pred_mse_heldout_test_ema": round(sum(mses_ema) / len(mses_ema), 4),
        "n_samples_per_sampler": TEST_N_PER_CLASS * 10,
        "main_guidance_w": main_guidance,
    }
    y = class_major_labels(TEST_N_PER_CLASS)
    torch.manual_seed(99)
    t0 = time.time()
    g_ddpm = diff.sample_ddpm(model, y.clone(), guidance=main_guidance,
                              null_label=NULL_LABEL)
    sec_ddpm = time.time() - t0
    torch.manual_seed(99)
    t0 = time.time()
    g_ddim = diff.sample_ddim(model, y.clone(), steps=50, guidance=main_guidance,
                              null_label=NULL_LABEL)
    sec_ddim = time.time() - t0
    results["ddpm"] = {"steps": ckpt["T"],
                       "fidelity_pct": fidelity(clf, g_ddpm, y, device),
                       "seconds_for_samples": round(sec_ddpm, 1),
                       "diversity_pairwise_l2": pairwise_diversity(g_ddpm)}
    results["ddim50"] = {"steps": 50,
                         "fidelity_pct": fidelity(clf, g_ddim, y, device),
                         "seconds_for_samples": round(sec_ddim, 1),
                         "diversity_pairwise_l2": pairwise_diversity(g_ddim)}
    results["real_data_diversity_pairwise_l2"] = pairwise_diversity(x_test)
    if cfg_ablation:
        results["guidance_ablation_fidelity_pct"] = {}
        results["guidance_ablation_diversity"] = {}
        for w in cfg_ablation:
            torch.manual_seed(99)
            g = diff.sample_ddim(model, y.clone(), steps=50, guidance=w,
                                 null_label=NULL_LABEL)
            results["guidance_ablation_fidelity_pct"][str(w)] = fidelity(clf, g, y, device)
            results["guidance_ablation_diversity"][str(w)] = pairwise_diversity(g)
    if grid_prefix:
        order = grid_order(TEST_N_PER_CLASS)
        save_grid(g_ddpm[order], f"{grid_prefix}_ddpm.png", cols=10)
        save_grid(g_ddim[order], f"{grid_prefix}_ddim50.png", cols=10)
        torch.save({"ddpm": g_ddpm, "ddim": g_ddim, "labels": y},
                   f"{grid_prefix}_samples.pt")
    return results
