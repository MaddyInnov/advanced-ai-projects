"""Generate sample grids: DDPM (full T steps) vs DDIM (50 steps) vs CFG sweep."""
import time

import numpy as np
import torch
from PIL import Image

from diffusion import Diffusion
from model import SmallUNet


def load_model(device="cpu"):
    ckpt = torch.load("checkpoint.pt", map_location=device, weights_only=False)
    model = SmallUNet().to(device)
    model.load_state_dict(ckpt["ema"])  # EMA weights sample noticeably better
    model.eval()
    return model, ckpt["T"]


def save_grid(imgs: torch.Tensor, path: str, cols: int = 8):
    """imgs: (N,1,16,16) in [-1,1] -> PNG grid scaled x8 for visibility."""
    arr = ((imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()[:, 0]
    n = arr.shape[0]
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * 18, cols * 18), dtype=np.uint8)
    for i, im in enumerate(arr):
        r, c = divmod(i, cols)
        # place 16x16 tile with 2px gap; whole grid is upscaled at the end
        grid[r * 18: r * 18 + 16, c * 18: c * 18 + 16] = im
    img = Image.fromarray(grid).resize((cols * 18 * 4, rows * 18 * 4), Image.NEAREST)
    img.save(path)
    print(f"saved {path}  ({n} samples)")


def balanced_labels(n_per_class=8, device="cpu"):
    return torch.tensor([c for c in range(4) for _ in range(n_per_class)], device=device)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model, T = load_model()
    diff = Diffusion(T=T, schedule="cosine")
    y = balanced_labels(8)

    t0 = time.time(); ddpm = diff.sample_ddpm(model, y, guidance=1.0); t_ddpm = time.time() - t0
    t0 = time.time(); ddim = diff.sample_ddim(model, y, steps=50, guidance=1.0); t_ddim = time.time() - t0
    save_grid(ddpm, "samples_ddpm.png")
    save_grid(ddim, "samples_ddim50.png")
    outs = {"ddpm": ddpm, "ddim": ddim, "labels": y}
    for w in (0.0, 1.5, 3.0):
        g = diff.sample_ddim(model, y, steps=50, guidance=w)
        save_grid(g, f"samples_cfg_w{w}.png")
        outs[f"cfg_{w}"] = g
    print(f"DDPM {T} steps: {t_ddpm:.1f}s | DDIM 50 steps: {t_ddim:.1f}s for {len(y)} samples")
    torch.save(outs, "samples.pt")


if __name__ == "__main__":
    main()
