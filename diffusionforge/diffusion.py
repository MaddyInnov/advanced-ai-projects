"""DDPM diffusion from scratch: schedules, forward process, DDPM & DDIM samplers.

Implements:
  * linear (Ho et al. 2020) and cosine (Nichol & Dhariwal 2021) beta schedules
  * forward process  q(x_t | x_0) = N(sqrt(ab_t) x_0, (1-ab_t) I)
  * reverse DDPM ancestral sampling
  * DDIM (Song et al. 2021) deterministic sampling (eta=0) on a timestep subsequence
  * classifier-free guidance (Ho & Salimans 2022):
        eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
    so w=0 is unconditional, w=1 is plain conditional, w>1 amplifies conditioning.
"""
import math
import torch


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.999)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int = 128):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = timesteps.float()[:, None] * freqs[None, :].to(timesteps.device)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Diffusion:
    def __init__(self, T: int = 300, schedule: str = "cosine", device: str = "cpu"):
        self.T = T
        self.device = device
        betas = cosine_beta_schedule(T) if schedule == "cosine" else linear_beta_schedule(T)
        self.betas = betas.to(device)
        self.alphas = (1.0 - betas).to(device)
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_ab = torch.sqrt(self.alphas_bar)
        self.sqrt_1m_ab = torch.sqrt(1.0 - self.alphas_bar)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)
        # posterior variance of q(x_{t-1} | x_t, x_0)
        ab_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_bar[:-1]])
        self.posterior_var = betas.to(device) * (1.0 - ab_prev) / (1.0 - self.alphas_bar)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape):
        out = a.gather(0, t)
        return out.reshape(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise=None):
        """Forward process: sample x_t given clean x_0."""
        if noise is None:
            noise = torch.randn_like(x0)
        s_ab = self._extract(self.sqrt_ab, t, x0.shape)
        s_1m = self._extract(self.sqrt_1m_ab, t, x0.shape)
        return s_ab * x0 + s_1m * noise, noise

    def predict_x0(self, x_t, t, eps):
        s_ab = self._extract(self.sqrt_ab, t, x_t.shape)
        s_1m = self._extract(self.sqrt_1m_ab, t, x_t.shape)
        return (x_t - s_1m * eps) / s_ab

    @torch.no_grad()
    def _guided_eps(self, model, x, t, y, guidance: float, null_label: int):
        if guidance == 1.0 or y is None:
            tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            return model(x, tt, y)
        tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        y_null = torch.full_like(y, null_label)
        eps_cond = model(x, tt, y)
        eps_uncond = model(x, tt, y_null)
        return eps_uncond + guidance * (eps_cond - eps_uncond)

    @torch.no_grad()
    def sample_ddpm(self, model, y, guidance: float = 1.0, null_label: int = 4, shape=(1, 16, 16)):
        model.eval()
        n = y.shape[0]
        x = torch.randn(n, *shape, device=self.device)
        for ti in reversed(range(self.T)):
            eps = self._guided_eps(model, x, ti, y, guidance, null_label)
            beta = self.betas[ti]
            alpha = self.alphas[ti]
            ab = self.alphas_bar[ti]
            ab_prev = self.alphas_bar[ti - 1] if ti > 0 else torch.tensor(1.0, device=self.device)
            # Predict x0 and clamp it to the data range: without this, early
            # (high-noise) prediction errors compound over 300 stochastic
            # steps and the chain drifts off-manifold (measured: fidelity
            # collapsed to ~chance; DDIM, which clamps x0 every step, did not).
            x0_pred = ((x - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab)).clamp(-1, 1)
            # Posterior mean of q(x_{t-1} | x_t, x0) — the stable form.
            coef_x0 = torch.sqrt(ab_prev) * beta / (1.0 - ab)
            coef_xt = torch.sqrt(alpha) * (1.0 - ab_prev) / (1.0 - ab)
            x = coef_x0 * x0_pred + coef_xt * x
            if ti > 0:
                x = x + torch.sqrt(self.posterior_var[ti]) * torch.randn_like(x)
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_ddim(self, model, y, steps: int = 50, guidance: float = 1.0, null_label: int = 4, shape=(1, 16, 16)):
        model.eval()
        n = y.shape[0]
        x = torch.randn(n, *shape, device=self.device)
        times = torch.linspace(0, self.T - 1, steps).long().tolist()
        times = list(reversed(times))  # T-1 ... 0
        for i, ti in enumerate(times):
            t = torch.full((n,), ti, device=self.device, dtype=torch.long)
            eps = self._guided_eps(model, x, ti, y, guidance, null_label)
            ab_t = self.alphas_bar[ti]
            x0_pred = (x - torch.sqrt(1.0 - ab_t) * eps) / torch.sqrt(ab_t)
            x0_pred = x0_pred.clamp(-1, 1)
            if i == len(times) - 1:
                x = x0_pred
            else:
                ab_next = self.alphas_bar[times[i + 1]]
                x = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1.0 - ab_next) * eps
        return x.clamp(-1, 1)
