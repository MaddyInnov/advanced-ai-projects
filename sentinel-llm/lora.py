"""
LoRA (Low-Rank Adaptation) implemented FROM SCRATCH in PyTorch.
Author: Ashutosh Sahoo

Math:
  For a frozen pretrained weight W0 ∈ R^{d_out x d_in},
  LoRA learns a low-rank update ΔW = B A, where
    A ∈ R^{r x d_in}  (init ~ N(0, sigma))
    B ∈ R^{d_out x r} (init 0, so ΔW = 0 at start)
  Forward:  h = W0 x + (alpha / r) * B A x
  Only A and B are trained -> r*(d_in + d_out) params instead of d_in*d_out.
  e.g. 256x128 layer, r=8: 3,072 LoRA params vs 32,768 full = 9.4%

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language
Models", ICLR 2022 (arXiv:2106.09685).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with trainable low-rank A/B matrices."""

    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Freeze base weight + bias
        for p in self.base.parameters():
            p.requires_grad = False

        # A: (r, in) — random init; B: (out, r) — zero init so delta starts at 0
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # x: (..., in) -> @ A.T: (..., r) -> @ B.T: (..., out)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def merged_weight(self) -> torch.Tensor:
        """Weight with LoRA merged in — for inference export / inspection."""
        with torch.no_grad():
            return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


def inject_lora(model: nn.Module, target_suffixes=("fc1", "fc2"), r: int = 8, alpha: int = 16, dropout: float = 0.05):
    """
    Replace nn.Linear modules whose name ends with any target suffix
    with LoRALinear wrappers. Returns list of replaced names.
    Works recursively; replaces in-place on parent modules.
    """
    replaced = []

    def _walk(parent, prefix=""):
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(full.endswith(s) or name == s for s in target_suffixes):
                setattr(parent, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced.append(full)
            else:
                _walk(child, full)

    _walk(model)
    return replaced


def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_lora_only(model: nn.Module):
    """Freeze everything, then unfreeze only LoRA A/B parameters."""
    freeze_all(model)
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            count += module.lora_A.numel() + module.lora_B.numel()
    return count


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_pct": round(100 * trainable / total, 2) if total else 0.0}


if __name__ == "__main__":
    # Quick self-test
    lin = nn.Linear(128, 256)
    lora = LoRALinear(lin, r=8, alpha=16)
    x = torch.randn(4, 128)
    y = lora(x)
    assert y.shape == (4, 256), y.shape
    stats = count_parameters(lora)
    print(f"LoRALinear self-test OK: output {tuple(y.shape)}, params={stats}")
    # At init, LoRA contribution is zero (B=0) so output == base output
    assert torch.allclose(y, lin(x), atol=1e-6), "B=0 init should make LoRA a no-op at start"
    print("Init check OK: LoRA is a no-op at initialisation (B = 0).")
