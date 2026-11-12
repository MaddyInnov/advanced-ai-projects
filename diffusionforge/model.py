"""Small conditional U-Net for 16x16 single-channel images.

Design: residual blocks with GroupNorm + SiLU, sinusoidal time embedding and
class embedding (including a learned NULL embedding for classifier-free
guidance) injected additively into every residual block. Down path
16 -> 8 -> 4, bottleneck, up path with skip connections.
"""
import torch
import torch.nn as nn

from diffusion import sinusoidal_embedding

EMB_DIM = 128
NULL_LABEL = 4  # index of the unconditional / dropped class embedding


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(EMB_DIM, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.emb_proj(self.act(emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    def __init__(self, num_classes: int = 4, base: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(nn.Linear(EMB_DIM, EMB_DIM), nn.SiLU(), nn.Linear(EMB_DIM, EMB_DIM))
        self.class_emb = nn.Embedding(num_classes + 1, EMB_DIM)  # +1 = null label
        self.conv_in = nn.Conv2d(1, base, 3, padding=1)
        self.block1 = ResBlock(base, base)            # 16x16
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)   # -> 8x8
        self.block2 = ResBlock(base * 2, base * 2)    # 8x8
        self.down2 = nn.Conv2d(base * 2, base * 2, 4, stride=2, padding=1)  # -> 4x4
        self.mid = ResBlock(base * 2, base * 2)       # 4x4
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 4, stride=2, padding=1)  # -> 8x8
        self.block3 = ResBlock(base * 4, base * 2)    # concat skip2
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)      # -> 16x16
        self.block4 = ResBlock(base * 2, base)        # concat skip1
        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, x, t, y):
        emb = self.time_mlp(sinusoidal_embedding(t, EMB_DIM).to(x.device)) + self.class_emb(y)
        h = self.conv_in(x)
        s1 = self.block1(h, emb)                 # 32 @ 16
        s2 = self.block2(self.down1(s1), emb)    # 64 @ 8
        m = self.mid(self.down2(s2), emb)        # 64 @ 4
        u = self.block3(torch.cat([self.up2(m), s2], dim=1), emb)   # 64 @ 8
        u = self.block4(torch.cat([self.up1(u), s1], dim=1), emb)   # 32 @ 16
        return self.out(u)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
