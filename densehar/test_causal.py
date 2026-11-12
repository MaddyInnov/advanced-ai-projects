"""Causality test for the causal DenseHAR variant: perturbing inputs
after position T must not change any output at positions <= T."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_real import build_model, count_params

torch.manual_seed(0)


def check(model, tag):
    model.eval()
    x = torch.randn(2, 6, 512)
    T = 300
    x2 = x.clone()
    x2[:, :, T + 1:] = torch.randn_like(x2[:, :, T + 1:]) * 5
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x2)
    outs1 = y1 if isinstance(y1, tuple) else (y1,)
    outs2 = y2 if isinstance(y2, tuple) else (y2,)
    for i, (a, b) in enumerate(zip(outs1, outs2)):
        assert a.shape == (2, 512, a.shape[-1]), (tag, i, a.shape)
        diff = (a[:, :T + 1] - b[:, :T + 1]).abs().max().item()
        assert diff == 0.0, f"{tag} output {i}: future leaked into past ({diff})"
        d_future = (a[:, T + 1:] - b[:, T + 1:]).abs().max().item()
        assert d_future > 0, f"{tag} output {i}: perturbation had no effect?!"
    print(f"{tag}: CAUSAL OK (params {count_params(model):,})")


def main():
    check(build_model(48, 12, True, arch="causal", boundary_head=True),
          "causal refined + boundary head")
    check(build_model(48, 12, True, arch="causal", boundary_head=False),
          "causal refined")
    check(build_model(48, 6, False, arch="causal"), "causal plain")
    # sanity: the offline model must FAIL this test (it is non-causal)
    m = build_model(48, 12, True, arch="densehar", boundary_head=True)
    m.eval()
    x = torch.randn(2, 6, 512)
    x2 = x.clone(); x2[:, :, 301:] = torch.randn_like(x2[:, :, 301:]) * 5
    with torch.no_grad():
        d = (m(x)[1][:, :301] - m(x2)[1][:, :301]).abs().max().item()
    assert d > 0, "offline model unexpectedly causal — test is broken"
    print(f"offline refined: non-causal as expected (past diff {d:.4f})")
    print("ALL CAUSALITY TESTS PASS")


if __name__ == "__main__":
    main()
