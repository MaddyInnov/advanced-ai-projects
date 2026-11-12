"""Unit checks for the transition-graph loss (train_real.py).

Run: python3 test_trans_graph_loss.py
Checks:
  1. Mask: semantic edges legal, non-edges illegal (12-class, labels that
     do not themselves contain the tested pair).
  2. Loss ~0 for a confident LEGAL sequence (standing -> stand-to-sit ->
     sitting); loss large for a confident ILLEGAL switch (sitting -> walking).
  3. Gradient flows (loss.backward() populates .grad).
  4. Loss is a scalar in [0, 1] and invariant to batch size.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from train_real import (transition_graph_mask, transition_graph_loss,
                        HAPT12_EDGES)


def one_hot_logits(labels, strength=10.0):
    """Logits that put ~all mass on the given label per timestep."""
    B, L = labels.shape
    C = 12
    logits = torch.zeros(B, L, C)
    logits.scatter_(2, torch.tensor(labels)[..., None], strength)
    return logits


def main():
    # --- 1. mask -----------------------------------------------------------
    # Labels with only standing(4) -> stand-to-sit(6) -> sitting(3) runs.
    Y = np.array([[4] * 50 + [6] * 30 + [3] * 50,
                  [3] * 40 + [7] * 20 + [4] * 70], dtype=np.int64)
    mask = transition_graph_mask(Y, 12)
    assert mask[4, 6] and mask[6, 3] and mask[3, 7] and mask[7, 4], "semantic edges must be legal"
    assert mask[3, 3] and mask[4, 4], "self-loops must be legal"
    assert not mask[3, 0], "sitting -> walking must be ILLEGAL (not observed, not semantic)"
    assert not mask[3, 4], "sitting -> standing direct must be ILLEGAL (must go via sit-to-stand)"
    assert not mask[6, 4], "stand-to-sit -> standing must be ILLEGAL (target is sitting)"
    for u, v in HAPT12_EDGES:
        assert mask[u, v], f"edge {u}->{v} must be legal"
    print(f"mask OK: {int(mask.sum())}/144 legal pairs from this label set")

    legal_t = torch.from_numpy(mask)

    # --- 2. loss values ----------------------------------------------------
    legal_seq = np.array([[4] * 50 + [6] * 30 + [3] * 50])
    illegal_seq = np.array([[3] * 60 + [0] * 60])  # sitting -> walking (illegal)
    l_legal = transition_graph_loss(one_hot_logits(legal_seq), legal_t).item()
    l_illegal = transition_graph_loss(one_hot_logits(illegal_seq), legal_t).item()
    print(f"loss legal-seq={l_legal:.6f}  illegal-seq={l_illegal:.4f}")
    assert l_legal < 0.01, f"legal sequence should cost ~0, got {l_legal}"
    # Mean over timesteps: ONE illegal switch in 119 adjacent pairs costs
    # ~1/119 of full mass. The loss is a rate, not a count.
    assert 0.005 < l_illegal < 0.02, f"one illegal switch should cost ~1/119, got {l_illegal}"
    assert l_illegal > 10 * l_legal, "illegal switch must cost >> legal sequence"

    # uniform predictions: illegal mass = fraction of illegal pairs.
    uni = torch.zeros(2, 64, 12)
    l_uni = transition_graph_loss(uni, legal_t).item()
    n_illegal = 144 - int(mask.sum())
    expect = n_illegal / 144.0
    print(f"loss uniform={l_uni:.4f} (expected ~{expect:.4f})")
    assert abs(l_uni - expect) < 0.02, f"uniform loss {l_uni} != {expect}"

    # --- 3. gradient flows -------------------------------------------------
    logits = one_hot_logits(illegal_seq, strength=2.0).requires_grad_(True)
    loss = transition_graph_loss(logits, legal_t)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0, "gradient must be non-zero"
    print("gradient OK")

    # --- 4. scalar + batch invariance --------------------------------------
    assert loss.dim() == 0 and 0.0 <= float(loss) <= 1.0
    l1 = transition_graph_loss(one_hot_logits(legal_seq), legal_t).item()
    l2 = transition_graph_loss(
        torch.cat([one_hot_logits(legal_seq), one_hot_logits(legal_seq)]),
        legal_t).item()
    assert abs(l1 - l2) < 1e-6, "loss must not depend on batch size"
    print("scalar/batch-invariance OK")
    print("ALL TRANSITION-GRAPH LOSS CHECKS PASSED")


if __name__ == "__main__":
    main()
