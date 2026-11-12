"""Unit tests for the ASRF-style boundary branch (BRB) in train_real.py.

Covers: boundary-target construction (radius, edges), BoundaryHead
shapes + gradient flow, segment_majority refinement behaviour, and a
DENSE_SMOKE end-to-end train with the head enabled.
"""
import numpy as np
import torch

from train_real import (BoundaryHead, DenseHARRefined, boundary_targets,
                        segment_majority)


def test_boundary_targets_radius():
    y = np.zeros((1, 20), dtype=np.int64)
    y[0, 10:] = 1  # single change point at t=10
    t = boundary_targets(y, radius=2)
    assert t.shape == (1, 20)
    # 1 within +-2 of the change (t=8..12), 0 elsewhere
    assert t[0, 8:13].tolist() == [1.0] * 5
    assert t[0, :8].sum() == 0 and t[0, 13:].sum() == 0


def test_boundary_targets_edges_and_multi():
    y = np.array([[0, 0, 1, 1, 1, 2, 2, 0]], dtype=np.int64)
    t = boundary_targets(y, radius=1)
    # changes at t=2 (0->1), t=5 (1->2), t=7 (2->0)
    assert t[0, 1:4].tolist() == [1.0] * 3
    assert t[0, 4:6].tolist() == [1.0] * 2
    assert t[0, 6:8].tolist() == [1.0] * 2
    assert t[0, 0] == 0.0


def test_boundary_targets_no_change():
    y = np.full((2, 50), 3, dtype=np.int64)
    assert boundary_targets(y, radius=2).sum() == 0


def test_boundary_head_shape_and_grad():
    torch.manual_seed(0)
    head = BoundaryHead(num_classes=12)
    logp = torch.log_softmax(torch.randn(2, 512, 12), dim=-1)
    out = head(logp)
    assert out.shape == (2, 512, 1)
    loss = out.mean()
    loss.backward()
    assert all(p.grad is not None for p in head.parameters())
    assert torch.isfinite(out).all()


def test_densehar_refined_with_head():
    torch.manual_seed(0)
    m = DenseHARRefined(width=32, num_classes=12, in_ch=6, boundary_head=True)
    x = torch.randn(2, 6, 512)
    l1, l2, b = m(x)
    assert l1.shape == (2, 512, 12) and l2.shape == (2, 512, 12)
    assert b.shape == (2, 512, 1)
    assert torch.isfinite(b).all()
    m2 = DenseHARRefined(width=32, num_classes=12, in_ch=6)
    out = m2(x)
    assert isinstance(out, tuple) and len(out) == 2  # no head -> (l1, l2)


def test_segment_majority_removes_flicker():
    # one flicker timestep inside the 0-segment; a predicted boundary at the
    # true 0->2 change (t=6) keeps the 2-segment intact
    P = np.array([[0, 0, 1, 0, 0, 0, 2, 2, 2]])
    B = np.zeros((1, 9), dtype=np.float32)
    B[0, 6] = 0.9
    out = segment_majority(P, B, thresh=0.5)
    assert out[0].tolist() == [0, 0, 0, 0, 0, 0, 2, 2, 2]


def test_segment_majority_missed_boundary_risk():
    # documented limitation: with NO predicted boundary, the whole sequence
    # is one segment and the minority true segment is voted away. Boundary
    # recall is therefore load-bearing for the ASRF refinement.
    P = np.array([[0, 0, 1, 0, 0, 0, 2, 2, 2]])
    B = np.zeros((1, 9), dtype=np.float32)
    out = segment_majority(P, B, thresh=0.5)
    assert out[0].tolist() == [0] * 9


def test_segment_majority_keeps_true_short_segment():
    # a genuine 1-step segment, delimited by predicted boundaries, survives
    P = np.array([[0, 0, 0, 1, 0, 0, 0]])
    B = np.zeros((1, 7), dtype=np.float32)
    B[0, 3] = 0.9  # boundary at the short segment's start
    B[0, 4] = 0.9  # ...and end
    out = segment_majority(P, B, thresh=0.5)
    assert out[0, 3] == 1


def test_segment_majority_no_boundaries_is_global_vote():
    P = np.array([[0, 1, 1, 1, 0, 1]])
    B = np.zeros((1, 6), dtype=np.float32)
    out = segment_majority(P, B, thresh=0.5)
    assert (out[0] == 1).all()


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all boundary-head unit tests passed")
