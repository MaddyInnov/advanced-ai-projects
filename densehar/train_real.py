"""Train DenseHAR + window-128 baseline on REAL HAPT data (data_hapt.npz).

Protocol (honest, fixed):
- Subject-disjoint splits from the loader: train 21 / val 4 / test 5 subjects.
- Model selection STRICTLY on val (best val macro F1 checkpoint).
- Test evaluated exactly ONCE per candidate configuration, after selection.
- Every experiment (success or failure) is appended to results_real.json.

Subcommands:
  phase1   DenseHAR config A (width 48, boundary-weighted CE + MS-TCN
           smoothing lambda=0.15, 14 epochs) + window-128 broadcast baseline
           (10 epochs). Both val-selected.
  exp      One configurable dense experiment (phase-2 levers):
             --name N --epochs E --seed S --classes {6,12} --augment
             --class-weight --refine --lr LR --width W
           classes=12 trains on the original 12 HAPT labels (transitions as
           their own classes) and maps predicted probabilities to 6 classes
           for scoring (probabilities summed over each target's preimage).
  median   Majority-filter post-processing; window tuned on VAL only, then
           applied once to the test predictions of --model checkpoint.
  ensemble Probability-averaging ensemble of --models ckpt,ckpt,... (val
           reported, test evaluated once).

Checkpoints: per-epoch resumable state ckpt_state_<tag>.pt; best-val model
model_real_<tag>.pt. Set DENSE_SMOKE=1 for a tiny pipeline check.
"""
import os
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
import sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import DenseHARNet, WindowClassifier, count_params
from metrics import macro_f1, segmental_metrics, change_points

SMOKE = os.environ.get("DENSE_SMOKE") == "1"
RESULTS = os.path.join(HERE, "results_real_smoke.json" if SMOKE else "results_real.json")
PREFIX = "smoke_" if SMOKE else ""
MAP12_TO_6 = np.array([0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 5, 4])
# HARTH (and any future single-label-space dataset) has no 12->6 dual
# structure: its 12 classes are the real classes. When set, model_probs
# must NOT aggregate. Set from the exp --dataset flag in main().
SINGLE_LABEL_12 = False
ACTIVITIES_6 = ["walking", "walking_upstairs", "walking_downstairs",
                "sitting", "standing", "laying"]


# ---------------------------------------------------------------- data
def load_data(dataset="hapt"):
    d = np.load(os.path.join(HERE, f"data_{dataset}.npz"))
    out = {}
    for sp, name in ((0, "train"), (1, "val"), (2, "test")):
        m = d["split"] == sp
        out[name] = dict(X=d["X"][m], y6=d["y6"][m], y12=d["y12"][m],
                         y_trans=d["y_trans"][m], subjects=d["subjects"][m])
    out["meta"] = dict(train_subjects=d["train_subjects"].tolist(),
                       val_subjects=d["val_subjects"].tolist(),
                       test_subjects=d["test_subjects"].tolist(),
                       norm_mean=d["norm_mean"].tolist(),
                       norm_std=d["norm_std"].tolist())
    return out


# ---------------------------------------------------------------- metrics
def eval_from_preds(P, Y, Tmask):
    """Full dense metric suite from predictions (mirrors metrics.evaluate_dense)
    plus accuracy restricted to transition-class timesteps (y_trans mask)."""
    acc = float((P == Y).mean())
    # Class count inferred from the data: HAPT's 6-class evals are
    # unchanged (max label 5 -> C=6); single-label datasets like HARTH
    # (12 classes) now get a correct macro-F1 instead of C=6.
    C = int(max(Y.max(), P.max())) + 1
    mf1, per = macro_f1(Y.ravel(), P.ravel(), C=C)
    mask = np.zeros_like(Y, dtype=bool)
    for b in range(len(Y)):
        for cp in change_points(Y[b]):
            mask[b, max(0, cp - 10):min(Y.shape[1], cp + 11)] = True
    trans_acc = float((P[mask] == Y[mask]).mean()) if mask.any() else 0.0
    tp = fp = fn = 0
    for b in range(len(Y)):
        tcps = change_points(Y[b]); matched = set()
        for pc in change_points(P[b]):
            hit = [i for i, tc in enumerate(tcps)
                   if abs(int(pc) - int(tc)) <= 10 and i not in matched]
            if hit:
                tp += 1; matched.add(hit[0])
            else:
                fp += 1
        fn += len(tcps) - len(matched)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    bf1 = 2 * prec * rec / max(prec + rec, 1e-9)
    res = dict(accuracy=acc, macro_f1=mf1, per_class_f1=per,
               per_class_f1_named=({ACTIVITIES_6[i]: per[i] for i in range(6)}
                                   if C == 6 else
                                   {str(i): per[i] for i in range(C)}),
               transition_window_accuracy=trans_acc,
               transition_window_timesteps=int(mask.sum()),
               boundary_f1=float(bf1), boundary_precision=float(prec),
               boundary_recall=float(rec), boundary_tp=int(tp),
               boundary_fp=int(fp), boundary_fn=int(fn))
    res.update(segmental_metrics(Y, P))
    if Tmask is not None and Tmask.any():
        res["transition_class_timestep_accuracy"] = float((P[Tmask] == Y[Tmask]).mean())
        res["transition_class_timesteps"] = int(Tmask.sum())
        res["non_transition_timestep_accuracy"] = float((P[~Tmask] == Y[~Tmask]).mean())
    return res


# ---------------------------------------------------------------- models
class RefineStage(nn.Module):
    """MS-TCN-style refinement stage: dilated residual blocks over stage-1
    log-probabilities."""

    def __init__(self, num_classes, hidden=64, dilations=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.inp = nn.Conv1d(num_classes, hidden, 1)
        self.blocks = nn.ModuleList([
            nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d) for d in dilations])
        self.out = nn.Conv1d(hidden, num_classes, 1)

    def forward(self, logp):  # (B,L,C) -> (B,L,C) logits
        x = torch.relu(self.inp(logp.transpose(1, 2)))
        for conv in self.blocks:
            x = x + torch.relu(conv(x))
        return self.out(x).transpose(1, 2)


class DenseHARRefined(nn.Module):
    def __init__(self, width=48, num_classes=6, hidden=64, in_ch=6,
                 boundary_head=False):
        super().__init__()
        self.base = DenseHARNet(in_ch=in_ch, width=width, num_classes=num_classes)
        self.refine = RefineStage(num_classes, hidden)
        self.bhead = BoundaryHead(num_classes) if boundary_head else None

    def forward(self, x):
        l1 = self.base(x)
        l2 = self.refine(torch.log_softmax(l1, dim=-1))
        if self.bhead is not None:
            return l1, l2, self.bhead(torch.log_softmax(l1, dim=-1))
        return l1, l2


class BoundaryHead(nn.Module):
    """ASRF-style boundary regression branch (BRB): class-agnostic
    per-timestep boundary logit, decoupled from the classification head.
    Reads stage-1 log-probabilities (the backbone output); a small dilated
    stack gives it local context around each candidate boundary. Zero
    inference cost when unused; the head itself is ~10K params."""

    def __init__(self, num_classes, hidden=32, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.inp = nn.Conv1d(num_classes, hidden, 1)
        self.blocks = nn.ModuleList([
            nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d)
            for d in dilations])
        self.out = nn.Conv1d(hidden, 1, 1)

    def forward(self, logp):  # (B,L,C) -> (B,L,1) boundary logits
        x = torch.relu(self.inp(logp.transpose(1, 2)))
        for conv in self.blocks:
            x = x + torch.relu(conv(x))
        return self.out(x).transpose(1, 2)


# ---------------------------------------------------------------------------
# Causal DenseHAR (2026-11-10): the streaming variant. Every temporal
# operation is causal — left-only conv padding, per-position channel
# norms (no temporal statistics), per-position SE gate, repeat
# upsampling — so output at time t depends only on inputs <= t, in
# training and at inference alike. Contract matches DenseHARRefined:
# forward returns (l1, l2) or (l1, l2, boundary).
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation=1, stride=1):
        super().__init__()
        self.left = (k - 1) if stride > 1 else dilation * (k - 1)
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=dilation,
                              stride=stride)

    def forward(self, x):
        if self.left:
            x = torch.nn.functional.pad(x, (self.left, 0))
        return self.conv(x)


class PosNorm(nn.Module):
    """LayerNorm over channels at each position (no temporal mixing)."""

    def __init__(self, ch):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class CausalSE(nn.Module):
    """Per-position channel gate (SE without the temporal squeeze)."""

    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, ch // r), nn.ReLU(),
                                nn.Linear(ch // r, ch), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.transpose(1, 2)).transpose(1, 2)
        return x * w


class CausalInceptionBlock(nn.Module):
    def __init__(self, in_ch, branch_ch=16, out_ch=48):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(CausalConv1d(in_ch, branch_ch, k), PosNorm(branch_ch),
                          nn.ReLU())
            for k in (1, 3, 5, 11)])
        self.proj = nn.Sequential(nn.Conv1d(branch_ch * 4, out_ch, 1),
                                  PosNorm(out_ch), nn.ReLU())
        self.se = CausalSE(out_ch)
        self.short = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch \
            else nn.Identity()

    def forward(self, x):
        y = self.proj(torch.cat([b(x) for b in self.branches], dim=1))
        return torch.relu(self.se(y) + self.short(x))


class CausalTemporalBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.conv = nn.Sequential(CausalConv1d(ch, ch, 3, dilation=dilation),
                                  PosNorm(ch), nn.ReLU())

    def forward(self, x):
        return torch.relu(x + self.conv(x))


class CausalDenseHARNet(nn.Module):
    def __init__(self, in_ch=6, num_classes=6, width=48, branch_ch=16):
        super().__init__()
        self.width = width
        self.stem = nn.Sequential(
            CausalConv1d(in_ch, width, 5, stride=2), PosNorm(width), nn.ReLU())
        self.inc1 = CausalInceptionBlock(width, branch_ch, width)
        self.inc2 = CausalInceptionBlock(width, branch_ch, width)
        self.temp = nn.Sequential(CausalTemporalBlock(width, 1),
                                  CausalTemporalBlock(width, 2),
                                  CausalTemporalBlock(width, 4),
                                  CausalTemporalBlock(width, 8))
        self.head = nn.Conv1d(width, num_classes, 1)

    def forward(self, x):  # (B,6,L) -> (B,L,C); half-rate encoder, repeat x2
        L = x.shape[-1]
        x = self.temp(self.inc2(self.inc1(self.stem(x))))
        x = x.repeat_interleave(2, dim=-1)[..., :L]
        return self.head(x).transpose(1, 2)


class CausalRefineStage(nn.Module):
    def __init__(self, num_classes, hidden=64,
                 dilations=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.inp = nn.Conv1d(num_classes, hidden, 1)
        self.blocks = nn.ModuleList([
            CausalConv1d(hidden, hidden, 3, dilation=d) for d in dilations])
        self.out = nn.Conv1d(hidden, num_classes, 1)

    def forward(self, logp):
        x = torch.relu(self.inp(logp.transpose(1, 2)))
        for conv in self.blocks:
            x = x + torch.relu(conv(x))
        return self.out(x).transpose(1, 2)


class CausalBoundaryHead(nn.Module):
    def __init__(self, num_classes, hidden=32, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.inp = nn.Conv1d(num_classes, hidden, 1)
        self.blocks = nn.ModuleList([
            CausalConv1d(hidden, hidden, 3, dilation=d) for d in dilations])
        self.out = nn.Conv1d(hidden, 1, 1)

    def forward(self, logp):
        x = torch.relu(self.inp(logp.transpose(1, 2)))
        for conv in self.blocks:
            x = x + torch.relu(conv(x))
        return self.out(x).transpose(1, 2)


class CausalDenseHARRefined(nn.Module):
    def __init__(self, width=48, num_classes=6, hidden=64, in_ch=6,
                 boundary_head=False):
        super().__init__()
        self.base = CausalDenseHARNet(in_ch=in_ch, width=width,
                                      num_classes=num_classes)
        self.refine = CausalRefineStage(num_classes, hidden)
        self.bhead = CausalBoundaryHead(num_classes) if boundary_head else None

    def forward(self, x):
        l1 = self.base(x)
        l2 = self.refine(torch.log_softmax(l1, dim=-1))
        if self.bhead is not None:
            return l1, l2, self.bhead(torch.log_softmax(l1, dim=-1))
        return l1, l2


class _DilatedResidual(nn.Module):
    """Single-dilation residual layer (MS-TCN refinement stages)."""

    def __init__(self, hidden, dilation):
        super().__init__()
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=dilation,
                              dilation=dilation)
        self.out = nn.Conv1d(hidden, hidden, 1)

    def forward(self, x):
        return x + self.out(torch.relu(self.conv(x)))


class _DualDilatedResidual(nn.Module):
    """MS-TCN++ prediction-generation layer: two dilated branches with
    complementary dilations (d and its mirror in the dilation schedule),
    summed before the 1x1 output projection."""

    def __init__(self, hidden, d1, d2):
        super().__init__()
        self.ca = nn.Conv1d(hidden, hidden, 3, padding=d1, dilation=d1)
        self.cb = nn.Conv1d(hidden, hidden, 3, padding=d2, dilation=d2)
        self.out = nn.Conv1d(hidden, hidden, 1)

    def forward(self, x):
        return x + self.out(torch.relu(self.ca(x)) + torch.relu(self.cb(x)))


class _TCNStage(nn.Module):
    """One TCN stage: 1x1 input projection -> residual layers -> 1x1 head.
    Returns logits (B, T, C)."""

    def __init__(self, in_dim, hidden, num_classes, dilations, dual=False):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, hidden, 1)
        L = len(dilations)
        if dual:
            self.body = nn.ModuleList([
                _DualDilatedResidual(hidden, d, dilations[L - 1 - i])
                for i, d in enumerate(dilations)])
        else:
            self.body = nn.ModuleList([
                _DilatedResidual(hidden, d) for d in dilations])
        self.head = nn.Conv1d(hidden, num_classes, 1)

    def forward(self, x):  # (B, in_dim, T) -> (B, T, C)
        h = torch.relu(self.inp(x))
        for m in self.body:
            h = m(h)
        return self.head(h).transpose(1, 2)


class MSTCNppNet(nn.Module):
    """MS-TCN++ (Li et al., AAAI 2020) same-split baseline: a
    prediction-generation stage with dual dilated layers, then refinement
    stages over the previous stage's log-probabilities.
    Input (B, C_in, T) -> tuple of per-stage logits, each (B, T, C)."""

    DILS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

    def __init__(self, hidden=64, num_classes=12, in_ch=6, n_refine=3):
        super().__init__()
        self.pg = _TCNStage(in_ch, hidden, num_classes, self.DILS, dual=True)
        self.refines = nn.ModuleList([
            _TCNStage(num_classes, hidden, num_classes, self.DILS, dual=False)
            for _ in range(n_refine)])

    def forward(self, x):
        outs = [self.pg(x)]
        for st in self.refines:
            logp = torch.log_softmax(outs[-1], dim=-1).transpose(1, 2)
            outs.append(st(logp))
        return tuple(outs)


def build_model(width, num_classes, refine, in_ch=6, arch="densehar",
                boundary_head=False):
    if arch == "mstcnpp":
        return MSTCNppNet(hidden=width, num_classes=num_classes, in_ch=in_ch)
    if arch == "causal":
        if refine:
            return CausalDenseHARRefined(width=width, num_classes=num_classes,
                                         in_ch=in_ch,
                                         boundary_head=boundary_head)
        return CausalDenseHARNet(in_ch=in_ch, width=width,
                                 num_classes=num_classes)
    if refine:
        return DenseHARRefined(width=width, num_classes=num_classes, in_ch=in_ch,
                               boundary_head=boundary_head)
    return DenseHARNet(in_ch=in_ch, width=width, num_classes=num_classes)


def model_probs(model, X, num_classes, refine, bs=32):
    """6-class per-timestep probabilities (12-class probs are summed over
    each 6-class target's preimage)."""
    model.eval(); probs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out = model(torch.tensor(X[i:i + bs]).permute(0, 2, 1))
            if isinstance(out, tuple):
                # (l1, l2, boundary) -> refined logits are element -2;
                # (stage1..stageN) MS-TCN++ -> final stage is element -1.
                out = out[-2] if len(out) == 3 else out[-1]
            probs.append(torch.softmax(out, dim=-1).numpy())
    P = np.concatenate(probs, axis=0)
    if num_classes == 12 and not SINGLE_LABEL_12:
        P6 = np.zeros(P.shape[:2] + (6,), dtype=np.float64)
        for c12 in range(12):
            P6[..., MAP12_TO_6[c12]] += P[..., c12]
        return P6
    return P


def model_boundary_probs(model, X, bs=32):
    """Per-timestep boundary probability (B, L) from the BRB head, or None
    when the model has no boundary head."""
    if getattr(model, "bhead", None) is None:
        return None
    model.eval(); probs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out = model(torch.tensor(X[i:i + bs]).permute(0, 2, 1))
            probs.append(torch.sigmoid(out[-1]).squeeze(-1).numpy())
    return np.concatenate(probs, axis=0)


def save_model(path, model, meta):
    torch.save(dict(state_dict=model.state_dict(), meta=meta), path)


def load_model(path):
    blob = torch.load(path, map_location="cpu")
    meta = blob["meta"]
    if meta.get("kind") == "baseline":
        model = WindowClassifier()
    else:
        model = build_model(meta["width"], meta["num_classes"], meta["refine"],
                            in_ch=meta.get("in_ch", 6),
                            arch=meta.get("arch", "densehar"),
                            boundary_head=meta.get("boundary_head", False))
    model.load_state_dict(blob["state_dict"]); model.eval()
    return model, meta


# ---------------------------------------------------------------- training
def boundary_weights(y, extra=3.0):
    B, L = y.shape
    w = np.ones((B, L), dtype=np.float32)
    idx = np.arange(L)
    for b in range(B):
        bounds = np.where(np.diff(y[b]) != 0)[0] + 1
        if len(bounds):
            w[b] = 1.0 + extra * np.exp(
                -np.min(np.abs(idx[None, :] - bounds[:, None]), axis=0) / 10.0)
    return w


def _rot_matrix(rng, max_deg=40.0):
    """Random axis-angle rotation matrix (Rodrigues), angle U(-max_deg, max_deg)."""
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)).astype(np.float32)


def augment_batch(Xb, Yb, rng, mean, std):
    """Physics-respecting augmentation (train only), per IMPROVEMENT_RESEARCH
    Technique 4: joint acc+gyro rotation (+-40 deg, same matrix both triads,
    applied in RAW units), smooth magnitude warping, global scaling, small
    time-warp (labels warped identically), jitter. NO permutation (would
    destroy dense sequence structure). Xb (B,512,6) normalised float32,
    Yb (B,512) int64."""
    Xb = Xb.copy(); Yb = Yb.copy()
    B, L, _ = Xb.shape
    mean = np.asarray(mean, dtype=np.float32); std = np.asarray(std, dtype=np.float32)
    for i in range(B):
        raw = Xb[i] * std + mean
        if rng.random() < 0.5:
            R = _rot_matrix(rng)
            raw[:, :3] = raw[:, :3] @ R.T
            raw[:, 3:] = raw[:, 3:] @ R.T
        if rng.random() < 0.5:
            knots = rng.normal(1.0, 0.2, 4).astype(np.float32)
            curve = np.interp(np.linspace(0, 3, L), np.arange(4), knots).astype(np.float32)
            raw = raw * curve[:, None]
        if rng.random() < 0.5:
            raw = raw * np.float32(rng.normal(1.0, 0.1))
        if rng.random() < 0.3:
            f = rng.uniform(0.9, 1.1)
            src = np.clip(np.linspace(0, L - 1, L) * f, 0, L - 1)
            lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, L - 1)
            frac = (src - lo).astype(np.float32)
            raw = raw[lo] * (1 - frac[:, None]) + raw[hi] * frac[:, None]
            Yb[i] = Yb[i][np.round(src).astype(int)]
        x = (raw - mean) / std
        if rng.random() < 0.5:
            x = x + rng.normal(0, 0.05, x.shape).astype(np.float32)
        Xb[i] = x.astype(np.float32)
    return Xb, Yb


def _moving_avg(a, w=51):
    """(N,L) moving average with edge padding (cumsum implementation)."""
    pad = w // 2
    ap = np.pad(a, ((0, 0), (pad, pad)), mode="edge")
    cs = np.cumsum(ap, axis=1, dtype=np.float64)
    cs = np.concatenate([np.zeros((a.shape[0], 1)), cs], axis=1)
    return ((cs[:, w:] - cs[:, :-w]) / w).astype(np.float32)


def _f11_raw_extra(X6, meta):
    """Unstandardised gravity(3) + |acc| + |gyro| channels from normalised
    6-channel input."""
    mean = np.array(meta["norm_mean"], dtype=np.float32)
    std = np.array(meta["norm_std"], dtype=np.float32)
    raw = X6 * std + mean
    acc, gyr = raw[:, :, :3], raw[:, :, 3:]
    grav = np.stack([_moving_avg(acc[:, :, c]) for c in range(3)], axis=-1)
    mag_a = np.linalg.norm(acc, axis=-1, keepdims=True)
    mag_g = np.linalg.norm(gyr, axis=-1, keepdims=True)
    return np.concatenate([grav, mag_a, mag_g], axis=-1).astype(np.float32)


def f11_extra_std(X6, meta):
    """Standardised extra channels (uses TRAIN stats stored in meta)."""
    F = _f11_raw_extra(X6, meta)
    fm = np.array(meta["f11_mean"], dtype=np.float32)
    fs = np.array(meta["f11_std"], dtype=np.float32)
    return ((F - fm) / fs).astype(np.float32)


def feature_pack(data):
    """Technique 8: append gravity estimate (1 s moving average of raw acc,
    3 ch) + signal magnitudes (|acc|, |gyro|) to the 6 raw channels -> 11 ch.
    New channels standardised with TRAIN-split statistics. Gravity direction
    is the posture cue for the sitting/standing/laying pair."""
    Ftr = _f11_raw_extra(data["train"]["X"], data["meta"])
    fmean = Ftr.reshape(-1, 5).mean(axis=0); fstd = Ftr.reshape(-1, 5).std(axis=0)
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in data.items()}
    out["meta"]["f11_mean"] = fmean.tolist(); out["meta"]["f11_std"] = fstd.tolist()
    for split in ("train", "val", "test"):
        F = Ftr if split == "train" else _f11_raw_extra(data[split]["X"], data["meta"])
        out[split]["X"] = np.concatenate(
            [data[split]["X"], (F - fmean) / fstd], axis=-1).astype(np.float32)
    return out


def smooth_loss(logits):
    logp = torch.log_softmax(logits, dim=-1)
    return torch.clamp((logp[:, 1:] - logp[:, :-1]) ** 2, 0, 4).mean()


def boundary_targets(y, radius=2):
    """ASRF-style boundary labels: 1.0 within `radius` steps of any true
    label change (change point itself included), else 0.0. y (B, L) int."""
    B, L = y.shape
    tgt = np.zeros((B, L), dtype=np.float32)
    idx = np.arange(L)
    for b in range(B):
        bounds = np.where(np.diff(y[b]) != 0)[0] + 1
        if len(bounds):
            d = np.min(np.abs(idx[None, :] - bounds[:, None]), axis=0)
            tgt[b] = (d <= radius).astype(np.float32)
    return tgt


def segment_majority(P, Bprob, thresh=0.5):
    """ASRF refinement: split each predicted sequence at timesteps where the
    boundary head fires (Bprob > thresh), then assign every segment its
    majority class — suppressing within-segment flicker while preserving
    true short segments the head separates. P (B, L) int labels,
    Bprob (B, L) float in [0, 1]."""
    B, L = P.shape
    out = P.copy()
    for b in range(B):
        cuts = np.where(Bprob[b] > thresh)[0]
        edges = np.concatenate([[0], cuts, [L]])
        for s, e in zip(edges[:-1], edges[1:]):
            if e - s <= 1:
                continue
            vals, cnts = np.unique(P[b, s:e], return_counts=True)
            out[b, s:e] = vals[np.argmax(cnts)]
    return out


# ---------------------------------------------------------------------------
# Transition-graph loss (RoboSubtaskNet-style, 2026; queue step 2, 2026-11-08)
# Penalizes probability mass on adjacent-label switches that are ILLEGAL
# under HAPT's transition structure. In 12-class space the legal graph is
# well-defined: a basic activity may stay, go to one of its outgoing named
# transitions, or (for the dynamic set) switch to another dynamic activity /
# standing; a transition class may stay or go to its target basic activity.
# The legal set is the UNION of (a) pairs actually observed adjacently in
# the training labels and (b) the semantic edges below, so a protocol
# assumption can never make a real training switch illegal.
# Loss = mean over timesteps of sum_{(a,b) illegal} p_t(a) * p_{t+1}(b).
# Zero when the model puts all mass on legal switches; directly penalizes
# the flicker patterns (e.g. sitting -> walking -> sitting) that drive
# over-segmentation. In 6-class space the graph is nearly complete (y6 maps
# transitions to their target, so standing->sitting is observed), so the
# loss is only meaningful in 12-class mode.
# ---------------------------------------------------------------------------
# 12-class indices: 0 walking, 1 upstairs, 2 downstairs, 3 sitting,
# 4 standing, 5 laying, 6 stand-to-sit, 7 sit-to-stand, 8 sit-to-lie,
# 9 lie-to-sit, 10 stand-to-lie, 11 lie-to-stand.
HAPT12_EDGES = [(4, 6), (4, 10),          # standing -> stand-to-sit / stand-to-lie
                (3, 7), (3, 8),          # sitting -> sit-to-stand / sit-to-lie
                (5, 9), (5, 11),         # laying -> lie-to-sit / lie-to-stand
                (6, 3), (7, 4), (8, 5),  # transition -> target basic
                (9, 3), (10, 5), (11, 4)]
HAPT12_DYNAMIC = [0, 1, 2, 4]  # walking/upstairs/downstairs/standing switch freely


def transition_graph_mask(Y, num_classes):
    """Legal adjacent-label pairs as a (C, C) bool array."""
    legal = np.zeros((num_classes, num_classes), dtype=bool)
    np.fill_diagonal(legal, True)
    a = Y[:, :-1].ravel()
    b = Y[:, 1:].ravel()
    legal[a, b] = True  # every adjacent pair observed in training labels
    if num_classes == 12:
        for u, v in HAPT12_EDGES:
            legal[u, v] = True
        for u in HAPT12_DYNAMIC:
            for v in HAPT12_DYNAMIC:
                legal[u, v] = True
    return legal


def transition_graph_loss(logits, legal_mask):
    """Mean illegal-switch probability mass. logits (B, L, C); legal_mask
    (C, C) bool tensor on the same device as logits."""
    p = torch.softmax(logits, dim=-1)
    joint = p[:, :-1, :, None] * p[:, 1:, None, :]  # (B, L-1, C, C)
    return (joint * (~legal_mask)).sum(dim=(-2, -1)).mean()


def train_dense(tag, data, epochs, seed, width=48, lr=3e-3, num_classes=6,
                class_weight=False, augment=False, refine=False,
                use_bw=True, smooth_lambda=0.15, features="raw6",
                eval_test=True, trans_graph_lambda=0.0, trans_graph_stage="all",
                arch="densehar", boundary_lambda=0.0, boundary_radius=2,
                boundary_thresh=0.5, distill_from=None, distill_alpha=0.0,
                distill_theta=10):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    Xtr, Xval = data["train"]["X"], data["val"]["X"]
    in_ch = Xtr.shape[-1]
    Ytr_key = "y12" if num_classes == 12 else "y6"
    Ytr, Yval6 = data["train"][Ytr_key], data["val"]["y6"]
    Tval = data["val"]["y_trans"]
    if SMOKE:
        Xtr, Ytr = Xtr[:128], Ytr[:128]
    bw = boundary_weights(Ytr) if use_bw else np.ones_like(Ytr, dtype=np.float32)
    legal_t = None
    if trans_graph_lambda > 0:
        legal = transition_graph_mask(Ytr, num_classes)
        legal_t = torch.from_numpy(legal)
        n_legal = int(legal.sum())
        print(f"[{tag}] transition-graph loss on: {n_legal}/{num_classes * num_classes} "
              f"legal adjacent pairs (lambda={trans_graph_lambda})", flush=True)
    if class_weight:
        # Median-frequency balancing (ASRF's choice; IMPROVEMENT_RESEARCH Tech 6)
        counts = np.bincount(Ytr.ravel(), minlength=num_classes).astype(np.float64)
        freq = counts / counts.sum()
        cw = np.median(freq) / np.maximum(freq, 1e-9)
        cw = np.clip(cw, 0.5, 10.0).astype(np.float32)
        bw = bw * cw[Ytr]
        print(f"[{tag}] class weights (median-freq): {np.round(cw, 2).tolist()}", flush=True)
    Wt = torch.tensor(bw)
    if boundary_lambda > 0 and not refine:
        raise SystemExit("--boundary-head needs --refine (BRB attaches to the "
                         "refined architecture)")
    if boundary_lambda > 0 and arch == "mstcnpp":
        raise SystemExit("--boundary-head is not wired for --mstcnpp")
    bhead = boundary_lambda > 0
    bcrit = None
    if bhead:
        Bt = boundary_targets(Ytr, radius=boundary_radius)
        pos = float(Bt.sum()); neg = float(Bt.size - pos)
        pos_w = min(neg / max(pos, 1.0), 50.0)
        bcrit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
        print(f"[{tag}] ASRF boundary head on: lambda={boundary_lambda} "
              f"radius={boundary_radius} boundary-positive {pos / Bt.size:.3f} "
              f"pos_weight={pos_w:.1f}", flush=True)
    model = build_model(width, num_classes, refine, in_ch=in_ch, arch=arch,
                        boundary_head=bhead)
    teacher = None; groups6 = None; nll = None
    if distill_from:
        # TBDM-style boundary distillation (adapted; NOT the published
        # cross-attention module): a frozen boundary-head teacher's
        # boundary-refined + min-duration labels on each (augmented) batch
        # become an auxiliary 6-class CE target for the student's final
        # stage, aggregated over the 12->6 preimage. Training only — the
        # student needs no boundary head or post-processing at inference.
        teacher, tmeta = load_model(os.path.join(HERE, distill_from))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        assert getattr(teacher, "bhead", None) is not None, \
            "distillation teacher needs a boundary head"
        assert tmeta["num_classes"] == num_classes, \
            "teacher/student class counts differ"
        groups6 = [[c for c in range(num_classes) if MAP12_TO_6[c] == g]
                   for g in range(6)]
        nll = nn.NLLLoss(reduction="none")
        print(f"[{tag}] TBDM-style boundary distillation on: teacher="
              f"{distill_from} alpha={distill_alpha} theta={distill_theta}",
              flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(reduction="none")
    state_path = os.path.join(HERE, f"ckpt_state_{PREFIX}{tag}.pt")
    best_path = os.path.join(HERE, f"model_real_{PREFIX}{tag}.pt")
    start_epoch, best_val_f1, history = 0, -1.0, []
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu")
        if st["epochs_total"] == epochs:
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"]); start_epoch = st["epoch"]
            best_val_f1, history = st["best_val_f1"], st["history"]
            print(f"[{tag}] resuming at epoch {start_epoch}", flush=True)
    t0 = time.time()
    for ep in range(start_epoch, epochs):
        model.train(); perm = np.random.permutation(len(Xtr)); tot, nb = 0.0, 0
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            Xb, Yb = Xtr[idx], Ytr[idx]
            if augment:
                if in_ch == 11:
                    # Augment the 6 physical channels in raw space, then
                    # recompute the derived gravity/magnitude channels from
                    # the augmented signal so features stay consistent.
                    X6, Yb = augment_batch(Xb[:, :, :6], Yb, rng,
                                           data["meta"]["norm_mean"],
                                           data["meta"]["norm_std"])
                    Xb = np.concatenate(
                        [X6, f11_extra_std(X6, data["meta"])], axis=-1)
                else:
                    Xb, Yb = augment_batch(Xb, Yb, rng, data["meta"]["norm_mean"],
                                           data["meta"]["norm_std"])
            Xt = torch.tensor(Xb).permute(0, 2, 1)
            Yt = torch.tensor(Yb); Wb = Wt[idx]
            opt.zero_grad()
            out = model(Xt)
            if bhead:
                b_logits = out[-1]
                outs = out[:-1]
            else:
                outs = out if isinstance(out, tuple) else (out,)
            loss = 0.0
            for j, o in enumerate(outs):
                loss = loss + (crit(o.reshape(-1, num_classes),
                                    Yt.reshape(-1)) * Wb.reshape(-1)).mean()
                if smooth_lambda > 0:
                    loss = loss + smooth_lambda * smooth_loss(o)
                if trans_graph_lambda > 0 and (
                        trans_graph_stage == "all" or j == len(outs) - 1):
                    loss = loss + trans_graph_lambda * transition_graph_loss(o, legal_t)
            if bhead:
                Bt = torch.tensor(boundary_targets(Yb, radius=boundary_radius))
                loss = loss + boundary_lambda * bcrit(
                    b_logits.squeeze(-1), Bt)
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(Xt)
                    t_logits = t_out[-2] if isinstance(t_out, tuple) else t_out
                    tp = torch.softmax(t_logits, dim=-1).numpy()
                    if num_classes == 12:
                        tp6 = np.zeros(tp.shape[:2] + (6,), dtype=np.float64)
                        for c12 in range(12):
                            tp6[..., MAP12_TO_6[c12]] += tp[..., c12]
                    else:
                        tp6 = tp
                    tb = torch.sigmoid(t_out[-1]).squeeze(-1).numpy()
                    t_lab = segment_majority(tp6.argmax(-1), tb,
                                             thresh=boundary_thresh)
                    t_lab = min_duration_relabel(t_lab, distill_theta)
                lp = torch.log_softmax(outs[-1], dim=-1)
                lp6 = torch.stack(
                    [torch.logsumexp(lp[..., idxs], dim=-1)
                     for idxs in groups6], dim=-1)
                dloss = (nll(lp6.reshape(-1, 6),
                             torch.tensor(t_lab).reshape(-1))
                         * Wb.reshape(-1)).mean()
                loss = loss + distill_alpha * dloss
            loss.backward(); opt.step(); tot += float(loss.item()); nb += 1
        sched.step()
        Pv = model_probs(model, Xval, num_classes, refine)
        v = eval_from_preds(Pv.argmax(-1), Yval6, Tval)
        hist_row = dict(epoch=ep + 1, train_loss=tot / max(nb, 1),
                        val_accuracy=v["accuracy"], val_macro_f1=v["macro_f1"],
                        val_transition_window_accuracy=v["transition_window_accuracy"],
                        val_edit=v["edit_score"])
        if bhead:
            Bv = model_boundary_probs(model, Xval)
            vb = eval_from_preds(
                segment_majority(Pv.argmax(-1), Bv, thresh=boundary_thresh),
                Yval6, Tval)
            hist_row["val_boundary_refined_seg25"] = vb["segmental_f1@25"]
            hist_row["val_boundary_refined_edit"] = vb["edit_score"]
            hist_row["val_boundary_refined_bf1"] = vb["boundary_f1"]
        history.append(hist_row)
        if v["macro_f1"] > best_val_f1:
            best_val_f1 = v["macro_f1"]
            save_model(best_path, model, dict(kind="dense", tag=tag, width=width,
                                              num_classes=num_classes, refine=refine,
                                              arch=arch, boundary_head=bhead,
                                              in_ch=in_ch, features=features,
                                              best_epoch=ep + 1))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=ep + 1,
                        epochs_total=epochs, best_val_f1=best_val_f1,
                        history=history), state_path)
        print(f"[{tag}] ep {ep + 1}/{epochs} loss {tot / max(nb, 1):.4f} "
              f"val acc {v['accuracy']:.4f} F1 {v['macro_f1']:.4f} "
              f"trans {v['transition_window_accuracy']:.4f} edit {v['edit_score']:.1f}",
              flush=True)
    secs = round(time.time() - t0, 1)
    model, meta = load_model(best_path)
    Pv = model_probs(model, data["val"]["X"], num_classes, refine)
    val_best = eval_from_preds(Pv.argmax(-1), data["val"]["y6"], data["val"]["y_trans"])
    val_bref = None
    if bhead:
        Bv = model_boundary_probs(model, data["val"]["X"])
        val_bref = eval_from_preds(
            segment_majority(Pv.argmax(-1), Bv, thresh=boundary_thresh),
            data["val"]["y6"], data["val"]["y_trans"])
    test = None
    if eval_test:
        Pt = model_probs(model, data["test"]["X"], num_classes, refine)
        test = eval_from_preds(Pt.argmax(-1), data["test"]["y6"], data["test"]["y_trans"])
    return dict(name=tag, kind=("dense_mstcnpp" if arch == "mstcnpp" else
                                "dense_refined" if refine else "dense"),
                config=dict(width=width, epochs=epochs, seed=seed, lr=lr,
                            num_classes=num_classes, class_weight=class_weight,
                            augment=augment, refine=refine, features=features,
                            boundary_weighted=use_bw, smoothing_lambda=smooth_lambda,
                            transition_graph_lambda=trans_graph_lambda,
                            trans_graph_stage=trans_graph_stage,
                            arch=arch, boundary_lambda=boundary_lambda,
                            boundary_radius=boundary_radius,
                            boundary_thresh=boundary_thresh,
                            distill_from=distill_from,
                            distill_alpha=distill_alpha,
                            distill_theta=distill_theta,
                            train_sequences=int(len(Xtr))),
                params=count_params(model), best_epoch=meta["best_epoch"],
                val=val_best, val_boundary_refined=val_bref, test=test,
                history=history, train_seconds=secs)


def train_baseline(tag, data, epochs=10, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    wins, labs = [], []
    Xtr, Ytr = data["train"]["X"], data["train"]["y6"]
    for i in range(len(Xtr)):
        for s in range(0, 512, 128):
            wins.append(Xtr[i, s:s + 128])
            labs.append(np.bincount(Ytr[i, s:s + 128], minlength=6).argmax())
    wins = np.stack(wins); labs = np.array(labs)
    if SMOKE:
        wins, labs = wins[:512], labs[:512]
    model = WindowClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(wins).permute(0, 2, 1); Yt = torch.tensor(labs)
    crit = nn.CrossEntropyLoss()
    state_path = os.path.join(HERE, f"ckpt_state_{PREFIX}{tag}.pt")
    best_path = os.path.join(HERE, f"model_real_{PREFIX}{tag}.pt")
    start_epoch, best_val_f1, history = 0, -1.0, []
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu")
        if st["epochs_total"] == epochs:
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"]); start_epoch = st["epoch"]
            best_val_f1, history = st["best_val_f1"], st["history"]
            print(f"[{tag}] resuming at epoch {start_epoch}", flush=True)

    def broadcast_preds(m, X):
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                xb = X[i:i + 32]
                w = np.stack([xb[:, s:s + 128] for s in range(0, 512, 128)], axis=1)
                p = m(torch.tensor(w.reshape(-1, 128, 6)).permute(0, 2, 1)).argmax(-1)
                out.append(np.repeat(p.numpy().reshape(-1, 4), 128, axis=1))
        return np.concatenate(out, axis=0)

    t0 = time.time()
    for ep in range(start_epoch, epochs):
        model.train(); perm = torch.randperm(len(Xt)); tot, nb = 0.0, 0
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]; opt.zero_grad()
            loss = crit(model(Xt[idx]), Yt[idx]); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        sched.step()
        v = eval_from_preds(broadcast_preds(model, data["val"]["X"]),
                            data["val"]["y6"], data["val"]["y_trans"])
        history.append(dict(epoch=ep + 1, train_loss=tot / max(nb, 1),
                            val_accuracy=v["accuracy"], val_macro_f1=v["macro_f1"]))
        if v["macro_f1"] > best_val_f1:
            best_val_f1 = v["macro_f1"]
            save_model(best_path, model, dict(kind="baseline", tag=tag, width=48,
                                              num_classes=6, refine=False,
                                              best_epoch=ep + 1))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=ep + 1,
                        epochs_total=epochs, best_val_f1=best_val_f1,
                        history=history), state_path)
        print(f"[{tag}] ep {ep + 1}/{epochs} loss {tot / max(nb, 1):.4f} "
              f"val acc {v['accuracy']:.4f} F1 {v['macro_f1']:.4f}", flush=True)
    secs = round(time.time() - t0, 1)
    model, meta = load_model(best_path)

    def bp(m, X):
        return broadcast_preds(m, X)
    val_best = eval_from_preds(bp(model, data["val"]["X"]), data["val"]["y6"],
                               data["val"]["y_trans"])
    test = eval_from_preds(bp(model, data["test"]["X"]), data["test"]["y6"],
                           data["test"]["y_trans"])
    return dict(name=tag, kind="baseline_window128",
                config=dict(epochs=epochs, seed=seed, lr=2e-3,
                            window=128, broadcast=True,
                            train_windows=int(len(wins))),
                params=count_params(model), best_epoch=meta["best_epoch"],
                val=val_best, test=test, history=history, train_seconds=secs)


# ---------------------------------------------------------------- results io
def load_results(data):
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    stats = {}
    sp = os.path.join(HERE, "data_hapt_stats.json")
    if os.path.exists(sp):
        with open(sp) as f:
            stats = json.load(f)
    return {
        "dataset": {
            "name": "HAPT (UCI ML Repository id 341) — real smartphone acc+gyro @50Hz",
            "source_zip": "~/workspace/goals/transition-aware-dense-har-research/code/data/hapt.zip",
            "stats_from_loader": stats,
            "split_subjects": data["meta"],
            "normalisation": "per-channel mean/std from TRAIN subjects' streams only",
            "sequencing": "512-step windows, stride 256, per (experiment,user) stream; "
                          "windows touching unlabelled samples dropped; boundary/transition "
                          "windows always kept, pure windows capped at <=30% of each split",
            "y6_mapping": "transition classes 7-12 mapped to TARGET activity "
                          "(stand-to-sit->sitting, sit-to-stand->standing, sit-to-lie->laying, "
                          "lie-to-sit->sitting, stand-to-lie->laying, lie-to-stand->standing); "
                          "y_trans marks timesteps whose original label was a transition class",
        },
        "protocol": "Subject-disjoint train(21)/val(4)/test(5). Model selection on val "
                    "macro F1 only. Test evaluated exactly once per candidate, after "
                    "selection. Every experiment logged, including failures.",
        "reference_windowed_hapt": {
            "model": "TinyHAR (author's earlier windowed HAPT work)",
            "window_accuracy": 0.9045, "macro_f1": 0.7942, "classes": 12,
            "protocol_note": "Windowed classification (128-sample windows, one label per "
                             "window), subject-disjoint split from the same research line. "
                             "NOT directly comparable: windowed scoring never has to label "
                             "transition timesteps densely, and the split/labels differ.",
        },
        "synthetic_reference": {
            "densehar_accuracy": 0.9534765625, "densehar_macro_f1": 0.9545498118833632,
            "baseline_accuracy": 0.857109375,
            "note": "Synthetic-data numbers from results.json — pipeline proof only, "
                    "superseded as headline by the real-data results below.",
        },
        "experiments": [],
    }


def append_result(res, exp):
    res["experiments"] = [e for e in res["experiments"] if e["name"] != exp["name"]]
    res["experiments"].append(exp)
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2)
    t = exp.get("test") or {}
    print(f"== {exp['name']}: val F1 {exp['val']['macro_f1']:.4f} | "
          f"test acc {t.get('accuracy', float('nan')):.4f} "
          f"F1 {t.get('macro_f1', float('nan')):.4f} "
          f"transWin {t.get('transition_window_accuracy', float('nan')):.4f} "
          f"transClass {t.get('transition_class_timestep_accuracy', float('nan')):.4f} "
          f"[saved to {os.path.basename(RESULTS)}]", flush=True)


# ---------------------------------------------------------------- filters
def gaussian_smooth_probs(P, sigma):
    """1D Gaussian smoothing of per-timestep probabilities (N,512,C)."""
    if sigma <= 0:
        return P
    r = int(3 * sigma)
    xs = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (xs / sigma) ** 2); k /= k.sum()
    out = np.empty_like(P)
    for b in range(len(P)):
        padded = np.pad(P[b], ((r, r), (0, 0)), mode="edge")
        for c in range(P.shape[-1]):
            out[b, :, c] = np.convolve(padded[:, c], k, mode="valid")
    return out


def min_duration_relabel(P, theta):
    """Iteratively merge predicted segments shorter than theta timesteps into
    their longer neighbour (ASRF-style relabeling; theta <= ~0.5 s = 25 samples
    so real ~1 s HAPT transitions survive)."""
    if theta <= 1:
        return P
    out = P.copy()
    for b in range(len(P)):
        lab = out[b].copy()
        while True:
            segs = segmental_segments(lab)
            if len(segs) <= 1:
                break
            short = [(e - s + 1, i) for i, (l, s, e) in enumerate(segs)
                     if e - s + 1 < theta]
            if not short:
                break
            _, i = min(short)
            l, s, e = segs[i]
            left = segs[i - 1] if i > 0 else None
            right = segs[i + 1] if i < len(segs) - 1 else None
            if left and right:
                nl = left[0] if (left[2] - left[1]) >= (right[2] - right[1]) else right[0]
            elif left:
                nl = left[0]
            else:
                nl = right[0]
            lab[s:e + 1] = nl
        out[b] = lab
    return out


def segmental_segments(a):
    segs = []; s = 0
    for i in range(1, len(a) + 1):
        if i == len(a) or a[i] != a[s]:
            segs.append((int(a[s]), s, i - 1)); s = i
    return segs


def postproc_labels(probs, sigma, theta, med_w):
    P = gaussian_smooth_probs(probs, sigma).argmax(-1)
    if med_w > 1:
        P = majority_filter(P, med_w)
    return min_duration_relabel(P, theta)


def majority_filter(P, w):
    if w <= 1:
        return P
    from numpy.lib.stride_tricks import sliding_window_view
    pad = w // 2
    out = np.empty_like(P)
    for b in range(len(P)):
        padded = np.pad(P[b], pad, mode="edge")
        win = sliding_window_view(padded, w)  # (512, w)
        out[b] = np.apply_along_axis(
            lambda v: np.bincount(v, minlength=6).argmax(), 1, win)
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("phase1")
    e = sub.add_parser("exp")
    e.add_argument("--name", required=True)
    e.add_argument("--epochs", type=int, default=14)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--classes", type=int, default=6, choices=(6, 12))
    e.add_argument("--augment", action="store_true")
    e.add_argument("--class-weight", action="store_true")
    e.add_argument("--refine", action="store_true")
    e.add_argument("--mstcnpp", action="store_true",
                   help="train the MS-TCN++ baseline instead of DenseHAR "
                        "(mutually exclusive with --refine)")
    e.add_argument("--causal", action="store_true",
                   help="train the causal streaming variant of DenseHAR "
                        "(left-pad convs, per-position norms; combines "
                        "with --refine and --boundary-head)")
    e.add_argument("--dataset", default="hapt",
                   help="dataset npz to load: data_<dataset>.npz "
                        "(hapt | harth)")
    e.add_argument("--lr", type=float, default=3e-3)
    e.add_argument("--width", type=int, default=48)
    e.add_argument("--no-bw", action="store_true",
                   help="plain CE weights (no boundary weighting)")
    e.add_argument("--smooth", type=float, default=0.15,
                   help="T-MSE smoothing-loss lambda (0 = off)")
    e.add_argument("--trans-graph", type=float, default=0.0,
                   help="transition-graph loss lambda (0 = off; penalizes "
                        "probability mass on illegal adjacent-label switches)")
    e.add_argument("--trans-graph-stage", choices=("all", "last"), default="all",
                   help="apply the transition-graph loss to all stage outputs "
                        "(default) or only the final/refined output")
    e.add_argument("--boundary-head", type=float, default=0.0,
                   help="ASRF-style boundary branch lambda (0 = off; needs "
                        "--refine): class-agnostic boundary regression head on "
                        "stage-1 log-probs, BCE vs GT boundaries, plus "
                        "boundary-gated segment-majority refinement at eval")
    e.add_argument("--boundary-radius", type=int, default=2,
                   help="GT boundary label half-width in timesteps")
    e.add_argument("--boundary-thresh", type=float, default=0.5,
                   help="predicted-boundary probability threshold for the "
                        "segment-majority refinement")
    e.add_argument("--distill-from", default=None,
                   help="teacher checkpoint WITH a boundary head: TBDM-style "
                        "boundary distillation (teacher's boundary-refined + "
                        "min-duration labels as auxiliary 6-class targets, "
                        "computed on the fly per augmented batch)")
    e.add_argument("--distill-alpha", type=float, default=0.0,
                   help="weight of the distillation loss")
    e.add_argument("--distill-theta", type=int, default=10,
                   help="min-duration theta applied to teacher labels")
    e.add_argument("--features11", action="store_true",
                   help="append gravity + magnitude channels (Technique 8)")
    e.add_argument("--no-test", action="store_true",
                   help="screening run: val only, do NOT evaluate test")
    m = sub.add_parser("median")
    m.add_argument("--model", required=True, help="model_real_<tag>.pt filename")
    m.add_argument("--name", required=True)
    pp = sub.add_parser("postproc")
    pp.add_argument("--model", required=True, help="model_real_<tag>.pt filename")
    pp.add_argument("--name", required=True)
    pp.add_argument("--no-test", action="store_true")
    cf = sub.add_parser("confirm")
    cf.add_argument("--model", required=True, help="model_real_<tag>.pt filename")
    cf.add_argument("--name", required=True)
    cf.add_argument("--boundary-refined", action="store_true",
                    help="evaluate the ASRF boundary-gated segment-majority "
                         "refinement instead of raw argmax (needs a checkpoint "
                         "with a boundary head)")
    cf.add_argument("--boundary-thresh", type=float, default=0.5,
                    help="predicted-boundary probability threshold")
    en = sub.add_parser("ensemble")
    en.add_argument("--models", required=True, help="comma-separated model filenames")
    en.add_argument("--name", required=True)
    en.add_argument("--postproc-from", default=None,
                    help="experiment name whose selected (sigma,theta,med) to apply")
    args = ap.parse_args()

    data = load_data(getattr(args, "dataset", "hapt"))
    res = load_results(data)
    print(f"splits: train {len(data['train']['X'])} val {len(data['val']['X'])} "
          f"test {len(data['test']['X'])}", flush=True)

    if args.cmd == "phase1":
        exp = train_dense(PREFIX + "p1_dense", data,
                          epochs=2 if SMOKE else 14, seed=0)
        append_result(res, exp)
        base = train_baseline(PREFIX + "p1_base", data,
                              epochs=2 if SMOKE else 10, seed=0)
        append_result(res, base)
        if not SMOKE:
            res["phase1"] = dict(densehar=exp["name"], baseline=base["name"])
            with open(RESULTS, "w") as f:
                json.dump(res, f, indent=2)
    elif args.cmd == "exp":
        global SINGLE_LABEL_12
        SINGLE_LABEL_12 = (args.dataset != "hapt")
        if args.mstcnpp and args.refine:
            raise SystemExit("--mstcnpp and --refine are mutually exclusive")
        if args.mstcnpp and args.causal:
            raise SystemExit("--mstcnpp and --causal are mutually exclusive")
        d = feature_pack(data) if args.features11 else data
        exp = train_dense(PREFIX + args.name, d, epochs=args.epochs,
                          seed=args.seed, width=args.width, lr=args.lr,
                          num_classes=args.classes,
                          class_weight=args.class_weight, augment=args.augment,
                          refine=args.refine, use_bw=not args.no_bw,
                          smooth_lambda=args.smooth,
                          trans_graph_lambda=args.trans_graph,
                          trans_graph_stage=args.trans_graph_stage,
                          arch=("mstcnpp" if args.mstcnpp else
                                "causal" if args.causal else "densehar"),
                          boundary_lambda=args.boundary_head,
                          boundary_radius=args.boundary_radius,
                          boundary_thresh=args.boundary_thresh,
                          distill_from=args.distill_from,
                          distill_alpha=args.distill_alpha,
                          distill_theta=args.distill_theta,
                          features="f11" if args.features11 else "raw6",
                          eval_test=not args.no_test)
        append_result(res, exp)
    elif args.cmd == "median":
        path = os.path.join(HERE, args.model)
        model, meta = load_model(path)
        if meta["kind"] != "dense":
            print("median filter applies to dense checkpoints", flush=True)
            return
        if meta.get("features") == "f11":
            data = feature_pack(data)
        Pv = model_probs(model, data["val"]["X"], meta["num_classes"],
                         meta["refine"]).argmax(-1)
        rows, best_w, best_f1 = [], 1, -1.0
        for w in (1, 3, 5, 7, 9, 11, 15, 21, 31):
            v = eval_from_preds(majority_filter(Pv, w), data["val"]["y6"],
                                data["val"]["y_trans"])
            rows.append(dict(window=w, val_accuracy=v["accuracy"],
                             val_macro_f1=v["macro_f1"],
                             val_transition_window_accuracy=v["transition_window_accuracy"],
                             val_edit=v["edit_score"]))
            print(f"[median] w={w} val acc {v['accuracy']:.4f} F1 {v['macro_f1']:.4f} "
                  f"edit {v['edit_score']:.1f}", flush=True)
            if v["macro_f1"] > best_f1:
                best_f1, best_w = v["macro_f1"], w
        Pt = model_probs(model, data["test"]["X"], meta["num_classes"],
                         meta["refine"]).argmax(-1)
        val_best = eval_from_preds(majority_filter(Pv, best_w), data["val"]["y6"],
                                   data["val"]["y_trans"])
        test = eval_from_preds(majority_filter(Pt, best_w), data["test"]["y6"],
                               data["test"]["y_trans"])
        append_result(res, dict(
            name=PREFIX + args.name, kind="median_filter_posthoc",
            config=dict(base_model=args.model, best_window=best_w,
                        window_selected_on="val macro F1", sweep=rows),
            params=None, best_epoch=None, val=val_best, test=test,
            history=[], train_seconds=0.0))
    elif args.cmd == "postproc":
        path = os.path.join(HERE, args.model)
        model, meta = load_model(path)
        if meta.get("kind") not in ("dense", "dense_refined"):
            print("postproc applies to dense checkpoints", flush=True)
            return
        if meta.get("features") == "f11":
            data = feature_pack(data)
        Pv = model_probs(model, data["val"]["X"], meta["num_classes"], meta["refine"])
        rows, best, best_key = [], None, (-1.0, -1.0)
        for sigma in (0, 4, 8):
            for theta in (0, 10, 15, 25):
                for med in (1, 11):
                    v = eval_from_preds(postproc_labels(Pv, sigma, theta, med),
                                        data["val"]["y6"], data["val"]["y_trans"])
                    rows.append(dict(sigma=sigma, theta=theta, median=med,
                                     val_accuracy=v["accuracy"],
                                     val_macro_f1=v["macro_f1"],
                                     val_transition_window_accuracy=v["transition_window_accuracy"],
                                     val_segmental_f1_25=v["segmental_f1@25"],
                                     val_edit=v["edit_score"],
                                     val_boundary_f1=v["boundary_f1"]))
                    key = (v["macro_f1"], v["segmental_f1@25"])
                    print(f"[postproc] sigma={sigma} theta={theta} med={med} "
                          f"val acc {v['accuracy']:.4f} F1 {v['macro_f1']:.4f} "
                          f"seg25 {v['segmental_f1@25']:.4f} edit {v['edit_score']:.1f}",
                          flush=True)
                    if key > best_key:
                        best_key, best = key, (sigma, theta, med)
        sigma, theta, med = best
        val_best = eval_from_preds(postproc_labels(Pv, sigma, theta, med),
                                   data["val"]["y6"], data["val"]["y_trans"])
        test = None
        if not args.no_test:
            Pt = model_probs(model, data["test"]["X"], meta["num_classes"],
                             meta["refine"])
            test = eval_from_preds(postproc_labels(Pt, sigma, theta, med),
                                   data["test"]["y6"], data["test"]["y_trans"])
        append_result(res, dict(
            name=PREFIX + args.name, kind="postproc_pack",
            config=dict(base_model=args.model, base_tag=meta["tag"],
                        selected=dict(sigma=sigma, theta=theta, median=med),
                        selected_on="val macro F1 (tie-break seg F1@25)", grid=rows),
            params=None, best_epoch=None, val=val_best, test=test,
            history=[], train_seconds=0.0))
    elif args.cmd == "confirm":
        path = os.path.join(HERE, args.model)
        model, meta = load_model(path)
        if meta.get("features") == "f11":
            data = feature_pack(data)
        if args.boundary_refined and getattr(model, "bhead", None) is None:
            raise SystemExit("--boundary-refined needs a checkpoint with a "
                             "boundary head")
        Pv = model_probs(model, data["val"]["X"], meta["num_classes"], meta["refine"])
        Pt = model_probs(model, data["test"]["X"], meta["num_classes"], meta["refine"])
        if args.boundary_refined:
            Bv = model_boundary_probs(model, data["val"]["X"])
            Bt = model_boundary_probs(model, data["test"]["X"])
            Pv_l = segment_majority(Pv.argmax(-1), Bv, thresh=args.boundary_thresh)
            Pt_l = segment_majority(Pt.argmax(-1), Bt, thresh=args.boundary_thresh)
        else:
            Pv_l, Pt_l = Pv.argmax(-1), Pt.argmax(-1)
        val = eval_from_preds(Pv_l, data["val"]["y6"], data["val"]["y_trans"])
        test = eval_from_preds(Pt_l, data["test"]["y6"], data["test"]["y_trans"])
        append_result(res, dict(
            name=PREFIX + args.name, kind="test_confirmation",
            config=dict(base_model=args.model, base_tag=meta["tag"],
                        boundary_refined=bool(args.boundary_refined),
                        boundary_thresh=args.boundary_thresh,
                        base_config_note="single test evaluation of a val-selected candidate"),
            params=None, best_epoch=meta.get("best_epoch"), val=val, test=test,
            history=[], train_seconds=0.0))
    elif args.cmd == "ensemble":
        files = [s.strip() for s in args.models.split(",")]
        loaded = [load_model(os.path.join(HERE, fn)) for fn in files]
        metas = [m for _, m in loaded]
        if any(m.get("features") == "f11" for m in metas):
            assert all(m.get("features") == "f11" for m in metas), \
                "ensemble members must share the feature pack"
            data = feature_pack(data)
        Pv, Pt = None, None
        for model, meta in loaded:
            pv = model_probs(model, data["val"]["X"], meta["num_classes"], meta["refine"])
            pt = model_probs(model, data["test"]["X"], meta["num_classes"], meta["refine"])
            Pv = pv if Pv is None else Pv + pv
            Pt = pt if Pt is None else Pt + pt
        pp_cfg = None
        if args.postproc_from:
            src = next((e for e in res["experiments"]
                        if e["name"] == args.postproc_from), None)
            if src is None:
                print(f"postproc source {args.postproc_from} not found", flush=True)
                return
            pp_cfg = src["config"]["selected"]
        if pp_cfg:
            Pv_l = postproc_labels(Pv, pp_cfg["sigma"], pp_cfg["theta"], pp_cfg["median"])
            Pt_l = postproc_labels(Pt, pp_cfg["sigma"], pp_cfg["theta"], pp_cfg["median"])
        else:
            Pv_l, Pt_l = Pv.argmax(-1), Pt.argmax(-1)
        val = eval_from_preds(Pv_l, data["val"]["y6"], data["val"]["y_trans"])
        test = eval_from_preds(Pt_l, data["test"]["y6"], data["test"]["y_trans"])
        append_result(res, dict(
            name=PREFIX + args.name, kind="ensemble_prob_average",
            config=dict(members=files, member_tags=[m["tag"] for m in metas],
                        postproc=pp_cfg),
            params=None, best_epoch=None, val=val, test=test,
            history=[], train_seconds=0.0))


if __name__ == "__main__":
    main()
