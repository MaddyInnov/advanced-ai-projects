"""Deployment sprint step 1: export DenseHAR champions to ONNX, verify
numerical parity vs PyTorch, and measure batch-1 CPU latency.

Models: the offline segmental champion (refine + BRB, 170,133 params)
and the causal streaming model (same size) — the two deployment
candidates. Run with the .venv-deploy interpreter (onnxruntime lives
there); train_real/torch come from system site-packages.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_real as tr  # noqa: E402


class Wrap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):  # x: (B, C, L)
        out = self.model(x)
        if isinstance(out, tuple):
            return out[-2], out[-1]  # refined logits, boundary logits
        return out, torch.zeros(x.shape[0], x.shape[2], 1)


def bench(sess, x, runs=200, warm=20):
    for _ in range(warm):
        sess.run(None, {"x": x})
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"x": x})
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return dict(mean_ms=float(ts.mean()), p50_ms=float(np.percentile(ts, 50)),
                p95_ms=float(np.percentile(ts, 95)))


def export_one(name, ckpt, in_ch, L=512):
    import onnxruntime as ort
    model, meta = tr.load_model(os.path.join(HERE, ckpt))
    model.eval()
    w = Wrap(model)
    dummy = torch.randn(1, in_ch, L)
    path = os.path.join(HERE, f"deploy_{name}.onnx")
    torch.onnx.export(w, dummy, path, input_names=["x"],
                      output_names=["logits", "boundary"],
                      dynamic_axes={"x": {0: "B", 2: "L"},
                                    "logits": {0: "B", 1: "L"},
                                    "boundary": {0: "B", 1: "L"}},
                      opset_version=17, dynamo=False)
    with torch.no_grad():
        ref_l, ref_b = w(dummy)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])
    got_l, got_b = sess.run(None, {"x": dummy.numpy()})
    parity = dict(logits_max_abs_diff=float(np.abs(got_l - ref_l.numpy()).max()),
                  boundary_max_abs_diff=float(np.abs(got_b - ref_b.numpy()).max()))
    size_kb = os.path.getsize(path) / 1024
    lat = {}
    for threads in (1, 4):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        s = ort.InferenceSession(path, sess_options=so,
                                 providers=["CPUExecutionProvider"])
        lat[f"threads{threads}_L512"] = bench(s, dummy.numpy())
        lat[f"threads{threads}_L256"] = bench(s, np.random.randn(1, in_ch, 256).astype(np.float32))
    print(f"{name}: size {size_kb:.0f} KB, parity {parity}")
    for k, v in lat.items():
        print(f"  {k}: mean {v['mean_ms']:.2f} ms p95 {v['p95_ms']:.2f} ms")
    return dict(model=name, ckpt=ckpt, onnx=os.path.basename(path),
                size_kb=round(size_kb), params=meta.get("n_params"),
                parity=parity, latency=lat)


def main():
    data = tr.load_data()
    in_ch = data["train"]["X"].shape[2]
    print("in_ch:", in_ch)
    out = [export_one("champion_bh10", "model_real_p2_aug_c12_refine_bh10.pt", in_ch),
           export_one("causal_bh10", "model_real_p2_aug_c12_causal_bh10.pt", in_ch)]
    json.dump(out, open(os.path.join(HERE, "deploy_bench.json"), "w"), indent=1)
    print("saved deploy_bench.json")


if __name__ == "__main__":
    main()
