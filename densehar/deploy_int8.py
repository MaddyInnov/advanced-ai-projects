"""Deployment sprint step 2: does INT8 hurt OUR architecture?

Dynamic (weight-only) INT8 quantization of the ONNX champions, then the
FULL HAPT val pipeline (raw + bref + bref/mindur10) through the INT8
models vs FP32. The literature is split on PTQ risk (-9.3 F1 in one
study, -0.21 in another) — so we measure, not assume.
"""
import json
import os
import sys

import numpy as np
import torch  # noqa: F401  (train_real needs it)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_real as tr  # noqa: E402
from onnxruntime.quantization import quantize_dynamic, QuantType  # noqa: E402
import onnxruntime as ort  # noqa: E402


def probs_from_onnx(path, X, bs=32):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])
    L, B = [], []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].transpose(0, 2, 1).astype(np.float32)
        l, b = sess.run(None, {"x": x})
        L.append(torch.log_softmax(torch.tensor(l), dim=-1).numpy())
        B.append(1 / (1 + np.exp(-b[..., 0])))
    P12 = np.exp(np.concatenate(L, axis=0))
    P6 = np.zeros(P12.shape[:2] + (6,), dtype=np.float64)
    for c12 in range(12):
        P6[..., tr.MAP12_TO_6[c12]] += P12[..., c12]
    return P6, np.concatenate(B, axis=0)


def pipeline(P6, B, Y, T, tag):
    raw = P6.argmax(-1)
    bref = tr.segment_majority(raw, B, thresh=0.5)
    md = tr.min_duration_relabel(bref, 10)
    out = {}
    for nm, P in (("raw", raw), ("bref", bref), ("bref_mindur10", md)):
        r = tr.eval_from_preds(P, Y, T)
        out[nm] = {k: r[k] for k in ("accuracy", "macro_f1", "boundary_f1",
                                     "segmental_f1@25", "edit_score")}
        print("%s %-14s acc %.4f F1 %.4f bF1 %.4f seg25 %.4f edit %.1f" %
              (tag, nm, r["accuracy"], r["macro_f1"], r["boundary_f1"],
               r["segmental_f1@25"], r["edit_score"]))
    return out


def main():
    data = tr.load_data()
    Xv, Yv, Tv = data["val"]["X"], data["val"]["y6"], data["val"]["y_trans"]
    results = {}
    for name in ("champion_bh10", "causal_bh10"):
        fp = os.path.join(HERE, f"deploy_{name}.onnx")
        ip = os.path.join(HERE, f"deploy_{name}_int8dyn.onnx")
        quantize_dynamic(fp, ip, weight_type=QuantType.QInt8)
        print(f"{name}: FP32 {os.path.getsize(fp)//1024} KB -> "
              f"INT8dyn {os.path.getsize(ip)//1024} KB")
        P6f, Bf = probs_from_onnx(fp, Xv)
        P6i, Bi = probs_from_onnx(ip, Xv)
        results[name] = dict(
            size_kb=dict(fp32=os.path.getsize(fp) // 1024,
                         int8dyn=os.path.getsize(ip) // 1024),
            fp32=pipeline(P6f, Bf, Yv, Tv, name + " FP32 "),
            int8dyn=pipeline(P6i, Bi, Yv, Tv, name + " INT8 "))
    json.dump(results, open(os.path.join(HERE, "deploy_int8.json"), "w"), indent=1)
    print("saved deploy_int8.json")


if __name__ == "__main__":
    main()
