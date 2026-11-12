"""Try ONNX export for mobile deployment; fall back to a note on failure."""
import os, torch
from model import DenseHARNet, count_params
here=os.path.dirname(__file__)
def main():
    try:
        import json as _json
        _cfg=_json.load(open(os.path.join(here,"best_config.json"))) if os.path.exists(os.path.join(here,"best_config.json")) else {"width":48}
        m=DenseHARNet(width=_cfg["width"])
        ckpt=os.path.join(here,"densehar.pt")
        if os.path.exists(ckpt): m.load_state_dict(torch.load(ckpt,map_location="cpu"))
        m.eval()
        out=os.path.join(here,"densehar.onnx")
        torch.onnx.export(m, torch.randn(1,6,512), out, input_names=["imu"], output_names=["logits"],
                          dynamic_axes={"imu":{0:"batch",2:"time"},"logits":{0:"batch",1:"time"}}, opset_version=14, dynamo=False)
        print(f"ONNX exported: {out} ({os.path.getsize(out)/1024:.1f} KB), params {count_params(m):,}")
    except Exception as e:
        with open(os.path.join(here,"ONNX_EXPORT_NOTE.md"),"w") as f:
            f.write(f"# ONNX export failed\n\n`{type(e).__name__}: {e}`\n\nFallback: use PyTorch Mobile / ExecuTorch, or export a fixed-length (1,6,512) graph without dynamic axes.\n")
        print(f"ONNX export failed, note written: {e}")
if __name__=="__main__":
    main()
