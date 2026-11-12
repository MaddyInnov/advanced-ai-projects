"""Demo: one synthetic sequence with transitions, true vs predicted timeline (RLE)."""
import os, numpy as np, torch
from model import DenseHARNet
from data_gen import ACTIVITIES

def rle(labels):
    out=[]; start=0
    for i in range(1,len(labels)+1):
        if i==len(labels) or labels[i]!=labels[start]:
            out.append(f"[{start}-{i-1}] {ACTIVITIES[labels[start]]}"); start=i
    return " -> ".join(out)

def main():
    here=os.path.dirname(__file__)
    d=np.load(os.path.join(here,"data","test.npz")); X,y=d["X"],d["y"]
    # pick a sequence with at least 3 segments
    idx=0
    for i in range(len(y)):
        if len(np.unique(y[i]))>=3: idx=i; break
    import json as _json
    _cfg=_json.load(open(os.path.join(here,"best_config.json"))) if os.path.exists(os.path.join(here,"best_config.json")) else {"width":48}
    model=DenseHARNet(width=_cfg["width"]); model.load_state_dict(torch.load(os.path.join(here,"densehar.pt"),map_location="cpu")); model.eval()
    with torch.no_grad():
        pred=model(torch.tensor(X[idx:idx+1]).permute(0,2,1)).argmax(-1).numpy()[0]
    print(f"Test sequence #{idx} (unseen subject {int(d['subjects'][idx])})")
    print("TRUE:", rle(y[idx]))
    print("PRED:", rle(pred))
    print(f"Per-timestep accuracy on this sequence: {(pred==y[idx]).mean()*100:.1f}%")

if __name__=="__main__":
    main()
