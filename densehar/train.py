"""DenseHAR training with SOTA protocol:
- Validation split = held-out subjects 10-11 from train file (train subjects 0-9 for tuning)
- Sweep: 3 configs (width / boundary-weight) x 6 epochs on validation
- Final: best config retrained on ALL train subjects, 12 epochs, evaluated on unseen test subjects 12-15
- Ablation on test: tuning-run plain vs boundary models (same width/epochs, sub-train only)
- Baseline: window-128 classifier broadcast to dense, same metrics incl. segmental F1@10/25/50 + edit score
"""
import os, json, time, numpy as np, torch, torch.nn as nn
from model import DenseHARNet, WindowClassifier, count_params
from metrics import evaluate_dense

HERE=os.path.dirname(__file__); torch.manual_seed(0); np.random.seed(0)
try: torch.set_num_threads(max(1,(os.cpu_count() or 2)))
except Exception: pass

def load(split):
    d=np.load(os.path.join(HERE,"data",f"{split}.npz")); return d["X"], d["y"], d["subjects"]

def boundary_weights(y, extra=3.0):
    B,L=y.shape; w=np.ones((B,L),dtype=np.float32); idx=np.arange(L)
    for b in range(B):
        bounds=np.where(np.diff(y[b])!=0)[0]+1
        if len(bounds): w[b]=1.0+extra*np.exp(-np.min(np.abs(idx[None,:]-bounds[:,None]),axis=0)/10.0)
    return torch.tensor(w)

def train_dense(width, use_weights, epochs, Xtr, Ytr, seed=0, lr=3e-3):
    torch.manual_seed(seed)
    model=DenseHARNet(width=width); opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    Xt=torch.tensor(Xtr).permute(0,2,1); Yt=torch.tensor(Ytr)
    Wt=boundary_weights(Ytr) if use_weights else torch.ones_like(Yt,dtype=torch.float32)
    crit=nn.CrossEntropyLoss(reduction="none")
    for ep in range(epochs):
        model.train(); perm=torch.randperm(len(Xt)); tot=0.0; nb=0
        for i in range(0,len(Xt),16):
            idx=perm[i:i+32]; opt.zero_grad()
            out=model(Xt[idx]); loss=(crit(out.reshape(-1,6),Yt[idx].reshape(-1))*Wt[idx].reshape(-1)).mean()
            if use_weights:  # MS-TCN-style smoothing loss (transition-aware package) reduces over-segmentation
                logp=torch.log_softmax(out,dim=-1); smooth=torch.clamp((logp[:,1:]-logp[:,:-1])**2,0,4).mean()
                loss=loss+0.15*smooth
            loss.backward(); opt.step(); tot+=loss.item(); nb+=1
        sched.step(); print(f"  w{width} bw={use_weights} ep{ep+1}/{epochs} loss {tot/max(nb,1):.4f}",flush=True)
    return model

def train_baseline(epochs, Xtr, Ytr):
    wins=[]; labs=[]
    for i in range(len(Xtr)):
        for s in range(0,512,128):
            wins.append(Xtr[i,s:s+128]); labs.append(np.bincount(Ytr[i,s:s+128],minlength=6).argmax())
    wins=np.stack(wins); labs=np.array(labs)
    torch.manual_seed(0); model=WindowClassifier(); opt=torch.optim.Adam(model.parameters(),lr=2e-3,weight_decay=1e-4)
    Xt=torch.tensor(wins).permute(0,2,1); Yt=torch.tensor(labs); crit=nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); perm=torch.randperm(len(Xt)); tot=0.0; nb=0
        for i in range(0,len(Xt),64):
            idx=perm[i:i+32]; opt.zero_grad(); loss=crit(model(Xt[idx]),Yt[idx]); loss.backward(); opt.step(); tot+=loss.item(); nb+=1
        print(f"  baseline ep{ep+1}/{epochs} loss {tot/max(nb,1):.4f}",flush=True)
    return model

def dense_fn(m):
    def fn(xb):
        m.eval()
        with torch.no_grad(): return m(torch.tensor(xb).permute(0,2,1)).argmax(-1).numpy()
    return fn

def _save_json(obj, name):
    with open(os.path.join(HERE,name),"w") as f: json.dump(obj,f,indent=2)

def stage_tuning():
    X,Y,S=load("train"); sub=S<10; val=S>=10
    Xsub,Ysub,Xval,Yval=X[sub],Y[sub],X[val],Y[val]
    print(f"sub-train {Xsub.shape} val {Xval.shape}",flush=True)
    configs=[
        dict(name="A_width48_boundary", width=48, boundary_weighted=True, epochs=6),
        dict(name="B_width48_plain",    width=48, boundary_weighted=False, epochs=6),
        dict(name="C_width32_boundary", width=32, boundary_weighted=True, epochs=6),
    ]
    tuning=[]
    if os.path.exists(os.path.join(HERE,"tuning.json")):
        tuning=json.load(open(os.path.join(HERE,"tuning.json")))
    done={r["name"] for r in tuning}
    for cfg in configs:
        if cfg["name"] in done:
            print(f"skip {cfg['name']} (saved)",flush=True); continue
        print(f"== Tuning {cfg['name']} ==",flush=True)
        m=train_dense(cfg["width"], cfg["boundary_weighted"], cfg["epochs"], Xsub, Ysub)
        v=evaluate_dense(dense_fn(m), Xval, Yval)
        torch.save(m.state_dict(), os.path.join(HERE, f"ckpt_{cfg['name']}.pt"))
        tuning.append({**cfg, "smoothing_lambda":(0.15 if cfg["boundary_weighted"] else 0.0), "params":count_params(m), "val":v})
        _save_json(tuning,"tuning.json")
        print(f"  val acc {v['accuracy']:.4f} macroF1 {v['macro_f1']:.4f} segF1@25 {v['segmental_f1@25']:.4f} edit {v['edit_score']:.1f} [saved]",flush=True)
    return tuning

def stage_final():
    t0=time.time()
    X,Y,S=load("train"); Xte,Yte,_=load("test")
    tuning=json.load(open(os.path.join(HERE,"tuning.json")))
    best=max(tuning, key=lambda r: r["val"]["macro_f1"])
    print(f"BEST by val macro-F1: {best['name']}",flush=True)
    print("== Final on full train, 10 epochs ==",flush=True)
    final=train_dense(best["width"], best["boundary_weighted"], 10, X, Y, seed=1)
    torch.save(final.state_dict(), os.path.join(HERE,"densehar.pt"))
    with open(os.path.join(HERE,"best_config.json"),"w") as f: json.dump({"width":best["width"],"name":best["name"]},f)
    test_final=evaluate_dense(dense_fn(final), Xte, Yte)
    def load_tuned(name,width):
        mm=DenseHARNet(width=width); mm.load_state_dict(torch.load(os.path.join(HERE,f"ckpt_{name}.pt"),map_location="cpu")); return mm
    abl_b=evaluate_dense(dense_fn(load_tuned("A_width48_boundary",48)), Xte, Yte)
    abl_p=evaluate_dense(dense_fn(load_tuned("B_width48_plain",48)), Xte, Yte)
    results={
        "protocol":"SOTA methodology on SYNTHETIC data only: tuning on held-out subjects 10-11, final test on unseen subjects 12-15. No real-benchmark SOTA claim.",
        "params_densehar_best": count_params(final), "params_baseline_window": count_params(WindowClassifier()),
        "tuning_runs": tuning, "best_config": {k:best[k] for k in ("name","width","boundary_weighted","epochs")},
        "final_epochs_full_train": 10, "densehar_final_test": test_final,
        "ablation_test_subtrain6ep_width48_boundary": abl_b,
        "ablation_test_subtrain6ep_width48_plain": abl_p,
        "train_seqs": int(len(X)), "test_seqs": int(len(Xte))}
    _save_json(results,"results.json"); print("results.json written (pre-baseline)",flush=True)
    print("== Baseline window-128 ==",flush=True)
    base=train_baseline(4, X, Y)
    torch.save(base.state_dict(), os.path.join(HERE,"baseline_window.pt"))
    def base_fn(xb):
        base.eval(); out=[]
        with torch.no_grad():
            for seq in xb:
                ps=[int(base(torch.tensor(seq[s:s+128]).T.unsqueeze(0)).argmax(-1)) for s in range(0,512,128)]
                out.append(np.repeat(ps,128))
        return np.stack(out)
    test_base=evaluate_dense(base_fn, Xte, Yte)
    results["baseline_window128_broadcast_test"]=test_base
    results["training_seconds_final_stage"]=round(time.time()-t0,1)
    results["windowed_collapse_at_transitions"]={
        "baseline_overall_accuracy": test_base["accuracy"],
        "baseline_transition_window_accuracy": test_base["transition_window_accuracy"],
        "densehar_transition_window_accuracy": test_final["transition_window_accuracy"],
        "note":"Gap between baseline overall and its transition-window accuracy quantifies windowed collapse at transitions."}
    _save_json(results,"results.json")
    print(json.dumps(results,indent=2),flush=True); print("DONE",flush=True)

def main():
    import sys
    stage=sys.argv[1] if len(sys.argv)>1 else "all"
    if stage in ("all","tuning"): stage_tuning()
    if stage in ("all","final"): stage_final()

if __name__=="__main__":
    main()
