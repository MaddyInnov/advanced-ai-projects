"""Dense + segmental metrics (MS-TCN style), pure numpy. No sklearn."""
import numpy as np

def macro_f1(y_true, y_pred, C=6):
    f1s=[]
    for c in range(C):
        tp=int(((y_true==c)&(y_pred==c)).sum()); fp=int(((y_true!=c)&(y_pred==c)).sum()); fn=int(((y_true==c)&(y_pred!=c)).sum())
        p=tp/max(tp+fp,1); r=tp/max(tp+fn,1); f1s.append(2*p*r/max(p+r,1e-9))
    return float(np.mean(f1s)), [float(x) for x in f1s]

def change_points(a): return np.where(np.diff(a)!=0)[0]+1

def segments(a):
    segs=[]; s=0
    for i in range(1,len(a)+1):
        if i==len(a) or a[i]!=a[s]: segs.append((int(a[s]),s,i-1)); s=i
    return segs

def _lev(a,b):
    # Levenshtein on label lists
    dp=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        prev=dp[0]; dp[0]=i
        for j,cb in enumerate(b,1):
            cur=dp[j]; dp[j]=min(dp[j]+1, dp[j-1]+1, prev+(ca!=cb)); prev=cur
    return dp[-1]

def segmental_metrics(Y, P, taus=(10,25,50)):
    """F1@tau: pred segment is TP if IoU with an unmatched GT segment of same label > tau/100.
    Edit score: (1 - lev/max_len)*100 on collapsed segment label sequences, averaged."""
    f1={t: [0,0,0] for t in taus}  # tp,fp,fn
    edit_scores=[]; edit_dists=[]
    for b in range(len(Y)):
        gt=segments(Y[b]); pr=segments(P[b])
        for t in taus:
            matched=set(); tp=fp=0
            for (lab,s,e) in pr:
                best=-1; best_i=-1
                for i,(gl,gs,ge) in enumerate(gt):
                    if i in matched or gl!=lab: continue
                    inter=max(0,min(e,ge)-max(s,gs)+1); union=(e-s+1)+(ge-gs+1)-inter
                    iou=inter/max(union,1)
                    if iou>best: best, best_i = iou, i
                if best>t/100: tp+=1; matched.add(best_i)
                else: fp+=1
            f1[t][0]+=tp; f1[t][1]+=fp; f1[t][2]+=len(gt)-len(matched)
        glabels=[s[0] for s in gt]; plabels=[s[0] for s in pr]
        d=_lev(plabels,glabels); edit_dists.append(d)
        edit_scores.append((1-d/max(len(plabels),len(glabels),1))*100)
    out={}
    for t,(tp,fp,fn) in f1.items():
        p=tp/max(tp+fp,1); r=tp/max(tp+fn,1)
        out[f"segmental_f1@{t}"]=float(2*p*r/max(p+r,1e-9))
    out["edit_score"]=float(np.mean(edit_scores)); out["edit_distance_mean"]=float(np.mean(edit_dists))
    return out

def evaluate_dense(predict_fn, X, Y):
    preds=[]; bs=32
    for i in range(0,len(X),bs): preds.append(predict_fn(X[i:i+bs]))
    P=np.concatenate(preds,axis=0)
    acc=float((P==Y).mean()); mf1,per=macro_f1(Y.ravel(),P.ravel())
    mask=np.zeros_like(Y,dtype=bool)
    for b in range(len(Y)):
        for cp in change_points(Y[b]): mask[b,max(0,cp-10):min(Y.shape[1],cp+11)]=True
    trans_acc=float((P[mask]==Y[mask]).mean()) if mask.any() else 0.0
    tp=fp=fn=0
    for b in range(len(Y)):
        tcps=change_points(Y[b]); matched=set()
        for pc in change_points(P[b]):
            hit=[i for i,tc in enumerate(tcps) if abs(int(pc)-int(tc))<=10 and i not in matched]
            if hit: tp+=1; matched.add(hit[0])
            else: fp+=1
        fn+=len(tcps)-len(matched)
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); bf1=2*prec*rec/max(prec+rec,1e-9)
    res=dict(accuracy=acc, macro_f1=mf1, per_class_f1=per, transition_window_accuracy=trans_acc,
             transition_timesteps=int(mask.sum()), boundary_f1=float(bf1), boundary_precision=float(prec),
             boundary_recall=float(rec), boundary_tp=int(tp), boundary_fp=int(fp), boundary_fn=int(fn))
    res.update(segmental_metrics(Y,P))
    return res
