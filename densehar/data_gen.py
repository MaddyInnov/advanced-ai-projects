"""Synthetic IMU generator for DenseHAR.

6 activities, 6 channels (acc xyz + gyro xyz), sequences length 512
with 2-4 segments and smoothed transitions. Train/test use disjoint subjects.
"""
import numpy as np, os

ACTIVITIES = ["walking","walking_upstairs","walking_downstairs","sitting","standing","laying"]
FS = 50.0

# activity profile: freq Hz, acc_amp, gyro_amp, gravity (3,), noise
PROFILES = {
    0: dict(freq=2.0, acc_amp=0.35, gyro_amp=0.45, gravity=np.array([0.0,0.0,1.0]), noise=0.05),
    1: dict(freq=1.7, acc_amp=0.42, gyro_amp=0.55, gravity=np.array([0.0,0.2,0.98]), noise=0.06),
    2: dict(freq=2.3, acc_amp=0.40, gyro_amp=0.50, gravity=np.array([0.0,-0.15,0.99]), noise=0.07),
    3: dict(freq=0.0, acc_amp=0.01, gyro_amp=0.02, gravity=np.array([0.0,0.70,0.71]), noise=0.02),
    4: dict(freq=0.0, acc_amp=0.02, gyro_amp=0.03, gravity=np.array([0.0,0.0,1.0]), noise=0.03),
    5: dict(freq=0.0, acc_amp=0.01, gyro_amp=0.015, gravity=np.array([1.0,0.0,0.10]), noise=0.015),
}

def subject_params(rng):
    return dict(amp_scale=rng.uniform(0.85,1.18), freq_scale=rng.uniform(0.9,1.1),
                noise_scale=rng.uniform(0.8,1.3), offset=rng.normal(0,0.03,6),
                phase=rng.uniform(0,2*np.pi))

def gen_segment(act, length, rng, subj, t0=0):
    p = PROFILES[act]; t = np.arange(t0, t0+length)/FS
    sig = np.zeros((length,6))
    grav = p["gravity"]/np.linalg.norm(p["gravity"])
    sig[:,:3] += grav[None,:]  # acc gravity
    if p["freq"]>0:
        f = p["freq"]*subj["freq_scale"]; ph = subj["phase"]
        wave = np.sin(2*np.pi*f*t+ph); wave2 = np.sin(2*np.pi*f*t*2+ph*0.5)
        a = p["acc_amp"]*subj["amp_scale"]
        sig[:,0] += a*wave; sig[:,1] += a*0.6*wave2; sig[:,2] += a*0.8*np.abs(wave)
        g = p["gyro_amp"]*subj["amp_scale"]
        sig[:,3] += g*wave2; sig[:,4] += g*0.7*wave; sig[:,5] += g*0.3*np.sin(2*np.pi*f*t+1.0)
        if act==2:  # downstairs impact spikes
            spikes = (np.sin(2*np.pi*f*t+ph)>0.92).astype(float)*0.3
            sig[:,2] += spikes
    else:
        # static: slow sway / breathing
        sway = 0.02*np.sin(2*np.pi*0.3*t+subj["phase"])
        sig[:,0]+=sway; sig[:,1]+=0.5*sway
    sig += rng.normal(0, p["noise"]*subj["noise_scale"], sig.shape)
    sig += subj["offset"][None,:]
    return sig

def gen_sequence(rng, subj):
    L=512; nseg=int(rng.integers(2,5))
    # partition
    cuts=sorted(rng.choice(np.arange(80, L-80), size=nseg-1, replace=False).tolist())
    bounds=[0]+cuts+[L]
    # activities: ensure consecutive differ
    acts=[]; prev=-1
    for i in range(nseg):
        a=int(rng.integers(0,6))
        while a==prev: a=int(rng.integers(0,6))
        acts.append(a); prev=a
    sig=np.zeros((L,6)); labels=np.zeros(L,dtype=np.int64)
    for i,act in enumerate(acts):
        s,e=bounds[i],bounds[i+1]
        seg=gen_segment(act, e-s, rng, subj, t0=s)
        sig[s:e]=seg; labels[s:e]=act
    # smooth transitions: 8 steps linear blend across boundary using smoothed signal convolution near boundary
    # Apply simple smoothing window around boundaries to mimic realistic transition (signal level)
    for b in cuts:
        lo=max(0,b-4); hi=min(L,b+4); w=np.linspace(0,1,hi-lo)[:,None]
        # blend is already abrupt in gravity; smooth by averaging with linear interpolation between edge values
        left=sig[lo:b].copy(); right=sig[b:hi].copy()
        # create ramp from value at b-1 to value at b
        # Use moving average smoothing in window
        window=sig[lo:hi].copy()
        kernel=np.array([0.25,0.5,0.25])
        for c in range(6):
            sm=np.convolve(np.pad(window[:,c],(1,1),mode='edge'),kernel,mode='valid')
            window[:,c]=sm
        sig[lo:hi]=window
    return sig.astype(np.float32), labels

def generate(n, subjects, seed):
    rng=np.random.default_rng(seed)
    subj_params={s: subject_params(np.random.default_rng(seed+1000+s)) for s in subjects}
    X=[]; Y=[]; S=[]
    for i in range(n):
        subj=int(rng.choice(subjects)); subj_rng=np.random.default_rng(rng.integers(0,1<<31))
        # Use main rng for sequence structure but per-seq rng seeded from it for reproducibility
        x,y=gen_sequence(rng, subj_params[subj])
        X.append(x.T)  # (6,512) store as (512,6)? Keep (N,512,6) for simplicity in file, transpose in loader
        # Actually X currently x is (512,6), x.T wrong; fix: append x
        X[-1]=x
        Y.append(y); S.append(subj)
    return np.stack(X).astype(np.float32), np.stack(Y), np.array(S)

def main():
    out=os.path.join(os.path.dirname(__file__),"data")
    os.makedirs(out, exist_ok=True)
    train_subjects=list(range(0,12)); test_subjects=list(range(12,16))  # unseen
    Xtr,Ytr,Str=generate(300, train_subjects, seed=42)
    Xte,Yte,Ste=generate(100, test_subjects, seed=123)
    np.savez_compressed(os.path.join(out,"train.npz"), X=Xtr, y=Ytr, subjects=Str)
    np.savez_compressed(os.path.join(out,"test.npz"), X=Xte, y=Yte, subjects=Ste)
    print(f"train {Xtr.shape} subjects {sorted(set(Str.tolist()))}")
    print(f"test  {Xte.shape} subjects {sorted(set(Ste.tolist()))}")
    print(f"label dist train {np.bincount(Ytr.ravel(), minlength=6)}")
    print(f"label dist test  {np.bincount(Yte.ravel(), minlength=6)}")

if __name__=="__main__":
    main()
