"""
Sentinel-LLM — Real-benchmark evaluation (MITRE ATT&CK tactics + CVE severity).
Author: Ashutosh Sahoo (real-data re-run, 2026-11-13)

Reuses the project's own architecture (model.py), from-scratch LoRA (lora.py)
and experimental protocol (train.py: brief base -> full-FT baseline -> LoRA
sweep selected on validation -> test touched once). Only the DATA is real now
(see load_real.py for provenance), plus macro-F1 reporting and, in Phase B,
principled improvement levers screened on a validation split carved from TRAIN.

Phases:
  baseline : per task — untrained / majority / TF-IDF-LR reference / brief base /
             full FT / LoRA sweep (val-selected) -> ONE test evaluation.
  phaseb   : candidate levers screened on inner-val (15% carved from train,
             seed 777). Test is NOT touched.
  final    : the valB-selected recipe is retrained on the full train split and
             evaluated on test exactly once (LoRA + full-FT under that recipe).

Usage:
  python eval_real.py baseline --task mitre|cve
  python eval_real.py phaseb   --task mitre|cve
  python eval_real.py final    --task mitre|cve
"""
import argparse
import copy
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict

import torch
import torch.nn as nn

torch.set_num_threads(4)

from model import SentinelClassifier, encode, tokenize, hash_token, VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM
from lora import inject_lora, unfreeze_lora_only, count_parameters
from train import Split, train_epochs  # the project's own split container + trainer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, "results_real.json")
BATCH = 32

# ---------------------------------------------------------------- metrics
def metrics_from_preds(preds, golds, labels):
    """Accuracy + macro F1 + per-class F1."""
    n = len(golds)
    acc = sum(int(p == g) for p, g in zip(preds, golds)) / n if n else 0.0
    f1s = {}
    for li, lab in enumerate(labels):
        tp = sum(1 for p, g in zip(preds, golds) if p == li and g == li)
        fp = sum(1 for p, g in zip(preds, golds) if p == li and g != li)
        fn = sum(1 for p, g in zip(preds, golds) if p != li and g == li)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s[lab] = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0
    macro = round(sum(f1s.values()) / len(f1s), 4) if f1s else 0.0
    return {"accuracy": round(acc, 4), "macro_f1": macro, "per_class_f1": f1s, "n": n}


def predict(model, split):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, split.n, 64):
            idxs = list(range(i, min(i + 64, split.n)))
            x, m, y = split.batch(idxs)
            preds.extend(model(x, m).argmax(-1).tolist())
    return preds


def eval_full(model, split, labels):
    preds = predict(model, split)
    out = metrics_from_preds(preds, split.ys, labels)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    model.eval(); total = 0.0
    with torch.no_grad():
        for i in range(0, split.n, 64):
            idxs = list(range(i, min(i + 64, split.n)))
            x, m, y = split.batch(idxs)
            total += loss_fn(model(x, m), y).item()
    out["loss"] = round(total / split.n, 4)
    return out


# ---------------------------------------------------------------- data
def load_task_rows(task):
    path = os.path.join(BASE_DIR, f"dataset_real_{task}.jsonl")
    return [json.loads(l) for l in open(path)]


def stratified_split(rows, seed=123):
    """Exactly train.py's stratified 70/15/15."""
    labels = sorted(set(r["label"] for r in rows))
    label2id = {l: i for i, l in enumerate(labels)}
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    tr, va, te = [], [], []
    rng = random.Random(seed)
    for lab, items in by_label.items():
        rng.shuffle(items)
        c1, c2 = int(len(items) * 0.70), int(len(items) * 0.85)
        tr.extend(items[:c1]); va.extend(items[c1:c2]); te.extend(items[c2:])
    for s in (tr, va, te):
        rng.shuffle(s)
    return tr, va, te, labels, label2id


def era_split(rows):
    """CVE temporal split: eras are the split (load_real.py sampling rule)."""
    labels = ["CRITICAL", "HIGH", "LOW", "MEDIUM"]  # sorted(set(...)) equivalent
    labels = sorted(set(r["label"] for r in rows))
    label2id = {l: i for i, l in enumerate(labels)}
    tr = [r for r in rows if r["split_era"] == "train_2020_2022"]
    va = [r for r in rows if r["split_era"] == "val_2023"]
    te = [r for r in rows if r["split_era"] == "test_2024"]
    rng = random.Random(123)
    for s in (tr, va, te):
        rng.shuffle(s)
    return tr, va, te, labels, label2id


def get_splits(task):
    rows = load_task_rows(task)
    if task == "mitre":
        tr, va, te, labels, l2i = stratified_split(rows)
        split_desc = "stratified 70/15/15 by tactic, seed 123 (same rule as train.py)"
    else:
        tr, va, te, labels, l2i = era_split(rows)
        split_desc = "TEMPORAL: train = CVE 2020-2022, val = 2023, test = 2024 (era = split)"
    return rows, tr, va, te, labels, l2i, split_desc


def save_results(update_fn):
    data = {}
    if os.path.exists(RESULTS_PATH):
        data = json.load(open(RESULTS_PATH))
    update_fn(data)
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return data


# ---------------------------------------------------------------- TF-IDF LR reference
def tfidf_lr_reference(task, tr, va, te):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    texts_tr = [r["instruction"] + " " + r["input"] for r in tr]
    texts_va = [r["instruction"] + " " + r["input"] for r in va]
    texts_te = [r["instruction"] + " " + r["input"] for r in te]
    labels = sorted(set(r["label"] for r in tr))
    l2i = {l: i for i, l in enumerate(labels)}
    y_tr = [l2i[r["label"]] for r in tr]; y_va = [l2i[r["label"]] for r in va]; y_te = [l2i[r["label"]] for r in te]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=60000)
    X_tr = vec.fit_transform(texts_tr); X_va = vec.transform(texts_va); X_te = vec.transform(texts_te)
    best = None
    for C in (1.0, 4.0):
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(X_tr, y_tr)
        m = metrics_from_preds(clf.predict(X_va).tolist(), y_va, labels)
        if best is None or m["macro_f1"] > best[1]["macro_f1"]:
            best = (C, m, clf)
    C, val_m, clf = best
    test_m = metrics_from_preds(clf.predict(X_te).tolist(), y_te, labels)
    return {"model": "TF-IDF(word 1-2g, sublinear) + multinomial logistic regression",
            "C_selected_on_val": C, "val": val_m, "test": test_m}


# ---------------------------------------------------------------- baseline
def run_baseline(task):
    t0 = time.time()
    rows, tr, va, te, labels, l2i, split_desc = get_splits(task)
    print(f"[{task}] rows={len(rows)} train={len(tr)} val={len(va)} test={len(te)} labels={len(labels)}")
    print(f"[{task}] split: {split_desc}")
    trS, vaS, teS = Split(tr, l2i), Split(va, l2i), Split(te, l2i)
    result = {
        "dataset_rows": len(rows), "train_size": len(tr), "val_size": len(va), "test_size": len(te),
        "labels": labels, "split": split_desc,
        "train_class_distribution": dict(Counter(r["label"] for r in tr).most_common()),
        "val_class_distribution": dict(Counter(r["label"] for r in va).most_common()),
        "test_class_distribution": dict(Counter(r["label"] for r in te).most_common()),
    }

    # untrained + majority references (test)
    torch.manual_seed(42); random.seed(42)
    base_model = SentinelClassifier(num_labels=len(labels))
    result["untrained_test"] = eval_full(base_model, teS, labels)
    maj = Counter(r["label"] for r in tr).most_common(1)[0][0]
    result["majority_class_test"] = metrics_from_preds([l2i[maj]] * len(te), [l2i[r["label"]] for r in te], labels)
    result["majority_class"] = maj
    print(f"[{task}] untrained test acc={result['untrained_test']['accuracy']} majority acc={result['majority_class_test']['accuracy']}")

    # TF-IDF LR reference (val-selected C, test once)
    print(f"[{task}] TF-IDF LR reference...")
    result["tfidf_lr_reference"] = tfidf_lr_reference(task, tr, va, te)
    print(f"[{task}] TF-IDF LR test: {result['tfidf_lr_reference']['test']}")

    # Stage A: brief base (1 epoch, all params) — identical to train.py
    print(f"[{task}] Stage A brief base...")
    t = time.time()
    train_epochs(base_model, trS, epochs=1, lr=3e-3)
    result["base_training_time_sec"] = round(time.time() - t, 1)
    result["brief_base_test"] = eval_full(base_model, teS, labels)
    result["brief_base_val"] = eval_full(base_model, vaS, labels)
    base_state = copy.deepcopy(base_model.state_dict())
    result["total_params_base"] = count_parameters(base_model)["total"]
    print(f"[{task}] base test: {result['brief_base_test']}")

    # Stage B: full fine-tuning (8 epochs) — identical to train.py
    print(f"[{task}] Stage B full FT...")
    ft = SentinelClassifier(num_labels=len(labels)); ft.load_state_dict(copy.deepcopy(base_state))
    t = time.time()
    train_epochs(ft, trS, epochs=8, lr=3e-3)
    result["full_ft_training_time_sec"] = round(time.time() - t, 1)
    ft_stats = count_parameters(ft)
    result["full_ft"] = {"trainable_params": ft_stats["trainable"], "epochs": 8, "lr": 3e-3,
                         "val": eval_full(ft, vaS, labels), "test": eval_full(ft, teS, labels)}
    print(f"[{task}] full FT test: {result['full_ft']['test']}")

    # Stage C: LoRA sweep selected on val — identical configs to train.py
    print(f"[{task}] Stage C LoRA sweep...")
    sweep_cfgs = [{"r": 4, "alpha": 8, "lr": 8e-3}, {"r": 8, "alpha": 16, "lr": 8e-3},
                  {"r": 16, "alpha": 32, "lr": 8e-3}, {"r": 8, "alpha": 16, "lr": 3e-3}]
    sweep, best = [], None
    for cfg in sweep_cfgs:
        torch.manual_seed(7); random.seed(7)
        m = SentinelClassifier(num_labels=len(labels)); m.load_state_dict(copy.deepcopy(base_state))
        inject_lora(m, target_suffixes=("fc1", "fc2"), r=cfg["r"], alpha=cfg["alpha"], dropout=0.05)
        unfreeze_lora_only(m)
        stats = count_parameters(m)
        t = time.time()
        train_epochs(m, trS, epochs=10, lr=cfg["lr"], seed=7, verbose=False)
        el = round(time.time() - t, 1)
        val_m = eval_full(m, vaS, labels)
        entry = {**cfg, "epochs": 10, "trainable_params": stats["trainable"],
                 "trainable_pct": stats["trainable_pct"], "val_accuracy": val_m["accuracy"],
                 "val_macro_f1": val_m["macro_f1"], "val_loss": val_m["loss"], "training_time_sec": el}
        sweep.append(entry)
        print(f"[{task}]   LoRA r={cfg['r']} lr={cfg['lr']}: val acc={val_m['accuracy']} f1={val_m['macro_f1']} ({el}s)")
        key = (val_m["accuracy"], -val_m["loss"], -stats["trainable"])
        if best is None or key > best[0]:
            best = (key, cfg, m, stats, val_m)
    _, bcfg, bmodel, bstats, bval = best
    # Stage D: test touched once for the selected LoRA config
    lora_test = eval_full(bmodel, teS, labels)
    result["lora_sweep"] = sweep
    result["lora_best"] = {"config": bcfg, "trainable_params": bstats["trainable"],
                           "trainable_pct": bstats["trainable_pct"], "total_params": bstats["total"],
                           "val": bval, "test": lora_test}
    result["protocol"] = ("train.py protocol: brief base 1 epoch -> full FT 8 epochs; LoRA sweep "
                          "4 configs x 10 epochs selected on val (acc, tie: lower val loss, fewer params); "
                          "test evaluated once per finished model")
    result["total_time_sec"] = round(time.time() - t0, 1)
    print(f"[{task}] LoRA best r={bcfg['r']}: test {lora_test}")

    def upd(data):
        data.setdefault("tasks", {})[task] = {**data.get("tasks", {}).get(task, {}), "baseline": result}
    save_results(upd)
    print(f"[{task}] baseline written to results_real.json ({result['total_time_sec']}s)")


# ================================================================ Phase B
CHAR_VOCAB = 8192

def encode_hybrid(text, vocab_size=CHAR_VOCAB):
    """Word tokens + char 3-5-grams of each word, hashed into one vocab."""
    ids = [hash_token(t, vocab_size) for t in tokenize(text)]
    for tok in tokenize(text):
        w = f"#{tok}#"
        for n in (3, 4, 5):
            for i in range(len(w) - n + 1):
                ids.append(hash_token("c:" + w[i:i + n], vocab_size))
    return ids or [hash_token("<empty>", vocab_size)]


class SplitB(Split):
    """Split with pluggable encoder + optional IDF-weighted pooling mask."""
    def __init__(self, rows, label2id, mode="word", idf=None):
        self.rows = rows
        self.mode = mode
        self.idf = idf
        if mode == "wordchar":
            self.enc = [encode_hybrid(r["instruction"] + " " + r["input"]) for r in rows]
        else:
            self.enc = [encode(r["instruction"] + " " + r["input"]) for r in rows]
        self.ys = [label2id[r["label"]] for r in rows]
        self.tasks = [r["task"] for r in rows]
        self.n = len(rows)

    def batch(self, idxs):
        batch = [self.enc[i] for i in idxs]
        maxlen = max(len(b) for b in batch)
        x = torch.zeros(len(batch), maxlen, dtype=torch.long)
        m = torch.zeros(len(batch), maxlen, dtype=torch.float)
        for i, b in enumerate(batch):
            x[i, :len(b)] = torch.tensor(b, dtype=torch.long)
            if self.idf is not None:
                m[i, :len(b)] = torch.tensor([self.idf.get(t, 1.0) for t in b], dtype=torch.float)
            else:
                m[i, :len(b)] = 1.0
        y = torch.tensor([self.ys[i] for i in idxs], dtype=torch.long)
        return x, m, y


def compute_idf(rows, mode):
    df = Counter()
    for r in rows:
        enc = encode_hybrid(r["instruction"] + " " + r["input"]) if mode == "wordchar" else encode(r["instruction"] + " " + r["input"])
        for t in set(enc):
            df[t] += 1
    N = len(rows)
    return {t: math.log((N + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def train_epochs_es(model, split, epochs, lr, val_split, labels, seed=42, class_weight=None,
                    label_smoothing=0.0, ordinal_lambda=0.0, label_order=None, patience=4):
    """Like train_epochs2 but with early stopping on val macro-F1; returns (model, best_val_metrics, epochs_run)."""
    torch.manual_seed(seed); random.seed(seed)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = torch.tensor(class_weight, dtype=torch.float) if class_weight else None
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    best_state, best_m, bad = None, None, 0
    for ep in range(epochs):
        order = list(range(split.n)); random.shuffle(order)
        for i in range(0, split.n, BATCH):
            idxs = order[i:i + BATCH]
            x, m, y = split.batch(idxs)
            logits = model(x, m)
            loss = loss_fn(logits, y)
            if ordinal_lambda > 0 and label_order is not None:
                probs = torch.softmax(logits, dim=-1)
                pos = torch.tensor([label_order[int(t)] for t in y], dtype=torch.float)
                exp_pos = probs @ torch.arange(logits.size(1), dtype=torch.float)
                loss = loss + ordinal_lambda * torch.mean(torch.abs(exp_pos - pos))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        vm = eval_full(model, val_split, labels)
        if best_m is None or vm["macro_f1"] > best_m["macro_f1"]:
            best_m, best_state, bad = vm, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
        model.train()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_m, ep + 1


def train_plain(model, split, epochs, lr, seed=42, class_weight=None, label_smoothing=0.0,
                ordinal_lambda=0.0, label_order=None):
    """Fixed-epoch trainer supporting class weights / label smoothing / ordinal loss
    (train.py's train_epochs supports none of these; semantics otherwise identical)."""
    torch.manual_seed(seed); random.seed(seed)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = torch.tensor(class_weight, dtype=torch.float) if class_weight else None
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    for ep in range(epochs):
        order = list(range(split.n)); random.shuffle(order)
        for i in range(0, split.n, BATCH):
            idxs = order[i:i + BATCH]
            x, m, y = split.batch(idxs)
            logits = model(x, m)
            loss = loss_fn(logits, y)
            if ordinal_lambda > 0 and label_order is not None:
                probs = torch.softmax(logits, dim=-1)
                pos = torch.tensor([label_order[int(t)] for t in y], dtype=torch.float)
                exp_pos = probs @ torch.arange(logits.size(1), dtype=torch.float)
                loss = loss + ordinal_lambda * torch.mean(torch.abs(exp_pos - pos))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


def inner_split(tr_rows, seed=777):
    """Carve inner-val (15%) from TRAIN, stratified."""
    by = defaultdict(list)
    for r in tr_rows:
        by[r["label"]].append(r)
    rng = random.Random(seed)
    sub, iv = [], []
    for lab, items in by.items():
        items = list(items); rng.shuffle(items)
        cut = int(len(items) * 0.85)
        sub.extend(items[:cut]); iv.extend(items[cut:])
    rng.shuffle(sub); rng.shuffle(iv)
    return sub, iv


def make_model(num_labels, mode):
    if mode == "wordchar":
        return SentinelClassifier(num_labels=num_labels, vocab_size=CHAR_VOCAB)
    return SentinelClassifier(num_labels=num_labels)


def brief_base(subS, mode, num_labels):
    torch.manual_seed(42); random.seed(42)
    m = make_model(num_labels, mode)
    train_epochs(m, subS, epochs=1, lr=3e-3, verbose=False)
    return copy.deepcopy(m.state_dict())


def lora_from(base_state, mode, num_labels, r=8, alpha=16):
    m = make_model(num_labels, mode)
    m.load_state_dict(copy.deepcopy(base_state))
    inject_lora(m, target_suffixes=("fc1", "fc2"), r=r, alpha=alpha, dropout=0.05)
    unfreeze_lora_only(m)
    return m


def run_phaseb(task):
    t0 = time.time()
    rows, tr, va, te, labels, l2i, split_desc = get_splits(task)
    sub, iv = inner_split(tr)
    print(f"[phaseB {task}] subtrain={len(sub)} inner-val={len(iv)} (carved from train; test untouched)")
    attempts = []

    def record(name, desc, val_m, extra=None):
        e = {"candidate": name, "description": desc,
             "val_accuracy": val_m["accuracy"], "val_macro_f1": val_m["macro_f1"]}
        if extra:
            e.update(extra)
        attempts.append(e)
        print(f"[phaseB {task}] {name}: inner-val acc={val_m['accuracy']} macroF1={val_m['macro_f1']}")

    order_idx = None
    if task == "cve":
        order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        order_idx = {l2i[l]: i for i, l in enumerate(order)}

    # Pre-build splits per mode
    splits = {}
    bases = {}
    for mode in ("word", "wordchar"):
        idf = compute_idf(sub, mode) if mode else None
        splits[(mode, False)] = (SplitB(sub, l2i, mode), SplitB(iv, l2i, mode))
        splits[(mode, True)] = (SplitB(sub, l2i, mode, idf), SplitB(iv, l2i, mode, idf))
        bases[mode] = brief_base(splits[(mode, False)][0], mode, len(labels))

    def screen(name, desc, mode, use_idf, epochs=10, lr=8e-3, cw=False, ls=0.0, ol=0.0, es=False):
        s_tr, s_iv = splits[(mode, use_idf)]
        m = lora_from(bases[mode], mode, len(labels))
        class_weight = None
        if cw:
            freq = Counter(r["label"] for r in sub)
            class_weight = [1.0 / math.sqrt(freq[lab]) for lab in labels]
            mean_w = sum(class_weight) / len(class_weight)
            class_weight = [w_ / mean_w for w_ in class_weight]
        if es:
            m, vm, ran = train_epochs_es(m, s_tr, epochs=epochs, lr=lr, val_split=s_iv, labels=labels,
                                         seed=7, class_weight=class_weight, label_smoothing=ls,
                                         ordinal_lambda=ol, label_order=order_idx)
            record(name, desc, vm, {"epochs_run": ran})
            return vm
        train_plain(m, s_tr, epochs=epochs, lr=lr, seed=7, class_weight=class_weight,
                    label_smoothing=ls, ordinal_lambda=ol, label_order=order_idx)
        vm = eval_full(m, s_iv, labels)
        record(name, desc, vm)
        return vm

    # C0 control — baseline recipe (word features, LoRA r=8, 10 ep)
    screen("C0_control_word", "Baseline recipe on inner split: word features, LoRA r=8, 10 epochs", "word", False)
    # C1 char+word hybrid features
    screen("C1_wordchar", "Word + char 3-5-gram hashed features (vocab 8192), LoRA r=8, 10 epochs", "wordchar", False)
    # C2 hybrid + IDF-weighted pooling
    screen("C2_wordchar_idf", "C1 + IDF-weighted mean pooling (tf-idf style weighting)", "wordchar", True)
    # C3 hybrid + class weights 1/sqrt(freq)
    screen("C3_wordchar_classw", "C1 + per-class loss weights 1/sqrt(freq) (mean-normalised)", "wordchar", False, cw=True)
    # C4 hybrid + label smoothing + longer training w/ early stop
    screen("C4_wordchar_ls_long", "C1 + label smoothing 0.1, up to 24 epochs, early stop on inner-val macro-F1",
           "wordchar", False, epochs=24, ls=0.1, es=True)
    if task == "cve":
        # C5 ordinal-aware loss
        screen("C5_wordchar_ordinal", "C1 + ordinal penalty lambda=0.5 on expected class position (LOW<MEDIUM<HIGH<CRITICAL)",
               "wordchar", False, ol=0.5)
        # C6 component-derive reference (TF-IDF LR per CVSS component -> CVSS 3.1 formula -> tier), valB only
        c6 = component_derive_reference(sub, iv)
        attempts.append({"candidate": "C6_component_derive_tfidf", "description":
                         "TF-IDF LR predicts 8 CVSS vector components; tier derived via the official CVSS v3.1 base-score formula (Dixit et al. lever). Reference method, not a LoRA recipe.",
                         "val_accuracy": c6["accuracy"], "val_macro_f1": c6["macro_f1"]})
        print(f"[phaseB {task}] C6_component_derive: inner-val acc={c6['accuracy']} macroF1={c6['macro_f1']}")
    if task == "mitre":
        # C7 multi-label head (sigmoid/BCE) on wordchar features
        c7 = multilabel_screen(sub, iv, labels, l2i, bases["wordchar"])
        attempts.append({"candidate": "C7_multilabel_wordchar", "description":
                         "Multi-label sigmoid head (BCE) on wordchar features, LoRA r=8; threshold tuned on inner-val. Reports multi-label metrics + argmax single-label metrics.",
                         "val_accuracy": c7["argmax_metrics"]["accuracy"], "val_macro_f1": c7["argmax_metrics"]["macro_f1"],
                         "multilabel": c7})
        print(f"[phaseB {task}] C7_multilabel: argmax acc={c7['argmax_metrics']['accuracy']} f1={c7['argmax_metrics']['macro_f1']} mlF1={c7['micro_f1']}")

    # Selection: best LoRA-recipe candidate by inner-val macro-F1 (C6 is a reference, excluded;
    # C7 is eligible as a recipe — its selection score is its argmax macro-F1 for comparability).
    eligible = [a for a in attempts if not a["candidate"].startswith("C6")]
    best = max(eligible, key=lambda a: a["val_macro_f1"])
    for a in attempts:
        a["decision"] = "KEEP (selected)" if a["candidate"] == best["candidate"] else "drop"
    out = {"screening_split": "inner-val = stratified 15% carved from TRAIN (seed 777); subtrain = remaining 85%",
           "attempts": attempts, "selected_candidate": best["candidate"],
           "selection_rule": "highest inner-val macro-F1 among LoRA-recipe candidates (C6 reference excluded)",
           "screening_time_sec": round(time.time() - t0, 1)}

    def upd(data):
        data.setdefault("tasks", {})[task] = {**data.get("tasks", {}).get(task, {}), "phase_b_screening": out}
    save_results(upd)
    print(f"[phaseB {task}] selected: {best['candidate']} (inner-val macroF1 {best['val_macro_f1']}) — written to results_real.json")


# ---------------------------------------------------------------- MITRE multi-label
def multilabel_screen(sub, iv, labels, l2i, base_state, epochs=10, lr=8e-3):
    class MLSplit(SplitB):
        def __init__(self, rows, label2id, all_labels_list):
            super().__init__(rows, label2id, "wordchar")
            self.multi = []
            for r in rows:
                v = torch.zeros(len(all_labels_list))
                for lab in r["all_labels"]:
                    v[all_labels_list.index(lab)] = 1.0
                self.multi.append(v)
    all_labels = labels  # sorted label list == tactic list
    s_tr, s_iv = MLSplit(sub, l2i, all_labels), MLSplit(iv, l2i, all_labels)
    m = lora_from(base_state, "wordchar", len(labels))
    torch.manual_seed(7); random.seed(7)
    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        order = list(range(s_tr.n)); random.shuffle(order)
        for i in range(0, s_tr.n, BATCH):
            idxs = order[i:i + BATCH]
            x, mk, _ = s_tr.batch(idxs)
            y = torch.stack([s_tr.multi[j] for j in idxs])
            loss = loss_fn(m(x, mk), y)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, s_iv.n, 64):
            idxs = list(range(i, min(i + 64, s_iv.n)))
            x, mk, _ = s_iv.batch(idxs)
            probs.extend(torch.sigmoid(m(x, mk)).tolist())
    gold_multi = [s_iv.multi[j] for j in range(s_iv.n)]
    # threshold tuned on this same inner-val (screening only; documented)
    best_t, best_f1 = 0.5, -1
    for t in (0.2, 0.3, 0.4, 0.5):
        tp = fp = fn = 0
        for pi, g in zip(probs, gold_multi):
            for li in range(len(labels)):
                pred = 1 if pi[li] >= t else 0
                if pred and g[li]: tp += 1
                elif pred and not g[li]: fp += 1
                elif not pred and g[li]: fn += 1
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    argmax_preds = [max(range(len(labels)), key=lambda li: p[li]) for p in probs]
    argmax_m = metrics_from_preds(argmax_preds, s_iv.ys, labels)
    return {"threshold": best_t, "micro_f1": round(best_f1, 4), "argmax_metrics": argmax_m}


# ---------------------------------------------------------------- CVE component-derive
CVSS_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

def cvss31_base_score(v):
    """Official CVSS v3.1 base score from component dict {AV,AC,PR,UI,S,C,I,A}."""
    AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}[v["AV"]]
    AC = {"L": 0.77, "H": 0.44}[v["AC"]]
    UI = {"N": 0.85, "R": 0.62}[v["UI"]]
    S = v["S"]
    PR = {"N": 0.85, "L": 0.62 if S == "U" else 0.68, "H": 0.27 if S == "U" else 0.50}[v["PR"]]
    C = {"H": 0.56, "L": 0.22, "N": 0.0}[v["C"]]
    I = {"H": 0.56, "L": 0.22, "N": 0.0}[v["I"]]
    A = {"H": 0.56, "L": 0.22, "N": 0.0}[v["A"]]
    ISC = 1 - (1 - C) * (1 - I) * (1 - A)
    exploit = 8.22 * AV * AC * PR * UI
    if S == "U":
        impact = 6.42 * ISC
        base = min(impact + exploit, 10) if impact > 0 else 0.0
    else:
        impact = 7.52 * (ISC - 0.029) - 3.25 * (ISC - 0.02) ** 15
        base = min(1.08 * (impact + exploit), 10) if impact > 0 else 0.0
    # roundup (CVSS v3.1 spec, integer-precision method)
    import decimal
    def roundup(x):
        int_input = int(round(x * 100000))
        if int_input % 10000 == 0:
            return int_input / 100000.0
        return (math.floor(int_input / 10000) + 1) / 10.0
    return roundup(base)


def tier_of(score):
    if score == 0.0: return "NONE"
    if score < 4.0: return "LOW"
    if score < 7.0: return "MEDIUM"
    if score < 9.0: return "HIGH"
    return "CRITICAL"


def component_derive_reference(sub, iv):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    comps = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
    texts_tr = [r["instruction"] + " " + r["input"] for r in sub]
    texts_iv = [r["instruction"] + " " + r["input"] for r in iv]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=60000)
    X_tr = vec.fit_transform(texts_tr); X_iv = vec.transform(texts_iv)
    models = {}
    for c in comps:
        y = [r["cvss_vector"].get(c) for r in sub]
        if len(set(y)) < 2:
            models[c] = ("const", y[0]); continue
        clf = LogisticRegression(C=4.0, max_iter=1000)
        clf.fit(X_tr, y)
        models[c] = ("clf", clf)
    preds = []
    for j, r in enumerate(iv):
        v = {}
        for c in comps:
            kind, obj = models[c]
            v[c] = obj if kind == "const" else obj.predict(X_iv[j])[0]
        try:
            preds.append(tier_of(cvss31_base_score(v)))
        except Exception:
            preds.append("MEDIUM")
    labels = sorted(set(r["label"] for r in sub))
    l2i = {l: i for i, l in enumerate(labels)}
    return metrics_from_preds([l2i.get(p, l2i["MEDIUM"]) for p in preds], [l2i[r["label"]] for r in iv], labels)


# ---------------------------------------------------------------- Phase B ablations (round 2)
def run_phaseb2(task):
    """Round-2 screening: (i) corrected class-weight run — in round 1 the C3 candidate's
    class weights were silently dropped by a trainer-routing bug (it therefore duplicated
    C1 exactly); (ii) ablations of the winning C4 bundle to attribute its gain
    (features vs label smoothing vs longer training). Same inner-val, test untouched."""
    t0 = time.time()
    data = json.load(open(RESULTS_PATH))
    screening = data["tasks"][task]["phase_b_screening"]
    attempts = screening["attempts"]
    for a in attempts:
        if a["candidate"] == "C3_wordchar_classw" and "invalid_run" not in a:
            a["invalid_run"] = ("Round-1 run dropped class_weight (trainer routing bug) and so "
                                "duplicated C1; superseded by A1_wordchar_classw_fixed.")
            a["decision"] = "invalid (bug) — superseded by A1"
    rows, tr, va, te, labels, l2i, split_desc = get_splits(task)
    sub, iv = inner_split(tr)
    order_idx = None
    if task == "cve":
        order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        order_idx = {l2i[l]: i for i, l in enumerate(order)}
    print(f"[phaseB2 {task}] subtrain={len(sub)} inner-val={len(iv)}")

    def record(name, desc, val_m, extra=None):
        e = {"candidate": name, "description": desc,
             "val_accuracy": val_m["accuracy"], "val_macro_f1": val_m["macro_f1"]}
        if extra:
            e.update(extra)
        attempts.append(e)
        print(f"[phaseB2 {task}] {name}: inner-val acc={val_m['accuracy']} macroF1={val_m['macro_f1']}")

    bases, splits = {}, {}
    for mode in ("word", "wordchar"):
        s_tr, s_iv = SplitB(sub, l2i, mode), SplitB(iv, l2i, mode)
        splits[mode] = (s_tr, s_iv)
        bases[mode] = brief_base(s_tr, mode, len(labels))

    # A1: wordchar + class weights, correctly applied, plain 10 epochs
    freq = Counter(r["label"] for r in sub)
    cw = [1.0 / math.sqrt(freq[lab]) for lab in labels]
    mw = sum(cw) / len(cw); cw = [w_ / mw for w_ in cw]
    m = lora_from(bases["wordchar"], "wordchar", len(labels))
    train_plain(m, splits["wordchar"][0], epochs=10, lr=8e-3, seed=7, class_weight=cw)
    record("A1_wordchar_classw_fixed",
           "C1 + per-class loss weights 1/sqrt(freq), CORRECTED run (round-1 C3 dropped the weights).",
           eval_full(m, splits["wordchar"][1], labels))
    # A2: word features + label smoothing + long training (ES) — C4 minus char features
    m = lora_from(bases["word"], "word", len(labels))
    m, vm, ran = train_epochs_es(m, splits["word"][0], epochs=24, lr=8e-3, val_split=splits["word"][1],
                                 labels=labels, seed=7, label_smoothing=0.1)
    record("A2_word_ls_long", "Word features + label smoothing 0.1, up to 24 epochs, early stop on inner-val macro-F1 (C4 ablation: no char features).",
           vm, {"epochs_run": ran})
    # A3: wordchar + long training (ES), no label smoothing — C4 minus smoothing
    m = lora_from(bases["wordchar"], "wordchar", len(labels))
    m, vm, ran = train_epochs_es(m, splits["wordchar"][0], epochs=24, lr=8e-3, val_split=splits["wordchar"][1],
                                 labels=labels, seed=7, label_smoothing=0.0)
    record("A3_wordchar_long_nols", "Wordchar features, up to 24 epochs, early stop on inner-val macro-F1, NO label smoothing (C4 ablation).",
           vm, {"epochs_run": ran})

    eligible = [a for a in attempts if not a["candidate"].startswith("C6") and "invalid_run" not in a]
    best = max(eligible, key=lambda a: a["val_macro_f1"])
    for a in attempts:
        if "invalid_run" in a:
            continue
        a["decision"] = "KEEP (selected)" if a["candidate"] == best["candidate"] else "drop"
    screening["selected_candidate"] = best["candidate"]
    screening["round2_note"] = ("Round 2 added the corrected class-weight run (A1) and ablations of the "
                                "round-1 winner bundle (A2/A3); selection recomputed over all valid attempts.")
    screening["round2_time_sec"] = round(time.time() - t0, 1)

    def upd(d):
        d["tasks"][task]["phase_b_screening"] = screening
    save_results(upd)
    print(f"[phaseB2 {task}] selected after round 2: {best['candidate']} (inner-val macroF1 {best['val_macro_f1']})")


# ================================================================ final
RECIPE_MAP = {
    "C0_control_word":        dict(mode="word", idf=False, cw=False, ls=0.0, ol=0.0, epochs=10, multilabel=False),
    "C1_wordchar":            dict(mode="wordchar", idf=False, cw=False, ls=0.0, ol=0.0, epochs=10, multilabel=False),
    "C2_wordchar_idf":        dict(mode="wordchar", idf=True, cw=False, ls=0.0, ol=0.0, epochs=10, multilabel=False),
    "C3_wordchar_classw":     dict(mode="wordchar", idf=False, cw=True, ls=0.0, ol=0.0, epochs=10, multilabel=False),
    "C4_wordchar_ls_long":    dict(mode="wordchar", idf=False, cw=False, ls=0.1, ol=0.0, epochs=24, multilabel=False),
    "C5_wordchar_ordinal":    dict(mode="wordchar", idf=False, cw=False, ls=0.0, ol=0.5, epochs=10, multilabel=False),
    "C7_multilabel_wordchar": dict(mode="wordchar", idf=False, cw=False, ls=0.0, ol=0.0, epochs=10, multilabel=True),
    "A1_wordchar_classw_fixed": dict(mode="wordchar", idf=False, cw=True, ls=0.0, ol=0.0, epochs=10, multilabel=False),
    "A2_word_ls_long":        dict(mode="word", idf=False, cw=False, ls=0.1, ol=0.0, epochs=24, multilabel=False),
    "A3_wordchar_long_nols":  dict(mode="wordchar", idf=False, cw=False, ls=0.0, ol=0.0, epochs=24, multilabel=False),
}

def run_final(task):
    data = json.load(open(RESULTS_PATH))
    sel = data["tasks"][task]["phase_b_screening"]["selected_candidate"]
    recipe = RECIPE_MAP[sel]
    print(f"[final {task}] selected recipe {sel}: {recipe}")
    rows, tr, va, te, labels, l2i, split_desc = get_splits(task)
    mode = recipe["mode"]
    idf = compute_idf(tr, mode) if recipe["idf"] else None
    trS = SplitB(tr, l2i, mode, idf); teS = SplitB(te, l2i, mode, idf)
    order_idx = None
    if task == "cve":
        order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        order_idx = {l2i[l]: i for i, l in enumerate(order)}
    class_weight = None
    if recipe["cw"]:
        freq = Counter(r["label"] for r in tr)
        class_weight = [1.0 / math.sqrt(freq[lab]) for lab in labels]
        mw = sum(class_weight) / len(class_weight)
        class_weight = [w_ / mw for w_ in class_weight]

    out = {"selected_candidate": sel, "recipe": {k: v for k, v in recipe.items()},
           "note": "Retrained on FULL train split; test evaluated exactly once for this Phase-B final config."}
    t0 = time.time()

    if recipe["multilabel"]:
        # Multi-label final: base brief -> LoRA (BCE) and full FT (BCE), evaluate both once on test.
        class MLSplitF(SplitB):
            def __init__(self, rows_, l2i_):
                super().__init__(rows_, l2i_, mode, idf)
                self.multi = []
                for r in rows_:
                    v = torch.zeros(len(labels))
                    for lab in r["all_labels"]:
                        v[labels.index(lab)] = 1.0
                    self.multi.append(v)
        trM, teM = MLSplitF(tr, l2i), MLSplitF(te, l2i)
        def train_ml(model, epochs, lr):
            model.train()
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
            fn = nn.BCEWithLogitsLoss()
            torch.manual_seed(7); random.seed(7)
            for ep in range(epochs):
                order_ = list(range(trM.n)); random.shuffle(order_)
                for i in range(0, trM.n, BATCH):
                    idxs = order_[i:i + BATCH]
                    x, mk, _ = trM.batch(idxs)
                    y = torch.stack([trM.multi[j] for j in idxs])
                    loss = fn(model(x, mk), y)
                    opt.zero_grad(); loss.backward(); opt.step()
            return model
        def eval_ml(model):
            model.eval(); probs = []
            with torch.no_grad():
                for i in range(0, teM.n, 64):
                    idxs = list(range(i, min(i + 64, teM.n)))
                    x, mk, _ = teM.batch(idxs)
                    probs.extend(torch.sigmoid(model(x, mk)).tolist())
            thr = data["tasks"][task]["phase_b_screening"]["attempts"]
            thr = [a for a in thr if a["candidate"] == sel][0]["multilabel"]["threshold"]
            tp = fp = fn_ = 0
            for pi, gi in zip(probs, range(teM.n)):
                g = teM.multi[gi]
                for li in range(len(labels)):
                    pred = 1 if pi[li] >= thr else 0
                    if pred and g[li]: tp += 1
                    elif pred and not g[li]: fp += 1
                    elif not pred and g[li]: fn_ += 1
            micro = 2 * tp / (2 * tp + fp + fn_) if (2 * tp + fp + fn_) else 0.0
            argmax_preds = [max(range(len(labels)), key=lambda li: p[li]) for p in probs]
            return {"threshold_from_inner_val": thr, "multilabel_micro_f1": round(micro, 4),
                    "single_label_argmax": metrics_from_preds(argmax_preds, teM.ys, labels)}
        # base
        torch.manual_seed(42); random.seed(42)
        bm = make_model(len(labels), mode)
        train_epochs(bm, trS, epochs=1, lr=3e-3, verbose=False)
        bstate = copy.deepcopy(bm.state_dict())
        # LoRA ML
        lm = lora_from(bstate, mode, len(labels))
        lm = train_ml(lm, recipe["epochs"], 8e-3)
        out["lora_test"] = eval_ml(lm)
        out["lora_trainable_params"] = count_parameters(lm)["trainable"]
        # Full FT ML
        fm = make_model(len(labels), mode); fm.load_state_dict(copy.deepcopy(bstate))
        fm = train_ml(fm, 8, 3e-3)
        out["full_ft_test"] = eval_ml(fm)
        out["full_ft_trainable_params"] = count_parameters(fm)["trainable"]
    else:
        # base brief on full train
        torch.manual_seed(42); random.seed(42)
        bm = make_model(len(labels), mode)
        train_epochs(bm, trS, epochs=1, lr=3e-3, verbose=False)
        bstate = copy.deepcopy(bm.state_dict())
        # LoRA final (plain fixed-epoch retrain of the screened recipe; no val/test used)
        lm = lora_from(bstate, mode, len(labels))
        _train_final(lm, trS, recipe, labels, class_weight, order_idx)
        out["lora_test"] = eval_full(lm, teS, labels)
        stats = count_parameters(lm)
        out["lora_trainable_params"] = stats["trainable"]; out["lora_trainable_pct"] = stats["trainable_pct"]
        out["total_params"] = stats["total"]
        # Full FT final (baseline FT schedule, same features/loss recipe)
        fm = make_model(len(labels), mode); fm.load_state_dict(copy.deepcopy(bstate))
        _train_final_ft(fm, trS, recipe, labels, class_weight, order_idx)
        out["full_ft_test"] = eval_full(fm, teS, labels)
        out["full_ft_trainable_params"] = count_parameters(fm)["trainable"]
    out["total_time_sec"] = round(time.time() - t0, 1)

    def upd(data):
        data.setdefault("tasks", {})[task] = {**data.get("tasks", {}).get(task, {}), "phase_b_final": out}
    save_results(upd)
    print(f"[final {task}] {json.dumps(out, indent=2)[:1200]}")


def _train_final(model, split, recipe, labels, class_weight, order_idx):
    """Plain fixed-epoch LoRA training for the final config (no early stop, no val)."""
    torch.manual_seed(7); random.seed(7)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=8e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=recipe["epochs"])
    w = torch.tensor(class_weight, dtype=torch.float) if class_weight else None
    fn = nn.CrossEntropyLoss(weight=w, label_smoothing=recipe["ls"])
    for ep in range(recipe["epochs"]):
        order = list(range(split.n)); random.shuffle(order)
        for i in range(0, split.n, BATCH):
            idxs = order[i:i + BATCH]
            x, m, y = split.batch(idxs)
            logits = model(x, m)
            loss = fn(logits, y)
            if recipe["ol"] > 0 and order_idx is not None:
                probs = torch.softmax(logits, dim=-1)
                pos = torch.tensor([order_idx[int(t)] for t in y], dtype=torch.float)
                exp_pos = probs @ torch.arange(logits.size(1), dtype=torch.float)
                loss = loss + recipe["ol"] * torch.mean(torch.abs(exp_pos - pos))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


def _train_final_ft(model, split, recipe, labels, class_weight, order_idx):
    rec = dict(recipe); rec["epochs"] = 8
    torch.manual_seed(42); random.seed(42)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8)
    w = torch.tensor(class_weight, dtype=torch.float) if class_weight else None
    fn = nn.CrossEntropyLoss(weight=w, label_smoothing=recipe["ls"])
    for ep in range(8):
        order = list(range(split.n)); random.shuffle(order)
        for i in range(0, split.n, BATCH):
            idxs = order[i:i + BATCH]
            x, m, y = split.batch(idxs)
            logits = model(x, m)
            loss = fn(logits, y)
            if recipe["ol"] > 0 and order_idx is not None:
                probs = torch.softmax(logits, dim=-1)
                pos = torch.tensor([order_idx[int(t)] for t in y], dtype=torch.float)
                exp_pos = probs @ torch.arange(logits.size(1), dtype=torch.float)
                loss = loss + recipe["ol"] * torch.mean(torch.abs(exp_pos - pos))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["baseline", "phaseb", "phaseb2", "final"])
    ap.add_argument("--task", required=True, choices=["mitre", "cve"])
    args = ap.parse_args()
    {"baseline": run_baseline, "phaseb": run_phaseb, "phaseb2": run_phaseb2, "final": run_final}[args.phase](args.task)
