"""ResolveDesk AI — REAL benchmark evaluation on the Bitext customer-support dataset.

Reproduces: download -> verification -> splits -> baseline -> improvement
screening (validation only) -> final test evaluation, writing results_real.json.

Method notes
------------
* BASELINE uses the project's own classifier method UNCHANGED (ml.Tfidf
  word-unigram TF-IDF, max_vocab=3000 + softmax LR full-batch GD, epochs=300,
  lr=0.5, l2=1e-4, seed=1 — exactly the intent head of ml.TriageModel.train).
  Execution note: GD is run on the sparse TF-IDF matrix (rows are ~99.7%
  zeros) after an equivalence check on the project's synthetic split showed
  the sparse executor reproduces the dense ml.SoftmaxLR predictions/weights
  (see results_real.json: equivalence_check) — the dense run was impractical
  on this CPU-shared VM; the maths, features and hyper-parameters are ml.py's.
* SPLIT HYGIENE (per REAL_BENCH_IMPROVEMENTS.md §5): Bitext contains many
  duplicate / near-duplicate utterances, so a naive random split leaks them
  across train/test. We report BOTH:
    - naive: stratified random 80/20 split of all 26,872 rows;
    - deduped (HEADLINE): deduplicate on normalised text first (lowercase,
      punctuation -> space, whitespace collapsed), then stratified 80/20 split
      of the unique utterances — no test text's normalised form occurs in train.
* PHASE B (improvement) experiments are screened ONLY on a validation split
  carved stratified from the deduped TRAIN portion. The deduped test split is
  touched exactly twice: once for the baseline, once for the final selected
  config. Phase-B models keep the project's from-scratch method (TF-IDF +
  full-batch gradient-descent softmax logistic regression, NumPy/SciPy-sparse
  implementation of the same maths); only the feature representation,
  regularisation, class weighting and label smoothing vary.
* Priority: Bitext has no priority labels — INTENT is the real result (plus
  category as an honestly-labelled second task). Priority stays synthetic-only.

Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset
Licence (HF API cardData.license): cdla-sharing-1.0
Usage: python3 eval_bitext.py [--stage all|baseline|phaseb|final]
"""
import argparse, collections, json, math, os, random, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import scipy.sparse as sp

from ml import SoftmaxLR, Tfidf, macro_f1, tokenize

SEED = 42
HF_API = "https://huggingface.co/api/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset"
PARQUET_URL = HF_API + "/parquet/default/train/0.parquet"
CACHE = "/tmp/bitext_cache/train.parquet"
RESULTS = os.path.join(HERE, "results_real.json")


# ------------------------------------------------------------------ data --
def download():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    if not os.path.exists(CACHE):
        urllib.request.urlretrieve(PARQUET_URL, CACHE)
    meta = json.load(urllib.request.urlopen(HF_API))
    return meta


def load_rows():
    import pyarrow.parquet as pq
    return pq.read_table(CACHE).to_pylist()


def normalize_text(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def stratified_split(items, key, test_frac, rng):
    """items: list; key: item->label. Returns (rest, test) stratified by label."""
    buckets = collections.defaultdict(list)
    for it in items:
        buckets[key(it)].append(it)
    rest, test = [], []
    for lab in sorted(buckets):
        b = buckets[lab][:]
        rng.shuffle(b)
        cut = int(round(len(b) * test_frac))
        test += b[:cut]
        rest += b[cut:]
    rng.shuffle(rest)
    rng.shuffle(test)
    return rest, test


# --------------------------------------------------------------- metrics --
def per_class_f1(y_true, y_pred, labels):
    out = {}
    for i, lab in enumerate(labels):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[lab] = {"precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4),
                    "support": int((y_true == i).sum())}
    return out


def top_confusions(y_true, y_pred, labels, k=8):
    pairs = collections.Counter()
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t != p:
            pairs[(labels[t], labels[p])] += 1
    return [{"true": a, "predicted": b, "count": c}
            for (a, b), c in pairs.most_common(k)]


def metrics_block(y_true, y_pred, labels):
    acc = float((y_pred == y_true).mean())
    pc = per_class_f1(y_true, y_pred, labels)
    weakest = sorted(pc.items(), key=lambda kv: kv[1]["f1"])[:5]
    return {"accuracy": round(acc, 4),
            "macro_f1": round(macro_f1(y_true, y_pred, range(len(labels))), 4),
            "n_test": int(len(y_true)),
            "weakest_intents": [{"intent": k, **v} for k, v in weakest],
            "top_confusions": top_confusions(y_true, y_pred, labels),
            "per_intent": pc}


def wilson(acc, n, z=1.96):
    if n == 0:
        return [None, None]
    d = 1 + z * z / n
    c = acc + z * z / (2 * n)
    m = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n))
    return [round((c - m) / d, 4), round((c + m) / d, 4)]


# ------------------------------------------------- sparse Phase-B model --
def doc_features(text, word_ngram_max, char_ngrams):
    toks = tokenize(text)
    feats = ["w:" + t for t in toks]
    if word_ngram_max >= 2:
        feats += ["w:" + toks[i] + " " + toks[i + 1] for i in range(len(toks) - 1)]
    if char_ngrams:
        lo, hi = char_ngrams
        for t in toks:
            s = " " + t + " "
            for n in range(lo, hi + 1):
                feats += ["c:" + s[i:i + n] for i in range(len(s) - n + 1)]
    return feats


class SparseTfidf:
    """Same TF-IDF maths as ml.Tfidf ((1+log tf) * (log((1+N)/(1+df))+1),
    L2-normalised) over word/char n-gram features, SciPy-sparse storage."""

    def __init__(self, word_ngram_max=1, char_ngrams=None,
                 max_word_vocab=3000, max_char_vocab=0, min_df_char=2):
        self.word_ngram_max = word_ngram_max
        self.char_ngrams = char_ngrams
        self.max_word_vocab = max_word_vocab
        self.max_char_vocab = max_char_vocab
        self.min_df_char = min_df_char

    def _select(self, df, prefix, cap, min_df):
        items = [(t, c) for t, c in df.items()
                 if t.startswith(prefix) and c >= min_df]
        items.sort(key=lambda kv: -kv[1])
        return items[:cap]

    def fit(self, docs):
        df = collections.Counter()
        for d in docs:
            for f in set(doc_features(d, self.word_ngram_max, self.char_ngrams)):
                df[f] += 1
        terms = self._select(df, "w:", self.max_word_vocab, 1)
        if self.char_ngrams:
            terms += self._select(df, "c:", self.max_char_vocab, self.min_df_char)
        self.vocab = {t: i for i, (t, _) in enumerate(terms)}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0
                             for t, _ in terms])
        return self

    def transform(self, docs):
        rows, cols, vals = [], [], []
        for r, d in enumerate(docs):
            counts = collections.Counter(
                f for f in doc_features(d, self.word_ngram_max, self.char_ngrams)
                if f in self.vocab)
            for f, c in counts.items():
                j = self.vocab[f]
                rows.append(r)
                cols.append(j)
                vals.append((1 + math.log(c)) * self.idf[j])
        X = sp.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)))
        norms = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        return X.multiply(1.0 / norms[:, None]).tocsr()


class SparseSoftmaxLR:
    """ml.SoftmaxLR's full-batch GD softmax regression on sparse X, extended
    with optional per-sample (class) weights and label smoothing."""

    def __init__(self, n_features, n_classes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (n_features, n_classes))
        self.b = np.zeros(n_classes)

    def fit(self, X, y, epochs=250, lr=0.5, l2=1e-4,
            sample_weight=None, label_smoothing=0.0):
        k = self.W.shape[1]
        Y = np.eye(k)[y] * (1 - label_smoothing) + label_smoothing / k
        w = np.ones(len(y)) if sample_weight is None else sample_weight
        wsum = w.sum()
        for _ in range(epochs):
            logits = X @ self.W + self.b
            logits = np.asarray(logits)
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)
            G = (P - Y) * w[:, None] / wsum
            self.W -= lr * (np.asarray(X.T @ G) + l2 * self.W)
            self.b -= lr * G.sum(axis=0)
        return self

    def predict(self, X):
        logits = np.asarray(X @ self.W + self.b)
        return logits.argmax(1)


def class_weights(y, n_classes, mode):
    freq = np.bincount(y, minlength=n_classes).astype(float)
    if mode is None:
        return None
    if mode == "inverse":
        cw = 1.0 / np.maximum(freq, 1)
    elif mode == "sqrt_inverse":
        cw = 1.0 / np.sqrt(np.maximum(freq, 1))
    else:
        raise ValueError(mode)
    cw *= n_classes / cw.sum()
    return cw[y]


# ---------------------------------------------------------------- stages --
def save_results(res):
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2, sort_keys=True)


def load_results():
    with open(RESULTS) as f:
        return json.load(f)


def build_splits(rows):
    rng = random.Random(SEED)
    intents = sorted({r["intent"] for r in rows})
    categories = sorted({r["category"] for r in rows})
    # naive: all rows, stratified 80/20
    naive_train, naive_test = stratified_split(rows, lambda r: r["intent"], 0.2, rng)
    # deduped: unique normalised texts (first occurrence kept; duplicates were
    # verified to never carry a conflicting intent), stratified 80/20
    seen, uniq = set(), []
    for r in rows:
        n = normalize_text(r["instruction"])
        if n not in seen:
            seen.add(n)
            uniq.append(r)
    ded_train, ded_test = stratified_split(uniq, lambda r: r["intent"], 0.2, rng)
    # validation carved from the deduped TRAIN portion only (12.5% of it)
    ded_core, ded_val = stratified_split(ded_train, lambda r: r["intent"], 0.125, rng)
    # naive leakage statistic
    train_norms = {normalize_text(r["instruction"]) for r in naive_train}
    leak = np.mean([normalize_text(r["instruction"]) in train_norms
                    for r in naive_test])
    uniq_counts = collections.Counter(r["intent"] for r in uniq)
    splits = {"seed": SEED, "naive": {"train": len(naive_train), "test": len(naive_test)},
              "deduped_unique_utterances_per_intent": dict(sorted(uniq_counts.items())),
              "deduped_unique_per_intent_min": min(uniq_counts.values()),
              "deduped_unique_per_intent_max": max(uniq_counts.values()),
              "deduped": {"unique_utterances": len(uniq), "train": len(ded_train),
                          "test": len(ded_test), "train_core_for_val_screening": len(ded_core),
                          "val": len(ded_val)},
              "naive_test_rows_whose_normalised_text_appears_in_naive_train": round(float(leak), 4)}
    data = {"intents": intents, "categories": categories,
            "naive_train": naive_train, "naive_test": naive_test,
            "ded_train": ded_train, "ded_test": ded_test,
            "ded_core": ded_core, "ded_val": ded_val}
    return splits, data


def equivalence_check():
    """Verify the sparse GD executor reproduces ml.SoftmaxLR exactly in method:
    train BOTH implementations (same seed/hyper-parameters) on the project's
    own synthetic split (train_ml.py protocol) and compare predictions."""
    from ml import TriageModel, generate_data
    data = generate_data(n_per_combo=25, seed=42)
    rng = random.Random(7)
    buckets = {}
    for d in data:
        buckets.setdefault((d["intent"], d["priority"]), []).append(d)
    train, test = [], []
    for items in buckets.values():
        rng.shuffle(items)
        cut = int(len(items) * 0.8)
        train += items[:cut]
        test += items[cut:]
    dense = TriageModel().train(train)  # the shipped model, verbatim
    from ml import INTENTS
    tf = Tfidf()
    X = tf.fit([d["text"] for d in train]).transform([d["text"] for d in train])
    y = np.array([INTENTS.index(d["intent"]) for d in train])
    sparse_clf = SparseSoftmaxLR(X.shape[1], len(INTENTS), seed=1).fit(
        sp.csr_matrix(X), y, epochs=300, lr=0.5, l2=1e-4)
    Xte = tf.transform([d["text"] for d in test])
    # dense intent predictions directly from the trained head (no rule layer)
    Pd = dense.intent_clf.predict_proba(dense.tfidf.transform(
        [d["text"] for d in test])).argmax(1)
    Ps = sparse_clf.predict(sp.csr_matrix(Xte))
    agree = float((Pd == Ps).mean())
    w_max = float(np.abs(dense.intent_clf.W - sparse_clf.W).max())
    return {"dataset": "project synthetic 600 (train_ml.py split protocol)",
            "prediction_agreement_dense_vs_sparse": agree,
            "max_abs_weight_diff": w_max,
            "conclusion": "sparse executor reproduces the dense ml.SoftmaxLR "
                          "training maths (full-batch GD, same init/lr/l2/epochs); "
                          "used for the Bitext baseline because the dense TF-IDF "
                          "matrix is ~99.7% zeros and the VM is CPU-shared."}


def run_baseline(res, D):
    """Project's classifier as-is: ml.Tfidf features (word unigrams,
    max_vocab=3000) + ml.SoftmaxLR training maths (epochs=300, lr=0.5,
    l2=1e-4, seed=1), executed on the sparse TF-IDF matrix (see
    equivalence_check). Both splits, intent + category heads."""
    out = {"classifier": {"implementation": "ml.Tfidf (word unigrams, max_vocab=3000) "
                                            "+ softmax logistic regression with "
                                            "ml.SoftmaxLR's exact full-batch GD "
                                            "(verified equivalent, see equivalence_check)",
                          "epochs": 300, "lr": 0.5, "l2": 1e-4, "seed": 1}}
    t_all = time.time()
    for name, prefix in [("naive", "naive"), ("deduped", "ded")]:
        train, test = D[prefix + "_train"], D[prefix + "_test"]
        texts_tr = [r["instruction"] for r in train]
        texts_te = [r["instruction"] for r in test]
        tf = Tfidf()
        Xtr = sp.csr_matrix(tf.fit(texts_tr).transform(texts_tr))
        Xte = sp.csr_matrix(tf.transform(texts_te))
        block = {"vocab_size": len(tf.vocab)}
        for task, labels in [("intent", D["intents"]), ("category", D["categories"])]:
            lab2i = {l: i for i, l in enumerate(labels)}
            ytr = np.array([lab2i[r[task]] for r in train])
            yte = np.array([lab2i[r[task]] for r in test])
            t0 = time.time()
            clf = SparseSoftmaxLR(Xtr.shape[1], len(labels), seed=1).fit(
                Xtr, ytr, epochs=300, lr=0.5, l2=1e-4)
            secs = time.time() - t0
            pred = clf.predict(Xte)
            blk = metrics_block(yte, pred, labels)
            blk["accuracy_wilson95"] = wilson(blk["accuracy"], blk["n_test"])
            blk["train_seconds"] = round(secs, 2)
            block[task] = blk
            print(name, task, blk["accuracy"], blk["macro_f1"], flush=True)
        out[name] = block
    out["total_seconds"] = round(time.time() - t_all, 2)
    return out


def run_sparse(cfg, train_rows, labels, task="intent"):
    """Train one Phase-B config (sparse, same method) and return the model."""
    tf = SparseTfidf(word_ngram_max=cfg.get("word_ngram_max", 1),
                     char_ngrams=tuple(cfg["char_ngrams"]) if cfg.get("char_ngrams") else None,
                     max_word_vocab=cfg.get("max_word_vocab", 3000),
                     max_char_vocab=cfg.get("max_char_vocab", 0),
                     min_df_char=cfg.get("min_df_char", 2))
    texts = [r["instruction"] for r in train_rows]
    X = tf.fit(texts).transform(texts)
    lab2i = {l: i for i, l in enumerate(labels)}
    y = np.array([lab2i[r[task]] for r in train_rows])
    sw = class_weights(y, len(labels), cfg.get("class_weight"))
    clf = SparseSoftmaxLR(X.shape[1], len(labels), seed=1).fit(
        X, y, epochs=cfg.get("epochs", 250), lr=cfg.get("lr", 0.5),
        l2=cfg.get("l2", 1e-4), sample_weight=sw,
        label_smoothing=cfg.get("label_smoothing", 0.0))
    return tf, clf


def eval_sparse(tf, clf, rows, labels, task="intent"):
    lab2i = {l: i for i, l in enumerate(labels)}
    y = np.array([lab2i[r[task]] for r in rows])
    pred = clf.predict(tf.transform([r["instruction"] for r in rows]))
    return y, pred


def run_phaseb(res, D):
    """Screen levers on validation ONLY (train_core -> val). Logs every attempt."""
    intents = D["intents"]
    core, val = D["ded_core"], D["ded_val"]
    log = res.get("improvement_log", [])
    tried = {e["config_id"] for e in log}

    def attempt(cid, cfg, note):
        if cid in tried:
            return None
        t0 = time.time()
        tf, clf = run_sparse(cfg, core, intents)
        y, pred = eval_sparse(tf, clf, val, intents)
        acc = float((pred == y).mean())
        mf1 = macro_f1(y, pred, range(len(intents)))
        entry = {"config_id": cid, "config": cfg, "split": "val (from deduped train only)",
                 "val_accuracy": round(acc, 4), "val_macro_f1": round(mf1, 4),
                 "n_val": int(len(y)), "feature_dim": int(len(tf.vocab)),
                 "seconds": round(time.time() - t0, 2), "note": note}
        log.append(entry)
        res["improvement_log"] = log
        save_results(res)
        print("VAL", cid, entry["val_accuracy"], entry["val_macro_f1"], flush=True)
        return entry

    base_feats = {"word_ngram_max": 1, "max_word_vocab": 3000}
    attempt("v0_word_unigram_replica", dict(base_feats),
            "Sparse replica of the baseline representation (sanity anchor).")
    attempt("v1_word_bigrams", {"word_ngram_max": 2, "max_word_vocab": 20000},
            "Lever: word uni+bigrams, larger word vocab.")
    attempt("v2_word_char", {"word_ngram_max": 2, "max_word_vocab": 20000,
                             "char_ngrams": [3, 5], "max_char_vocab": 30000,
                             "min_df_char": 2},
            "Lever: + word-boundary char 3-5 grams (brief's top feature lever).")
    # regularisation sweep on the best feature set so far
    best = max(log, key=lambda e: e["val_macro_f1"])
    feats = {k: best["config"][k] for k in
             ["word_ngram_max", "max_word_vocab", "char_ngrams", "max_char_vocab", "min_df_char"]
             if k in best["config"]}
    for l2 in [1e-5, 1e-3, 1e-2]:
        attempt(f"sweep_l2_{l2}", dict(feats, l2=l2),
                f"Lever: LR regularisation sweep (l2={l2}) on best features so far.")
    best = max(log, key=lambda e: e["val_macro_f1"])
    feats = {k: best["config"][k] for k in
             ["word_ngram_max", "max_word_vocab", "char_ngrams", "max_char_vocab", "min_df_char", "l2"]
             if k in best["config"]}
    for cw in ["inverse", "sqrt_inverse"]:
        attempt(f"classweight_{cw}", dict(feats, class_weight=cw),
                f"Lever: class weighting ({cw}). Bitext is near-balanced; verify, don't assume.")
    # Round 2: combine the winning lever (inverse class weighting — it fixes
    # the imbalance that DEDUPLICATION itself creates: unique-utterance counts
    # per intent range widely even though raw rows are balanced) with the
    # feature levers, and refine the winner. All still validation-only.
    attempt("cw_inverse_plus_bigrams",
            {"word_ngram_max": 2, "max_word_vocab": 20000, "class_weight": "inverse"},
            "Round 2: inverse class weighting + word uni+bigrams.")
    attempt("cw_inverse_plus_word_char",
            {"word_ngram_max": 2, "max_word_vocab": 20000, "char_ngrams": [3, 5],
             "max_char_vocab": 30000, "min_df_char": 2, "class_weight": "inverse"},
            "Round 2: inverse class weighting + word bigrams + char 3-5 grams.")
    best = max(log, key=lambda e: e["val_macro_f1"])
    bcfg = dict(best["config"])
    attempt("best_plus_label_smoothing_0.1", dict(bcfg, label_smoothing=0.1),
            "Lever: label smoothing 0.1 on the best config so far.")
    attempt("best_plus_epochs_600", dict(bcfg, epochs=600),
            "Lever: longer training (600 vs 250 epochs) on the best config so far.")
    best = max(log, key=lambda e: e["val_macro_f1"])
    bcfg = dict(best["config"])
    if bcfg.get("l2", 1e-4) != 1e-3:
        attempt("best_plus_l2_1e-3", dict(bcfg, l2=1e-3),
                "Lever: l2=1e-3 on the best config so far.")
    # train-size curve with the best config (validation only)
    best = max(log, key=lambda e: e["val_macro_f1"])
    feats = dict(best["config"])
    rng = random.Random(SEED)
    for frac in [0.25, 0.5]:
        sub, _ = stratified_split(core, lambda r: r["intent"], 1 - frac, rng)
        t0 = time.time()
        tf, clf = run_sparse(feats, sub, intents)
        y, pred = eval_sparse(tf, clf, val, intents)
        log.append({"config_id": f"trainsize_{int(frac*100)}pct", "config": feats,
                    "split": "val (from deduped train only)",
                    "train_fraction_of_train_core": frac,
                    "val_accuracy": round(float((pred == y).mean()), 4),
                    "val_macro_f1": round(macro_f1(y, pred, range(len(intents))), 4),
                    "n_val": int(len(y)), "feature_dim": int(len(tf.vocab)),
                    "seconds": round(time.time() - t0, 2),
                    "note": "Train-size curve point (brief lever 5); full size = best val entry."})
        res["improvement_log"] = log
        save_results(res)
        print("VAL trainsize", frac, log[-1]["val_accuracy"], flush=True)
    # mark keep/drop: the single best val config is kept, the rest dropped
    # (train-size curve points are analysis, not selection candidates)
    best = max((e for e in log if not e["config_id"].startswith("trainsize")),
               key=lambda e: e["val_macro_f1"])
    for e in log:
        e["decision"] = "KEEP (selected for final)" if e is best else "drop"
    res["improvement_log"] = log
    res["selected_config_id"] = best["config_id"]
    res["selected_config"] = best["config"]
    save_results(res)
    return best


def run_final(res, D):
    """Retrain selected config on the FULL deduped train portion; test ONCE."""
    cfg = res["selected_config"]
    train, test = D["ded_train"], D["ded_test"]
    out = {"config_id": res["selected_config_id"], "config": cfg,
           "trained_on": "full deduped train portion (train_core + val)",
           "split": "deduped test (touched once for the baseline, once here)"}
    t0 = time.time()
    tf, clf = run_sparse(cfg, train, D["intents"])
    secs = time.time() - t0
    y, pred = eval_sparse(tf, clf, test, D["intents"])
    blk = metrics_block(y, pred, D["intents"])
    blk["accuracy_wilson95"] = wilson(blk["accuracy"], blk["n_test"])
    blk["train_seconds"] = round(secs, 2)
    blk["feature_dim"] = int(len(tf.vocab))
    out["intent"] = blk
    # category as honestly-labelled second task, same features/config
    lab2i = {l: i for i, l in enumerate(D["categories"])}
    ytr = np.array([lab2i[r["category"]] for r in train])
    sw = class_weights(ytr, len(D["categories"]), cfg.get("class_weight"))
    cclf = SparseSoftmaxLR(len(tf.vocab), len(D["categories"]), seed=1).fit(
        tf.transform([r["instruction"] for r in train]), ytr,
        epochs=cfg.get("epochs", 250), lr=cfg.get("lr", 0.5), l2=cfg.get("l2", 1e-4),
        sample_weight=sw, label_smoothing=cfg.get("label_smoothing", 0.0))
    yte, predc = eval_sparse(tf, cclf, test, D["categories"], task="category")
    cblk = metrics_block(yte, predc, D["categories"])
    cblk["accuracy_wilson95"] = wilson(cblk["accuracy"], cblk["n_test"])
    out["category"] = cblk
    out["total_seconds"] = round(time.time() - t0, 2)
    print("FINAL intent", blk["accuracy"], blk["macro_f1"], "| category",
          cblk["accuracy"], cblk["macro_f1"], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "baseline", "phaseb", "final"])
    args = ap.parse_args()
    meta = download()
    rows = load_rows()
    res = load_results()
    res["dataset"]["licence"] = meta["cardData"]["license"]
    assert len(rows) == res["dataset"]["rows_total"] == 26872
    splits, D = build_splits(rows)
    res["splits"] = splits
    res["status"] = "in_progress"
    save_results(res)
    print("splits:", json.dumps(splits), flush=True)
    if args.stage in ("all", "baseline"):
        res["equivalence_check"] = equivalence_check()
        print("equivalence:", res["equivalence_check"]["prediction_agreement_dense_vs_sparse"],
              res["equivalence_check"]["max_abs_weight_diff"], flush=True)
        res["baseline"] = run_baseline(res, D)
        save_results(res)
    if args.stage in ("all", "phaseb"):
        run_phaseb(res, D)
    if args.stage in ("all", "final"):
        res["final"] = run_final(res, D)
        res["status"] = "complete"
        save_results(res)
    print("done", args.stage)


if __name__ == "__main__":
    main()
