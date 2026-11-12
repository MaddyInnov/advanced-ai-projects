"""
AgentForge-RAG — REAL benchmark evaluation on SciFact (BEIR). Author: Ashutosh Sahoo

Re-runs the project's own retrieval ablation ladder (rag.py Retriever modes)
on a real public benchmark instead of the self-authored corpus in evaluate.py.

Dataset: SciFact, distributed by BEIR.
  Source: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
  Corpus: 5,183 scientific abstracts (title + text). Queries: 1,109 total;
  the official test split (qrels/test.tsv) has 300 queries / 339 qrel rows,
  all with grade 1 (binary relevance, as distributed). The train split
  (qrels/train.tsv, 809 queries / 919 rows) is used ONLY as the validation /
  dev set for the Phase-B improvement screening — test qrels are never tuned on.

Protocol (honest, no leakage):
  * Stage "baseline": the three original rungs on the test queries, using
    rag.Retriever exactly as shipped (defaults: alpha=0.55 score fusion,
    BM25 k1=1.5 / b=0.75, project query expansion inside the hybrid modes).
    This is the ONE baseline test evaluation.
  * Stage "dev": improvement levers screened ONLY on train/dev queries
    (a deterministic 250-query slice for screening, full 809 for the final
    dev confirmation). Every attempt is logged to results_real.json.
  * Stage "final": the single dev-selected pipeline gets exactly ONE more
    test evaluation.

Each SciFact abstract is indexed as ONE chunk (document-level retrieval is
the BEIR standard, and abstracts average ~214 words, close to the project's
200-word chunk size); chunk doc_id = SciFact corpus _id, so retrieved chunks
map 1:1 to qrels corpus-ids. Text indexed = title + ". " + abstract text.

Metrics (computed from qrels, standard BEIR definitions):
  Recall@k  = mean over queries of |relevant ∩ top-k| / |relevant|
  nDCG@10   = DCG with gain = grade (binary here), log2 discount, / IDCG
  MRR@10    = mean reciprocal rank of first relevant doc within top-10

SciFact does NOT benchmark the groundedness / abstention machinery — no
numbers for those are produced here; they remain from the internal eval only.

Run:  python3 eval_scifact.py --stage baseline|dev|final|all
Outputs: results_real.json (updated incrementally after every rung/attempt).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
import time
import urllib.request
import zipfile

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

import re

import numpy as np  # noqa: E402
import rag as rag_mod  # noqa: E402
from rag import Chunk, Retriever, TFIDFEmbedder, BM25Index, expand_query, _cross_score, content_tokens  # noqa: E402
from rag import STOPWORDS as RAG_STOPWORDS, _sentences as rag_sentences  # noqa: E402

DATA_DIR = HERE / "data"
ZIP_PATH = DATA_DIR / "scifact.zip"
SF_DIR = DATA_DIR / "scifact"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
RESULTS_PATH = HERE / "results_real.json"

BASELINE_CONFIG = {
    "retriever": "rag.Retriever (project code, unmodified)",
    "alpha_tfidf_fusion_weight": 0.55,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "chunking": "one chunk per SciFact document (title + abstract); doc-level BEIR protocol",
    "rungs": {
        "naive_tfidf_only": "mode='tfidf': TF-IDF cosine on the raw query, no query expansion",
        "hybrid_tfidf_bm25": "mode='hybrid': TF-IDF + BM25 score fusion (BM25 via saturating transform x/(x+8)) WITH the project's query expansion (rag.expand_query)",
        "hybrid_plus_rerank": "mode='hybrid_rerank': hybrid as above + project cross-score reranker (_cross_score) over the top-20 fused pool, combined 0.55*fused + 0.45*cross",
    },
}


# ------------------------------------------------------------------ data
def ensure_data() -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    if not (SF_DIR / "corpus.jsonl").exists():
        if not ZIP_PATH.exists():
            print(f"[data] downloading {SCIFACT_URL}", flush=True)
            urllib.request.urlretrieve(SCIFACT_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH) as z:
            bad = z.testzip()
            if bad is not None:
                raise RuntimeError(f"zip integrity check failed at {bad}")
            z.extractall(DATA_DIR)
    corpus = [json.loads(l) for l in open(SF_DIR / "corpus.jsonl", encoding="utf-8")]
    queries = {json.loads(l)["_id"]: json.loads(l)["text"]
               for l in open(SF_DIR / "queries.jsonl", encoding="utf-8")}

    def load_qrels(split: str):
        qrels: dict[str, dict[str, int]] = {}
        rows = 0
        with open(SF_DIR / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                rows += 1
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
        return qrels, rows

    test_qrels, test_rows = load_qrels("test")
    train_qrels, train_rows = load_qrels("train")
    grades = sorted({g for q in test_qrels.values() for g in q.values()})
    info = {
        "dataset": "SciFact (BEIR)",
        "source_url": SCIFACT_URL,
        "provenance": "BEIR benchmark distribution of SciFact (Wadden et al., EMNLP 2020), "
                      "downloaded from the UKP Lab BEIR datasets server",
        "corpus_documents": len(corpus),
        "queries_total_in_queries_jsonl": len(queries),
        "test_queries": len(test_qrels),
        "test_qrel_rows": test_rows,
        "test_qrel_grades": grades,
        "train_dev_queries": len(train_qrels),
        "train_dev_qrel_rows": train_rows,
        "split_used_for_tuning": "train/dev qrels ONLY (Phase-B screening); test qrels never tuned on",
    }
    print(f"[data] corpus={len(corpus)} test_queries={len(test_qrels)} "
          f"test_qrel_rows={test_rows} grades={grades} train_dev_queries={len(train_qrels)}", flush=True)
    return {"corpus": corpus, "queries": queries, "test_qrels": test_qrels,
            "train_qrels": train_qrels, "info": info}


def doc_text(d: dict) -> str:
    title = (d.get("title") or "").strip()
    text = (d.get("text") or "").strip()
    return f"{title}. {text}" if title else text


def build_chunks(corpus: list[dict]) -> list[Chunk]:
    return [Chunk(doc_id=d["_id"], chunk_id=0, text=doc_text(d),
                  metadata={"source": d["_id"], "title": d.get("title", "")})
            for d in corpus]


# ------------------------------------------------------------------ metrics
def metrics_for_ranking(ranked_doc_ids: list[str], qrel: dict[str, int]) -> dict:
    """Standard BEIR metrics for one query from a ranked doc-id list."""
    rels = [qrel.get(d, 0) for d in ranked_doc_ids]
    n_rel = sum(1 for g in qrel.values() if g > 0)

    def recall_at(k: int) -> float:
        return sum(1 for g in rels[:k] if g > 0) / n_rel if n_rel else 0.0

    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(rels[:10]))
    ideal = sorted((g for g in qrel.values() if g > 0), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    mrr = 0.0
    for i, g in enumerate(rels[:10]):
        if g > 0:
            mrr = 1.0 / (i + 1)
            break
    return {"recall_at_3": recall_at(3), "recall_at_5": recall_at(5),
            "ndcg_at_10": dcg / idcg if idcg else 0.0, "mrr_at_10": mrr}


def aggregate(per_query: list[dict]) -> dict:
    keys = ["recall_at_3", "recall_at_5", "ndcg_at_10", "mrr_at_10"]
    return {k: round(float(np.mean([p[k] for p in per_query])), 4) for k in keys}


# ------------------------------------------------------------------ results io
def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {}


def save_results(res: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[io] wrote {RESULTS_PATH}", flush=True)


# ------------------------------------------------------------------ stage: baseline
def stage_baseline(data: dict) -> None:
    res = load_results()
    res["dataset"] = data["info"]
    res["baseline_config"] = BASELINE_CONFIG
    res["not_benchmarked_by_scifact"] = (
        "Groundedness / abstention / injection-guardrail numbers are NOT measured by "
        "SciFact (it has no unanswerable or adversarial split and no generated answers); "
        "those figures remain from the internal self-authored eval in results.json only.")
    save_results(res)  # checkpoint: dataset + config recorded before any rung

    chunks = build_chunks(data["corpus"])
    t0 = time.time()
    retriever = Retriever(chunks)  # project defaults: alpha=0.55, BM25 k1=1.5/b=0.75
    index_s = time.time() - t0
    print(f"[index] built in {index_s:.1f}s over {len(chunks)} chunks, "
          f"vocab={len(retriever.embedder.vocab)}", flush=True)
    res["baseline_config"]["index_build_seconds"] = round(index_s, 2)
    res["baseline_config"]["n_chunks_indexed"] = len(chunks)
    res["baseline_config"]["vocab_size"] = len(retriever.embedder.vocab)

    test_ids = sorted(data["test_qrels"].keys(), key=lambda x: int(x))
    rungs = [("naive_tfidf_only", "tfidf"),
             ("hybrid_tfidf_bm25", "hybrid"),
             ("hybrid_plus_rerank", "hybrid_rerank")]
    res.setdefault("baseline_test", {})
    for rung_name, mode in rungs:
        per_q, lat = [], []
        t_start = time.time()
        for qid in test_ids:
            q = data["queries"][qid]
            s = time.perf_counter()
            hits = retriever.retrieve(q, top_k=10, mode=mode)
            lat.append((time.perf_counter() - s) * 1000)
            per_q.append(metrics_for_ranking([h.chunk.doc_id for h in hits],
                                             data["test_qrels"][qid]))
        entry = aggregate(per_q)
        entry["mean_latency_ms"] = round(statistics.mean(lat), 2)
        entry["median_latency_ms"] = round(statistics.median(lat), 2)
        entry["runtime_seconds_all_queries"] = round(time.time() - t_start, 2)
        entry["n_queries"] = len(test_ids)
        res["baseline_test"][rung_name] = entry
        save_results(res)  # checkpoint after EVERY rung
        print(f"[baseline:{rung_name}] {entry}", flush=True)
    print("[baseline] done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["baseline", "dev", "dev2", "final", "all"], default="baseline")
    args = ap.parse_args()
    data = ensure_data()
    if args.stage in ("baseline", "all"):
        stage_baseline(data)
    if args.stage == "dev2":
        stage_dev2(data)
    if args.stage in ("dev", "final", "all"):
        stage_dev_and_final(data, only_final=(args.stage == "final"))


# ------------------------------------------------------------------ Phase B
# Improvement phase (brief: REAL_BENCH_IMPROVEMENTS.md, AgentForge section).
# All screening happens on train/dev queries ONLY. The final selected pipeline
# gets exactly one test evaluation (stage "final").

DEV_SLICE_N = 250
ADOPT_EPS = 0.003  # minimum dev-slice nDCG@10 gain required to adopt a lever

# ------------------------------------------------ Porter stemmer (Porter
# 1980, implemented from the published algorithm) for the tokenisation lever.
def _p_cons(w: str, i: int) -> bool:
    ch = w[i]
    if ch in "aeiou":
        return False
    if ch == "y":
        return not _p_cons(w, i - 1) if i > 0 else True
    return True


def _p_m(w: str) -> int:
    n, i, L = 0, 0, len(w)
    while i < L and _p_cons(w, i):
        i += 1
    while i < L:
        while i < L and not _p_cons(w, i):
            i += 1
        if i >= L:
            break
        n += 1
        while i < L and _p_cons(w, i):
            i += 1
    return n


def _p_vowel_in(w: str) -> bool:
    return any(not _p_cons(w, i) for i in range(len(w)))


def _p_doublec(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _p_cons(w, len(w) - 1)


def _p_cvc(w: str) -> bool:
    return (len(w) >= 3 and _p_cons(w, len(w) - 3) and not _p_cons(w, len(w) - 2)
            and _p_cons(w, len(w) - 1) and w[-1] not in "wxy")


def porter_stem(w: str) -> str:
    if len(w) <= 2:
        return w
    # Step 1a
    if w.endswith("sses"):
        w = w[:-4] + "ss"
    elif w.endswith("ies"):
        w = w[:-3] + "i"
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    # Step 1b
    if w.endswith("eed"):
        if _p_m(w[:-3]) > 0:
            w = w[:-1]
    else:
        cut = None
        for suf in ("ing", "ed"):
            if w.endswith(suf) and _p_vowel_in(w[:-len(suf)]):
                cut = w[:-len(suf)]
                break
        if cut is not None:
            w = cut
            if w.endswith(("at", "bl", "iz")):
                w = w + "e"
            elif _p_doublec(w) and w[-1] not in "lsz":
                w = w[:-1]
            elif _p_m(w) == 1 and _p_cvc(w):
                w = w + "e"
    # Step 1c
    if w.endswith("y") and _p_vowel_in(w[:-1]):
        w = w[:-1] + "i"
    # Step 2
    step2 = {"ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
             "izer": "ize", "abli": "able", "alli": "al", "entli": "ent",
             "eli": "e", "ousli": "ous", "ization": "ize", "ation": "ate",
             "ator": "ate", "alism": "al", "iveness": "ive", "fulness": "ful",
             "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
             "logi": "log"}
    for suf in sorted(step2, key=len, reverse=True):
        if w.endswith(suf) and _p_m(w[:-len(suf)]) > 0:
            w = w[:-len(suf)] + step2[suf]
            break
    # Step 3
    step3 = {"icate": "ic", "ative": "", "alize": "al", "iciti": "ic",
             "ical": "ic", "ful": "", "ness": ""}
    for suf in sorted(step3, key=len, reverse=True):
        if w.endswith(suf) and _p_m(w[:-len(suf)]) > 0:
            w = w[:-len(suf)] + step3[suf]
            break
    # Step 4
    step4 = ["ement", "ance", "ence", "able", "ible", "ment", "ant", "ent",
             "ion", "ism", "ate", "iti", "ous", "ive", "ize", "al", "er", "ic", "ou"]
    for suf in sorted(step4, key=len, reverse=True):
        if w.endswith(suf):
            stem = w[:-len(suf)]
            ok = _p_m(stem) > 1 and (suf != "ion" or (stem and stem[-1] in "st"))
            if ok:
                w = stem
            break
    # Step 5a
    if w.endswith("e"):
        stem = w[:-1]
        m = _p_m(stem)
        if m > 1 or (m == 1 and not _p_cvc(stem)):
            w = stem
    # Step 5b
    if _p_m(w) > 1 and _p_doublec(w) and w.endswith("l"):
        w = w[:-1]
    return w


_HYPHEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")


def content_tokens_v2(text: str) -> list[str]:
    """Tokenisation lever: Porter-stemmed tokens + hyphenated compounds kept
    both split (project default) and joined (covid-19 -> covid19)."""
    out = [porter_stem(t) for t in rag_mod.tokenize(text) if t not in RAG_STOPWORDS]
    for m in _HYPHEN_RE.finditer(text.lower()):
        joined = "".join(m.group(0).split("-"))
        if len(joined) > 1:
            out.append(porter_stem(joined))
    return out


_ORIG_CONTENT_TOKENS = rag_mod.content_tokens


def set_tokenizer(v2: bool) -> None:
    rag_mod.content_tokens = content_tokens_v2 if v2 else _ORIG_CONTENT_TOKENS


# ------------------------------------------------ Fast BM25 (postings-based;
# mathematically identical to rag.BM25Index.scores — verified in the dev run —
# parameterised by k1/b so the sweep does not rebuild indexes)
class FastBM25:
    def __init__(self, index: BM25Index):
        self.idf = index.idf
        self.doc_len = index.doc_len
        self.avgdl = index.avgdl
        self.n = len(index.tf)
        post: dict[str, list[tuple[int, int]]] = {}
        for i, counts in enumerate(index.tf):
            for tok, f in counts.items():
                post.setdefault(tok, []).append((i, f))
        self.postings = {t: (np.array([d for d, _ in v], dtype=np.int64),
                             np.array([f for _, f in v], dtype=np.float64))
                         for t, v in post.items()}

    def scores(self, query_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        for tok in dict.fromkeys(query_tokens):
            entry = self.postings.get(tok)
            if entry is None:
                continue
            docs, fs = entry
            denom = fs + k1 * (1 - b + b * self.doc_len[docs] / self.avgdl)
            out[docs] += self.idf[tok] * (fs * (k1 + 1)) / denom
        return out

    def matrix(self, queries_tokens: list[list[str]], k1: float, b: float) -> np.ndarray:
        return np.vstack([self.scores(t, k1, b) for t in queries_tokens])


def _sat(x: np.ndarray) -> np.ndarray:
    return x / (x + 8.0)  # the project's BM25 saturating transform


def _rank_matrix(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    vals = np.broadcast_to(np.arange(scores.shape[1]), scores.shape)
    np.put_along_axis(ranks, order, vals, axis=1)
    return ranks


def eval_score_matrix(scores: np.ndarray, doc_ids: list[str], qids: list[str],
                      qrels: dict) -> dict:
    top = np.argsort(-scores, axis=1)[:, :10]
    per = [metrics_for_ranking([doc_ids[int(j)] for j in row], qrels[qid])
           for row, qid in zip(top, qids)]
    return aggregate(per)


# ------------------------------------------------ Pipeline family: the
# project's components (TFIDFEmbedder / BM25Index / expand_query /
# _cross_score) assembled under one selectable config spec.
class Family:
    """spec keys: tokenizer ('v1'|'v2'), bm25_k1, bm25_b,
    field_weights (None or (w_title, w_abstract)),
    fusion ('score'|'rrf'), fusion_weight (alpha for score fusion,
    w_tfidf for RRF), tfidf_title_beta (0 = off),
    rerank (None or {'pool': int, 'mix': float, 'features': 'base'|'ext'})"""

    def __init__(self, data: dict, spec: dict):
        self.spec = spec
        set_tokenizer(spec.get("tokenizer", "v1") == "v2")
        corpus = data["corpus"]
        self.doc_ids = [d["_id"] for d in corpus]
        self.chunks = build_chunks(corpus)
        self.titles = [(d.get("title") or "").strip() for d in corpus]
        self.texts = [(d.get("text") or "").strip() for d in corpus]
        self.embedder = TFIDFEmbedder().fit([c.text for c in self.chunks])
        self.matrix = self.embedder.embed_many([c.text for c in self.chunks])
        self.bm25 = FastBM25(BM25Index(self.chunks))
        self.bm25_title = self.bm25_text = None
        if spec.get("field_weights"):
            tc = [Chunk(d["_id"], 0, self.titles[i]) for i, d in enumerate(corpus)]
            xc = [Chunk(d["_id"], 0, self.texts[i]) for i, d in enumerate(corpus)]
            self.bm25_title = FastBM25(BM25Index(tc))
            self.bm25_text = FastBM25(BM25Index(xc))
        self.title_embedder = self.title_matrix = None
        if spec.get("tfidf_title_beta"):
            self.title_embedder = TFIDFEmbedder().fit(self.titles)
            self.title_matrix = self.title_embedder.embed_many(self.titles)
        self.title_token_sets = [set(content_tokens(t)) for t in self.titles]

    # -- token helpers respect the currently-set tokenizer
    def q_tokens(self, q: str, expanded: bool = True) -> list[str]:
        return content_tokens(expand_query(q) if expanded else q)

    def tfidf_scores(self, queries: list[str]) -> np.ndarray:
        rows = []
        for q in queries:
            qv = self.embedder.embed(expand_query(q)).astype(self.matrix.dtype, copy=False)
            s = self.matrix @ qv if np.linalg.norm(qv) else np.zeros(len(self.chunks))
            if self.spec.get("tfidf_title_beta"):
                tv = self.title_embedder.embed(expand_query(q)).astype(
                    self.title_matrix.dtype, copy=False)
                ts = self.title_matrix @ tv if np.linalg.norm(tv) else np.zeros(len(self.chunks))
                s = s + float(self.spec["tfidf_title_beta"]) * ts
            rows.append(s)
        return np.vstack(rows)

    def bm25_scores(self, queries: list[str]) -> np.ndarray:
        k1, b = self.spec.get("bm25_k1", 1.5), self.spec.get("bm25_b", 0.75)
        toks = [self.q_tokens(q) for q in queries]
        if self.spec.get("field_weights"):
            wt, wa = self.spec["field_weights"]
            return (wt * self.bm25_title.matrix(toks, k1, b)
                    + wa * self.bm25_text.matrix(toks, k1, b))
        return self.bm25.matrix(toks, k1, b)

    def fused_scores(self, queries: list[str],
                     tfidf_s: np.ndarray | None = None,
                     bm25_s: np.ndarray | None = None) -> np.ndarray:
        if tfidf_s is None:
            tfidf_s = self.tfidf_scores(queries)
        if bm25_s is None:
            bm25_s = self.bm25_scores(queries)
        w = float(self.spec.get("fusion_weight", 0.55))
        if self.spec.get("fusion", "score") == "rrf":
            return w / (60 + _rank_matrix(tfidf_s)) + (1 - w) / (60 + _rank_matrix(bm25_s))
        return w * tfidf_s + (1 - w) * _sat(bm25_s)

    def _feat_ext(self, qi: int, query: str, doc_idx: int) -> tuple[float, float]:
        """Extra rerank features: title term coverage + IDF-weighted
        best-sentence overlap."""
        q_toks = set(self.q_tokens(query))
        if not q_toks:
            return 0.0, 0.0
        title_cov = len(q_toks & self.title_token_sets[doc_idx]) / len(q_toks)
        idf = self.bm25.idf
        denom = sum(idf.get(t, 0.0) for t in q_toks) or 1.0
        best = 0.0
        for sent in rag_sentences(self.chunks[doc_idx].text):
            s_toks = set(content_tokens(sent))
            best = max(best, sum(idf.get(t, 0.0) for t in q_toks & s_toks) / denom)
        return title_cov, best

    def rerank_rankings(self, fused: np.ndarray, queries: list[str]) -> list[list[str]]:
        rr = self.spec["rerank"]
        pool_n, mix = int(rr["pool"]), float(rr["mix"])
        order = np.argsort(-fused, axis=1)[:, :pool_n]
        rankings = []
        for qi, q in enumerate(queries):
            pool = [int(j) for j in order[qi]]
            scored = []
            for j in pool:
                cross = _cross_score(expand_query(q), self.chunks[j])
                if rr.get("features") == "ext":
                    tc, si = self._feat_ext(qi, q, j)
                    cross = 0.6 * cross + 0.2 * tc + 0.2 * si
                scored.append(((1 - mix) * float(fused[qi, j]) + mix * cross, j))
            scored.sort(key=lambda x: -x[0])
            rankings.append([self.doc_ids[j] for _, j in scored[:10]])
        return rankings

    def evaluate(self, queries: list[str], qids: list[str], qrels: dict,
                 fused: np.ndarray | None = None) -> dict:
        if fused is None:
            fused = self.fused_scores(queries)
        if self.spec.get("rerank"):
            rankings = self.rerank_rankings(fused, queries)
            per = [metrics_for_ranking(r, qrels[qid]) for r, qid in zip(rankings, qids)]
            return aggregate(per)
        return eval_score_matrix(fused, self.doc_ids, qids, qrels)


BASELINE_SPEC = {"tokenizer": "v1", "bm25_k1": 1.5, "bm25_b": 0.75,
                 "field_weights": None, "fusion": "score", "fusion_weight": 0.55,
                 "tfidf_title_beta": 0,
                 "rerank": {"pool": 20, "mix": 0.45, "features": "base"}}

def _log_attempt(res: dict, attempt: dict) -> None:
    res.setdefault("improvement_log", []).append(attempt)
    save_results(res)
    print(f"[attempt] {attempt['lever']} :: {attempt['config']} -> "
          f"dev nDCG@10={attempt['metrics']['ndcg_at_10']} "
          f"decision={attempt['decision']}", flush=True)


def _attempt(lever, config, metrics, ref, decision, notes, dev_set=f"train/dev slice (first {DEV_SLICE_N} train queries)"):
    return {"lever": lever, "config": config, "dev_set": dev_set,
            "n_queries": DEV_SLICE_N, "metrics": metrics,
            "reference_ndcg_at_10": ref, "decision": decision, "notes": notes}


def stage_dev_and_final(data: dict, only_final: bool = False) -> None:
    res = load_results()
    if only_final:
        _run_final(data, res)
        return

    res["published_bar"] = {
        "source": "REAL_BENCH_IMPROVEMENTS.md (BEIR paper, Thakur et al. 2021)",
        "bm25_scifact_ndcg_at_10": 0.665,
        "pure_lexical_frontier_estimate": "0.66-0.70 nDCG@10",
        "note": "Absolute comparison valid only under the official protocol "
                "(full corpus, test queries, title+abstract) — which this run uses.",
    }
    res["dev_protocol"] = {
        "screening_set": f"first {DEV_SLICE_N} SciFact train/dev queries (qrels/train.tsv order)",
        "confirmation_set": "all 809 train/dev queries",
        "test_evaluations_allowed": "baseline ladder (done) + exactly one final-pipeline evaluation",
        "adoption_rule": f"adopt a lever only if dev-slice nDCG@10 improves by >= {ADOPT_EPS}",
        "rm3_expectation": "brief documents RM3 HURTS on SciFact (query drift on claim-style queries); "
                           "run only as a dev A/B, expected rejection",
    }
    # Porter stemmer self-check (classic Porter sample stems)
    checks = {"caresses": "caress", "ponies": "poni", "ties": "ti", "cats": "cat",
              "agreed": "agre", "motoring": "motor", "hopping": "hop",
              "relational": "relat", "probate": "probat", "rate": "rate",
              "cease": "ceas", "controll": "control", "roll": "roll"}
    got = {w: porter_stem(w) for w in checks}
    res["dev_protocol"]["porter_stemmer_self_check"] = {
        "passed": got == checks, "results": got}
    print(f"[dev] porter self-check passed={got == checks}", flush=True)
    save_results(res)

    qrels_tr = data["train_qrels"]
    dev_all_ids = list(qrels_tr.keys())
    dev_ids = dev_all_ids[:DEV_SLICE_N]
    dev_queries = [data["queries"][q] for q in dev_ids]

    # ---------- Step 0: dev diagnostics with the authentic project Retriever
    chunks = build_chunks(data["corpus"])
    retriever = Retriever(chunks)
    # FastBM25 equivalence verification vs rag.BM25Index.scores
    fast0 = FastBM25(retriever.bm25)
    equiv = True
    for q in dev_queries[:3]:
        a = retriever.bm25.scores(expand_query(q))
        b = fast0.scores(content_tokens(expand_query(q)), 1.5, 0.75)
        equiv = equiv and bool(np.allclose(a, b, atol=1e-9))
    print(f"[dev] FastBM25 == BM25Index.scores: {equiv}", flush=True)
    diag = {}
    for name, mode in [("tfidf_only", "tfidf"), ("bm25_only_component_check", None),
                       ("hybrid_default", "hybrid"), ("hybrid_rerank_default", "hybrid_rerank")]:
        if mode is None:
            per = [metrics_for_ranking(
                [chunks[int(j)].doc_id for j in np.argsort(
                    -fast0.scores(content_tokens(expand_query(q)), 1.5, 0.75))[:10]],
                qrels_tr[qid]) for q, qid in zip(dev_queries, dev_ids)]
            diag[name] = aggregate(per)
        else:
            per = [metrics_for_ranking(
                [h.chunk.doc_id for h in retriever.retrieve(q, top_k=10, mode=mode)],
                qrels_tr[qid]) for q, qid in zip(dev_queries, dev_ids)]
            diag[name] = aggregate(per)
        print(f"[dev-diag] {name}: {diag[name]}", flush=True)
    n_expanded = sum(1 for q in dev_queries if expand_query(q) != q)
    diag["queries_changed_by_project_expansion"] = f"{n_expanded}/{len(dev_queries)}"
    res["dev_baseline_slice"] = diag
    save_results(res)
    del retriever, fast0

    # ---------- Screening with Family (project components, selectable spec)
    cur_spec = {"tokenizer": "v1", "bm25_k1": 1.5, "bm25_b": 0.75,
                "field_weights": None, "fusion": "score", "fusion_weight": 0.55,
                "tfidf_title_beta": 0, "rerank": None}
    fam = Family(data, cur_spec)
    tfidf_s = fam.tfidf_scores(dev_queries)
    bm25_def = fam.bm25_scores(dev_queries)
    cur_metrics = eval_score_matrix(fam.fused_scores(dev_queries, tfidf_s, bm25_def),
                                    fam.doc_ids, dev_ids, qrels_tr)
    cur_fused = fam.fused_scores(dev_queries, tfidf_s, bm25_def)
    print(f"[dev] starting point (hybrid first-stage, defaults): {cur_metrics}", flush=True)

    # ---------- Step 1: BM25 k1/b grid (BM25-only, brief lever 3)
    best_kb, best_kb_m = None, None
    for k1 in (0.8, 1.0, 1.2, 1.5):
        for b in (0.35, 0.5, 0.65, 0.75):
            m = eval_score_matrix(fam.bm25.matrix(
                [fam.q_tokens(q) for q in dev_queries], k1, b), fam.doc_ids, dev_ids, qrels_tr)
            dec = "screened"
            _log_attempt(res, _attempt("bm25_k1_b_grid", {"bm25_k1": k1, "bm25_b": b},
                                       m, diag["bm25_only_component_check"]["ndcg_at_10"],
                                       dec, "BM25-only component screening on dev slice"))
            if best_kb_m is None or m["ndcg_at_10"] > best_kb_m["ndcg_at_10"]:
                best_kb, best_kb_m = (k1, b), m
    _log_attempt(res, _attempt("bm25_k1_b_selected", {"bm25_k1": best_kb[0], "bm25_b": best_kb[1]},
                               best_kb_m, diag["bm25_only_component_check"]["ndcg_at_10"],
                               "adopted_as_component",
                               "Best BM25-only (k1,b) on dev slice; carried into fusion screening"))
    cur_spec["bm25_k1"], cur_spec["bm25_b"] = best_kb
    bm25_cur = fam.bm25.matrix([fam.q_tokens(q) for q in dev_queries], best_kb[0], best_kb[1])

    # ---------- Step 2: fusion — score-fusion alpha sweep + RRF (brief lever 5)
    fusion_candidates = []
    for alpha in (0.0, 0.15, 0.3, 0.45, 0.55, 0.7, 0.85, 1.0):
        fused = alpha * tfidf_s + (1 - alpha) * _sat(bm25_cur)
        m = eval_score_matrix(fused, fam.doc_ids, dev_ids, qrels_tr)
        fusion_candidates.append(("score", alpha, m, fused))
        _log_attempt(res, _attempt("fusion_score_alpha", {"fusion": "score", "alpha": alpha,
                                                          "bm25_k1": best_kb[0], "bm25_b": best_kb[1]},
                                   m, cur_metrics["ndcg_at_10"], "screened",
                                   "Score fusion (project saturating BM25 transform), alpha sweep"))
    for w in (0.3, 0.5, 0.7):
        fused = w / (60 + _rank_matrix(tfidf_s)) + (1 - w) / (60 + _rank_matrix(bm25_cur))
        m = eval_score_matrix(fused, fam.doc_ids, dev_ids, qrels_tr)
        fusion_candidates.append(("rrf", w, m, fused))
        _log_attempt(res, _attempt("fusion_rrf", {"fusion": "rrf", "k": 60, "w_tfidf": w,
                                                  "bm25_k1": best_kb[0], "bm25_b": best_kb[1]},
                                   m, cur_metrics["ndcg_at_10"], "screened",
                                   "Reciprocal-rank fusion of TF-IDF and BM25 rankings"))
    fusion_candidates.sort(key=lambda x: -x[2]["ndcg_at_10"])
    f_type, f_w, f_m, f_fused = fusion_candidates[0]
    adopted = f_m["ndcg_at_10"] >= cur_metrics["ndcg_at_10"] + ADOPT_EPS
    _log_attempt(res, _attempt("fusion_selected", {"fusion": f_type, "weight": f_w},
                               f_m, cur_metrics["ndcg_at_10"],
                               "adopted" if adopted else "kept_default",
                               "Best fusion on dev slice vs default hybrid first-stage"))
    if adopted:
        cur_spec["fusion"], cur_spec["fusion_weight"] = f_type, f_w
        cur_metrics, cur_fused = f_m, f_fused

    # ---------- Step 2.5: RM3 pseudo-relevance feedback A/B (brief: expected HARM)
    from collections import Counter
    for a_prf in (0.5, 0.3):
        rows_t, rows_b = [], []
        for qi, q in enumerate(dev_queries):
            fb = [int(j) for j in np.argsort(-cur_fused[qi])[:10]]
            term_w: dict[str, float] = {}
            for d in fb:
                w_doc = float(cur_fused[qi, d])
                for t, f in Counter(content_tokens(fam.chunks[d].text)).items():
                    term_w[t] = term_w.get(t, 0.0) + w_doc * f * fam.bm25.idf.get(t, 0.0)
            q_toks = set(content_tokens(expand_query(q)))
            top_terms = [t for t, _ in sorted(term_w.items(), key=lambda x: -x[1])
                         if t not in q_toks][:15]
            qv = fam.embedder.embed(expand_query(q))
            centroid = fam.matrix[fb].mean(axis=0)
            q2 = (1 - a_prf) * qv + a_prf * centroid
            rows_t.append(fam.matrix @ q2)
            exp_toks = content_tokens(expand_query(q)) + top_terms
            rows_b.append(fam.bm25.scores(exp_toks, cur_spec["bm25_k1"], cur_spec["bm25_b"]))
        alpha = cur_spec["fusion_weight"] if cur_spec["fusion"] == "score" else 0.55
        fused_rm3 = alpha * np.vstack(rows_t) + (1 - alpha) * _sat(np.vstack(rows_b))
        m = eval_score_matrix(fused_rm3, fam.doc_ids, dev_ids, qrels_tr)
        delta = m["ndcg_at_10"] - cur_metrics["ndcg_at_10"]
        if delta >= ADOPT_EPS:
            dec = "positive_on_dev_but_not_shipped"
            note = ("RM3 beat the reference on this dev slice, contrary to the brief's "
                    "documented SciFact evidence; per the brief's explicit do-not-ship "
                    "guidance for RM3 on SciFact it was NOT adopted into the final pipeline.")
        else:
            dec = "dropped"
            note = ("Negative result: RM3 did not beat the reference on dev — the brief's "
                    "documented SciFact RM3 harm (query drift on short claim queries) is "
                    "consistent with this measurement. Not shipped.")
        _log_attempt(res, _attempt("rm3_prf", {"alpha_prf": a_prf, "feedback_docs": 10,
                                               "expansion_terms": 15},
                                   m, cur_metrics["ndcg_at_10"], dec, note))

    # ---------- Step 3: field weighting (brief lever 4) + TF-IDF title boost
    for wt in (1.5, 2.0, 3.0):
        spec_try = dict(cur_spec, field_weights=(wt, 1.0), rerank=None)
        fam_try = Family(data, spec_try)
        m = eval_score_matrix(fam_try.fused_scores(dev_queries), fam.doc_ids, dev_ids, qrels_tr)
        gain = m["ndcg_at_10"] - cur_metrics["ndcg_at_10"]
        dec = "adopted" if gain >= ADOPT_EPS else "dropped"
        _log_attempt(res, _attempt("field_weighting_bm25f", {"w_title": wt, "w_abstract": 1.0},
                                   m, cur_metrics["ndcg_at_10"], dec,
                                   "BM25F-style: separate title/abstract BM25 indexes, "
                                   "score = w_t*BM25(title) + w_a*BM25(abstract), then fused as current spec"))
        if dec == "adopted":
            cur_spec = spec_try
            fam = fam_try
            cur_metrics = m
            cur_fused = fam.fused_scores(dev_queries)
            tfidf_s = fam.tfidf_scores(dev_queries)
        else:
            del fam_try
    for beta in (0.25, 0.5):
        spec_try = dict(cur_spec, tfidf_title_beta=beta, rerank=None)
        fam_try = Family(data, spec_try)
        m = eval_score_matrix(fam_try.fused_scores(dev_queries), fam.doc_ids, dev_ids, qrels_tr)
        gain = m["ndcg_at_10"] - cur_metrics["ndcg_at_10"]
        dec = "adopted" if gain >= ADOPT_EPS else "dropped"
        _log_attempt(res, _attempt("tfidf_title_boost", {"tfidf_title_beta": beta},
                                   m, cur_metrics["ndcg_at_10"], dec,
                                   "TF-IDF side field boost: body+title cosine + beta * title-only cosine"))
        if dec == "adopted":
            cur_spec = spec_try
            fam = fam_try
            cur_metrics = m
            cur_fused = fam.fused_scores(dev_queries)
            tfidf_s = fam.tfidf_scores(dev_queries)
        else:
            del fam_try

    # ---------- Step 4: tokenisation v2 (Porter stemming + compounds, brief lever 2)
    spec_try = dict(cur_spec, tokenizer="v2", rerank=None)
    fam_try = Family(data, spec_try)
    m = eval_score_matrix(fam_try.fused_scores(dev_queries), fam.doc_ids, dev_ids, qrels_tr)
    gain = m["ndcg_at_10"] - cur_metrics["ndcg_at_10"]
    dec = "adopted" if gain >= ADOPT_EPS else "dropped"
    _log_attempt(res, _attempt("tokenisation_v2", {"tokenizer": "porter_stem + hyphen_compounds"},
                               m, cur_metrics["ndcg_at_10"], dec,
                               "Same selected config family, rebuilt with stemmed/compound tokenisation"))
    if dec == "adopted":
        cur_spec = spec_try
        fam = fam_try
        cur_metrics = m
        cur_fused = fam.fused_scores(dev_queries)
        tfidf_s = fam.tfidf_scores(dev_queries)
    else:
        del fam_try
        set_tokenizer(cur_spec.get("tokenizer", "v1") == "v2")

    # ---------- Step 5: reranker (pool / mix / extended features, brief lever 6)
    pool_n = 50
    pool_order = np.argsort(-cur_fused, axis=1)[:, :pool_n]
    cross_arr = np.zeros((len(dev_queries), pool_n))
    ext_arr = np.zeros((len(dev_queries), pool_n))
    for qi, q in enumerate(dev_queries):
        for pj, j in enumerate(pool_order[qi]):
            j = int(j)
            c = _cross_score(expand_query(q), fam.chunks[j])
            cross_arr[qi, pj] = c
            tc, si = fam._feat_ext(qi, q, j)
            ext_arr[qi, pj] = 0.6 * c + 0.2 * tc + 0.2 * si

    def _rerank_metrics(pool_use: int, mix: float, feats: np.ndarray) -> dict:
        per = []
        for qi, qid in enumerate(dev_ids):
            pool = [int(j) for j in pool_order[qi][:pool_use]]
            comb = [(1 - mix) * float(cur_fused[qi, j]) + mix * float(feats[qi, pj])
                    for pj, j in enumerate(pool)]
            order_local = np.argsort(-np.array(comb))[:10]
            per.append(metrics_for_ranking([fam.doc_ids[pool[i]] for i in order_local],
                                           qrels_tr[qid]))
        return aggregate(per)

    rerank_candidates = []
    for pool_use, mix, feats, fname in [(20, 0.45, cross_arr, "base"),
                                        (50, 0.45, cross_arr, "base"),
                                        (50, 0.3, cross_arr, "base"),
                                        (50, 0.6, cross_arr, "base"),
                                        (50, 0.45, ext_arr, "ext")]:
        m = _rerank_metrics(pool_use, mix, feats)
        rerank_candidates.append((pool_use, mix, fname, m))
        _log_attempt(res, _attempt("rerank", {"pool": pool_use, "mix": mix, "features": fname},
                                   m, cur_metrics["ndcg_at_10"], "screened",
                                   "Lexical rerank of the fused pool; ext = cross-score + "
                                   "title coverage + IDF-weighted best-sentence overlap"))
    rerank_candidates.sort(key=lambda x: -x[3]["ndcg_at_10"])
    p_use, p_mix, p_feat, p_m = rerank_candidates[0]
    if p_m["ndcg_at_10"] >= cur_metrics["ndcg_at_10"] + ADOPT_EPS:
        cur_spec["rerank"] = {"pool": p_use, "mix": p_mix, "features": p_feat}
        _log_attempt(res, _attempt("rerank_selected", dict(cur_spec["rerank"]), p_m,
                                   cur_metrics["ndcg_at_10"], "adopted",
                                   "Best rerank config on dev slice"))
        cur_metrics = p_m
    else:
        _log_attempt(res, _attempt("rerank_selected", {"pool": p_use, "mix": p_mix,
                                                       "features": p_feat}, p_m,
                                   cur_metrics["ndcg_at_10"], "dropped",
                                   "No rerank config beat the first stage by the adoption epsilon on dev"))

    # ---------- Step 6: full-dev (809) confirmation — candidate vs project baseline
    all_ids = dev_all_ids
    all_queries = [data["queries"][q] for q in all_ids]
    fam_base = Family(data, BASELINE_SPEC)
    base_full = fam_base.evaluate(all_queries, all_ids, qrels_tr)
    del fam_base
    set_tokenizer(cur_spec.get("tokenizer", "v1") == "v2")
    cand_full = fam.evaluate(all_queries, all_ids, qrels_tr)
    print(f"[dev-full] baseline={base_full} candidate={cand_full}", flush=True)
    if cand_full["ndcg_at_10"] > base_full["ndcg_at_10"]:
        final_spec, how = cur_spec, "candidate beat project baseline on full dev"
    else:
        final_spec, how = BASELINE_SPEC, "candidate did not beat project baseline on full dev; baseline selected (plateau)"
    res["final_selection"] = {
        "selected_spec": final_spec,
        "selection": how,
        "candidate_spec": cur_spec,
        "candidate_full_dev_metrics": cand_full,
        "baseline_full_dev_metrics": base_full,
        "full_dev_n_queries": len(all_ids),
        "candidate_dev_slice_metrics": cur_metrics,
    }
    save_results(res)
    print(f"[selection] {how}: {final_spec}", flush=True)

    _run_final(data, res)


def stage_dev2(data: dict) -> None:
    """Resume of the dev stage after an OOM kill during Step 3 (two live
    Family objects + embed_many transient exceeded the shared memory cap).
    Steps 0-2.5 are already logged in results_real.json; this runs Steps 3-6
    and the final test evaluation with strict one-Family-at-a-time memory
    discipline (screening families use float32 TF-IDF matrices)."""
    import gc
    res = load_results()
    log = res["improvement_log"]
    kb = next(a for a in log if a["lever"] == "bm25_k1_b_selected")["config"]
    fus = next(a for a in log if a["lever"] == "fusion_selected")
    cur_spec = {"tokenizer": "v1", "bm25_k1": kb["bm25_k1"], "bm25_b": kb["bm25_b"],
                "field_weights": None, "fusion": fus["config"]["fusion"],
                "fusion_weight": fus["config"]["weight"], "tfidf_title_beta": 0,
                "rerank": None}
    cur_metrics = fus["metrics"]
    if not any(a.get("lever") == "run_note" for a in log):
        res["improvement_log"].append({
            "lever": "run_note",
            "config": {"resumed_from_step": 3, "reason": "first dev process was killed "
                       "(OOM) during the Step-3 Family rebuild; steps 0-2.5 results are "
                       "from that run and are unchanged"},
            "dev_set": f"train/dev slice (first {DEV_SLICE_N} train queries)",
            "n_queries": DEV_SLICE_N, "metrics": cur_metrics,
            "reference_ndcg_at_10": cur_metrics["ndcg_at_10"],
            "decision": "resumed", "notes": "State reconstructed from the logged "
                       "bm25_k1_b_selected and fusion_selected attempts."})
        save_results(res)
    print(f"[dev2] resumed with spec {cur_spec}, slice metrics {cur_metrics}", flush=True)

    qrels_tr = data["train_qrels"]
    dev_all_ids = list(qrels_tr.keys())
    dev_ids = dev_all_ids[:DEV_SLICE_N]
    dev_queries = [data["queries"][q] for q in dev_ids]

    def build(spec, dtype32=True):
        fam = Family(data, spec)
        if dtype32:
            fam.matrix = fam.matrix.astype(np.float32)
            if fam.title_matrix is not None:
                fam.title_matrix = fam.title_matrix.astype(np.float32)
        return fam

    # ---------- Step 3: field weighting + TF-IDF title boost (one Family at a time)
    for lever, patch in [("field_weighting_bm25f", None), ("tfidf_title_boost", None)]:
        values = (1.5, 2.0, 3.0) if lever == "field_weighting_bm25f" else (0.25, 0.5)
        for v in values:
            spec_try = dict(cur_spec, rerank=None)
            if lever == "field_weighting_bm25f":
                spec_try["field_weights"] = (v, 1.0)
                cfg = {"w_title": v, "w_abstract": 1.0}
                note = ("BM25F-style: separate title/abstract BM25 indexes, "
                        "score = w_t*BM25(title) + w_a*BM25(abstract), then fused as current spec")
            else:
                spec_try["tfidf_title_beta"] = v
                cfg = {"tfidf_title_beta": v}
                note = "TF-IDF side field boost: body+title cosine + beta * title-only cosine"
            fam_try = build(spec_try)
            m = eval_score_matrix(fam_try.fused_scores(dev_queries), fam_try.doc_ids,
                                  dev_ids, qrels_tr)
            del fam_try
            gc.collect()
            gain = m["ndcg_at_10"] - cur_metrics["ndcg_at_10"]
            dec = "adopted" if gain >= ADOPT_EPS else "dropped"
            _log_attempt(res, _attempt(lever, cfg, m, cur_metrics["ndcg_at_10"], dec, note))
            if dec == "adopted":
                cur_spec = spec_try
                cur_metrics = m

    # ---------- Step 4: tokenisation v2
    spec_try = dict(cur_spec, tokenizer="v2", rerank=None)
    fam_try = build(spec_try)
    m = eval_score_matrix(fam_try.fused_scores(dev_queries), fam_try.doc_ids,
                          dev_ids, qrels_tr)
    del fam_try
    gc.collect()
    gain = m["ndcg_at_10"] - cur_metrics["ndcg_at_10"]
    dec = "adopted" if gain >= ADOPT_EPS else "dropped"
    _log_attempt(res, _attempt("tokenisation_v2",
                               {"tokenizer": "porter_stem + hyphen_compounds"},
                               m, cur_metrics["ndcg_at_10"], dec,
                               "Same selected config family, rebuilt with stemmed/compound tokenisation"))
    if dec == "adopted":
        cur_spec = spec_try
        cur_metrics = m
    set_tokenizer(cur_spec.get("tokenizer", "v1") == "v2")

    # ---------- Step 5: reranker screening on the adopted family's fused scores
    fam = build(cur_spec)
    cur_fused = fam.fused_scores(dev_queries)
    pool_n = 50
    pool_order = np.argsort(-cur_fused, axis=1)[:, :pool_n]
    cross_arr = np.zeros((len(dev_queries), pool_n))
    ext_arr = np.zeros((len(dev_queries), pool_n))
    for qi, q in enumerate(dev_queries):
        for pj, j in enumerate(pool_order[qi]):
            j = int(j)
            c = _cross_score(expand_query(q), fam.chunks[j])
            cross_arr[qi, pj] = c
            tc, si = fam._feat_ext(qi, q, j)
            ext_arr[qi, pj] = 0.6 * c + 0.2 * tc + 0.2 * si

    def _rerank_metrics(pool_use: int, mix: float, feats: np.ndarray) -> dict:
        per = []
        for qi, qid in enumerate(dev_ids):
            pool = [int(j) for j in pool_order[qi][:pool_use]]
            comb = [(1 - mix) * float(cur_fused[qi, j]) + mix * float(feats[qi, pj])
                    for pj, j in enumerate(pool)]
            order_local = np.argsort(-np.array(comb))[:10]
            per.append(metrics_for_ranking([fam.doc_ids[pool[i]] for i in order_local],
                                           qrels_tr[qid]))
        return aggregate(per)

    rerank_candidates = []
    for pool_use, mix, feats, fname in [(20, 0.45, cross_arr, "base"),
                                        (50, 0.45, cross_arr, "base"),
                                        (50, 0.3, cross_arr, "base"),
                                        (50, 0.6, cross_arr, "base"),
                                        (50, 0.45, ext_arr, "ext")]:
        m = _rerank_metrics(pool_use, mix, feats)
        rerank_candidates.append((pool_use, mix, fname, m))
        _log_attempt(res, _attempt("rerank", {"pool": pool_use, "mix": mix, "features": fname},
                                   m, cur_metrics["ndcg_at_10"], "screened",
                                   "Lexical rerank of the fused pool; ext = cross-score + "
                                   "title coverage + IDF-weighted best-sentence overlap"))
    rerank_candidates.sort(key=lambda x: -x[3]["ndcg_at_10"])
    p_use, p_mix, p_feat, p_m = rerank_candidates[0]
    if p_m["ndcg_at_10"] >= cur_metrics["ndcg_at_10"] + ADOPT_EPS:
        cur_spec["rerank"] = {"pool": p_use, "mix": p_mix, "features": p_feat}
        _log_attempt(res, _attempt("rerank_selected", dict(cur_spec["rerank"]), p_m,
                                   cur_metrics["ndcg_at_10"], "adopted",
                                   "Best rerank config on dev slice"))
        cur_metrics = p_m
    else:
        _log_attempt(res, _attempt("rerank_selected", {"pool": p_use, "mix": p_mix,
                                                       "features": p_feat}, p_m,
                                   cur_metrics["ndcg_at_10"], "dropped",
                                   "No rerank config beat the first stage by the adoption epsilon on dev"))
    del fam, cur_fused, cross_arr, ext_arr
    gc.collect()

    # ---------- Step 6: full-dev (809) confirmation, one Family at a time (float64)
    all_ids = dev_all_ids
    all_queries = [data["queries"][q] for q in all_ids]
    fam_base = Family(data, BASELINE_SPEC)
    base_full = fam_base.evaluate(all_queries, all_ids, qrels_tr)
    del fam_base
    gc.collect()
    fam_cand = Family(data, cur_spec)
    cand_full = fam_cand.evaluate(all_queries, all_ids, qrels_tr)
    del fam_cand
    gc.collect()
    print(f"[dev-full] baseline={base_full} candidate={cand_full}", flush=True)
    if cand_full["ndcg_at_10"] > base_full["ndcg_at_10"]:
        final_spec, how = cur_spec, "candidate beat project baseline on full dev"
    else:
        final_spec, how = BASELINE_SPEC, ("candidate did not beat project baseline on "
                                          "full dev; baseline selected (plateau)")
    res["final_selection"] = {
        "selected_spec": final_spec,
        "selection": how,
        "candidate_spec": cur_spec,
        "candidate_full_dev_metrics": cand_full,
        "baseline_full_dev_metrics": base_full,
        "full_dev_n_queries": len(all_ids),
        "candidate_dev_slice_metrics": cur_metrics,
    }
    save_results(res)
    print(f"[selection] {how}: {final_spec}", flush=True)
    _run_final(data, res)


def _run_final(data: dict, res: dict) -> None:
    """Exactly ONE test evaluation of the dev-selected pipeline."""
    if "final_test" in res:
        print("[final] already recorded — refusing a second final test evaluation", flush=True)
        return
    sel = res.get("final_selection")
    if not sel:
        print("[final] no final_selection recorded; run --stage dev first", flush=True)
        return
    spec = sel["selected_spec"]
    test_qrels = data["test_qrels"]
    test_ids = sorted(test_qrels.keys(), key=lambda x: int(x))
    test_queries = [data["queries"][q] for q in test_ids]
    if spec == BASELINE_SPEC:
        res["final_test"] = {
            "note": "Selected pipeline IS the project baseline (hybrid+rerank); its test "
                    "metrics are the baseline ladder's — no second test run was performed.",
            "config": spec,
            "metrics": res["baseline_test"]["hybrid_plus_rerank"],
        }
        save_results(res)
        print("[final] selected == baseline rung; recorded without re-running test", flush=True)
        return
    fam = Family(data, spec)
    per, lat = [], []
    for q, qid in zip(test_queries, test_ids):
        t0 = time.perf_counter()
        fused = fam.fused_scores([q])
        if spec.get("rerank"):
            ranking = fam.rerank_rankings(fused, [q])[0]
        else:
            ranking = [fam.doc_ids[int(j)] for j in np.argsort(-fused[0])[:10]]
        lat.append((time.perf_counter() - t0) * 1000)
        per.append(metrics_for_ranking(ranking, test_qrels[qid]))
    m = aggregate(per)
    m["mean_latency_ms"] = round(statistics.mean(lat), 2)
    m["median_latency_ms"] = round(statistics.median(lat), 2)
    m["n_queries"] = len(test_ids)
    res["final_test"] = {
        "note": "Single test evaluation of the dev-selected pipeline (test touched "
                "exactly twice overall: baseline ladder + this run).",
        "config": spec,
        "metrics": m,
    }
    save_results(res)
    print(f"[final] test metrics: {m}", flush=True)


if __name__ == "__main__":
    main()
