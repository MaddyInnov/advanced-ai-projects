"""
AgentForge-RAG — core retrieval pipeline. Author: Ashutosh Sahoo

Zero external APIs, zero heavy dependencies (stdlib + numpy only).

Pipeline: load -> chunk (overlap + metadata) -> hybrid retrieval
(TF-IDF cosine + BM25 keyword scoring, score-fused) -> query expansion ->
cross-scoring rerank of the top candidate pool -> grounded, cited,
extractive answer composition with an abstain path.

This mirrors the architecture of current production / SOTA RAG stacks
(dense+sparse hybrid, expansion, reranker — e.g. hybrid BM25+vector with a
cross-encoder reranker) re-implemented in zero-dependency form: TF-IDF
stands in for the dense retriever, BM25 is implemented from scratch, and
the reranker is a deterministic query/chunk cross-scorer standing in for a
cross-encoder. Interfaces are kept so each stage swaps 1:1 for its neural
production counterpart (see README). evaluate.py ablates naive TF-IDF-only
against this full stack so the gain is measured, not asserted.

Honesty note: the default embedder is a TF-IDF / hashing-style local
baseline for a zero-dependency demo. It is NOT a neural embedding model.
The `Embedder` interface exists precisely so production can swap in
sentence-transformers + FAISS/pgvector by changing one class (see README).
"""
from __future__ import annotations

import hashlib
import math
import pathlib
import re
from dataclasses import dataclass, field

import numpy as np

STOPWORDS = set("""a an the is are was were be been being of to in on for and or not
with by at from as it its this that these those you your we our they their he she
must should shall will would can could may might do does did have has had what
when how long who whom which where why if then than so such into about after
before during within without per via all any each other more most""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+")
SENT_RE = re.compile(r"(?<=[.!?])\s+")

# Small, auditable synonym/expansion map — query rewriting step.
EXPANSIONS: dict[str, list[str]] = {
    "mfa": ["multi", "factor", "authentication"],
    "vpn": ["virtual", "private", "network", "remote", "access"],
    "byod": ["bring", "your", "own", "device", "personal", "device"],
    "rto": ["recovery", "time", "objective"],
    "rpo": ["recovery", "point", "objective"],
    "phishing": ["suspicious", "email", "report"],
    "ransomware": ["ransom", "encryption", "encrypted"],
    "password": ["passwords", "credential", "credentials"],
    "offboarding": ["leaver", "leaving", "departing", "offboard"],
    "onboarding": ["joiner", "joiners", "new", "employee"],
    "cvss": ["severity", "score", "vulnerability"],
    "dlp": ["data", "loss", "prevention"],
    "owasp": ["secure", "coding", "developers"],
    # General domain synonyms (how users actually phrase things vs. how
    # policy documents phrase them) — the paraphrase split in evaluate.py
    # exists to measure whether these help.
    "fake": ["phishing", "suspicious"],
    "worst": ["severity", "critical", "incident"],
    "react": ["response", "incident"],
    "locked": ["encrypted", "encryption", "ransomware"],
    "proof": ["authentication", "identity"],
    "flaws": ["vulnerabilities", "vulnerability"],
    "programmers": ["developers", "coding"],
    "crypto": ["cryptocurrency"],
    "coins": ["cryptocurrency", "mining"],
    "secret": ["password", "credential"],
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


# ---------------------------------------------------------------- embedders
class Embedder:
    """Interface: production swap-in point (e.g. SentenceTransformerEmbedder)."""

    def fit(self, texts: list[str]) -> "Embedder":
        return self

    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts]) if texts else np.zeros((0, 0))


class TFIDFEmbedder(Embedder):
    """Vocab TF-IDF implemented in numpy: TF * IDF, L2-normalised.

    Also exposes a fixed-size hashing projection (`hash_embed`) which is the
    classic hashing-vectoriser trick — included so the interface mirrors how
    a fixed-dimension dense embedder behaves.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)

    def fit(self, texts: list[str]) -> "TFIDFEmbedder":
        df: dict[str, int] = {}
        for text in texts:
            for tok in set(content_tokens(text)):
                df[tok] = df.get(tok, 0) + 1
        self.vocab = {tok: i for i, tok in enumerate(sorted(df))}
        n = max(len(texts), 1)
        self.idf = np.array(
            [math.log((1 + n) / (1 + df[tok])) + 1.0 for tok in sorted(df)],
            dtype=np.float64,
        )
        return self

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float64)
        toks = content_tokens(text)
        if not toks or not len(self.vocab):
            return vec
        for tok in toks:
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        vec = vec * self.idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    @staticmethod
    def hash_embed(text: str, dim: int = 512) -> np.ndarray:
        """Fixed-dimension hashing projection (stateless alternative)."""
        vec = np.zeros(dim, dtype=np.float64)
        for tok in content_tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


# ---------------------------------------------------------------- chunking
@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"[{self.doc_id}:{self.chunk_id}]"


def load_documents(corpus_dir: str | pathlib.Path) -> list[tuple[str, str]]:
    docs = []
    for path in sorted(pathlib.Path(corpus_dir).glob("*")):
        if path.suffix.lower() in {".md", ".txt"} and path.is_file():
            docs.append((path.stem, path.read_text(encoding="utf-8", errors="ignore")))
    return docs


def chunk_text(text: str, chunk_words: int = 120, overlap_words: int = 30) -> list[str]:
    """Word-window chunking with overlap. Deterministic and testable."""
    if chunk_words <= overlap_words:
        raise ValueError("chunk_words must be greater than overlap_words")
    # Strip markdown headings markers but keep heading words (they carry signal)
    clean = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    words = clean.split()
    if not words:
        return []
    chunks, start, step = [], 0, chunk_words - overlap_words
    while start < len(words):
        piece = words[start:start + chunk_words]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks


def build_chunks(corpus_dir, chunk_words: int = 120, overlap_words: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id, text in load_documents(corpus_dir):
        for i, piece in enumerate(chunk_text(text, chunk_words, overlap_words)):
            chunks.append(Chunk(doc_id=doc_id, chunk_id=i, text=piece,
                                metadata={"source": doc_id, "chunk_index": i,
                                          "n_words": len(piece.split())}))
    return chunks


# ---------------------------------------------------------------- retriever
def expand_query(query: str) -> str:
    """Query rewriting/expansion: add synonyms for known abbreviations/terms."""
    toks = tokenize(query)
    extra: list[str] = []
    for tok in toks:
        extra.extend(EXPANSIONS.get(tok, []))
    # also trigger expansion when a phrase's key term appears
    return query + (" " + " ".join(extra) if extra else "")


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


class BM25Index:
    """Okapi BM25 implemented from scratch (stdlib + numpy), k1=1.5, b=0.75."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [content_tokens(c.text) for c in chunks]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        df: dict[str, int] = {}
        for toks in self.doc_tokens:
            for tok in set(toks):
                df[tok] = df.get(tok, 0) + 1
        n = len(chunks)
        # Robertson/Spärck-Jones IDF, floored at 0
        self.idf = {t: max(math.log(1 + (n - c + 0.5) / (c + 0.5)), 0.0) for t, c in df.items()}
        self.tf: list[dict[str, int]] = []
        for toks in self.doc_tokens:
            counts: dict[str, int] = {}
            for tok in toks:
                counts[tok] = counts.get(tok, 0) + 1
            self.tf.append(counts)

    def scores(self, query: str) -> np.ndarray:
        q_toks = list(dict.fromkeys(content_tokens(query)))  # unique terms
        out = np.zeros(len(self.tf), dtype=np.float64)
        if not q_toks or self.avgdl == 0:
            return out
        for i, counts in enumerate(self.tf):
            denom_norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
            s = 0.0
            for tok in q_toks:
                f = counts.get(tok, 0)
                if f:
                    s += self.idf.get(tok, 0.0) * (f * (self.k1 + 1)) / (f + denom_norm)
            out[i] = s
        return out


def _cross_score(query: str, chunk: Chunk) -> float:
    """Deterministic reranker cross-score (cross-encoder stand-in):
    query-term coverage of the chunk + best-sentence overlap + phrase bonus."""
    q_toks = set(content_tokens(query))
    if not q_toks:
        return 0.0
    chunk_toks = set(content_tokens(chunk.text))
    coverage = len(q_toks & chunk_toks) / len(q_toks)
    best_sent = 0.0
    for sent in _sentences(chunk.text):
        s_toks = set(content_tokens(sent))
        if s_toks:
            best_sent = max(best_sent, len(q_toks & s_toks) / len(q_toks))
    q_low, c_low = query.lower(), chunk.text.lower()
    bigrams = [" ".join(p) for p in zip(tokenize(query), tokenize(query)[1:])]
    phrase = sum(1 for bg in bigrams if bg in c_low) / max(len(bigrams), 1) if bigrams else 0.0
    return 0.45 * coverage + 0.40 * best_sent + 0.15 * phrase


class Retriever:
    """Modes: 'tfidf' (naive baseline — cosine only, no expansion),
    'hybrid' (TF-IDF + BM25 fusion + expansion), 'hybrid_rerank' (default,
    + cross-score rerank over the top-20 hybrid candidate pool)."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None,
                 alpha: float = 0.55):
        self.chunks = chunks
        self.alpha = alpha  # TF-IDF weight in fusion; BM25 gets 1-alpha
        self.embedder = embedder or TFIDFEmbedder().fit([c.text for c in chunks])
        self.matrix = self.embedder.embed_many([c.text for c in chunks])
        self.bm25 = BM25Index(chunks)

    def _tfidf_scores(self, query: str) -> np.ndarray:
        q = self.embedder.embed(query)
        if not np.linalg.norm(q):
            return np.zeros(len(self.chunks))
        return self.matrix @ q

    def retrieve(self, query: str, top_k: int = 5, mode: str = "hybrid_rerank") -> list[Retrieved]:
        if not len(self.chunks):
            return []
        if mode == "tfidf":
            scores = self._tfidf_scores(query)  # naive: raw query, no expansion
            order = np.argsort(-scores)[:top_k]
            return [Retrieved(self.chunks[int(i)], float(scores[int(i)]))
                    for i in order if scores[int(i)] > 0]

        expanded = expand_query(query)
        tfidf = self._tfidf_scores(expanded)
        bm25_raw = self.bm25.scores(expanded)
        # Saturating BM25 transform keeps the fused score on an ABSOLUTE scale
        # (0 when nothing matches), which the abstention threshold depends on.
        # Min-max normalisation would force a top hit near 1.0 even for
        # nonsense queries, so it is deliberately not used here.
        bm25_norm = bm25_raw / (bm25_raw + 8.0)
        fused = self.alpha * tfidf + (1 - self.alpha) * bm25_norm
        if mode == "hybrid":
            order = np.argsort(-fused)[:top_k]
            return [Retrieved(self.chunks[int(i)], float(fused[int(i)]))
                    for i in order if fused[int(i)] > 0]

        # hybrid_rerank: rerank top-20 pool; keep fused score as reported score
        # for the best hit so the abstention threshold stays on a stable scale,
        # but order by the combined rerank score.
        pool = np.argsort(-fused)[:20]
        reranked = []
        for i in pool:
            if fused[int(i)] <= 0:
                continue
            cross = _cross_score(expanded, self.chunks[int(i)])
            combined = 0.55 * float(fused[int(i)]) + 0.45 * cross
            reranked.append((combined, int(i)))
        reranked.sort(key=lambda x: -x[0])
        return [Retrieved(self.chunks[i], float(fused[i])) for _, i in reranked[:top_k]]


# ---------------------------------------------------------------- answering
ABSTAIN = "I don't have enough information in the provided documents to answer that."

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(text.replace("\n", " ")) if len(s.strip()) > 20]


def compose_answer(query: str, retrieved: list[Retrieved],
                   threshold: float = 0.08, max_sentences: int = 4) -> dict:
    """Extractive grounded composer: answer sentences are verbatim from
    retrieved chunks and each carries a [doc:chunk] citation. Abstains when
    the best retrieval score is below `threshold`."""
    if not retrieved or retrieved[0].score < threshold:
        return {"answer": ABSTAIN, "abstained": True, "citations": [], "sentences": []}
    q_toks = set(content_tokens(expand_query(query)))
    scored: list[tuple[float, str, Retrieved]] = []
    seen: set[str] = set()
    for r in retrieved:
        for sent in _sentences(r.chunk.text):
            key = sent.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            overlap = len(q_toks & set(content_tokens(sent)))
            # rank by query overlap, tie-break by chunk score
            scored.append((overlap + r.score, sent, r))
    scored.sort(key=lambda x: -x[0])
    picked = [s for s in scored if s[0] > 0.1][:max_sentences]
    if not picked:
        return {"answer": ABSTAIN, "abstained": True, "citations": [], "sentences": []}
    lines, citations, sentences = [], [], []
    for _, sent, r in picked:
        clean = re.sub(r"\s+", " ", sent).strip()
        lines.append(f"{clean} {r.chunk.citation}")
        sentences.append(clean)
        if r.chunk.citation not in citations:
            citations.append(r.chunk.citation)
    return {"answer": " ".join(lines), "abstained": False,
            "citations": citations, "sentences": sentences}


class RAGPipeline:
    """Convenience wrapper used by demo/api/evaluate."""

    # Defaults below are the values chosen by the evaluate.py tuning sweep
    # (see results.json "tuning"): chunk 200/20, abstention threshold 0.20.
    def __init__(self, corpus_dir, threshold: float = 0.20, mode: str = "hybrid_rerank",
                 chunk_words: int = 200, overlap_words: int = 20):
        self.chunks = build_chunks(corpus_dir, chunk_words, overlap_words)
        self.retriever = Retriever(self.chunks)
        self.threshold = threshold
        self.mode = mode
        self.chunk_words, self.overlap_words = chunk_words, overlap_words

    def retrieve(self, query: str, top_k: int = 5, mode: str | None = None) -> list[Retrieved]:
        return self.retriever.retrieve(query, top_k=top_k, mode=mode or self.mode)

    def answer(self, query: str, top_k: int = 5) -> dict:
        retrieved = self.retrieve(query, top_k=top_k)
        result = compose_answer(query, retrieved, threshold=self.threshold)
        result["retrieved"] = [
            {"doc": r.chunk.doc_id, "chunk": r.chunk.chunk_id,
             "score": round(r.score, 4), "citation": r.chunk.citation}
            for r in retrieved
        ]
        result["top_score"] = round(retrieved[0].score, 4) if retrieved else 0.0
        return result
