"""
AgentForge-RAG — multi-agent orchestration. Author: Ashutosh Sahoo

Router -> Retriever -> (Composer) -> Verifier, orchestrated with a trace log.
Each agent is a small class with a clear contract; agents are deliberately
deterministic here so behaviour is testable without an LLM API key. In
production each agent's internals can be backed by an LLM call while keeping
the same interfaces and trace format.
"""
from __future__ import annotations

import time

from guardrails import guard_input, redact_pii
from rag import ABSTAIN, RAGPipeline, content_tokens

INCIDENT_TERMS = {"incident", "ransomware", "phishing", "breach", "malware",
                  "attack", "outage", "severity", "containment", "forensic",
                  "encrypted", "ransom", "compromised", "triage"}
POLICY_TERMS = {"policy", "password", "vpn", "access", "retention", "training",
                "device", "byod", "classification", "acceptable", "backup",
                "patch", "vulnerability", "onboarding", "offboarding", "privacy"}


class _Traced:
    name = "agent"

    def _log(self, trace: list, action: str, detail: str, t0: float) -> None:
        trace.append({"agent": self.name, "action": action, "detail": detail,
                      "ms": round((time.perf_counter() - t0) * 1000, 2)})


class RouterAgent(_Traced):
    name = "router"

    def route(self, query: str, trace: list) -> str:
        t0 = time.perf_counter()
        toks = set(content_tokens(query))
        kind = "incident" if toks & INCIDENT_TERMS else \
            "policy" if toks & POLICY_TERMS else "general"
        self._log(trace, "classify", f"query_type={kind}", t0)
        return kind


class RetrieverAgent(_Traced):
    name = "retriever"

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def run(self, query: str, query_type: str, trace: list, top_k: int = 5) -> dict:
        t0 = time.perf_counter()
        result = self.pipeline.answer(query, top_k=top_k)
        docs = [r["citation"] for r in result["retrieved"]]
        self._log(trace, "retrieve+compose",
                  f"type={query_type} top_k={top_k} hits={docs} "
                  f"top_score={result['top_score']} abstained={result['abstained']}", t0)
        return result


class VerifierAgent(_Traced):
    """Grounding check: every answer sentence must overlap retrieved text.

    A sentence is grounded if >=50% of its content tokens appear in the
    union of retrieved chunk tokens (extractive answers score ~100%).
    Free-form LLM answers that invent facts fail this check — that is the point.
    """
    name = "verifier"

    def verify(self, result: dict, pipeline: RAGPipeline, trace: list) -> dict:
        t0 = time.perf_counter()
        if result.get("abstained"):
            out = {"grounded": True, "pass": True, "per_sentence": [],
                   "note": "abstention is grounded by definition"}
            self._log(trace, "verify", out["note"], t0)
            return out
        retrieved_text = " ".join(
            c.text for c in pipeline.chunks
            if any(c.citation == cit for cit in result.get("citations", []))
        )
        retrieved_toks = set(content_tokens(retrieved_text))
        per_sentence, all_ok = [], True
        for sent in result.get("sentences", []):
            toks = set(content_tokens(sent))
            ratio = (len(toks & retrieved_toks) / len(toks)) if toks else 1.0
            ok = ratio >= 0.5
            all_ok = all_ok and ok
            per_sentence.append({"sentence": sent[:90], "overlap": round(ratio, 3), "grounded": ok})
        out = {"grounded": all_ok, "pass": all_ok, "per_sentence": per_sentence,
               "note": "all sentences grounded" if all_ok else "UNGROUNDED sentence(s) detected"}
        self._log(trace, "verify", out["note"], t0)
        return out


class Orchestrator:
    def __init__(self, corpus_dir, **pipeline_kwargs):
        self.pipeline = RAGPipeline(corpus_dir, **pipeline_kwargs)
        self.router = RouterAgent()
        self.retriever = RetrieverAgent(self.pipeline)
        self.verifier = VerifierAgent()

    def ask(self, query: str, top_k: int = 5) -> dict:
        trace: list[dict] = []
        t0 = time.perf_counter()

        guard = guard_input(query)
        trace.append({"agent": "guardrails", "action": "input_check",
                      "detail": f"blocked={guard['blocked']} pii={guard['pii_found']}",
                      "ms": 0.0})
        if guard["blocked"]:
            return {"answer": "Request blocked by guardrails: prompt-injection pattern detected.",
                    "blocked": True, "abstained": False, "query_type": None,
                    "citations": [], "verification": None, "trace": trace,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

        clean_query = guard["redacted_text"]
        query_type = self.router.route(clean_query, trace)
        result = self.retriever.run(clean_query, query_type, trace, top_k=top_k)
        verification = self.verifier.verify(result, self.pipeline, trace)

        answer = result["answer"]
        if not result["abstained"] and not verification["pass"]:
            answer = ABSTAIN + " (An initial answer failed grounding verification.)"
            result = {**result, "abstained": True}
        answer = redact_pii(answer)  # output-side PII redaction

        return {"answer": answer, "blocked": False,
                "abstained": bool(result["abstained"]), "query_type": query_type,
                "citations": result["citations"], "retrieved": result["retrieved"],
                "top_score": result["top_score"], "verification": verification,
                "trace": trace,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


# Backwards-friendly alias
AgentOrchestrator = Orchestrator
