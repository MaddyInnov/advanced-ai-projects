"""
AgentForge-RAG — end-to-end demo. Author: Ashutosh Sahoo
Run:  python3 demo.py
Runs 4 sample queries through Router -> Retriever -> Verifier with traces:
a policy question, an incident question, a paraphrased question (where the
hybrid+rerank stack earns its keep), and an unanswerable question (abstain).
"""
from __future__ import annotations

import pathlib

from agents import Orchestrator

CORPUS = pathlib.Path(__file__).parent / "corpus"

QUERIES = [
    "How many characters must a password be at least and how often do passwords expire?",
    "What is the company policy on paying a ransom in a ransomware attack?",
    "I think I received a fake email trying to steal my login — what should I do first?",
    "What is the recipe for baking a chocolate sourdough cake in an oven?",
]


def main() -> None:
    orch = Orchestrator(CORPUS)
    print(f"Loaded {len({c.doc_id for c in orch.pipeline.chunks})} documents, "
          f"{len(orch.pipeline.chunks)} chunks | mode={orch.pipeline.mode} "
          f"threshold={orch.pipeline.threshold}\n" + "=" * 78)
    for i, q in enumerate(QUERIES, 1):
        out = orch.ask(q)
        print(f"\n[{i}] Q: {q}")
        print(f"    type={out['query_type']}  top_score={out['top_score']}  "
              f"latency={out['latency_ms']} ms  abstained={out['abstained']}")
        print(f"    A: {out['answer']}")
        print("    trace:")
        for step in out["trace"]:
            print(f"      - {step['agent']}.{step['action']}: {step['detail']} ({step['ms']} ms)")
        if out["verification"]:
            print(f"    verifier: pass={out['verification']['pass']} — {out['verification']['note']}")
    print("\n" + "=" * 78 + "\nDemo complete.")


if __name__ == "__main__":
    main()
