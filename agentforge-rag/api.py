"""
AgentForge-RAG — HTTP API. Author: Ashutosh Sahoo

FastAPI is not installed in this environment, so this is a stdlib
`http.server` app with the same contract a FastAPI version would expose:

    GET  /health          -> {"status": "ok", "documents": N, "chunks": M}
    POST /ask  {"question": "...", "top_k": 5}
                        -> {"answer", "citations", "query_type", "abstained",
                            "blocked", "verification", "trace", "latency_ms", ...}

Run:  python3 api.py [port]        (default 8000)
Swap note: with FastAPI installed, the handler bodies port 1:1 to
@app.post("/ask") — the Orchestrator interface does not change.
"""
from __future__ import annotations

import json
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agents import Orchestrator

ORCH = Orchestrator(pathlib.Path(__file__).parent / "corpus")


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (stdlib naming)
        if self.path.rstrip("/") in ("/health", ""):
            self._json(200, {"status": "ok",
                             "documents": len({c.doc_id for c in ORCH.pipeline.chunks}),
                             "chunks": len(ORCH.pipeline.chunks)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/ask":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = str(payload.get("question") or payload.get("query") or "").strip()
            top_k = int(payload.get("top_k", 5))
        except Exception as exc:  # bad JSON / bad types
            self._json(400, {"error": f"invalid request: {exc}"})
            return
        if not question:
            self._json(400, {"error": "field 'question' is required"})
            return
        self._json(200, ORCH.ask(question, top_k=max(1, min(top_k, 10))))

    def log_message(self, fmt, *args):  # keep stdout clean
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"AgentForge-RAG API on http://127.0.0.1:{port}  (POST /ask, GET /health)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
