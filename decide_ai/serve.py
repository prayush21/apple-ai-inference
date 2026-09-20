"""Serve the NLI decider over local HTTP, speaking Jev's wire format.

Until ``CoreAI.framework`` ships (macOS 27) the Swift ``RemoteDecider`` talks
to this server, which runs the same ``.aimodel`` through the Core AI Python
runtime. The request and response are the same JSON as the Vercel AI
Gateway's ``/v1/evaluate``, so one client and one bench harness hit both.

    python -m decide_ai.serve                        # static asset on :8770
    python -m decide_ai.serve --asset dynamic --max-len 64

Endpoints (JSON):
    GET  /info    -> {"asset", "function"(s), "max_len", "temperature", "score", "host"}
    POST /decide  {"model": "...", "state": "...",
                   "questions": {name: {"type": "boolean", "instructions": "...", "hypothesis"?: "..."}},
                   "padded_len"?: 64}        # bench-only hint, pins L
              ->  {"model", "answers": {name: {"type": "boolean", "probability": p}},
                   "usage": {"inputTokens", "outputTokens": 0},
                   "timing": {"ms_tokenize", "ms_infer", "ms_total", "batch", "padded_len"},
                   "raw": {name: [P(contradiction), P(entailment), P(neutral)]}}
Only ``boolean`` questions are implemented; ``choice`` / ``score`` get a 400
naming them as step 5 work.
"""

from __future__ import annotations

import argparse
import json
import platform
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .convert import DEFAULT_OUT_DIR
from .decider import LABELS, LocalDecider, make_backend

DEFAULT_PORT = 8770


def info_payload(decider: LocalDecider) -> dict:
    m = decider.model
    return {
        "asset": m.path.name,
        "functions": sorted(m.functions),
        "static_shapes": sorted([list(k) for k in m.static_shapes]),
        "max_len": decider.max_len,
        "temperature": decider.temperature,
        "score": decider.score,
        "labels": list(LABELS),
        "load_ms": round(m.load_ms, 1),
        "host": {"machine": platform.machine(), "macos": platform.mac_ver()[0], "python": platform.python_version()},
    }


def make_handler(decider: LocalDecider):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/info":
                self._json(200, info_payload(decider))
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/decide":
                self._json(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError as e:
                self._json(400, {"error": f"invalid JSON: {e}"})
                return
            state = payload.get("state")
            if not isinstance(state, str):
                self._json(400, {"error": "state must be a string"})
                return
            err = LocalDecider.validate(payload.get("questions"))
            if err:
                self._json(400, {"error": err})
                return
            # Optional bench hint: pin the padded length (must be <= --max-len).
            length = payload.get("padded_len")
            if length is not None and (not isinstance(length, int) or not 4 <= length <= decider.max_len):
                self._json(400, {"error": f"padded_len must be an int in [4, {decider.max_len}]"})
                return
            try:
                answers, usage, timing, raw = decider.evaluate(state, payload["questions"], length=length)
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": str(e)})
                return
            self._json(200, {"model": payload.get("model", f"local/{decider.model.path.stem}"),
                             "answers": answers, "usage": usage, "timing": timing, "raw": raw})

        def log_message(self, fmt, *args):  # quiet
            pass

    return Handler


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--asset", choices=["static", "dynamic"], default="static")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--temperature", type=float, help="override models/decide/calibration.json")
    ap.add_argument("--score", choices=["entail", "entail_vs_contra"], default="entail")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = ap.parse_args(argv)

    decider = make_backend(f"local-{a.asset}", models_dir=a.models_dir, max_len=a.max_len,
                           temperature=a.temperature, score=a.score)
    info = info_payload(decider)
    print(f"loaded {info['asset']} in {info['load_ms']} ms; functions={info['functions']}; "
          f"T={info['temperature']}; max_len={a.max_len}")
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(decider))
    print(f"serving /decide on http://127.0.0.1:{a.port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
