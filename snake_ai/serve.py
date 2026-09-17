"""Serve the Core AI model over local HTTP so the Swift app can use it today.

The Swift ``ModelPlayer`` needs ``CoreAI.framework`` (macOS 27). Until then
this server runs the *same* ``.aimodel`` through the Core AI Python runtime,
and the app's ``RemoteModelPlayer`` talks to it. Responsibilities are split
exactly as in the in-process path: the app extracts the 16 features and picks
the safe argmax; the server only runs inference and returns logits.

    python -m snake_ai.serve                      # decode asset on :8765
    python -m snake_ai.serve --variant stateless
    python -m snake_ai.serve --models-dir models/minimax_teacher --port 8766 --tag "minimax-taught"

Endpoints (JSON):
    GET  /info                     -> {"asset", "function", "variant", "tag", "states": [...]}
    POST /reset                    -> fresh KV caches for a new game
    POST /act {"features": [16]}   -> {"logits": [4], "ms": inference time}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from coreai.runtime import AIModel, NDArray

ASSETS = {
    "decode": ("SnakeTransformerDecode.aimodel", "main_decode"),
    "stateful": ("SnakeTransformerStateful.aimodel", "main"),
    "stateless": ("SnakeTransformer.aimodel", "main"),
}


class ModelSession:
    """One loaded function plus per-game state (KV caches or history)."""

    def __init__(self, models_dir: Path, variant: str, tag: str = "") -> None:
        self.variant = variant
        self.tag = tag
        self.asset, self.function_name = ASSETS[variant]
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.model = self._await(AIModel.load(models_dir / self.asset))
        self.function = self.model.load_function(self.function_name)
        self.desc = self.function.desc
        self.lock = threading.Lock()
        self.reset()

    def _await(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def reset(self) -> None:
        self.position = 0
        self.history: list[list[float]] = []
        self.states = {
            name: NDArray(data=np.zeros(self.desc.state_descriptor(name).shape, np.float32))
            for name in self.desc.state_names
        }

    def act(self, features: list[float]) -> tuple[list[float], float]:
        with self.lock:
            t0 = time.perf_counter()
            if self.variant == "stateless":
                self.history.append(features)
                x = np.asarray(self.history, np.float32)[None]
                out = self._await(self.function({"features": NDArray(data=x)}))
            else:
                x = np.asarray([features], np.float32)[None]
                pos = np.array([[self.position]], np.int32)
                out = self._await(
                    self.function(
                        {"features": NDArray(data=x), "position_ids": NDArray(data=pos)},
                        state=self.states,
                    )
                )
                self.position += 1
            ms = (time.perf_counter() - t0) * 1e3
            return out["logits"].numpy()[0, -1].tolist(), ms


def make_handler(session: ModelSession):
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
                self._json(200, {
                    "asset": session.asset,
                    "function": session.function_name,
                    "variant": session.variant,
                    "tag": session.tag,
                    "states": list(session.desc.state_names),
                })
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/reset":
                with session.lock:
                    session.reset()
                self._json(200, {"ok": True})
            elif self.path == "/act":
                feats = payload.get("features")
                if not isinstance(feats, list) or len(feats) != 16:
                    self._json(400, {"error": "features must be a list of 16 floats"})
                    return
                try:
                    logits, ms = session.act([float(v) for v in feats])
                except Exception as e:  # noqa: BLE001
                    self._json(500, {"error": str(e)})
                    return
                self._json(200, {"logits": logits, "ms": ms})
            else:
                self._json(404, {"error": "not found"})

        def log_message(self, fmt, *args):  # quiet
            pass

    return Handler


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--variant", choices=list(ASSETS), default="decode")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--tag", default="", help="free-text label shown by clients, e.g. which teacher trained it")
    a = ap.parse_args(argv)

    session = ModelSession(a.models_dir, a.variant, a.tag)
    print(f"loaded {session.asset} ({session.function_name}); states={list(session.desc.state_names)}")
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(session))
    print(f"serving Core AI model on http://127.0.0.1:{a.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
