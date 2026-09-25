"""Minimal client for TypeSafe's Jev through the Vercel AI Gateway (``/v1/evaluate``).

    export AI_GATEWAY_API_KEY=...     # read from the environment only, never written anywhere
    python -m decide_ai.jev "I was charged twice." refund="Is the customer asking for money back?"

``evaluate(state, questions)`` returns ``(answers, usage, ms_roundtrip)`` with
exactly the JSON the gateway sends back, so ``calibrate.py`` / ``bench.py``
treat Jev and the local model as interchangeable backends. Behaviour that
the 2026-09-19 probe (``scripts/probe_jev.py``) made necessary:

* **Disk cache** ``data/decide/jev_cache.jsonl``, keyed by sha256 of the
  canonical request body. Reruns are free and deterministic; ``use_cache=False``
  neither reads nor writes it (repeatability and latency runs need fresh calls).
* **Pacing** at <= 25 req/min (>= 2.4 s between live calls; the gateway limit
  is 30/min per team). Cache hits do not sleep.
* **429** -> parse ``Retry after Ns`` from the body (or the ``Retry-After``
  header), sleep, retry. **5xx and transport errors** (read timeout, reset)
  -> exponential backoff, 5 attempts, counted in ``transient_errors``. Other
  4xx are raised immediately.
* **TLS**: this venv's CPython is a python.org build that does not see the
  macOS keychain, so ``urlopen`` needs ``certifi``'s CA bundle.

If ``AI_GATEWAY_API_KEY`` is unset, ``evaluate`` raises ``JevUnavailable``;
callers print a one-line skip and carry on with the local backends.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = "https://ai-gateway.vercel.sh/v1/evaluate"
MODEL = "typesafe-ai/jev"
CACHE_PATH = Path("data/decide/jev_cache.jsonl")
MIN_GAP_S = 2.4
KEY_ENV = "AI_GATEWAY_API_KEY"

try:
    import certifi

    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover - certifi is a transitive dep of huggingface_hub
    SSL_CTX = ssl.create_default_context()


class JevUnavailable(RuntimeError):
    """Raised when no API key is set; the rest of the pipeline treats it as a skip."""


class JevHTTPError(RuntimeError):
    def __init__(self, code: int, body: str) -> None:
        super().__init__(f"HTTP {code}: {body[:300]}")
        self.code = code
        self.body = body


def available() -> bool:
    return bool(os.environ.get(KEY_ENV))


def request_body(state: str, questions: dict[str, dict], model: str = MODEL) -> bytes:
    """Canonical JSON (sorted keys) so equal requests hash equally."""
    return json.dumps({"model": model, "state": state, "questions": questions}, sort_keys=True).encode()


def cache_key(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


# --------------------------------------------------------------------------- cache

_caches: dict[Path, dict[str, dict]] = {}


def _load_cache(path: Path) -> dict[str, dict]:
    if path not in _caches:
        cache: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    cache[rec["key"]] = rec  # last write wins
        _caches[path] = cache
    return _caches[path]


def _append_cache(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    _load_cache(path)[rec["key"]] = rec


# --------------------------------------------------------------------------- pacing

_last_live_call = 0.0


def _pace() -> None:
    global _last_live_call
    wait = MIN_GAP_S - (time.perf_counter() - _last_live_call)
    if wait > 0:
        time.sleep(wait)
    _last_live_call = time.perf_counter()


_RETRY_RE = re.compile(r"retry[ -]after[:\s]*(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _retry_after_seconds(err: urllib.error.HTTPError, body: str) -> float:
    hdr = err.headers.get("Retry-After") if err.headers else None
    if hdr and hdr.strip().isdigit():
        return float(hdr)
    m = _RETRY_RE.search(body)
    return float(m.group(1)) if m else 5.0


def _post(body: bytes, timeout: float) -> tuple[dict, float]:
    """One live call with pacing + retry policy. Returns (response json, ms)."""
    req = urllib.request.Request(URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {os.environ[KEY_ENV]}",
        "Content-Type": "application/json",
    })
    backoff = 1.0
    for attempt in range(5):
        _pace()
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
                out = json.loads(r.read())
            return out, (time.perf_counter() - t0) * 1e3
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")
            if e.code == 429:
                wait = _retry_after_seconds(e, text)
                print(f"  jev: 429 rate limited, sleeping {wait:.0f}s", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            if 500 <= e.code < 600 and attempt < 4:
                print(f"  jev: HTTP {e.code}, retrying in {backoff:.0f}s", file=sys.stderr, flush=True)
                time.sleep(backoff)
                backoff *= 2
                continue
            raise JevHTTPError(e.code, text) from None
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as e:
            transient_errors.append(f"{type(e).__name__}: {e}")
            if attempt == 4:
                raise
            print(f"  jev: {type(e).__name__}, retrying in {backoff:.0f}s", file=sys.stderr, flush=True)
            time.sleep(backoff)
            backoff *= 2
    raise JevHTTPError(429, "gave up after repeated 429s")


# One entry per transport-level failure that was retried (bench records the count).
transient_errors: list[str] = []


# --------------------------------------------------------------------------- API

def evaluate(
    state: str,
    questions: dict[str, dict],
    *,
    model: str = MODEL,
    use_cache: bool = True,
    cache_path: Path = CACHE_PATH,
    timeout: float = 30.0,
) -> tuple[dict, dict, float]:
    """POST ``/v1/evaluate``. ``questions`` is Jev's own shape
    ``{name: {"type": "boolean", "instructions": ...}}``.

    Returns ``(answers, usage, ms_roundtrip)``; on a cache hit ``ms_roundtrip``
    is the latency recorded when the entry was made (see ``last_was_cached``).
    """
    global last_was_cached
    if not available():
        raise JevUnavailable(f"{KEY_ENV} is not set; skipping Jev")
    body = request_body(state, questions, model)
    key = cache_key(body)
    if use_cache:
        hit = _load_cache(cache_path).get(key)
        if hit is not None:
            last_was_cached = True
            return hit["response"]["answers"], hit["response"].get("usage", {}), hit["ms"]
    out, ms = _post(body, timeout)
    last_was_cached = False
    if use_cache:
        _append_cache(cache_path, {
            "key": key,
            "request": json.loads(body),
            "response": out,
            "ms": ms,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    return out["answers"], out.get("usage", {}), ms


last_was_cached = False


def boolean_questions(instructions: dict[str, str]) -> dict[str, dict]:
    """``{name: question text}`` -> Jev's ``questions`` block, boolean type."""
    return {k: {"type": "boolean", "instructions": v} for k, v in instructions.items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state")
    ap.add_argument("questions", nargs="+", help="name=question text")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    qs = boolean_questions(dict(q.split("=", 1) for q in a.questions))
    try:
        answers, usage, ms = evaluate(a.state, qs, use_cache=not a.no_cache)
    except JevUnavailable as e:
        sys.exit(str(e))
    for k, v in answers.items():
        print(f"{k:24s} {v['probability']:.3f}")
    print(f"usage={usage} ms={ms:.0f} cached={last_was_cached}")


if __name__ == "__main__":
    main()
