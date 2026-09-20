"""The local "System One" decider: tokenizer + Core AI runtime + Jev-shaped answers.

    model = LocalModel.load(Path("models/decide"), asset="static")
    decider = LocalDecider(model, BPETokenizer.from_hf(hf_dir), max_len=128, temperature=T)
    answers, usage, timing, raw = decider.evaluate(state, questions)

``questions`` is Jev's request shape, ``answers`` is Jev's response shape
(``{name: {"type": "boolean", "probability": p}}``), so ``calibrate.py`` and
``bench.py`` can swap this for ``JevDecider`` and change nothing else.

Each question becomes an NLI hypothesis (``hypothesis`` if the caller gave one,
else ``instructions`` verbatim); N questions are one batched forward pass
``[N, L] -> [N, 3]``. ``P(yes)`` is the entailment probability after
temperature scaling of the logits (``T`` from ``models/decide/calibration.json``,
1.0 = raw). ``score="entail_vs_contra"`` instead renormalises over
entailment + contradiction, ignoring neutral; ``raw`` carries all three so a
caller can recompute either.

Static asset: the request is padded up to the smallest enumerated ``(N, L)``
that holds it (extra rows are all-``<pad>`` with a zero mask and are dropped
from the output). Dynamic asset: ``L`` is the longest pair in the batch rounded
up to a multiple of 16, capped at ``max_len``.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from coreai.runtime import AIModel, NDArray

from .convert import DYNAMIC_ASSET, STATIC_ASSET, static_function_name
from .tokenize import BPETokenizer

ASSETS = {"dynamic": DYNAMIC_ASSET, "static": STATIC_ASSET}
CALIBRATION_PATH = Path("models/decide/calibration.json")
LABELS = ("contradiction", "entailment", "neutral")  # config.json id2label of this checkpoint
_STATIC_RE = re.compile(r"main_n(\d+)_l(\d+)$")


class LocalModel:
    """One loaded ``.aimodel`` with a private event-loop thread, so callers
    (HTTP handler threads, plain scripts) run inference synchronously."""

    def __init__(self, path: Path, asset: str) -> None:
        self.path = path
        self.asset = asset
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        t0 = time.perf_counter()
        self.model = self._await(AIModel.load(path))
        self.functions = {name: self.model.load_function(name) for name in self.model.function_names}
        self.load_ms = (time.perf_counter() - t0) * 1e3
        # Static asset: {(N, L): function name}; dynamic: {} and "main".
        self.static_shapes: dict[tuple[int, int], str] = {}
        for name in self.functions:
            m = _STATIC_RE.match(name)
            if m:
                self.static_shapes[(int(m.group(1)), int(m.group(2)))] = name
        self.lock = threading.Lock()

    @classmethod
    def load(cls, models_dir: Path, asset: str = "static") -> "LocalModel":
        return cls(models_dir / ASSETS[asset], asset)

    def _await(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    @property
    def lengths(self) -> list[int]:
        return sorted({length for _, length in self.static_shapes})

    @property
    def batches(self) -> list[int]:
        return sorted({n for n, _ in self.static_shapes})

    def function_for(self, n: int, length: int) -> tuple[str, int, int]:
        """-> (function name, padded N, padded L) for a request of n rows x length tokens."""
        if not self.static_shapes:
            return "main", n, length
        fits = [(nn, ll) for nn, ll in self.static_shapes if nn >= n and ll >= length]
        if not fits:
            raise ValueError(f"no static function holds N={n}, L={length}; have {sorted(self.static_shapes)}")
        nn, ll = min(fits)
        return self.static_shapes[(nn, ll)], nn, ll

    def run(self, ids: np.ndarray, mask: np.ndarray, *, function: str | None = None) -> np.ndarray:
        """logits [N, 3] for int32 ``ids`` / ``mask`` [N, L]; pads to a static shape if needed."""
        n, length = ids.shape
        name, nn, ll = self.function_for(n, length) if function is None else (function, *ids.shape)
        if (nn, ll) != (n, length):
            ids = _pad_rows(ids, nn, ll, fill=1)
            mask = _pad_rows(mask, nn, ll, fill=0)
        with self.lock:
            out = self._await(self.functions[name]({"input_ids": NDArray(data=ids), "attention_mask": NDArray(data=mask)}))
        return out["logits"].numpy()[:n]


def _pad_rows(x: np.ndarray, n: int, length: int, *, fill: int) -> np.ndarray:
    out = np.full((n, length), fill, dtype=np.int32)
    out[: x.shape[0], : x.shape[1]] = x
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def probabilities(logits: np.ndarray, temperature: float = 1.0, score: str = "entail") -> np.ndarray:
    """P(yes) per row from [N, 3] logits in LABELS order."""
    p = softmax(logits / temperature)
    e, c = p[:, LABELS.index("entailment")], p[:, LABELS.index("contradiction")]
    if score == "entail":
        return e
    if score == "entail_vs_contra":
        return e / (e + c)
    raise ValueError(score)


def load_temperature(path: Path = CALIBRATION_PATH) -> float:
    if path.exists():
        return float(json.loads(path.read_text())["temperature"])
    return 1.0


def pick_length(needed: int, max_len: int, static_lengths: list[int]) -> int:
    """Padded length for a batch whose longest pair has ``needed`` tokens."""
    if static_lengths:
        for length in static_lengths:
            if length >= needed and length <= max_len:
                return length
        return min(max(static_lengths), max_len)
    return min(max_len, ((needed + 15) // 16) * 16)


@dataclass
class Timing:
    ms_tokenize: float
    ms_infer: float
    ms_total: float
    batch: int
    padded_len: int

    def as_dict(self) -> dict:
        return {"ms_tokenize": round(self.ms_tokenize, 3), "ms_infer": round(self.ms_infer, 3),
                "ms_total": round(self.ms_total, 3), "batch": self.batch, "padded_len": self.padded_len}


class LocalDecider:
    name = "local"

    def __init__(self, model: LocalModel, tokenizer: BPETokenizer, *, max_len: int = 128,
                 temperature: float = 1.0, score: str = "entail") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.temperature = temperature
        self.score = score
        self.name = f"local-{model.asset}"

    @staticmethod
    def validate(questions: dict) -> str | None:
        """Jev-compatible request validation. Returns an error message or None."""
        if not isinstance(questions, dict) or not questions:
            return "questions must be a non-empty object"
        for name, q in questions.items():
            if not isinstance(q, dict):
                return f"question {name!r} must be an object"
            qtype = q.get("type")
            if qtype in ("choice", "score"):
                return f"question {name!r}: type {qtype!r} is step 5 work; only 'boolean' is supported"
            if qtype != "boolean":
                return f"question {name!r}: unsupported type {qtype!r}"
            if not isinstance(q.get("hypothesis", q.get("instructions")), str):
                return f"question {name!r}: 'instructions' (or 'hypothesis') must be a string"
        return None

    def evaluate(self, state: str, questions: dict, *, length: int | None = None) -> tuple[dict, dict, dict, dict]:
        """-> (answers, usage, timing, raw). ``length`` pins the padded L (bench)."""
        t0 = time.perf_counter()
        names = list(questions)
        pairs = [(state, questions[n].get("hypothesis") or questions[n]["instructions"]) for n in names]
        needed = max(self.tokenizer.count_tokens(a, b) for a, b in pairs)
        padded = length or pick_length(needed, self.max_len, self.model.lengths)
        ids, mask = self.tokenizer.encode_batch(pairs, padded)
        t1 = time.perf_counter()
        logits = self.model.run(ids, mask)
        t2 = time.perf_counter()
        probs = probabilities(logits, self.temperature, self.score)
        raw = softmax(logits)
        answers = {n: {"type": "boolean", "probability": float(probs[i])} for i, n in enumerate(names)}
        usage = {"inputTokens": int(mask.sum()), "outputTokens": 0}
        t3 = time.perf_counter()
        timing = Timing((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t0) * 1e3, len(names), padded).as_dict()
        return answers, usage, timing, {n: [float(v) for v in raw[i]] for i, n in enumerate(names)}


class JevDecider:
    """Same ``evaluate`` shape as ``LocalDecider``, backed by ``decide_ai.jev``."""

    name = "jev"

    def __init__(self, *, use_cache: bool = True) -> None:
        self.use_cache = use_cache

    def evaluate(self, state: str, questions: dict, *, length: int | None = None) -> tuple[dict, dict, dict, dict]:
        from . import jev

        # Jev gets the question form only; ``hypothesis`` is a local-model concern.
        qs = {n: {"type": q["type"], "instructions": q["instructions"]} for n, q in questions.items()}
        answers, usage, ms = jev.evaluate(state, qs, use_cache=self.use_cache)
        timing = {"ms_total": ms, "batch": len(questions), "cached": jev.last_was_cached}
        return answers, usage, timing, {}


def make_backend(name: str, *, models_dir: Path = Path("models/decide"), hf_dir: Path | None = None,
                 max_len: int = 128, temperature: float | None = None, score: str = "entail",
                 use_cache: bool = True):
    """``local-static`` | ``local-dynamic`` | ``jev`` -> a decider with ``.evaluate``."""
    if name == "jev":
        return JevDecider(use_cache=use_cache)
    if name.startswith("local-"):
        from .download import model_dir

        asset = name.split("-", 1)[1]
        model = LocalModel.load(models_dir, asset)
        tok = BPETokenizer.from_hf(hf_dir or model_dir())
        T = load_temperature() if temperature is None else temperature
        return LocalDecider(model, tok, max_len=max_len, temperature=T, score=score)
    raise ValueError(f"unknown backend {name!r}")
