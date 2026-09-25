"""Laya (``convaiinnovations/laya``) as a third decide backend, in plain PyTorch on CPU.

    python -m decide_ai.laya --state "Order #48213 never showed up."          # the five triage questions
    python -m decide_ai.laya --state "..." --question refund "Is the customer asking for a refund?"

Laya is an open-weight (Apache 2.0) reimplementation of Jev's "System One"
design: ModernBERT-large (395M) + a 2-layer decision head, 421M params, RL
trained against strictly proper scoring rules, every question answered in one
forward pass with no generated tokens. Its model card benchmarks against Jev
throughout but never measured it; this repo has, so ``LayaDecider`` puts it
on the same holdout and latency matrix as ``JevDecider`` / ``LocalDecider``.

The checkpoint's own ``rl_agent_api.py`` (in ``models/decide/hf/laya``, put on
``sys.path`` rather than vendored) is the whole inference surface. Its
``system_one(state, questions)`` takes Jev's request shape except the boolean
type is spelled ``noul`` (yes / no / unknown; ``noul`` = P(yes)), so this
wrapper maps ``boolean -> noul`` on the way in and ``noul -> probability`` on
the way out. It is sent the ``instructions`` (question) form, like Jev — that
is what it was trained on; ``hypothesis`` is an NLI concern and is ignored.

Facts the numbers depend on:

* the agent applies its own per-question-type temperature from
  ``rl_agent_config.json`` (``noul:2`` -> 1.98). That is the shipped
  calibration and ``probability`` reports it as-is; ``raw`` carries the
  ``noul`` value, the ``act_probability`` and the shipped temperature so
  ``calibrate.py`` can refit a ``T`` on our split and show both;
* ``timing`` is ``ms_total`` around ``system_one`` only — the agent exposes
  no tokenize / infer split, and it pads to its own sequence (``max_len``
  512), so there is no ``padded_len`` either;
* load is ~63 s on an M2 (fp16 safetensors -> fp32 module), so the agent is
  loaded lazily on the first call and ``load_ms`` is recorded;
* output is deterministic (no sampling, no autocast on CPU) — the test and
  ``calibrate.py`` assert bit-identical probabilities across two calls.

``choice`` / ``score`` are rejected with the same 400-style error as the
local decider: Laya supports them, but step 1 is boolean-only and the holdout
has no labels for them — that is the step-5 opening.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .decider import LocalDecider

DEFAULT_HF_DIR = Path("models/decide/hf/laya")
REPO = "convaiinnovations/laya"


def available(hf_dir: Path = DEFAULT_HF_DIR) -> bool:
    return (hf_dir / "model.safetensors").exists() and (hf_dir / "rl_agent_api.py").exists()


def model_info(hf_dir: Path = DEFAULT_HF_DIR) -> dict:
    """The ``model`` block for bench records: what is on disk, without loading it."""
    import struct

    import torch
    import transformers

    params, dtypes = 0, {}
    with (hf_dir / "model.safetensors").open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    for k, v in header.items():
        if k == "__metadata__":
            continue
        count = 1
        for d in v["shape"]:
            count *= d
        params += count
        dtypes[v["dtype"]] = dtypes.get(v["dtype"], 0) + 1
    cfg = json.loads((hf_dir / "rl_agent_config.json").read_text())
    return {"repo": REPO, "encoder": cfg["encoder"], "params": params,
            "dtype_on_disk": max(dtypes, key=dtypes.get), "tensor_dtypes": dtypes,
            "safetensors_mb": round((hf_dir / "model.safetensors").stat().st_size / 1e6),
            "max_len": cfg["max_len"], "temperature_noul": cfg["temperature_by_options"].get("noul:2"),
            "torch": torch.__version__, "transformers": transformers.__version__, "device": "cpu",
            "compute_dtype": "fp32"}


class LayaDecider:
    """Same ``evaluate(state, questions) -> (answers, usage, timing, raw)`` as ``LocalDecider``."""

    name = "laya"

    def __init__(self, hf_dir: Path = DEFAULT_HF_DIR, *, device: str = "cpu") -> None:
        self.hf_dir = Path(hf_dir)
        self.device = device
        self._agent = None
        self.load_ms: float | None = None
        self.temperature: float | None = None  # the shipped noul temperature, once loaded

    validate = staticmethod(LocalDecider.validate)

    @property
    def agent(self):
        if self._agent is None:
            if not available(self.hf_dir):
                raise FileNotFoundError(f"Laya checkpoint not found under {self.hf_dir}")
            model_dir = str(self.hf_dir.resolve())
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)
            from rl_agent_api import RLAgent  # noqa: PLC0415  (lives in the checkpoint)

            t0 = time.perf_counter()
            self._agent = RLAgent(model_dir, device=self.device)
            self.load_ms = (time.perf_counter() - t0) * 1e3
            self.temperature = float(self._agent.temperature_by_options.get("noul:2", self._agent.temperature[2]))
        return self._agent

    def load(self) -> float:
        """Force the (slow) load; returns ``load_ms``."""
        self.agent
        return self.load_ms

    def evaluate(self, state: str, questions: dict, *, length: int | None = None) -> tuple[dict, dict, dict, dict]:
        """``length`` is accepted for signature parity and ignored: Laya pads to its own sequence."""
        err = self.validate(questions)
        if err:
            raise ValueError(err)
        # boolean -> noul; question form only, no hypothesis.
        qs = {n: {"type": "noul", "instructions": q["instructions"]} for n, q in questions.items()}
        agent = self.agent
        t0 = time.perf_counter()
        out = agent.system_one(state, qs)
        ms = (time.perf_counter() - t0) * 1e3
        answers, raw = {}, {}
        for n, a in out["answers"].items():
            answers[n] = {"type": "boolean", "probability": float(a["noul"])}
            raw[n] = {"noul": float(a["noul"]), "act_probability": float(a["rl_agent"]["act_probability"]),
                      "temperature": self.temperature}
        usage = {"inputTokens": int(out["usage"]["input_tokens"]), "outputTokens": int(out["usage"]["output_tokens"])}
        timing = {"ms_total": ms, "batch": len(questions)}
        return answers, usage, timing, raw


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", type=Path, default=DEFAULT_HF_DIR)
    ap.add_argument("--state", required=True)
    ap.add_argument("--question", nargs=2, action="append", metavar=("NAME", "INSTRUCTIONS"),
                    help="repeatable; default is the five triage questions")
    ap.add_argument("--repeat", type=int, default=1, help="call N times and print each ms")
    a = ap.parse_args(argv)

    if a.question:
        questions = {n: {"type": "boolean", "instructions": s} for n, s in a.question}
    else:
        from .holdout import load_questions

        questions = {q: {"type": "boolean", **v} for q, v in load_questions().items()}
    decider = LayaDecider(a.hf_dir)
    print(f"loading {a.hf_dir} ...", flush=True)
    print(f"loaded in {decider.load() / 1e3:.1f} s; shipped noul temperature {decider.temperature:.3f}")
    for i in range(a.repeat):
        answers, usage, timing, raw = decider.evaluate(a.state, questions)
        print(f"call {i + 1}: {timing['ms_total']:.0f} ms, {usage['inputTokens']} input tokens")
    for n, ans in answers.items():
        print(f"  {n:24s} {ans['probability']:.4f}   act {raw[n]['act_probability']:.3f}")


if __name__ == "__main__":
    main()
