"""Pre-tokenized bench inputs for the in-process Swift ``Decider`` (no Swift BPE until step 2).

    python -m decide_ai.bench_ids            # -> data/decide/bench_ids.json

Same inputs as ``decide_ai.bench``: every holdout state paired with each of the
16 questions in ``data/decide/bench_questions.json`` (their ``hypothesis``),
encoded by ``BPETokenizer.encode_batch`` at each L in ``STATIC_LENGTHS``.
Rows are stored unpadded; the reader right-pads with ``pad_id`` and sets the
mask to 1 for the stored tokens, which reproduces ``encode_batch`` exactly
(checked here). ``reference`` holds logits for the first state x all 16
questions from the Python Core AI path, so the Swift side can check itself
before timing anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .bench import BENCH_QUESTIONS_PATH, bench_inputs
from .convert import DEFAULT_OUT_DIR, STATIC_LENGTHS
from .decider import LocalModel
from .download import model_dir
from .tokenize import BPETokenizer

DEFAULT_PATH = Path("data/decide/bench_ids.json")
PAD_ID = 1


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_OUT_DIR)
    a = ap.parse_args(argv)

    tok = BPETokenizer.from_hf(model_dir())
    names = list(json.loads(BENCH_QUESTIONS_PATH.read_text()))
    states, questions = bench_inputs(len(names))
    hyps = [questions[n].get("hypothesis") or questions[n]["instructions"] for n in names]

    ids_by_len: dict[str, list[list[list[int]]]] = {}
    for length in STATIC_LENGTHS:
        per_state = []
        for s in states:
            ids, mask = tok.encode_batch([(s, h) for h in hyps], length)
            rows = []
            for r, m in zip(ids, mask):
                k = int(m.sum())
                assert m[:k].all() and not m[k:].any() and (r[k:] == PAD_ID).all(), "mask is not a right-padded prefix"
                rows.append([int(v) for v in r[:k]])
            per_state.append(rows)
        ids_by_len[str(length)] = per_state

    model = LocalModel.load(a.models_dir, "dynamic")
    ids, mask = tok.encode_batch([(states[0], h) for h in hyps], 64)
    logits = model.run(ids, mask)

    record = {
        "schema": "decide-bench-ids/1",
        "tokenizer": str(model_dir()),
        "pad_id": PAD_ID,
        "questions": names,
        "state_count": len(states),
        "ids": ids_by_len,
        "reference": {"state": 0, "L": 64, "asset": "dynamic", "logits": np.round(logits, 6).tolist()},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"wrote {a.out}: {len(states)} states x {len(names)} questions at L={list(STATIC_LENGTHS)}")


if __name__ == "__main__":
    main()
