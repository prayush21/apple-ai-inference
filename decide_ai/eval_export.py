"""The triage holdout as samples for the Swift Evaluations port (Xcode 27).

    python -m decide_ai.eval_export          # -> data/decide/eval/triage_holdout.json

One sample per holdout state, in holdout order:

* ``expected``: the reviewed labels as target probabilities, true -> 1,
  false -> 0, ``"unsure"`` -> 0.5 (Evaluations needs the expected value and
  the subject's output to be the same type; 0.5 is also what ``hedging`` in
  ``calibrate.py`` measures against);
* ``ids``: the five hypotheses paired with the state, tokenized exactly as the
  ``local-dynamic`` backend of ``calibrate.py`` does (``pick_length`` at
  ``max_len=128``: the longest pair rounded up to a multiple of 16), stored
  unpadded; the reader right-pads with ``pad_id`` to ``padded_len``;
* ``jev_key``: the sha256 of the canonical Jev request ``calibrate.py`` sent,
  so the Swift side can replay ``data/decide/jev_cache.jsonl``.

Neither model's answers are in here; the Swift subjects produce them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import jev
from .decider import pick_length
from .download import model_dir
from .holdout import QUESTION_KEYS, load_holdout, load_questions
from .tokenize import BPETokenizer

DEFAULT_PATH = Path("data/decide/eval/triage_holdout.json")
PAD_ID = 1
TARGET = {True: 1.0, False: 0.0, "unsure": 0.5}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--max-len", type=int, default=128)
    a = ap.parse_args(argv)

    tok = BPETokenizer.from_hf(model_dir())
    questions = {q: {"type": "boolean", **v} for q, v in load_questions().items()}
    jev_questions = {n: {"type": q["type"], "instructions": q["instructions"]} for n, q in questions.items()}
    samples = []
    for r in load_holdout():
        pairs = [(r["state"], questions[q]["hypothesis"]) for q in QUESTION_KEYS]
        padded = pick_length(max(tok.count_tokens(s, h) for s, h in pairs), a.max_len, [])
        ids, mask = tok.encode_batch(pairs, padded)
        rows = []
        for row, m in zip(ids, mask):
            k = int(m.sum())
            assert m[:k].all() and not m[k:].any() and (row[k:] == PAD_ID).all(), "mask is not a right-padded prefix"
            rows.append([int(v) for v in row[:k]])
        samples.append({
            "id": r["id"], "category": r["category"], "input": r["state"],
            "expected": {q: TARGET[r["labels"][q]] for q in QUESTION_KEYS},
            "padded_len": padded, "ids": rows,
            "jev_key": jev.cache_key(jev.request_body(r["state"], jev_questions)),
        })

    record = {"schema": "decide-eval-samples/1", "questions": list(QUESTION_KEYS), "pad_id": PAD_ID, "samples": samples}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {a.out}: {len(samples)} samples x {len(QUESTION_KEYS)} questions")


if __name__ == "__main__":
    main()
