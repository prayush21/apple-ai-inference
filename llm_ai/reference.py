"""Check ``llm_ai.model`` against Hugging Face ``transformers`` and save a golden run.

    python -m llm_ai.reference            # -> data/llm/golden.json

1. Logits of ``SmolLM`` vs ``LlamaForCausalLM`` (fp32, same bf16 weights) on
   three chat prompts: max |diff| must be < 1e-3.
2. ``SmolLMStateful`` (prefill + one token per step) must pick the same
   greedy tokens as ``SmolLM`` recomputing the full sequence, for 64 steps.
3. The prompt ids, the 64 greedy tokens and the last-position logits of the
   prompt go to ``data/llm/golden.json``, so later checks (Core AI verify,
   the Swift generator) compare against a file instead of loading
   ``transformers``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .model import SmolLM, SmolLMStateful, left_pad
from .tokenizer import DEFAULT_MODEL_DIR, ChatTokenizer

GOLDEN_PATH = Path("data/llm/golden.json")
PROMPTS = [
    "What is the capital of France?",
    "Write a haiku about a snake that plays video games.",
    "Explain in two sentences why the sky is blue.",
]


@torch.no_grad()
def greedy_stateless(model: SmolLM, ids: list[int], steps: int) -> list[int]:
    seq = list(ids)
    out = []
    for _ in range(steps):
        nxt = int(model(torch.tensor([seq]))[0, -1].argmax())
        out.append(nxt)
        seq.append(nxt)
    return out


@torch.no_grad()
def greedy_stateful(model: SmolLMStateful, ids: list[int], steps: int, prefill_len: int) -> list[int]:
    model.reset_cache()
    S = model.cfg.max_seq_len
    input_ids, position_ids = left_pad(ids, 0, prefill_len, S)
    logits = model(input_ids.long(), position_ids.long())
    out = []
    pos = len(ids)
    for _ in range(steps):
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        logits = model(torch.tensor([[nxt]]), torch.tensor([[pos]]))
        pos += 1
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--out", type=Path, default=GOLDEN_PATH)
    a = ap.parse_args(argv)

    tok = ChatTokenizer(a.model_dir)
    prompts = [tok.encode_chat(p) for p in PROMPTS]

    t0 = time.perf_counter()
    ours = SmolLM.from_hf(a.model_dir)
    print(f"SmolLM loaded in {time.perf_counter() - t0:.1f} s")

    from transformers import LlamaForCausalLM

    hf = LlamaForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.float32).eval()
    diffs = []
    with torch.no_grad():
        for text, ids in zip(PROMPTS, prompts):
            x = torch.tensor([ids])
            d = float((ours(x) - hf(x).logits).abs().max())
            diffs.append(d)
            print(f"  {len(ids):3d} tokens  max |logit diff| vs transformers = {d:.2e}  ({text[:40]!r})")
    del hf
    assert max(diffs) < 1e-3, f"logits differ from transformers by {max(diffs)}"

    stateful = SmolLMStateful.from_hf(a.model_dir)
    ids = prompts[0]
    ref = greedy_stateless(ours, ids, a.steps)
    got = greedy_stateful(stateful, ids, a.steps, prefill_len=128)
    print(f"  stateless greedy: {tok.decode(ref)!r}")
    assert got == ref, f"stateful diverges at step {next(i for i, (x, y) in enumerate(zip(got, ref)) if x != y)}"
    print(f"  stateful == stateless for {a.steps} greedy steps")

    with torch.no_grad():
        last_logits = ours(torch.tensor([ids]))[0, -1].numpy()
    record = {
        "schema": "llm-golden/1",
        "model_dir": str(a.model_dir),
        "prompt": PROMPTS[0],
        "prompt_ids": ids,
        "greedy_ids": ref,
        "greedy_text": tok.decode(ref),
        "prompt_last_logits_top8": [
            [int(i), round(float(last_logits[i]), 5)] for i in np.argsort(-last_logits)[:8]
        ],
        "transformers_max_abs_diff": max(diffs),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
