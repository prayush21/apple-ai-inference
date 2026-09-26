"""PyTorch vs Core AI (Python runtime) on a real chat prompt.

    python -m llm_ai.verify                  # stateful asset, 32 decode steps

The PyTorch reference is ``SmolLMStateful`` in **fp32** (itself checked
against ``transformers`` by ``llm_ai.reference``). Its greedy tokens are
teacher-forced into the Core AI asset so every step compares the same
context: prefill, then 32 decode steps, max |logit diff| per step and top-1
agreement. Pass: all 32 top-1 equal for an fp32 asset, >= 31/32 for fp16.

Also checks that the runtime mutates the states in place, that a reset
(fresh zeroed caches) reproduces the first prefill exactly, and that a prompt
longer than the prefill function (several chunks) matches PyTorch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import torch

from .model import SmolLMStateful, left_pad
from .runtime import COMPUTE_UNITS, DEFAULT_STATEFUL, StatefulLM
from .tokenizer import DEFAULT_MODEL_DIR, ChatTokenizer

VERIFY_PROMPT = "Give me three tips for writing clear commit messages."
# ~170 tokens with the chat template: three chunks through main_prefill_t64.
LONG_PROMPT = (
    "Here is a short story. A snake lived in a small terminal window. Every turn it looked at sixteen "
    "numbers describing the board and chose to go up, down, left or right. It had a memory of every move "
    "it had made, stored in a cache that never forgot, so it did not have to think about the past twice. "
    "One day a much larger model moved into the same machine. It had thirty-two layers and a vocabulary "
    "of forty-nine thousand words, and it also kept a cache, one slot per word it had read. The two of "
    "them shared the Neural Engine, the GPU and eight gigabytes of memory, and they did not always get "
    "along. Summarize the story in one sentence."
)


@torch.no_grad()
def torch_reference(model_dir: Path, ids: list[int], steps: int, prefill_len: int):
    """-> (greedy tokens, per-step logits [steps + 1, vocab]); row 0 is the prompt's last token."""
    m = SmolLMStateful.from_hf(model_dir)
    x, p = left_pad(ids, 0, prefill_len, m.cfg.max_seq_len)
    rows = [m(x.long(), p.long())[0, -1]]
    tokens = []
    for i in range(steps):
        tokens.append(int(rows[-1].argmax()))
        rows.append(m(torch.tensor([[tokens[-1]]]), torch.tensor([[len(ids) + i]]))[0, -1])
    return tokens, torch.stack(rows).numpy()


async def run(a: argparse.Namespace) -> dict:
    tok = ChatTokenizer(a.model_dir)
    ids = tok.encode_chat(a.prompt)

    lm = await StatefulLM.load(a.asset, a.compute)
    precision = "fp16" if lm.cache_dtype == np.float16 else "fp32"
    print(f"loaded {a.asset} ({precision}) in {lm.load_ms:.0f} ms: {sorted(lm.model.function_names)}")
    bucket = lm.bucket(len(ids))

    t0 = time.perf_counter()
    tokens, ref = torch_reference(a.model_dir, ids, a.steps, bucket)
    print(f"PyTorch fp32 reference in {time.perf_counter() - t0:.1f} s: {tok.decode(tokens)!r}")

    first = await lm.prefill_ids(ids)
    assert np.abs(lm.key_cache.numpy().astype(np.float32)).sum() > 0, "keyCache was not updated in place"
    got = [first]
    for t in tokens[:-1]:
        got.append(await lm.decode(t))
    got.append(await lm.decode(tokens[-1]))
    got = np.stack(got).astype(np.float32)

    diffs = np.abs(got - ref).max(axis=-1)
    top1 = got.argmax(-1) == ref.argmax(-1)
    for i, (d, ok) in enumerate(zip(diffs, top1)):
        tag = "prefill" if i == 0 else f"decode {i:2d}"
        mark = "" if ok else f"   <- top-1 differs: coreai {tok.decode([int(got[i].argmax())])!r} vs torch {tok.decode([int(ref[i].argmax())])!r}"
        print(f"  {tag}: max |logit diff| {d:.4f}{mark}")
    agree = int(top1[: a.steps].sum())  # the steps that chose a generated token
    need = a.steps if precision == "fp32" else a.steps - 1
    print(f"top-1 agreement {agree}/{a.steps} (need >= {need})")

    lm.reset()
    again = await lm.prefill_ids(ids)
    reset_diff = float(np.abs(again.astype(np.float32) - first.astype(np.float32)).max())
    print(f"reset: prefill after fresh caches differs from the first by {reset_diff:.2e}")

    lm.reset()
    long_ids = tok.encode_chat(LONG_PROMPT)
    chunked = (await lm.prefill_ids(long_ids)).astype(np.float32)
    from .model import SmolLM

    with torch.no_grad():
        full = SmolLM.from_hf(a.model_dir)(torch.tensor([long_ids]))[0, -1].numpy()
    long_diff = float(np.abs(chunked - full).max())
    long_top1 = int(chunked.argmax()) == int(full.argmax())
    print(f"chunked prefill of {len(long_ids)} tokens: max |logit diff| {long_diff:.4f}, top-1 {'agrees' if long_top1 else 'DIFFERS'}")

    record = {
        "asset": str(a.asset),
        "precision": precision,
        "prompt": a.prompt,
        "prompt_tokens": len(ids),
        "prefill_function": f"main_prefill_t{bucket}",
        "steps": a.steps,
        "max_abs_logit_diff_per_step": [round(float(d), 5) for d in diffs],
        "top1_agreement": agree,
        "diverging_steps": [int(i) for i in np.nonzero(~top1[: a.steps])[0]],
        "reset_diff": reset_diff,
        "chunked_prefill": {"prompt_tokens": len(long_ids), "max_abs_logit_diff": round(long_diff, 5), "top1_agrees": long_top1},
        "compute": a.compute,
        "load_ms": round(lm.load_ms, 1),
        "reference_text": tok.decode(tokens),
    }
    assert agree >= need, f"top-1 agreement {agree}/{a.steps}"
    assert reset_diff == 0.0
    assert long_top1, "chunked prefill top-1 differs from PyTorch"
    return record


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", type=Path, default=DEFAULT_STATEFUL)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--prompt", default=VERIFY_PROMPT)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--compute", choices=COMPUTE_UNITS, default="gpu")
    ap.add_argument("--json", type=Path, help="write the verify record here")
    a = ap.parse_args(argv)
    record = asyncio.run(run(a))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
