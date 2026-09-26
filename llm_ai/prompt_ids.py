"""Pre-tokenized prompts for the Swift ``ModelGenerator`` (no Swift BPE encoder yet).

    python -m llm_ai.prompt_ids              # -> data/llm/prompt_ids.json

``chat``: a few chat prompts (template applied) that ``llm-cli`` and
``LLMApp`` offer. ``bench``: the exact ids ``llm_ai.play --bench`` uses for
each prompt length, so Swift and Python time the same inputs. ``reference``:
greedy tokens for the first chat prompt from the PyTorch fp32 model, which
the Core AI fp16 asset matched 32/32 in ``llm_ai.verify``; ``llm-cli``
checks itself against them before timing anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .play import bench_prompt
from .tokenizer import DEFAULT_MODEL_DIR, ChatTokenizer
from .verify import VERIFY_PROMPT, torch_reference

DEFAULT_PATH = Path("data/llm/prompt_ids.json")
CHAT_PROMPTS = [
    VERIFY_PROMPT,
    "What is the capital of France?",
    "Write a haiku about a snake that plays video games.",
    "Explain in two sentences why the sky is blue.",
]
BENCH_LENGTHS = (16, 128, 512)
REFERENCE_STEPS = 32


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    a = ap.parse_args(argv)

    tok = ChatTokenizer(a.model_dir)
    chat = [{"prompt": p, "ids": tok.encode_chat(p)} for p in CHAT_PROMPTS]
    greedy, _ = torch_reference(a.model_dir, chat[0]["ids"], REFERENCE_STEPS, 64)
    record = {
        "schema": "llm-prompt-ids/1",
        "tokenizer": str(a.model_dir / "tokenizer.json"),
        "eos_id": tok.eos_id,
        "pad_id": tok.pad_id,
        "chat": chat,
        "bench": {str(n): bench_prompt(tok, n) for n in BENCH_LENGTHS},
        "reference": {"chat_index": 0, "greedy_ids": greedy, "text": tok.decode(greedy)},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"wrote {a.out}: {len(chat)} chat prompts, bench lengths {list(BENCH_LENGTHS)}")


if __name__ == "__main__":
    main()
