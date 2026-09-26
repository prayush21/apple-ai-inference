"""Tokenizer + chat template for SmolLM2-Instruct.

Encoding uses Hugging Face ``tokenizers`` (a Rust library, not
``transformers``). The chat template from ``tokenizer_config.json`` is
simple enough to spell out, which keeps Jinja out of the runtime path and
is the exact string the Swift side will have to build::

    <|im_start|>system\\n{system}<|im_end|>\\n
    <|im_start|>user\\n{content}<|im_end|>\\n
    <|im_start|>assistant\\n

(the default system message is inserted when the first message is not one).
"""

from __future__ import annotations

from pathlib import Path

from .download import DEFAULT_HF_DIR, DEFAULT_REPO

DEFAULT_MODEL_DIR = DEFAULT_HF_DIR / DEFAULT_REPO.split("/")[-1]
DEFAULT_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def chat_prompt(messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    out = []
    if messages and messages[0]["role"] != "system":
        out.append(f"{IM_START}system\n{DEFAULT_SYSTEM}{IM_END}\n")
    for m in messages:
        out.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n")
    return "".join(out)


class ChatTokenizer:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.eos_id = self.tok.token_to_id(IM_END)
        self.pad_id = self.tok.token_to_id("<|endoftext|>")

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_chat(self, user: str, system: str | None = None) -> list[int]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return self.encode(chat_prompt(msgs))

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)
