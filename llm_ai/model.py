"""SmolLM2 (Llama architecture) in plain ``torch``, written for ``torch.export``.

This is the LLM counterpart of ``snake_ai.model``: explicit softmax attention,
``register_buffer`` KV caches that ``coreai-torch`` turns into Core AI states,
and nothing from ``transformers`` at runtime — only its weight layout.

Milestone 1 ships the config only. ``LlamaConfig.from_hf`` reads the Hugging
Face ``config.json`` so the next step (the module itself) has the exact shapes
to build and a place to assert the checkpoint against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    max_position_embeddings: int
    # Fixed cache length used for the Core AI states. Kept well below the
    # model's 8k context: attention runs over the whole cache every step (see
    # docs/coreai-ecosystem.md §6), so this is a direct latency knob.
    max_seq_len: int = 1024

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def n_rep(self) -> int:
        """Grouped-query attention: how many query heads share one KV head."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf(cls, model_dir: Path, max_seq_len: int = 1024) -> "LlamaConfig":
        hf = json.loads((Path(model_dir) / "config.json").read_text())
        if hf.get("model_type") != "llama":
            raise ValueError(f"expected a llama-architecture checkpoint, got {hf.get('model_type')!r}")
        return cls(
            vocab_size=hf["vocab_size"],
            hidden_size=hf["hidden_size"],
            intermediate_size=hf["intermediate_size"],
            num_hidden_layers=hf["num_hidden_layers"],
            num_attention_heads=hf["num_attention_heads"],
            num_key_value_heads=hf["num_key_value_heads"],
            rms_norm_eps=hf["rms_norm_eps"],
            rope_theta=hf["rope_theta"],
            tie_word_embeddings=hf.get("tie_word_embeddings", False),
            max_position_embeddings=hf["max_position_embeddings"],
            max_seq_len=max_seq_len,
        )

    @property
    def kv_cache_shape(self) -> tuple[int, int, int, int, int]:
        """``[layers, 1, kv_heads, max_seq_len, head_dim]`` — one state tensor
        for all keys and one for all values, like the snake model."""
        return (self.num_hidden_layers, 1, self.num_key_value_heads, self.max_seq_len, self.head_dim)
