"""SmolLM2 (Llama architecture) in plain ``torch``, written for ``torch.export``.

This is the LLM counterpart of ``snake_ai.model``: explicit softmax attention,
``register_buffer`` KV caches that ``coreai-torch`` turns into Core AI states,
and nothing from ``transformers`` at runtime — only its weight layout.

``SmolLM``
    Stateless. ``input_ids [B, T] -> logits [B, T, vocab]``, positions
    ``0..T-1``. Every generated token recomputes the whole sequence.

``SmolLMStateful``
    Same parameters plus ``k_cache`` / ``v_cache`` buffers of shape
    ``[layers, 1, kv_heads, max_seq_len, head_dim]``.
    ``forward(input_ids [1, T], position_ids [1, T]) -> logits [1, 1, vocab]``
    for the *last* of the T tokens only: prefill never needs the other rows,
    and the vocabulary projection over a 512-token prompt would otherwise be
    most of the prefill cost. Prompts are padded on the **left** to a static
    length (see ``left_pad``), so the last row is always the real last token.

Weights load from the Hugging Face ``model.safetensors`` by name, and loading
fails unless every tensor in the file is consumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if hf.get("rope_scaling"):
            raise ValueError("rope_scaling is not implemented")
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


# ------------------------------------------------------------------ modules


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cos_sin(position_ids: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype):
    """``position_ids [B, T]`` -> ``cos, sin [B, 1, T, head_dim]`` (HF's
    non-interleaved layout: the frequency table is repeated, not interleaved)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = position_ids.to(torch.float32)[:, :, None] * inv_freq[None, None, :]  # [B, T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)[:, None]  # [B, 1, T, hd]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * hd, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * hd, cfg.hidden_size, bias=False)

    def project(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """``x [B, T, D]`` -> ``q [B, H, T, hd]``, ``k, v [B, KV, T, hd]`` with RoPE applied."""
        B, T, _ = x.shape
        c = self.cfg
        q = self.q_proj(x).view(B, T, c.num_attention_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def attend(
        self,
        q: torch.Tensor,  # [B, H, Tq, hd]
        k: torch.Tensor,  # [B, KV, Tk, hd]
        v: torch.Tensor,  # [B, KV, Tk, hd]
        allowed: torch.Tensor,  # [n_rep * Tq, Tk] bool, row r*Tq + t is query t of group member r
    ) -> torch.Tensor:
        B, H, Tq, hd = q.shape
        c = self.cfg
        # Grouped-query attention without materialising repeated keys: query
        # head h uses KV head h // n_rep (HF's repeat_kv), so fold the n_rep
        # heads of each group into the query-row axis and multiply against the
        # KV heads directly. This keeps the per-step read of the 1024-slot
        # cache at KV heads, not H.
        q = q.reshape(B, c.num_key_value_heads, c.n_rep * Tq, hd)
        scores = (q @ k.transpose(-1, -2)) * (hd**-0.5)  # [B, KV, n_rep*Tq, Tk]
        scores = scores.masked_fill(~allowed, float("-inf"))
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = (attn @ v).reshape(B, H, Tq, hd).transpose(1, 2).reshape(B, Tq, H * hd)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)


def expand_group_mask(allowed: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``[Tq, Tk]`` -> ``[n_rep * Tq, Tk]`` by explicit expand (gotcha 3:
    the comparison that built ``allowed`` already had equal operand shapes,
    and nothing downstream broadcasts)."""
    Tq, Tk = allowed.shape
    return allowed[None].expand(n_rep, Tq, Tk).reshape(n_rep * Tq, Tk)


class SmolLM(nn.Module):
    """Stateless causal LM: ``input_ids [B, T] -> logits [B, T, vocab]``."""

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.cfg.tie_word_embeddings:
            return x @ self.embed_tokens.weight.t()
        return self.lm_head(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        c = self.cfg
        x = self.embed_tokens(input_ids)
        positions = torch.arange(T, device=input_ids.device)
        cos, sin = rope_cos_sin(positions[None], c.head_dim, c.rope_theta, x.dtype)
        # Expanded to [T, T] on both sides before comparing (gotcha 3).
        rows = positions[:, None].expand(T, T)
        cols = positions[None, :].expand(T, T)
        allowed = expand_group_mask(cols <= rows, c.n_rep)
        for layer in self.layers:
            q, k, v = layer.self_attn.project(layer.input_layernorm(x), cos, sin)
            x = x + layer.self_attn.attend(q, k, v, allowed)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.logits(x)

    # ------------------------------------------------------------- weights
    @classmethod
    def from_hf(cls, model_dir: Path, *, dtype: torch.dtype = torch.float32, max_seq_len: int = 1024):
        """Build the module and load ``model.safetensors`` by name. Every
        tensor in the file must be used and every parameter must be filled."""
        from safetensors.torch import load_file

        cfg = LlamaConfig.from_hf(model_dir, max_seq_len=max_seq_len)
        model = cls(cfg)
        tensors = load_file(str(Path(model_dir) / "model.safetensors"))
        params = dict(model.named_parameters())
        unused = []
        for hf_name, t in tensors.items():
            name = hf_name.removeprefix("model.")
            if name == "lm_head.weight" and cfg.tie_word_embeddings:
                # Some exports store the tied head explicitly; it must equal the embedding.
                assert torch.equal(t, tensors["model.embed_tokens.weight"]), "tied lm_head differs from embed_tokens"
                continue
            if name not in params:
                unused.append(hf_name)
                continue
            p = params.pop(name)
            if p.shape != t.shape:
                raise ValueError(f"{hf_name}: checkpoint {tuple(t.shape)} vs module {tuple(p.shape)}")
            with torch.no_grad():
                p.copy_(t.to(torch.float32))
        if unused or params:
            raise ValueError(f"unused checkpoint tensors {unused}; unfilled parameters {sorted(params)}")
        return model.to(dtype).eval()


class SmolLMStateful(SmolLM):
    """KV-cached variant: ``forward(input_ids [1, T], position_ids [1, T]) -> logits [1, 1, vocab]``.

    ``k_cache`` / ``v_cache`` are ``[layers, 1, kv_heads, max_seq_len,
    head_dim]`` and are updated in place with ``copy_`` at the end of
    ``forward``, the pattern ``torch.export`` records as a buffer mutation and
    ``coreai-torch`` maps onto Core AI states. Token ``t`` writes cache slot
    ``position_ids[t]`` and attends to every slot ``j <= position_ids[t]``.
    Left padding (``left_pad``) points every pad row at the last slot, which
    real tokens never attend before a decode step has overwritten it.
    """

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__(cfg)
        self.register_buffer("k_cache", torch.zeros(cfg.kv_cache_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(cfg.kv_cache_shape), persistent=False)

    def reset_cache(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        c = self.cfg
        S = c.max_seq_len
        T = input_ids.shape[1]
        pos = position_ids[0]  # [T]
        x = self.embed_tokens(input_ids)
        cos, sin = rope_cos_sin(position_ids, c.head_dim, c.rope_theta, x.dtype)

        # [T, S] with both sides expanded explicitly (gotcha 3), then repeated
        # for the n_rep query heads folded into each KV group.
        slots = torch.arange(S, device=input_ids.device)
        allowed = slots[None, :].expand(T, S) <= pos[:, None].expand(T, S)
        allowed = expand_group_mask(allowed, c.n_rep)

        # scatter (-> scatter_along_axis) supports a dynamic T; index_put does
        # not (gotcha 2). Index is [1, KV, T, hd] along the slot axis.
        slot_index = pos.view(1, 1, T, 1).expand(1, c.num_key_value_heads, T, c.head_dim)

        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            q, k, v = layer.self_attn.project(layer.input_layernorm(x), cos, sin)
            k_full = torch.scatter(self.k_cache[i], 2, slot_index, k.to(self.k_cache.dtype))
            v_full = torch.scatter(self.v_cache[i], 2, slot_index, v.to(self.v_cache.dtype))
            new_k.append(k_full)
            new_v.append(v_full)
            x = x + layer.self_attn.attend(q, k_full.to(q.dtype), v_full.to(q.dtype), allowed)
            x = x + layer.mlp(layer.post_attention_layernorm(x))

        self.k_cache.copy_(torch.stack(new_k))
        self.v_cache.copy_(torch.stack(new_v))
        return self.logits(x[:, -1:])


def left_pad(ids: list[int], start: int, length: int, max_seq_len: int, pad_id: int = 0):
    """Pad a chunk of ``ids`` that begins at cache position ``start`` on the
    left to ``length`` tokens -> ``(input_ids [1, length], position_ids [1, length])``
    as int32. Pad rows get position ``max_seq_len - 1``: they write garbage
    to that slot (several times, in any order) and nothing real attends to it
    until the decode step at that position overwrites it first."""
    n = len(ids)
    if n > length:
        raise ValueError(f"{n} tokens do not fit a length-{length} call")
    if start + n >= max_seq_len:
        raise ValueError(f"positions {start}..{start + n - 1} leave no free slot below {max_seq_len}")
    pad = length - n
    input_ids = torch.tensor([[pad_id] * pad + list(ids)], dtype=torch.int32)
    position_ids = torch.tensor([[max_seq_len - 1] * pad + list(range(start, start + n))], dtype=torch.int32)
    return input_ids, position_ids
