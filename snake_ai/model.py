"""Snake action-prediction transformer, in two flavours.

``SnakeTransformer``
    Stateless. Takes the full game history ``features [B, T, 16]`` and returns
    ``logits [B, T, 4]``. Simple to author and convert, but every inference
    recomputes keys/values for the whole history, so latency grows
    quadratically with the number of moves.

``SnakeTransformerStateful``
    Same parameters, plus ``k_cache`` / ``v_cache`` buffers registered with
    ``register_buffer``. ``torch.export`` turns mutated buffers into mutable
    inputs, which ``coreai-torch`` converts into Core AI *states*: arguments
    that are read and updated in place during inference. Each call then only
    needs the *new* board features and their ``position_ids``.

Both are written with plain ``torch`` ops (explicit softmax attention, no
``nn.TransformerEncoder``) so the exported graph is small and easy to inspect
in the Core AI Debugger.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import FEATURE_DIM

N_ACTIONS = 4


@dataclass(frozen=True)
class SnakeModelConfig:
    feature_dim: int = FEATURE_DIM
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    n_actions: int = N_ACTIONS
    max_seq_len: int = 256

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class Block(nn.Module):
    """Pre-LN transformer block whose attention is split into ``qkv`` and
    ``attend`` so the stateful model can route k/v through the cache."""

    def __init__(self, cfg: SnakeModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        return q, k, v

    def attend(
        self,
        q: torch.Tensor,  # [B, Tq, D]
        k: torch.Tensor,  # [B, Tk, D]
        v: torch.Tensor,  # [B, Tk, D]
        allowed: torch.Tensor,  # [Tq, Tk] bool, True where attention is permitted
    ) -> torch.Tensor:
        B, Tq, D = q.shape
        Tk = k.shape[1]
        H, hd = self.cfg.n_heads, self.cfg.head_dim
        q = q.view(B, Tq, H, hd).transpose(1, 2)  # [B, H, Tq, hd]
        k = k.view(B, Tk, H, hd).transpose(1, 2)
        v = v.view(B, Tk, H, hd).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) * (hd**-0.5)  # [B, H, Tq, Tk]
        scores = scores.masked_fill(~allowed, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Tq, D)
        return self.proj(out)

    def feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln2(x))


class SnakeTransformer(nn.Module):
    """Stateless causal transformer: ``features [B, T, 16] -> logits [B, T, 4]``."""

    def __init__(self, cfg: SnakeModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SnakeModelConfig()
        c = self.cfg
        self.in_proj = nn.Linear(c.feature_dim, c.d_model)
        self.pos_emb = nn.Embedding(c.max_seq_len, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.ln_final = nn.LayerNorm(c.d_model)
        self.action_head = nn.Linear(c.d_model, c.n_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, T, _ = features.shape
        positions = torch.arange(T, device=features.device)
        x = self.in_proj(features) + self.pos_emb(positions)
        causal = torch.ones(T, T, dtype=torch.bool, device=features.device).tril()
        for block in self.blocks:
            q, k, v = block.project_qkv(x)
            x = x + block.attend(q, k, v, causal)
            x = x + block.feed_forward(x)
        return self.action_head(self.ln_final(x))

    # ----------------------------------------------------------- checkpoints
    def save_checkpoint(self, path: Path | str) -> None:
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict()}, path)

    @classmethod
    def load_checkpoint(cls, path: Path | str):
        ckpt = torch.load(path, map_location="cpu")
        model = cls(SnakeModelConfig(**ckpt["config"]))
        # strict=False lets the stateful subclass load weights that have no cache entries.
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        return model


class SnakeTransformerStateful(SnakeTransformer):
    """KV-cached variant. ``forward(features [1, T, 16], position_ids [1, T])``.

    ``k_cache`` / ``v_cache`` have shape ``[n_layers, 1, max_seq_len, d_model]``
    and are updated in place with ``copy_`` at the end of ``forward`` — the
    pattern ``torch.export`` recognises as a buffer mutation and that
    ``coreai-torch`` maps onto Core AI states.
    """

    def __init__(self, cfg: SnakeModelConfig | None = None) -> None:
        super().__init__(cfg)
        c = self.cfg
        self.register_buffer("k_cache", torch.zeros(c.n_layers, 1, c.max_seq_len, c.d_model))
        self.register_buffer("v_cache", torch.zeros(c.n_layers, 1, c.max_seq_len, c.d_model))

    def reset_cache(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()

    def forward(self, features: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        pos = position_ids[0]  # [T]
        x = self.in_proj(features) + self.pos_emb(position_ids)

        # New token t may attend to every cache slot j with j <= position_ids[t].
        # Both sides are expanded to [T, max_seq_len] explicitly: the Core AI
        # comparison kernel wants identical shapes rather than a two-sided
        # broadcast of [1, max_seq_len] against [T, 1].
        T = position_ids.shape[1]
        slots = torch.arange(self.cfg.max_seq_len, device=features.device)
        allowed = slots[None, :].expand(T, -1) <= pos[:, None].expand(-1, self.cfg.max_seq_len)

        # Scatter index for writing the T new rows into their cache slots.
        # ``scatter`` (not ``index_copy``/``index_put``) is used because
        # coreai-torch lowers it to ``scatter_along_axis``, which supports a
        # dynamic sequence dimension; index_put only allows dynamic dim 0.
        slot_index = pos.view(1, -1, 1).expand(1, -1, self.cfg.d_model)

        new_k, new_v = [], []
        for i, block in enumerate(self.blocks):
            q, k_new, v_new = block.project_qkv(x)
            # Read the previous keys/values and write the new ones into their slots.
            k_full = torch.scatter(self.k_cache[i], 1, slot_index, k_new)
            v_full = torch.scatter(self.v_cache[i], 1, slot_index, v_new)
            new_k.append(k_full)
            new_v.append(v_full)
            x = x + block.attend(q, k_full, v_full, allowed)
            x = x + block.feed_forward(x)

        # Update key/value caches in place.
        self.k_cache.copy_(torch.stack(new_k))
        self.v_cache.copy_(torch.stack(new_v))

        return self.action_head(self.ln_final(x))
