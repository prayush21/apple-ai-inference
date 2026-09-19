"""The NLI cross-encoder as a self-contained ``nn.Module``.

``cross-encoder/nli-MiniLM2-L6-H768`` is a 6-layer RoBERTa-style encoder
(MiniLMv2 distilled from RoBERTa-Large) with a 3-way NLI classification head.
This module re-implements it with plain tensor ops and loads the Hugging Face
state dict directly, so the graph ``torch.export`` sees is small, branch-free
and readable in the Core AI Debugger. ``transformers`` is only used by the
tests to check parity; nothing here imports it.

    ids, mask = ...                       # [N, L] int32 each, right-padded
    logits = NLICrossEncoder.from_pretrained(hf_dir)(ids, mask)   # [N, 3]

Label order comes from ``config.json`` (``id2label``): for this checkpoint
``[contradiction, entailment, neutral]``. ``LABELS`` is read from the config,
never assumed.

RoBERTa details that differ from a vanilla BERT and matter for parity:

* position ids start at ``pad_token_id + 1`` (= 2), so token *t* uses row
  ``t + 2`` of the 514-row position table. With right padding this is just
  ``arange(L) + 2`` for every row; padded positions never influence the
  ``<s>`` token because they are masked out of attention.
* one token-type embedding row (always added), no pooler; the classifier is
  ``dense → tanh → out_proj`` on the ``<s>`` hidden state.
* GELU is the exact (erf) form, LayerNorm eps 1e-5, post-LN blocks.

The attention mask is additive: ``(1 - mask) * -1e4`` broadcast explicitly to
``[N, H, L, L]`` before the add (see docs/coreai-ecosystem.md: comparison and
elementwise kernels want same-shaped operands).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_VALUE = -1e4


@dataclass(frozen=True)
class NLIConfig:
    vocab_size: int = 50265
    hidden: int = 768
    n_heads: int = 12
    n_layers: int = 6
    intermediate: int = 3072
    max_positions: int = 514
    pad_token_id: int = 1
    layer_norm_eps: float = 1e-5
    labels: tuple[str, ...] = ("contradiction", "entailment", "neutral")

    @property
    def head_dim(self) -> int:
        return self.hidden // self.n_heads

    @classmethod
    def from_hf(cls, hf_dir: Path) -> "NLIConfig":
        cfg = json.loads((Path(hf_dir) / "config.json").read_text())
        assert cfg["model_type"] == "roberta", cfg["model_type"]
        labels = tuple(cfg["id2label"][str(i)] for i in range(len(cfg["id2label"])))
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden=cfg["hidden_size"],
            n_heads=cfg["num_attention_heads"],
            n_layers=cfg["num_hidden_layers"],
            intermediate=cfg["intermediate_size"],
            max_positions=cfg["max_position_embeddings"],
            pad_token_id=cfg["pad_token_id"],
            layer_norm_eps=cfg["layer_norm_eps"],
            labels=labels,
        )


class Layer(nn.Module):
    """One post-LN encoder block with explicit softmax attention."""

    def __init__(self, cfg: NLIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.attn_out = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        self.ffn_in = nn.Linear(d, cfg.intermediate)
        self.ffn_out = nn.Linear(cfg.intermediate, d)
        self.ln2 = nn.LayerNorm(d, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        # x [N, L, D]; attn_bias [N, H, L, L] additive (0 or MASK_VALUE)
        N, L, D = x.shape
        H, hd = self.cfg.n_heads, self.cfg.head_dim
        q = self.query(x).view(N, L, H, hd).transpose(1, 2)  # [N, H, L, hd]
        k = self.key(x).view(N, L, H, hd).transpose(1, 2)
        v = self.value(x).view(N, L, H, hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (hd**-0.5) + attn_bias  # [N, H, L, L]
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v).transpose(1, 2).reshape(N, L, D)
        x = self.ln1(x + self.attn_out(ctx))
        x = self.ln2(x + self.ffn_out(F.gelu(self.ffn_in(x))))
        return x


class NLICrossEncoder(nn.Module):
    def __init__(self, cfg: NLIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.word_emb = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.pos_emb = nn.Embedding(cfg.max_positions, cfg.hidden)
        # RoBERTa has a single token-type row; keep it as a plain parameter.
        self.type_emb = nn.Parameter(torch.zeros(cfg.hidden))
        self.emb_ln = nn.LayerNorm(cfg.hidden, eps=cfg.layer_norm_eps)
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.n_layers))
        self.cls_dense = nn.Linear(cfg.hidden, cfg.hidden)
        self.cls_out = nn.Linear(cfg.hidden, len(cfg.labels))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids [N, L] int32, attention_mask [N, L] int32 (1 = real token) -> logits [N, 3]."""
        N, L = input_ids.shape
        H = self.cfg.n_heads
        input_ids = input_ids.to(torch.int64)
        position_ids = torch.arange(L, device=input_ids.device) + (self.cfg.pad_token_id + 1)
        x = self.word_emb(input_ids) + self.pos_emb(position_ids)[None] + self.type_emb
        x = self.emb_ln(x)

        # Additive mask, expanded to the full score shape so every elementwise
        # op downstream sees identical operand shapes.
        bias = (1.0 - attention_mask.to(torch.float32)) * MASK_VALUE  # [N, L]
        bias = bias.view(N, 1, 1, L).expand(N, H, L, L)

        for layer in self.layers:
            x = layer(x, bias)

        cls = x[:, 0]  # <s> token
        return self.cls_out(torch.tanh(self.cls_dense(cls)))

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_pretrained(cls, hf_dir: Path) -> "NLICrossEncoder":
        """Load the HF safetensors checkpoint into this module (eval mode)."""
        from safetensors.torch import load_file

        hf_dir = Path(hf_dir)
        cfg = NLIConfig.from_hf(hf_dir)
        sd = load_file(str(hf_dir / "model.safetensors"))
        model = cls(cfg)
        model.load_state_dict(remap_state_dict(sd, cfg), strict=True)
        return model.eval()


def remap_state_dict(sd: dict[str, torch.Tensor], cfg: NLIConfig) -> dict[str, torch.Tensor]:
    """HF ``roberta.*`` / ``classifier.*`` names -> this module's names."""
    p = "roberta."
    out = {
        "word_emb.weight": sd[p + "embeddings.word_embeddings.weight"],
        "pos_emb.weight": sd[p + "embeddings.position_embeddings.weight"],
        "type_emb": sd[p + "embeddings.token_type_embeddings.weight"][0],
        "emb_ln.weight": sd[p + "embeddings.LayerNorm.weight"],
        "emb_ln.bias": sd[p + "embeddings.LayerNorm.bias"],
        "cls_dense.weight": sd["classifier.dense.weight"],
        "cls_dense.bias": sd["classifier.dense.bias"],
        "cls_out.weight": sd["classifier.out_proj.weight"],
        "cls_out.bias": sd["classifier.out_proj.bias"],
    }
    pairs = {
        "query": "attention.self.query",
        "key": "attention.self.key",
        "value": "attention.self.value",
        "attn_out": "attention.output.dense",
        "ln1": "attention.output.LayerNorm",
        "ffn_in": "intermediate.dense",
        "ffn_out": "output.dense",
        "ln2": "output.LayerNorm",
    }
    for i in range(cfg.n_layers):
        for ours, theirs in pairs.items():
            for wb in ("weight", "bias"):
                out[f"layers.{i}.{ours}.{wb}"] = sd[f"{p}encoder.layer.{i}.{theirs}.{wb}"]
    return out
