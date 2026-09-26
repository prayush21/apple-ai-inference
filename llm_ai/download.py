"""Fetch SmolLM2-360M-Instruct weights and tokenizer from Hugging Face.

    python -m llm_ai.download                 # -> models/llm/hf/SmolLM2-360M-Instruct
    python -m llm_ai.download --repo HuggingFaceTB/SmolLM2-135M-Instruct

Only the files the rest of ``llm_ai`` reads are fetched: the single
``model.safetensors`` shard, ``config.json`` and the tokenizer. The directory
is gitignored; every other module takes ``--hf-dir`` pointing at it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_REPO = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEFAULT_HF_DIR = Path("models/llm/hf")

# Everything the model, tokenizer and generation code need, nothing else
# (no ONNX exports, no README images).
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def download(repo: str = DEFAULT_REPO, hf_dir: Path = DEFAULT_HF_DIR) -> Path:
    """Download ``repo`` into ``hf_dir/<model name>`` and return that path."""
    from huggingface_hub import snapshot_download

    target = hf_dir / repo.split("/")[-1]
    snapshot_download(repo_id=repo, local_dir=target, allow_patterns=ALLOW_PATTERNS)
    return target


def summarize(model_dir: Path) -> str:
    cfg = json.loads((model_dir / "config.json").read_text())
    size_mb = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file()) / 1e6
    return (
        f"{model_dir}: {cfg['model_type']} | layers={cfg['num_hidden_layers']} "
        f"hidden={cfg['hidden_size']} heads={cfg['num_attention_heads']} "
        f"kv_heads={cfg['num_key_value_heads']} vocab={cfg['vocab_size']} | {size_mb:.0f} MB on disk"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--hf-dir", type=Path, default=DEFAULT_HF_DIR)
    a = ap.parse_args(argv)
    print(summarize(download(a.repo, a.hf_dir)))


if __name__ == "__main__":
    main()
