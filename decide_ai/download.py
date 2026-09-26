"""Fetch the NLI cross-encoder weights and tokenizer from Hugging Face.

    python -m decide_ai.download            # -> models/decide/hf/nli-MiniLM2-L6-H768

Only the files the rest of ``decide_ai`` reads are fetched: ``config.json``,
the single ``model.safetensors`` shard and the tokenizer files. The ONNX /
OpenVINO exports and the ``pytorch_model.bin`` duplicate in the repo are
skipped (they would triple the download). The directory is gitignored; every
other module takes ``--hf-dir`` pointing at it.

Note on the checkpoint: ``cross-encoder/nli-MiniLM2-L6-H768`` is MiniLMv2
distilled from RoBERTa-Large, so ``config.json`` says ``model_type: roberta``
and the tokenizer is byte-level BPE (``vocab.json`` + ``merges.txt``), not
WordPiece. ``decide_ai.tokenize`` implements that BPE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_REPO = "cross-encoder/nli-MiniLM2-L6-H768"
DEFAULT_HF_DIR = Path("models/decide/hf")

ALLOW_PATTERNS = [
    "config.json",
    "model.safetensors",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def model_dir(hf_dir: Path = DEFAULT_HF_DIR, repo: str = DEFAULT_REPO) -> Path:
    return hf_dir / repo.split("/")[-1]


def download(repo: str = DEFAULT_REPO, hf_dir: Path = DEFAULT_HF_DIR) -> Path:
    """Download ``repo`` into ``hf_dir/<model name>`` and return that path."""
    from huggingface_hub import snapshot_download

    target = model_dir(hf_dir, repo)
    snapshot_download(repo_id=repo, local_dir=target, allow_patterns=ALLOW_PATTERNS)
    return target


def summarize(path: Path) -> str:
    cfg = json.loads((path / "config.json").read_text())
    size_mb = sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts) / 1e6
    labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
    return (
        f"{path}: {cfg['model_type']} | layers={cfg['num_hidden_layers']} hidden={cfg['hidden_size']} "
        f"heads={cfg['num_attention_heads']} vocab={cfg['vocab_size']} max_pos={cfg['max_position_embeddings']} "
        f"| labels={labels} | {size_mb:.0f} MB on disk"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--hf-dir", type=Path, default=DEFAULT_HF_DIR)
    a = ap.parse_args(argv)
    print(summarize(download(a.repo, a.hf_dir)))


if __name__ == "__main__":
    main()
