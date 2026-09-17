"""Convert the trained PyTorch model to a Core AI ``.aimodel`` asset.

Two variants are produced, mirroring the two stages of the talk:

    models/SnakeTransformer.aimodel          stateless, input  features[1, ?, 16]
    models/SnakeTransformerStateful.aimodel  KV-cached, inputs features[1, ?, 16],
                                             position_ids[1, ?]; states keyCache,
                                             valueCache [n_layers, 1, max_seq, d]

Pipeline for each:
    torch.export.export(...)              -> ExportedProgram (with dynamic dims)
    .run_decompositions(get_decomp_table) -> lower to the aten subset Core AI accepts
    TorchConverter().add_exported_program -> stage graph, name IO and states
    .to_coreai()                          -> AIProgram
    .save_asset(path)                     -> .aimodel bundle on disk
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coreai_torch
import torch

from .model import SnakeTransformer, SnakeTransformerStateful


def convert_stateless(ckpt: Path, out: Path) -> Path:
    pt_model = SnakeTransformer.load_checkpoint(ckpt)
    cfg = pt_model.cfg
    example = torch.randn(1, 5, cfg.feature_dim)

    # Sequence length is dynamic so the model is not traced with the static
    # sample length of 5. The upper bound is the positional-embedding table size.
    seq_len = torch.export.Dim("seq_len", min=1, max=cfg.max_seq_len)
    exported = torch.export.export(
        pt_model,
        args=(example,),
        dynamic_shapes={"features": {1: seq_len}},
    )
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())

    ai_program = (
        coreai_torch.TorchConverter()
        .add_exported_program(exported, input_names=["features"], output_names=["logits"])
        .to_coreai()
    )
    ai_program.optimize()
    ai_program.save_asset(out)
    return out


def convert_stateful(ckpt: Path, out: Path) -> Path:
    stateful = SnakeTransformerStateful.load_checkpoint(ckpt)
    cfg = stateful.cfg
    example_features = torch.randn(1, 5, cfg.feature_dim)
    example_position_ids = torch.arange(5)[None]

    # Both inputs share the same dynamic sequence dimension: a call may prefill
    # several steps at once or decode a single new step.
    seq_len = torch.export.Dim("seq_len", min=1, max=cfg.max_seq_len)
    exported = torch.export.export(
        stateful,
        args=(example_features, example_position_ids),
        dynamic_shapes={"features": {1: seq_len}, "position_ids": {1: seq_len}},
    )
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())

    ai_program = (
        coreai_torch.TorchConverter()
        .add_exported_program(
            exported,
            input_names=["features", "position_ids"],
            # One name per mutated buffer, in registration order (k_cache, v_cache).
            state_names=["keyCache", "valueCache"],
            output_names=["logits"],
        )
        .to_coreai()
    )
    # ``optimize()`` runs the pre-compilation rewrite pass. Among other things
    # it turns inputs tagged ``MutableBuffers.buffer_mutation`` into real Core AI
    # states (``!coreai.handle<tensor<...>>``); without it the runtime reports
    # keyCache/valueCache as ordinary inputs+outputs and ``States: []``.
    ai_program.optimize()
    ai_program.save_asset(out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/snake.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    ap.add_argument("--variant", choices=["stateless", "stateful", "both"], default="both")
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    if a.variant in ("stateless", "both"):
        p = convert_stateless(a.checkpoint, a.out_dir / "SnakeTransformer.aimodel")
        print(f"wrote {p}")
    if a.variant in ("stateful", "both"):
        p = convert_stateful(a.checkpoint, a.out_dir / "SnakeTransformerStateful.aimodel")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
