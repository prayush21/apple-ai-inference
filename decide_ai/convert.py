"""Convert the NLI cross-encoder to Core AI ``.aimodel`` assets.

    python -m decide_ai.convert                 # both assets into models/decide/
    python -m decide_ai.convert --variant dynamic

Two assets, mirroring the snake project's dynamic / static split:

    models/decide/NLICrossEncoder.aimodel        function ``main``
        input_ids [?, ?] int32, attention_mask [?, ?] int32 -> logits [?, 3]
        batch 1..32, seq 1..256 (torch.export Dims)
    models/decide/NLICrossEncoderStatic.aimodel  functions ``main_n{N}_l{L}``
        one fully static entrypoint per (N, L) in STATIC_SHAPES, via
        ``set_static_shape_config`` -> ``coreai.enumerated_shapes``; skips
        per-call "Function Type Inference" (~14 ms of a 330 ms call at
        N=1, L=64 on the local runtime).

Pipeline for each:
    torch.export.export(...)              -> ExportedProgram (with dynamic dims)
    .run_decompositions(get_decomp_table) -> lower to the aten subset Core AI accepts
    TorchConverter().add_exported_program -> stage graph, name IO
    .to_coreai()                          -> AIProgram
    [.set_static_shape_config]            -> enumerated static entrypoints
    .optimize()                           -> pre-compilation rewrite
    .save_asset(path)                     -> .aimodel bundle on disk

Everything the converter did or did not like is written up under
"BERT-style encoders" in docs/coreai-ecosystem.md.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import coreai_torch
import torch
from coreai.authoring import AIProgram
from torch.export import Dim, ExportedProgram

from .download import DEFAULT_HF_DIR, model_dir
from .model import NLICrossEncoder

DEFAULT_OUT_DIR = Path("models/decide")
DYNAMIC_ASSET = "NLICrossEncoder.aimodel"
STATIC_ASSET = "NLICrossEncoderStatic.aimodel"
MAX_BATCH = 32
MAX_SEQ = 256
STATIC_BATCHES = (1, 4, 8, 16)
STATIC_LENGTHS = (64, 128)


def static_config_name(n: int, length: int) -> str:
    return f"n{n}_l{length}"


def static_function_name(n: int, length: int) -> str:
    """``set_static_shape_config`` names the emitted function ``<graph>_<config>``."""
    return f"main_{static_config_name(n, length)}"


def build(hf_dir: Path) -> tuple[ExportedProgram, AIProgram]:
    """Export + convert, returning both in-memory programs (the coreai-torch
    debugging tools want the pair)."""
    model = NLICrossEncoder.from_pretrained(hf_dir)
    # The example batch must not be 1: torch.export would specialise a
    # size-1 dim to a constant and then reject the Dim constraint.
    example_ids = torch.full((2, 16), model.cfg.pad_token_id, dtype=torch.int32)
    example_ids[:, 0] = 0
    example_mask = torch.ones(2, 16, dtype=torch.int32)

    batch = Dim("batch", min=1, max=MAX_BATCH)
    seq = Dim("seq", min=1, max=MAX_SEQ)
    exported = torch.export.export(
        model,
        args=(example_ids, example_mask),
        dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
    )
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    program = (
        coreai_torch.TorchConverter()
        .add_exported_program(exported, input_names=["input_ids", "attention_mask"], output_names=["logits"])
        .to_coreai()
    )
    return exported, program


def convert_dynamic(hf_dir: Path, out: Path) -> Path:
    _, program = build(hf_dir)
    program.optimize()
    program.save_asset(out)
    return out


def convert_static(hf_dir: Path, out: Path, batches=STATIC_BATCHES, lengths=STATIC_LENGTHS) -> Path:
    """One static entrypoint per (N, L). ``shapes_config`` maps *new entrypoint
    name -> {input: shape}*; every input mentioned gets a
    ``coreai.enumerated_shapes`` attribute listing all of its specialisations,
    and ``optimize()`` emits one fully-typed function per entry. The dynamic
    ``main`` is dropped from this asset."""
    _, program = build(hf_dir)
    config = {
        static_config_name(n, length): {"input_ids": (n, length), "attention_mask": (n, length)}
        for n in batches
        for length in lengths
    }
    program.set_static_shape_config("main", config)
    program.optimize()
    program.save_asset(out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", type=Path, default=DEFAULT_HF_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--variant", choices=["dynamic", "static", "all"], default="all")
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = model_dir(a.hf_dir)

    if a.variant in ("dynamic", "all"):
        t0 = time.perf_counter()
        p = convert_dynamic(hf_dir, a.out_dir / DYNAMIC_ASSET)
        print(f"wrote {p} ({time.perf_counter() - t0:.0f} s)")
    if a.variant in ("static", "all"):
        t0 = time.perf_counter()
        p = convert_static(hf_dir, a.out_dir / STATIC_ASSET)
        names = [static_function_name(n, length) for n in STATIC_BATCHES for length in STATIC_LENGTHS]
        print(f"wrote {p} ({time.perf_counter() - t0:.0f} s): functions {names}")


if __name__ == "__main__":
    main()
