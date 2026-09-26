"""Convert SmolLM2 to Core AI ``.aimodel`` assets.

    python -m llm_ai.convert                          # stateful fp16 -> models/llm/SmolLM2Stateful.aimodel
    python -m llm_ai.convert --variant stateless      # comparison row only; delete after its numbers
    python -m llm_ai.convert --precision fp32

Stateful asset: one ``AIProgram`` exported with a dynamic T, then pinned by a
single ``set_static_shape_config`` call to two static entrypoints that share
the ``keyCache`` / ``valueCache`` states (gotchas 7, 8, 12)::

    main_prefill_t64   input_ids / position_ids [1, 64]
    main_decode                                 [1, 1]

A prompt is split into 64-token chunks and the last one is left-padded
(``model.left_pad``, ``runtime.StatefulLM.prefill_ids``). Every function
returns ``logits [1, 1, vocab]`` for the last token only.

Why only one prefill size: each static function gets its own specialized
copy of the weights at first load. With t16/t128/t512 + decode, loading the
fp16 asset on this 8 GB M2 peaked at 4.7 GB RSS and drained > 4 GB of disk
(gotcha 18). 64 covers a one-turn chat prompt with the default system
message (~40 tokens) in a single call.

Stateless asset (``SmolLM2.aimodel``): ``input_ids [1, T]`` dynamic up to
``max_seq_len``, ``logits [1, T, vocab]``.

Pipeline: ``torch.export`` (example T = 16, never 1: gotcha 11) ->
``run_decompositions(get_decomp_table())`` -> ``TorchConverter`` ->
``set_static_shape_config`` -> ``optimize()`` (states only exist after it,
gotcha 1) -> ``save_asset``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import coreai_torch
import torch
from coreai.authoring import AIProgram

from .model import SmolLM, SmolLMStateful
from .tokenizer import DEFAULT_MODEL_DIR

DEFAULT_OUT_DIR = Path("models/llm")
PREFILL_LENGTHS = (64,)
STATE_NAMES = ["keyCache", "valueCache"]
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def static_configs(prefill_lengths=PREFILL_LENGTHS) -> dict[str, dict[str, tuple[int, ...]]]:
    cfgs = {f"prefill_t{t}": {"input_ids": (1, t), "position_ids": (1, t)} for t in prefill_lengths}
    cfgs["decode"] = {"input_ids": (1, 1), "position_ids": (1, 1)}
    return cfgs


def build_stateful(model: SmolLMStateful, prefill_lengths=PREFILL_LENGTHS) -> AIProgram:
    S = model.cfg.max_seq_len
    example_ids = torch.zeros(1, 16, dtype=torch.long)
    example_pos = torch.arange(16)[None]
    # max is at least 16 so the size-16 example (never 1: gotcha 11) fits.
    seq = torch.export.Dim("seq", min=1, max=max(16, *prefill_lengths))
    assert max(prefill_lengths) < S
    exported = torch.export.export(
        model,
        args=(example_ids, example_pos),
        dynamic_shapes={"input_ids": {1: seq}, "position_ids": {1: seq}},
    )
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    program = (
        coreai_torch.TorchConverter()
        .add_exported_program(
            exported,
            input_names=["input_ids", "position_ids"],
            # One name per mutated buffer, in registration order (k_cache, v_cache).
            state_names=STATE_NAMES,
            output_names=["logits"],
        )
        .to_coreai()
    )
    # All configs in one call: a second call would drop the functions of the first (gotcha 12).
    program.set_static_shape_config("main", static_configs(prefill_lengths))
    program.optimize()
    return program


def build_stateless(model: SmolLM) -> AIProgram:
    seq = torch.export.Dim("seq", min=1, max=model.cfg.max_seq_len)
    exported = torch.export.export(
        model, args=(torch.zeros(1, 16, dtype=torch.long),), dynamic_shapes={"input_ids": {1: seq}}
    )
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    program = (
        coreai_torch.TorchConverter()
        .add_exported_program(exported, input_names=["input_ids"], output_names=["logits"])
        .to_coreai()
    )
    program.optimize()
    return program


def save(program: AIProgram, out: Path) -> float:
    """Replace ``out`` with the asset; return its size in MB."""
    if out.exists():
        shutil.rmtree(out)
    program.save_asset(out)
    return sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--variant", choices=["stateful", "stateless"], default="stateful")
    ap.add_argument("--precision", choices=list(DTYPES), default="fp16")
    a = ap.parse_args(argv)

    dtype = DTYPES[a.precision]
    t0 = time.perf_counter()
    if a.variant == "stateful":
        model = SmolLMStateful.from_hf(a.model_dir, dtype=dtype)
        program = build_stateful(model)
        out = a.out_dir / "SmolLM2Stateful.aimodel"
    else:
        model = SmolLM.from_hf(a.model_dir, dtype=dtype)
        program = build_stateless(model)
        out = a.out_dir / "SmolLM2.aimodel"
    del model
    size_mb = save(program, out)
    secs = time.perf_counter() - t0
    info = {"asset": str(out), "variant": a.variant, "precision": a.precision, "size_mb": round(size_mb, 1),
            "convert_s": round(secs, 1)}
    (out / "llm_convert.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
