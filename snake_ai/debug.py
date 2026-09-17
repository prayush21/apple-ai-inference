"""Python-side equivalents of the Core AI Instrument and Core AI Debugger.

Xcode 27 ships a Core AI Instrument (per-inference / per-op timing) and a
Core AI Debugger (inspect intermediate tensors, trace ops back to Python
source). ``coreai_torch.debugging`` exposes the same machinery from Python:

* ``benchmark_coreai_program`` runs the program under the runtime ``Profiler``
  and aggregates per-operation timings, grouped by the Python source line
  that produced each op (via the debug info recorded at conversion).
* ``create_comparator_for_programs`` runs the PyTorch ``ExportedProgram`` and
  the ``AIProgram`` op-by-op (with ``IntermediateLogger`` on the Core AI side)
  and bisects to the first op whose output diverges.
* ``Profiler`` directly, for raw begin/end events per inference.

    python -m snake_ai.debug benchmark   [--frames 40] [--runs 20] [--top 15]
    python -m snake_ai.debug compare     [--frames 10]
    python -m snake_ai.debug events      [--frames 10]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModel, NDArray, Profiler
from coreai_torch.debugging.benchmarker import benchmark_coreai_program
from coreai_torch.debugging.comparator import create_comparator_for_programs

from .convert import build_stateless
from .verify import sample_game_features


async def benchmark(ckpt: Path, frames: int, runs: int, top: int, annotate_out: Path) -> None:
    """Per-op timing table for one stateless inference over ``frames`` steps."""
    _, program = build_stateless(ckpt)
    features = sample_game_features(frames)
    print(f"benchmarking stateless model, features {features.shape}, {runs} runs")
    # Ops the runtime fuses away have no source mapping; the benchmarker warns
    # once per sample, which is just noise here.
    logging.getLogger("coreai_torch.debugging.benchmarker").setLevel(logging.ERROR)
    result = await benchmark_coreai_program(
        program, inputs={"features": torch.from_numpy(features)}, num_runs=runs
    )
    result.write_summary(sys.stdout, top_n=top)

    # Roll the per-op numbers up by op kind: which *kinds* of ops dominate?
    by_kind: Counter[str] = Counter()
    for op, measurement in result.get_operation_summary():
        if measurement.statistics is not None:
            by_kind[op.name] += measurement.statistics.average
    total = sum(by_kind.values()) or 1.0
    print("\ntime by op kind (sum of per-op averages):")
    for kind, ms in by_kind.most_common(8):
        print(f"  {kind:45s} {ms:8.3f} ms  {100 * ms / total:5.1f}%")

    # Group by the Python module / call stack that produced each op, then
    # print the dominant source file annotated line-by-line with timings —
    # the "trace back to your original Python source" view of the talk.
    modules = result.get_module_timings()
    print("\ntime by authoring module:")
    for name, mt in sorted(modules.items(), key=lambda kv: -(kv[1].total_time.average if kv[1].total_time else 0))[:8]:
        tot = mt.total_time.average if mt.total_time else 0.0
        print(f"  {name:45s} {tot:8.3f} ms  ({len(mt.get_all_operations())} ops)")
    if modules:
        top_module = max(modules.values(), key=lambda m: m.total_time.average if m.total_time else 0)
        annotate_out.parent.mkdir(parents=True, exist_ok=True)
        with annotate_out.open("w") as f:
            top_module.annotate_dominant_source(f)
        print(f"\nwrote source annotated with per-line op timings to {annotate_out}")


async def compare(ckpt: Path, frames: int, rtol: float, atol: float) -> None:
    """Op-by-op numerics: PyTorch ExportedProgram vs converted AIProgram."""
    exported, program = build_stateless(ckpt)
    features = sample_game_features(frames)
    comparator = await create_comparator_for_programs(exported, program, target_entry_point="main")
    result = await comparator.compare_with_tolerance(
        inputs={"features": torch.from_numpy(features)}, rtol=rtol, atol=atol
    )
    n = len(result.op_statuses)
    print(f"\ncompared {n} op pairs: {n - len(result.failed_nodes) - len(result.unknown_nodes)} pass, "
          f"{len(result.failed_nodes)} fail, {len(result.unknown_nodes)} unknown")
    for src, tgt in result.failed_nodes[:10]:
        print(f"  FAIL torch {getattr(src, 'target', src)}  ->  coreai {getattr(tgt, 'name', tgt)}")


async def events(models_dir: Path, frames: int) -> None:
    """Raw runtime ``Profiler`` events for a single stateless inference,
    aggregated by event kind. The same events feed the Core AI Instrument."""
    durations: dict[tuple[str, str], list[float]] = {}
    intervals: dict[int, tuple[str, str, int]] = {}
    next_id = [0]

    def on_begin(ev):
        next_id[0] += 1
        intervals[next_id[0]] = (ev.event_id, ev.phase, ev.timestamp)
        return next_id[0]

    def on_end(ev, iid):
        name, phase, t0 = intervals.pop(iid)
        durations.setdefault((phase, name), []).append((ev.timestamp - t0) / 1e6)

    def on_event(ev):
        durations.setdefault((ev.phase, ev.event_id), []).append(0.0)

    profiler = Profiler(on_log_event=on_event, on_log_event_begin=on_begin, on_log_event_end=on_end)
    model = await AIModel.load(models_dir / "SnakeTransformer.aimodel")
    fn = model.load_function("main", profiler=profiler)
    await fn({"features": NDArray(data=sample_game_features(frames))})

    total = sum(sum(v) for v in durations.values())
    print(f"profiler events for one inference over {frames} frames "
          f"({sum(len(v) for v in durations.values())} events, {total:.3f} ms inside intervals):")
    print(f"  {'phase':10s} {'event':40s} {'count':>6s} {'total ms':>9s} {'mean ms':>8s}")
    for (phase, name), v in sorted(durations.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {phase:10s} {name:40s} {len(v):6d} {sum(v):9.3f} {np.mean(v):8.4f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["benchmark", "compare", "events"])
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/snake.pt"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--annotate-out", type=Path, default=Path("models/SnakeTransformer.annotated.py.txt"))
    a = ap.parse_args(argv)

    if a.command == "benchmark":
        asyncio.run(benchmark(a.checkpoint, a.frames, a.runs, a.top, a.annotate_out))
    elif a.command == "compare":
        asyncio.run(compare(a.checkpoint, a.frames, a.rtol, a.atol))
    else:
        asyncio.run(events(a.models_dir, a.frames))


if __name__ == "__main__":
    main()
