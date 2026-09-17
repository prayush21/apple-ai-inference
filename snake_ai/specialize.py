"""Model specialization and caching, from the Python side.

The talk's "Specialization" section: a shipped ``.aimodel`` is a *source*
representation; before it can run it is specialized (compiled + executable
artifacts generated) for the device/OS, and the result is cached. The Swift
API exposes this as ``AIModelCache`` / ``AIModel.specialize(contentsOf:)`` /
``SpecializationOptions``. The Python runtime has the same moving parts:

* ``AIModel.load(path, specialization_options)`` specializes on first load and
  reuses the cache afterwards (``~/Library/Caches/coreai-cache/<os build>/``).
* ``SpecializationOptions`` restricts/prefers compute units and toggles debug
  builds. On the in-package runtime (macOS < 27) ``is_supported()`` is False:
  options are accepted but GPU/ANE delegation is an OS-runtime feature.

    python -m snake_ai.specialize            # time cold vs warm loads
    python -m snake_ai.specialize --warm-up  # "prepare AI features" ahead of time
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import time
from pathlib import Path

from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions

CACHE_DIR = Path.home() / "Library/Caches/coreai-cache"


def cache_summary() -> str:
    if not CACHE_DIR.exists():
        return f"{CACHE_DIR}: (absent)"
    n = sum(1 for p in CACHE_DIR.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in CACHE_DIR.rglob("*") if p.is_file())
    return f"{CACHE_DIR}: {n} files, {size / 1e6:.1f} MB"


async def timed_load(path: Path, options: SpecializationOptions | None = None) -> tuple[float, float]:
    t0 = time.perf_counter()
    model = await AIModel.load(path, options)
    t1 = time.perf_counter()
    model.load_function("main")
    t2 = time.perf_counter()
    return (t1 - t0) * 1e3, (t2 - t1) * 1e3


async def demo(models_dir: Path, clear_cache: bool) -> None:
    print(f"SpecializationOptions.is_supported() = {SpecializationOptions.is_supported()}  "
          f"(compute units on this machine: {[str(k) for k in ComputeUnitKind.available_kinds()]})")
    if clear_cache and CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print("cleared specialization cache")
    print("cache before:", cache_summary())

    for name in ("SnakeTransformer.aimodel", "SnakeTransformerStateful.aimodel"):
        path = models_dir / name
        cold = await timed_load(path)
        warm = await timed_load(path)
        print(f"{name:34s} cold load {cold[0]:6.1f} ms + function {cold[1]:6.1f} ms  |  "
              f"warm load {warm[0]:5.1f} ms + function {warm[1]:5.1f} ms")

    print("cache after: ", cache_summary())

    print("\nspecialization options (accepted by the local runtime, delegation needs the OS runtime):")
    for label, opts in [
        ("cpu_only", SpecializationOptions.cpu_only()),
        ("default", SpecializationOptions.default()),
        ("prefer GPU", SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())),
        ("default+debug", SpecializationOptions.default().with_debug(enabled=True)),
    ]:
        ms = await timed_load(models_dir / "SnakeTransformer.aimodel", opts)
        print(f"  {label:14s} allowed={[str(k) for k in opts.allowed_compute_unit_kinds]} "
              f"preferred={opts.preferred_compute_unit_kind}  -> load {ms[0] + ms[1]:.1f} ms")


async def warm_up(models_dir: Path) -> None:
    """What an app would do after download / opt-in: specialize now so the
    first interactive use is a cache hit."""
    for path in sorted(models_dir.glob("*.aimodel")):
        ms = await timed_load(path)
        print(f"prepared {path.name} in {ms[0] + ms[1]:.1f} ms")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--warm-up", action="store_true")
    ap.add_argument("--clear-cache", action="store_true", help="delete the local specialization cache first")
    a = ap.parse_args(argv)
    asyncio.run(warm_up(a.models_dir) if a.warm_up else demo(a.models_dir, a.clear_cache))


if __name__ == "__main__":
    main()
