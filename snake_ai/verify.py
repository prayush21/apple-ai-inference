"""Verify the converted ``.aimodel`` matches the PyTorch model numerically.

Uses the Core AI framework Python bindings (``coreai.runtime``) to load and
run the asset, exactly as the talk does before leaving the Python environment.

    python -m snake_ai.verify
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModel, NDArray

from .features import extract_features
from .game import SnakeGame
from .model import SnakeTransformer, SnakeTransformerStateful
from .policy import HeuristicPolicy


def sample_game_features(n_frames: int, seed: int = 0) -> np.ndarray:
    """Play ``n_frames`` of a real game so the inputs are realistic. -> [1, n, 16]"""
    game = SnakeGame(seed=seed)
    p = HeuristicPolicy()
    frames = []
    for _ in range(n_frames):
        frames.append(extract_features(game, 0))
        game.step({0: p.choose(game, 0), 1: p.choose(game, 1)})
    return np.asarray(frames, dtype=np.float32)[np.newaxis]


async def verify_stateless(ckpt: Path, asset: Path, n_frames: int, tol: float) -> float:
    pt_model = SnakeTransformer.load_checkpoint(ckpt)
    ai_model = await AIModel.load(asset)
    print(f"loaded {asset.name}: functions={ai_model.function_names}")
    function = ai_model.load_function("main")
    print(f"  signature: {function.desc}")

    features = sample_game_features(n_frames)

    with torch.no_grad():
        pytorch_logits = pt_model(torch.from_numpy(features)).numpy()[0, -1]

    result = await function({"features": NDArray(data=features)})
    coreai_logits = result["logits"].numpy()[0, -1]

    max_diff = float(np.max(np.abs(pytorch_logits - coreai_logits)))
    print(f"  pytorch {np.round(pytorch_logits, 4)}\n  coreai  {np.round(coreai_logits, 4)}")
    print(f"  max |diff| = {max_diff:.6f}  ({'OK' if max_diff < tol else 'FAIL'} tol {tol})")
    assert max_diff < tol
    return max_diff


async def verify_stateful(ckpt: Path, asset: Path, n_frames: int, tol: float) -> float:
    """Decode ``n_frames`` one step at a time through the Core AI stateful model,
    threading the KV caches, and compare the final logits with the stateless
    PyTorch model run on the full history."""
    pt_model = SnakeTransformer.load_checkpoint(ckpt)
    cfg = pt_model.cfg
    ai_model = await AIModel.load(asset)
    print(f"loaded {asset.name}: functions={ai_model.function_names}")
    function = ai_model.load_function("main")
    print(f"  signature: {function.desc}")
    desc = function.desc
    for n in desc.input_names:
        print(f"    input  {n:13s} {desc.input_descriptor(n)}")
    for n in desc.state_names:
        print(f"    state  {n:13s} {desc.state_descriptor(n)}")
    for n in desc.output_names:
        print(f"    output {n:13s} {desc.output_descriptor(n)}")

    features = sample_game_features(n_frames)
    with torch.no_grad():
        pytorch_logits = pt_model(torch.from_numpy(features)).numpy()[0, -1]

    # States are NDArrays that the runtime reads *and writes in place*.
    cache_shape = (cfg.n_layers, 1, cfg.max_seq_len, cfg.d_model)
    key_cache = NDArray(data=np.zeros(cache_shape, dtype=np.float32))
    value_cache = NDArray(data=np.zeros(cache_shape, dtype=np.float32))

    coreai_logits = None
    for t in range(n_frames):
        result = await function(
            {
                "features": NDArray(data=features[:, t : t + 1]),
                # The runtime does not coerce dtypes: coreai-torch maps int64
                # indices to int32, so the input must be int32 to match.
                "position_ids": NDArray(data=np.array([[t]], dtype=np.int32)),
            },
            state={"keyCache": key_cache, "valueCache": value_cache},
        )
        coreai_logits = result["logits"].numpy()[0, -1]

    # Sanity check: the cache must have been mutated by the runtime.
    assert np.abs(key_cache.numpy()).sum() > 0, "keyCache was not updated in place"

    max_diff = float(np.max(np.abs(pytorch_logits - coreai_logits)))
    print(f"  pytorch {np.round(pytorch_logits, 4)}\n  coreai  {np.round(coreai_logits, 4)}")
    print(f"  max |diff| = {max_diff:.6f}  ({'OK' if max_diff < tol else 'FAIL'} tol {tol})")
    assert max_diff < tol

    # Also confirm the PyTorch stateful module agrees (guards the authoring code).
    st = SnakeTransformerStateful.load_checkpoint(ckpt)
    with torch.no_grad():
        for t in range(n_frames):
            out = st(torch.from_numpy(features[:, t : t + 1]), torch.tensor([[t]]))
    assert np.max(np.abs(out.numpy()[0, -1] - pytorch_logits)) < 1e-4
    return max_diff


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/snake.pt"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--variant", choices=["stateless", "stateful", "both"], default="both")
    a = ap.parse_args(argv)

    async def run():
        if a.variant in ("stateless", "both"):
            await verify_stateless(a.checkpoint, a.models_dir / "SnakeTransformer.aimodel", a.frames, a.tol)
        if a.variant in ("stateful", "both"):
            await verify_stateful(a.checkpoint, a.models_dir / "SnakeTransformerStateful.aimodel", a.frames, a.tol)

    asyncio.run(run())


if __name__ == "__main__":
    main()
