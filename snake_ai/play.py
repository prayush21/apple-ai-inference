"""Play snake with the Core AI model driving snake 0 against the heuristic.

The two ``ModelPlayer`` classes are the Python counterparts of the Swift
``ModelPlayer`` in the talk:

* ``StatelessModelPlayer`` accumulates the full game history and feeds it to
  ``SnakeTransformer.aimodel`` every step (quadratic cost in game length).
* ``StatefulModelPlayer`` keeps ``keyCache`` / ``valueCache`` NDArrays and
  passes them as *states* to ``SnakeTransformerStateful.aimodel``, sending only
  the newest board features each step.

Per-inference latency is recorded so the difference is visible without
Instruments.

    python -m snake_ai.play --player stateful --render
    python -m snake_ai.play --player all --games 5       # latency comparison
    python -m snake_ai.play --player all --games 5 --json docs/bench/python.json

``--json`` writes a baseline record (load + per-move inference stats per
player, plus host info) so the numbers survive the terminal and can be diffed
against the Swift ``ModelPlayer`` on ``CoreAI.framework`` later. The app
writes the same schema from its "Save bench" button.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from coreai.runtime import AIModel, InferenceFunction, NDArray

from .features import FEATURE_DIM, extract_features
from .game import Direction, SnakeGame
from .policy import HeuristicPolicy


def predicted_direction(logits: np.ndarray, game: SnakeGame, snake_id: int, *, safe_only: bool) -> Direction:
    """Argmax over the 4 action logits. With ``safe_only`` the highest-scoring
    non-fatal move is taken instead (a tiny bit of app-side guard-railing)."""
    order = np.argsort(-logits)
    if safe_only:
        for a in order:
            if game.is_safe(snake_id, Direction(int(a))):
                return Direction(int(a))
    return Direction(int(order[0]))


class StatelessModelPlayer:
    def __init__(self, function: InferenceFunction, *, safe_only: bool = True) -> None:
        self.function = function
        self.safe_only = safe_only
        self.history: list[list[float]] = []
        self.latencies_ms: list[float] = []
        self.load_ms = 0.0  # set by load_player

    async def choose(self, game: SnakeGame, snake_id: int) -> Direction:
        self.history.append(extract_features(game, snake_id))
        features = np.asarray(self.history, dtype=np.float32)[np.newaxis]  # [1, T, 16]

        t0 = time.perf_counter()
        outputs = await self.function({"features": NDArray(data=features)})
        self.latencies_ms.append((time.perf_counter() - t0) * 1e3)

        logits = outputs["logits"].numpy()[0, -1]
        return predicted_direction(logits, game, snake_id, safe_only=self.safe_only)


class StatefulModelPlayer:
    """Works for both the dynamic (``main``) and static-shape (``main_decode``)
    stateful assets: the call shape is always [1, 1, 16] / [1, 1]."""

    def __init__(self, function: InferenceFunction, *, safe_only: bool = True) -> None:
        self.function = function
        self.safe_only = safe_only
        desc = function.desc
        # Allocate the states with the exact shape/dtype the function declares.
        self.key_cache = NDArray(data=np.zeros(desc.state_descriptor("keyCache").shape, np.float32))
        self.value_cache = NDArray(data=np.zeros(desc.state_descriptor("valueCache").shape, np.float32))
        self.max_context = desc.state_descriptor("keyCache").shape[2]
        self.position = 0
        self.latencies_ms: list[float] = []
        self.load_ms = 0.0  # set by load_player

    async def choose(self, game: SnakeGame, snake_id: int) -> Direction:
        if self.position >= self.max_context:
            raise RuntimeError("KV cache is full; game exceeded max context length")
        features = np.asarray([extract_features(game, snake_id)], dtype=np.float32)[np.newaxis]  # [1, 1, 16]
        position_ids = np.array([[self.position]], dtype=np.int32)

        t0 = time.perf_counter()
        outputs = await self.function(
            {"features": NDArray(data=features), "position_ids": NDArray(data=position_ids)},
            state={"keyCache": self.key_cache, "valueCache": self.value_cache},
        )
        self.latencies_ms.append((time.perf_counter() - t0) * 1e3)
        self.position += 1

        logits = outputs["logits"].numpy()[0, -1]
        return predicted_direction(logits, game, snake_id, safe_only=self.safe_only)


ASSETS = {
    "stateless": ("SnakeTransformer.aimodel", "main"),
    "stateful": ("SnakeTransformerStateful.aimodel", "main"),
    "decode": ("SnakeTransformerDecode.aimodel", "main_decode"),
}


async def load_player(kind: str, models_dir: Path, *, safe_only: bool):
    """Load the asset + function for ``kind``; ``player.load_ms`` records how
    long that took (first load in a process specializes and hits the on-disk
    cache in ``~/Library/Caches/coreai-cache``, later ones are near-free)."""
    asset, function_name = ASSETS[kind]
    t0 = time.perf_counter()
    model = await AIModel.load(models_dir / asset)
    function = model.load_function(function_name)
    load_ms = (time.perf_counter() - t0) * 1e3
    cls = StatelessModelPlayer if kind == "stateless" else StatefulModelPlayer
    player = cls(function, safe_only=safe_only)
    player.load_ms = load_ms
    return player


async def play_game(player, *, seed: int, render: bool, delay: float, max_steps: int) -> SnakeGame:
    game = SnakeGame(seed=seed, max_steps=max_steps)
    opponent = HeuristicPolicy(epsilon=0.05, seed=seed)
    while not game.is_over:
        actions = {}
        if game.snakes[0].alive:
            actions[0] = await player.choose(game, 0)
        if game.snakes[1].alive:
            actions[1] = opponent.choose(game, 1)
        game.step(actions)
        if render:
            sys.stdout.write("\x1b[2J\x1b[H")  # clear screen
            print(f"step {game.step_count}  A(model)={len(game.snakes[0])}  B(heuristic)={len(game.snakes[1])}")
            print(game.render())
            if player.latencies_ms:
                print(f"last inference: {player.latencies_ms[-1]:.2f} ms")
            await asyncio.sleep(delay)
    return game


def summarize(kind: str, games: list[SnakeGame], latencies: list[list[float]], loads_ms: list[float]) -> dict:
    """Print one line per player and return the same numbers as a record for
    ``--json``. Latency at the start of a game vs. late in the game shows the
    stateless scaling; first load vs. later loads shows the model cache."""
    wins = sum(g.winner == 0 for g in games)
    draws = sum(g.winner is None for g in games)
    steps = [g.step_count for g in games]
    lat = np.concatenate([np.asarray(l) for l in latencies])
    early = np.concatenate([np.asarray(l[:5]) for l in latencies])
    late = np.concatenate([np.asarray(l[-5:]) for l in latencies if len(l) >= 20])
    late_mean = float(late.mean()) if len(late) else float("nan")
    print(
        f"{kind:9s} games={len(games)} wins={wins} draws={draws} "
        f"avg steps={np.mean(steps):.1f} | load ms: first {loads_ms[0]:.1f}, "
        f"rest {np.mean(loads_ms[1:]) if len(loads_ms) > 1 else float('nan'):.1f} "
        f"| inference ms: mean {lat.mean():.2f}, p50 {np.median(lat):.2f}, "
        f"p95 {np.percentile(lat, 95):.2f}, first-5 {early.mean():.2f}, last-5 {late_mean:.2f}"
    )
    asset, function_name = ASSETS[kind]
    return {
        "player": kind,
        "asset": asset,
        "function": function_name,
        "games": len(games),
        "wins": wins,
        "draws": draws,
        "avg_steps": float(np.mean(steps)),
        "load_ms": {
            "first": loads_ms[0],
            "rest_mean": float(np.mean(loads_ms[1:])) if len(loads_ms) > 1 else None,
            "all": loads_ms,
        },
        "inference_ms": {
            "count": int(lat.size),
            "mean": float(lat.mean()),
            "p50": float(np.median(lat)),
            "p95": float(np.percentile(lat, 95)),
            "first_5": float(early.mean()),
            "last_5": late_mean if late_mean == late_mean else None,
        },
    }


def host_info() -> dict:
    """Enough about the machine to tell two baselines apart."""
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        chip = None
    try:
        from importlib.metadata import version
        coreai_version = version("coreai-core")
    except Exception:  # noqa: BLE001 - version is informational only
        coreai_version = None
    return {
        "machine": platform.machine(),
        "chip": chip,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "coreai_core": coreai_version,
    }


CACHE_DIR = Path.home() / "Library/Caches/coreai-cache"


def write_baseline(path: Path, a: argparse.Namespace, players: list[dict], *, cache_was_warm: bool) -> None:
    record = {
        "schema": "snake-bench/1",
        "runtime": "coreai.runtime (Python, in-process, CPU)",
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": host_info(),
        # First load in a process still reads the on-disk specialization cache;
        # delete CACHE_DIR before running to measure a truly cold load.
        "model_cache_warm": cache_was_warm,
        "config": {"games": a.games, "seed": a.seed, "max_steps": a.max_steps, "safe_only": not a.raw,
                   "models_dir": str(a.models_dir)},
        "players": players,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {path}")


async def run(a: argparse.Namespace) -> None:
    kinds = ["stateless", "stateful", "decode"] if a.player == "all" else [a.player]
    cache_was_warm = CACHE_DIR.exists()
    records = []
    for kind in kinds:
        games, lats, loads = [], [], []
        for i in range(a.games):
            player = await load_player(kind, a.models_dir, safe_only=not a.raw)
            g = await play_game(player, seed=a.seed + i, render=a.render, delay=a.delay, max_steps=a.max_steps)
            games.append(g)
            lats.append(player.latencies_ms)
            loads.append(player.load_ms)
            if a.render:
                print(f"game over: winner={'model' if g.winner == 0 else 'heuristic' if g.winner == 1 else 'draw'}")
        records.append(summarize(kind, games, lats, loads))
    if a.json:
        write_baseline(a.json, a, records, cache_was_warm=cache_was_warm)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player", choices=["stateless", "stateful", "decode", "all"], default="decode")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--render", action="store_true", help="animate the board in the terminal")
    ap.add_argument("--delay", type=float, default=0.08, help="seconds between rendered frames")
    ap.add_argument("--raw", action="store_true", help="take the raw argmax even if it is fatal")
    ap.add_argument("--json", type=Path, help="write the load/inference baseline record to this file")
    asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
