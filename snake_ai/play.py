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
    python -m snake_ai.play --player both --games 5      # latency comparison
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
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

    async def choose(self, game: SnakeGame, snake_id: int) -> Direction:
        self.history.append(extract_features(game, snake_id))
        features = np.asarray(self.history, dtype=np.float32)[np.newaxis]  # [1, T, 16]

        t0 = time.perf_counter()
        outputs = await self.function({"features": NDArray(data=features)})
        self.latencies_ms.append((time.perf_counter() - t0) * 1e3)

        logits = outputs["logits"].numpy()[0, -1]
        return predicted_direction(logits, game, snake_id, safe_only=self.safe_only)


class StatefulModelPlayer:
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


async def load_player(kind: str, models_dir: Path, *, safe_only: bool):
    if kind == "stateless":
        model = await AIModel.load(models_dir / "SnakeTransformer.aimodel")
        return StatelessModelPlayer(model.load_function("main"), safe_only=safe_only)
    model = await AIModel.load(models_dir / "SnakeTransformerStateful.aimodel")
    return StatefulModelPlayer(model.load_function("main"), safe_only=safe_only)


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


def summarize(label: str, games: list[SnakeGame], latencies: list[list[float]]) -> None:
    wins = sum(g.winner == 0 for g in games)
    draws = sum(g.winner is None for g in games)
    steps = [g.step_count for g in games]
    lat = np.concatenate([np.asarray(l) for l in latencies])
    # Latency at the start of a game vs. late in the game shows the scaling.
    early = np.concatenate([np.asarray(l[:5]) for l in latencies])
    late = np.concatenate([np.asarray(l[-5:]) for l in latencies if len(l) >= 20])
    print(
        f"{label:9s} games={len(games)} wins={wins} draws={draws} "
        f"avg steps={np.mean(steps):.1f} | inference ms: mean {lat.mean():.2f}, "
        f"first-5 {early.mean():.2f}, last-5 {late.mean() if len(late) else float('nan'):.2f}"
    )


async def run(a: argparse.Namespace) -> None:
    kinds = ["stateless", "stateful"] if a.player == "both" else [a.player]
    for kind in kinds:
        games, lats = [], []
        for i in range(a.games):
            player = await load_player(kind, a.models_dir, safe_only=not a.raw)
            g = await play_game(player, seed=a.seed + i, render=a.render, delay=a.delay, max_steps=a.max_steps)
            games.append(g)
            lats.append(player.latencies_ms)
            if a.render:
                print(f"game over: winner={'model' if g.winner == 0 else 'heuristic' if g.winner == 1 else 'draw'}")
        summarize(kind, games, lats)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player", choices=["stateless", "stateful", "both"], default="stateful")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--render", action="store_true", help="animate the board in the terminal")
    ap.add_argument("--delay", type=float, default=0.08, help="seconds between rendered frames")
    ap.add_argument("--raw", action="store_true", help="take the raw argmax even if it is fatal")
    asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
