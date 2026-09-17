"""Play the PyTorch model against the heuristic and report the win rate.

Validation accuracy measures imitation of the teacher; win rate measures
whether the snake actually survives. Both matter, and they disagree often
enough that training reports both.

    python -m snake_ai.evaluate --games 50
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .features import extract_features
from .game import Direction, SnakeGame
from .model import SnakeTransformer
from .policy import HeuristicPolicy


@dataclass
class EvalResult:
    games: int
    wins: int
    losses: int
    draws: int
    mean_steps: float
    mean_length: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.games

    def __str__(self) -> str:
        return (
            f"win rate {self.win_rate:.0%} ({self.wins}W/{self.losses}L/{self.draws}D over {self.games}), "
            f"avg steps {self.mean_steps:.0f}, avg model length {self.mean_length:.1f}"
        )


@torch.no_grad()
def choose_with_model(model: SnakeTransformer, history: list[list[float]], game: SnakeGame, snake_id: int, safe_only: bool) -> Direction:
    x = torch.tensor(history, dtype=torch.float32)[None]
    logits = model(x)[0, -1].numpy()
    for a in np.argsort(-logits):
        d = Direction(int(a))
        if not safe_only or game.is_safe(snake_id, d):
            return d
    return Direction(int(np.argmax(logits)))


def evaluate(model: SnakeTransformer, *, games: int, seed: int = 1000, max_steps: int = 250, safe_only: bool = True, opponent_epsilon: float = 0.05) -> EvalResult:
    model.eval()
    wins = losses = draws = 0
    steps, lengths = [], []
    for g in range(games):
        game = SnakeGame(seed=seed + g, max_steps=max_steps)
        opp = HeuristicPolicy(epsilon=opponent_epsilon, seed=seed + g)
        history: list[list[float]] = []
        while not game.is_over:
            actions = {}
            if game.snakes[0].alive:
                history.append(extract_features(game, 0))
                actions[0] = choose_with_model(model, history[-model.cfg.max_seq_len :], game, 0, safe_only)
            if game.snakes[1].alive:
                actions[1] = opp.choose(game, 1)
            game.step(actions)
        if game.winner == 0:
            wins += 1
        elif game.winner == 1:
            losses += 1
        else:
            draws += 1
        steps.append(game.step_count)
        lengths.append(len(game.snakes[0]))
    return EvalResult(games, wins, losses, draws, float(np.mean(steps)), float(np.mean(lengths)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/snake.pt"))
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--raw", action="store_true", help="take the raw argmax even if fatal")
    a = ap.parse_args(argv)
    model = SnakeTransformer.load_checkpoint(a.checkpoint)
    print(evaluate(model, games=a.games, safe_only=not a.raw))


if __name__ == "__main__":
    main()
