"""Hand-written snake policies.

``HeuristicPolicy`` is the "naive simulation" from the talk: it is used both to
generate imitation-learning data for the transformer and as the opponent when
playing the converted model.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Protocol

from .game import Direction, Point, SnakeGame


class Policy(Protocol):
    def choose(self, game: SnakeGame, snake_id: int) -> Direction: ...


class HeuristicPolicy:
    """Greedy food-seeking policy with a flood-fill safety check.

    For each non-fatal move, score = free reachable area (via BFS, capped) minus
    a small penalty proportional to the Manhattan distance to the food. With
    ``epsilon`` > 0 a random safe move is taken occasionally so the training
    data covers more states.
    """

    def __init__(self, epsilon: float = 0.0, seed: int | None = None) -> None:
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    def choose(self, game: SnakeGame, snake_id: int) -> Direction:
        snake = game.snakes[snake_id]
        safe = [d for d in Direction if game.is_safe(snake_id, d)]
        if not safe:
            return snake.direction  # doomed either way

        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return self.rng.choice(safe)

        occupied = game.occupied_cells()
        fx, fy = game.food
        best, best_score = safe[0], float("-inf")
        for d in safe:
            nx, ny = game.next_head(snake_id, d)
            area = _flood_fill_area(game, (nx, ny), occupied, cap=len(snake) * 2 + 8)
            dist = abs(fx - nx) + abs(fy - ny)
            score = area - 0.5 * dist
            # Deterministic tie-break: prefer continuing straight.
            if d == snake.direction:
                score += 0.01
            if score > best_score:
                best, best_score = d, score
        return best


def _flood_fill_area(game: SnakeGame, start: Point, occupied: set[Point], cap: int) -> int:
    """Number of free cells reachable from ``start`` (up to ``cap``)."""
    if not game.in_bounds(start) or start in occupied:
        return 0
    seen = {start}
    q: deque[Point] = deque([start])
    while q and len(seen) < cap:
        x, y = q.popleft()
        for d in Direction:
            dx, dy = d.delta
            p = (x + dx, y + dy)
            if game.in_bounds(p) and p not in occupied and p not in seen:
                seen.add(p)
                q.append(p)
    return len(seen)
