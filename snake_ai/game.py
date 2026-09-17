"""Two-player snake game engine.

Traditional snake rules: snakes grow by eating food and die when they hit a
wall, themselves, or the other snake. The last snake standing wins.

This module is pure Python (no torch) so it can be shared by the data
simulator, the training loop, the numerics verification script, and the
terminal player. The Swift package under ``SnakeCoreAI/Sources/SnakeEngine``
mirrors these rules 1:1 so features computed on either side are identical.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum


class Direction(IntEnum):
    """Absolute movement direction. The integer value is the action index the
    model predicts and the index used for one-hot encoding in features."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @property
    def delta(self) -> tuple[int, int]:
        return _DELTAS[self]

    @property
    def opposite(self) -> "Direction":
        return _OPPOSITES[self]

    @property
    def one_hot(self) -> list[float]:
        v = [0.0, 0.0, 0.0, 0.0]
        v[int(self)] = 1.0
        return v


# (dx, dy) with y growing downward, matching a screen / grid layout.
_DELTAS = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}
_OPPOSITES = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

Point = tuple[int, int]


@dataclass
class Snake:
    """A snake is a list of body cells, head first."""

    body: list[Point]
    direction: Direction
    alive: bool = True

    @property
    def head(self) -> Point:
        return self.body[0]

    def __len__(self) -> int:
        return len(self.body)


@dataclass
class SnakeGame:
    """Deterministic (given ``seed``) two-snake game on a ``width`` x ``height`` grid.

    Snake 0 starts on the left moving right; snake 1 starts on the right moving
    left. Both snakes move simultaneously each ``step``.
    """

    width: int = 12
    height: int = 12
    seed: int | None = None
    initial_length: int = 3
    max_steps: int = 256

    snakes: list[Snake] = field(default_factory=list)
    food: Point = (0, 0)
    step_count: int = 0
    rng: random.Random = field(default_factory=random.Random, repr=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        mid = self.height // 2
        left = [(self.initial_length - 1 - i, mid) for i in range(self.initial_length)]
        right = [
            (self.width - self.initial_length + i, mid)
            for i in range(self.initial_length)
        ]
        self.snakes = [
            Snake(body=left, direction=Direction.RIGHT),
            Snake(body=right, direction=Direction.LEFT),
        ]
        self.step_count = 0
        self.food = self._spawn_food()

    def _spawn_food(self) -> Point:
        occupied = {p for s in self.snakes for p in s.body}
        free = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        if not free:
            return (-1, -1)
        return self.rng.choice(free)

    # ---------------------------------------------------------------- queries
    def in_bounds(self, p: Point) -> bool:
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def occupied_cells(self) -> set[Point]:
        return {p for s in self.snakes if s.alive for p in s.body}

    def next_head(self, snake_id: int, direction: Direction) -> Point:
        hx, hy = self.snakes[snake_id].head
        dx, dy = direction.delta
        return (hx + dx, hy + dy)

    def is_safe(self, snake_id: int, direction: Direction) -> bool:
        """Would moving ``snake_id`` in ``direction`` be immediately fatal?

        Tail cells are treated as occupied (conservative): a snake only vacates
        its tail if it does not eat this step, and the opponent's move is
        unknown, so the safe choice is to avoid them.
        """
        snake = self.snakes[snake_id]
        if direction == snake.direction.opposite and len(snake) > 1:
            return False
        p = self.next_head(snake_id, direction)
        return self.in_bounds(p) and p not in self.occupied_cells()

    @property
    def alive_ids(self) -> list[int]:
        return [i for i, s in enumerate(self.snakes) if s.alive]

    @property
    def is_over(self) -> bool:
        return len(self.alive_ids) <= 1 or self.step_count >= self.max_steps

    @property
    def winner(self) -> int | None:
        """Index of the surviving snake, or None for a draw / not over."""
        alive = self.alive_ids
        if len(alive) == 1:
            return alive[0]
        if len(alive) == 2 and self.step_count >= self.max_steps:
            # Timeout: longer snake wins, equal length is a draw.
            l0, l1 = len(self.snakes[0]), len(self.snakes[1])
            if l0 != l1:
                return 0 if l0 > l1 else 1
        return None

    # ------------------------------------------------------------------- step
    def step(self, actions: dict[int, Direction]) -> None:
        """Advance the game one tick. ``actions`` maps snake_id -> direction.

        Snakes with no action (or dead snakes) keep their current direction.
        Reversing into yourself is ignored and the previous direction is kept.
        """
        if self.is_over:
            return

        # 1. Resolve directions.
        for i, snake in enumerate(self.snakes):
            if not snake.alive:
                continue
            d = actions.get(i, snake.direction)
            if d == snake.direction.opposite and len(snake) > 1:
                d = snake.direction
            snake.direction = d

        # 2. Compute new heads and whether each snake eats.
        new_heads: dict[int, Point] = {}
        eats: dict[int, bool] = {}
        for i in self.alive_ids:
            h = self.next_head(i, self.snakes[i].direction)
            new_heads[i] = h
            eats[i] = h == self.food

        # 3. Move bodies (grow if eating).
        for i in self.alive_ids:
            s = self.snakes[i]
            s.body.insert(0, new_heads[i])
            if not eats[i]:
                s.body.pop()

        # 4. Collision detection (simultaneous).
        dead: set[int] = set()
        for i in self.alive_ids:
            h = new_heads[i]
            if not self.in_bounds(h):
                dead.add(i)
                continue
            for j in self.alive_ids:
                body = self.snakes[j].body
                # For the mover's own body, skip the head cell itself.
                cells = body[1:] if j == i else body
                if h in cells:
                    dead.add(i)
                    break

        for i in dead:
            self.snakes[i].alive = False

        # 5. Respawn food if eaten.
        if any(eats.values()):
            self.food = self._spawn_food()

        self.step_count += 1

    # --------------------------------------------------------------- display
    def render(self) -> str:
        """ASCII board: ``A``/``a`` snake 0 head/body, ``B``/``b`` snake 1, ``*`` food."""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        fx, fy = self.food
        if self.in_bounds(self.food):
            grid[fy][fx] = "*"
        for i, s in enumerate(self.snakes):
            head_ch, body_ch = ("A", "a") if i == 0 else ("B", "b")
            if not s.alive:
                head_ch, body_ch = "x", "x"
            for k, (x, y) in enumerate(s.body):
                if self.in_bounds((x, y)):
                    grid[y][x] = head_ch if k == 0 else body_ch
        border = "+" + "-" * self.width + "+"
        rows = ["|" + "".join(r) + "|" for r in grid]
        return "\n".join([border, *rows, border])
