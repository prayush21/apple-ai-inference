"""A stronger teacher: adversarial lookahead over a territory evaluation.

``HeuristicPolicy`` scores one move ahead with a capped flood fill and treats
the opponent as static; it dies exclusively by being sealed into a pocket by
an opponent it never modelled. ``MinimaxPolicy`` fixes exactly that:

* **Search** — for each of my safe moves, the opponent picks the reply that
  is worst for me (sequential approximation of the simultaneous move; it is
  pessimistic, which is the safe side). ``depth`` counts my moves, so
  ``depth=1`` is 2-ply, ``depth=2`` is 4-ply. Alpha-beta pruning keeps
  ``depth=2`` affordable in pure Python.
* **Evaluation** — terminal outcomes, then a weighted sum of Voronoi territory
  (cells I reach strictly before the opponent: the classic Tron heuristic that
  sees cut-offs), a "trapped" penalty when my reachable region is smaller than
  my body, length difference (decides timeouts), and food proximity.
* **Simulation** — uses the real ``SnakeGame.step`` on a ``clone()``, so tails
  vacating, growth and simultaneous collisions all behave as in the game.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .game import Direction, Point, SnakeGame

INF = float("inf")


@dataclass(frozen=True)
class EvalWeights:
    """Territory keeps it alive; length and food make it actually grow, which
    decides games that reach the step limit (longer snake wins)."""

    win: float = 10_000.0
    territory: float = 1.0
    trapped: float = 200.0
    length: float = 30.0
    food: float = 15.0


def voronoi(game: SnakeGame, me: int) -> tuple[int, int]:
    """(cells I reach strictly first, cells the opponent reaches strictly first)
    by simultaneous BFS from both heads over free cells."""
    occupied = game.occupied_cells()
    a, b = game.snakes[me].head, game.snakes[1 - me].head
    dist_a = _bfs(game, a, occupied)
    dist_b = _bfs(game, b, occupied)
    mine = theirs = 0
    for p, da in dist_a.items():
        db = dist_b.get(p)
        if db is None or da < db:
            mine += 1
    for p, db in dist_b.items():
        da = dist_a.get(p)
        if da is None or db < da:
            theirs += 1
    return mine, theirs


def _bfs(game: SnakeGame, start: Point, occupied: set[Point]) -> dict[Point, int]:
    dist = {start: 0}
    q: deque[Point] = deque([start])
    while q:
        x, y = q.popleft()
        d = dist[(x, y)] + 1
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            p = (x + dx, y + dy)
            if game.in_bounds(p) and p not in occupied and p not in dist:
                dist[p] = d
                q.append(p)
    return dist


def evaluate(game: SnakeGame, me: int, w: EvalWeights) -> float:
    """Static score of ``game`` from snake ``me``'s point of view."""
    mine, opp = game.snakes[me], game.snakes[1 - me]
    if not mine.alive and not opp.alive:
        return 0.0
    if not mine.alive:
        return -w.win
    if not opp.alive:
        return w.win
    if game.step_count >= game.max_steps:  # timeout: longer snake wins
        return w.win * (1 if len(mine) > len(opp) else -1 if len(mine) < len(opp) else 0)

    my_terr, their_terr = voronoi(game, me)
    score = w.territory * (my_terr - their_terr)

    # A region smaller than my body is a slow death even if no move is fatal yet.
    dist = _bfs(game, mine.head, game.occupied_cells())
    if len(dist) - 1 < len(mine):
        score -= w.trapped

    score += w.length * (len(mine) - len(opp))

    # Food: reward being closer than the opponent, and being close at all,
    # both normalised by board size so the term is in [-2, 2].
    fx, fy = game.food
    scale = game.width + game.height
    my_d = (abs(fx - mine.head[0]) + abs(fy - mine.head[1])) / scale
    their_d = (abs(fx - opp.head[0]) + abs(fy - opp.head[1])) / scale
    score += w.food * ((their_d - my_d) + (1.0 - my_d))
    return score


class MinimaxPolicy:
    """Alpha-beta minimax over (my move, opponent reply) pairs."""

    def __init__(self, depth: int = 2, weights: EvalWeights | None = None) -> None:
        self.depth = depth
        self.w = weights or EvalWeights()
        self.nodes = 0

    def choose(self, game: SnakeGame, snake_id: int) -> Direction:
        self.nodes = 0
        best, best_score = game.snakes[snake_id].direction, -INF
        for d in self._ordered_moves(game, snake_id):
            score = self._opponent_reply(game, snake_id, d, self.depth, best_score, INF)
            if score > best_score:
                best, best_score = d, score
        return best

    # -- search -----------------------------------------------------------
    def _opponent_reply(self, game: SnakeGame, me: int, my_move: Direction, depth: int, alpha: float, beta: float) -> float:
        """Opponent minimises over its replies to ``my_move``."""
        opp = 1 - me
        value = INF
        for od in self._ordered_moves(game, opp):
            child = game.clone()
            child.step({me: my_move, opp: od})
            self.nodes += 1
            if child.is_over or depth <= 1:
                v = evaluate(child, me, self.w)
            else:
                v = self._my_turn(child, me, depth - 1, alpha, beta)
            value = min(value, v)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

    def _my_turn(self, game: SnakeGame, me: int, depth: int, alpha: float, beta: float) -> float:
        value = -INF
        for d in self._ordered_moves(game, me):
            v = self._opponent_reply(game, me, d, depth, alpha, beta)
            value = max(value, v)
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value

    @staticmethod
    def _ordered_moves(game: SnakeGame, snake_id: int) -> list[Direction]:
        """Safe moves first (current direction first for better pruning); if
        none are safe, the current direction so the snake still 'moves'."""
        snake = game.snakes[snake_id]
        if not snake.alive:
            return [snake.direction]
        safe = [d for d in Direction if game.is_safe(snake_id, d)]
        if not safe:
            return [snake.direction]
        safe.sort(key=lambda d: d != snake.direction)
        return safe
