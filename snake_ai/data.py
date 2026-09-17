"""Generate imitation-learning data by simulating games with the heuristic policy.

Each episode yields a trajectory of (features_t, action_t) for the learner
snake (snake 0). Because the model is a causal transformer over the game
history, trajectories are kept as sequences rather than shuffled into i.i.d.
samples.
"""

from __future__ import annotations

import numpy as np

from .features import FEATURE_DIM, extract_features
from .game import SnakeGame
from .policy import HeuristicPolicy


def simulate_episode(
    seed: int,
    *,
    width: int = 12,
    height: int = 12,
    max_steps: int = 256,
    epsilon: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one game and return ``(features [T, 16], actions [T])`` for snake 0.

    The learner's *teacher* acts greedily (no exploration noise) so labels are
    clean; the opponent uses epsilon-greedy so the learner sees varied states.
    """
    game = SnakeGame(width=width, height=height, seed=seed, max_steps=max_steps)
    teacher = HeuristicPolicy(epsilon=0.0)
    opponent = HeuristicPolicy(epsilon=epsilon, seed=seed + 1)

    feats: list[list[float]] = []
    acts: list[int] = []
    while not game.is_over and game.snakes[0].alive:
        a0 = teacher.choose(game, 0)
        a1 = opponent.choose(game, 1)
        feats.append(extract_features(game, 0))
        acts.append(int(a0))
        game.step({0: a0, 1: a1})

    return (
        np.asarray(feats, dtype=np.float32).reshape(-1, FEATURE_DIM),
        np.asarray(acts, dtype=np.int64),
    )


def build_dataset(
    n_episodes: int,
    *,
    seq_len: int,
    seed: int = 0,
    **episode_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate ``n_episodes`` games and pack them into fixed-length windows.

    Returns ``(features [N, seq_len, 16], actions [N, seq_len], mask [N, seq_len])``
    where ``mask`` is 1 for real timesteps and 0 for padding. Episodes longer
    than ``seq_len`` are split into non-overlapping windows.
    """
    xs, ys, ms = [], [], []
    for ep in range(n_episodes):
        f, a = simulate_episode(seed + ep, **episode_kwargs)
        for start in range(0, len(a), seq_len):
            fw, aw = f[start : start + seq_len], a[start : start + seq_len]
            n = len(aw)
            if n == 0:
                continue
            x = np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)
            y = np.zeros((seq_len,), dtype=np.int64)
            m = np.zeros((seq_len,), dtype=np.float32)
            x[:n], y[:n], m[:n] = fw, aw, 1.0
            xs.append(x)
            ys.append(y)
            ms.append(m)
    return np.stack(xs), np.stack(ys), np.stack(ms)
