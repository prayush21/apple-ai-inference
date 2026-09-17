"""Generate imitation-learning data by simulating games with the heuristic policy.

Each episode yields a trajectory of (features_t, action_t) for the learner
snake (snake 0). Because the model is a causal transformer over the game
history, trajectories are kept as sequences rather than shuffled into i.i.d.
samples.

Two kinds of rollouts:

* ``simulate_episode`` — the teacher drives snake 0 (plain behaviour cloning).
* ``simulate_dagger_episode`` — the *learner* drives snake 0 while the teacher
  labels every visited state (DAgger). Plain cloning only ever sees states the
  teacher reaches, so the learner never learns to recover from its own
  mistakes; DAgger closes that gap.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .features import FEATURE_DIM, extract_features
from .game import Direction, SnakeGame
from .policy import HeuristicPolicy

# An actor picks snake 0's move from the game and the feature history so far.
Actor = Callable[[SnakeGame, list[list[float]]], Direction]

Episode = tuple[np.ndarray, np.ndarray]  # features [T, 16], actions [T]


def simulate_episode(
    seed: int,
    *,
    width: int = 12,
    height: int = 12,
    max_steps: int = 256,
    epsilon: float = 0.1,
) -> Episode:
    """Run one game with the teacher driving snake 0.

    The teacher acts greedily (no exploration noise) so labels are clean; the
    opponent uses epsilon-greedy so the learner sees varied states.
    """
    teacher = HeuristicPolicy(epsilon=0.0)
    return simulate_dagger_episode(
        seed,
        lambda game, _history: teacher.choose(game, 0),
        width=width,
        height=height,
        max_steps=max_steps,
        epsilon=epsilon,
    )


def simulate_dagger_episode(
    seed: int,
    actor: Actor,
    *,
    width: int = 12,
    height: int = 12,
    max_steps: int = 256,
    epsilon: float = 0.1,
) -> Episode:
    """Run one game with ``actor`` driving snake 0; label each visited state
    with the teacher's action."""
    game = SnakeGame(width=width, height=height, seed=seed, max_steps=max_steps)
    teacher = HeuristicPolicy(epsilon=0.0)
    opponent = HeuristicPolicy(epsilon=epsilon, seed=seed + 1)

    feats: list[list[float]] = []
    acts: list[int] = []
    while not game.is_over and game.snakes[0].alive:
        feats.append(extract_features(game, 0))
        acts.append(int(teacher.choose(game, 0)))
        a0 = actor(game, feats)
        a1 = opponent.choose(game, 1)
        game.step({0: a0, 1: a1})

    return (
        np.asarray(feats, dtype=np.float32).reshape(-1, FEATURE_DIM),
        np.asarray(acts, dtype=np.int64),
    )


def pack_windows(episodes: list[Episode], seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split variable-length episodes into padded ``seq_len`` windows.

    Returns ``(features [N, seq_len, 16], actions [N, seq_len], mask [N, seq_len])``
    where ``mask`` is 1 for real timesteps and 0 for padding.
    """
    xs, ys, ms = [], [], []
    for f, a in episodes:
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


def build_dataset(
    n_episodes: int,
    *,
    seq_len: int,
    seed: int = 0,
    **episode_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate ``n_episodes`` teacher games and pack them into windows."""
    episodes = [simulate_episode(seed + ep, **episode_kwargs) for ep in range(n_episodes)]
    return pack_windows(episodes, seq_len)
