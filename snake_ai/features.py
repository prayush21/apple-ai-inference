"""Per-timestep board features fed to the model.

Layout (16 floats), matching the WWDC26 "Meet Core AI" snake demo:

    [0:4]   distance to wall  up/down/left/right, normalized to [0, 1]
    [4:6]   (dx, dy) to nearest food,           normalized to [-1, 1]
    [6:10]  own direction, one-hot               (up, down, left, right)
    [10:12] (dx, dy) to the opponent's head,     normalized to [-1, 1]
    [12:16] opponent direction, one-hot

The Swift ``FeatureExtractor`` in ``SnakeCoreAI/Sources/SnakeEngine`` must
produce byte-identical values so the converted model behaves the same in-app.
"""

from __future__ import annotations

from .game import SnakeGame

FEATURE_DIM = 16


def extract_features(game: SnakeGame, snake_id: int) -> list[float]:
    me = game.snakes[snake_id]
    opp = game.snakes[1 - snake_id]
    hx, hy = me.head
    w, h = game.width, game.height

    # Distance to wall in each direction (number of free cells before the wall),
    # normalized by the board dimension so values lie in [0, 1].
    d_up = hy / (h - 1)
    d_down = (h - 1 - hy) / (h - 1)
    d_left = hx / (w - 1)
    d_right = (w - 1 - hx) / (w - 1)

    # Vector to food, normalized to [-1, 1].
    fx, fy = game.food
    d_food_x = (fx - hx) / (w - 1)
    d_food_y = (fy - hy) / (h - 1)

    # Vector to the opponent's head. A dead opponent contributes zeros.
    if opp.alive:
        ox, oy = opp.head
        d_opp_x = (ox - hx) / (w - 1)
        d_opp_y = (oy - hy) / (h - 1)
        opp_dir = opp.direction.one_hot
    else:
        d_opp_x = d_opp_y = 0.0
        opp_dir = [0.0, 0.0, 0.0, 0.0]

    features: list[float] = []
    features += [d_up, d_down, d_left, d_right]
    features += [d_food_x, d_food_y]
    features += me.direction.one_hot
    features += [d_opp_x, d_opp_y]
    features += opp_dir
    assert len(features) == FEATURE_DIM
    return features
