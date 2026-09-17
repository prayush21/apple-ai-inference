from snake_ai.features import FEATURE_DIM, extract_features
from snake_ai.game import Direction, SnakeGame
from snake_ai.policy import HeuristicPolicy


def test_initial_layout():
    g = SnakeGame(width=12, height=12, seed=0)
    assert g.snakes[0].head == (2, 6) and g.snakes[0].direction == Direction.RIGHT
    assert g.snakes[1].head == (9, 6) and g.snakes[1].direction == Direction.LEFT
    assert g.food not in g.occupied_cells()


def test_wall_collision_kills():
    g = SnakeGame(width=6, height=6, seed=0)
    for _ in range(10):
        g.step({0: Direction.UP, 1: Direction.DOWN})
    # Snake 1 starts one cell closer to its wall, so it dies first and the
    # game ends immediately with snake 0 as the last snake standing.
    assert not g.snakes[1].alive and g.snakes[0].alive
    assert g.is_over and g.winner == 0


def test_reverse_is_ignored():
    g = SnakeGame(seed=0)
    g.step({0: Direction.LEFT})  # reverse into own body -> keep RIGHT
    assert g.snakes[0].direction == Direction.RIGHT
    assert g.snakes[0].head == (3, 6)


def test_eating_grows():
    g = SnakeGame(seed=0)
    g.food = (3, 6)  # directly ahead of snake 0
    g.step({})
    assert len(g.snakes[0]) == 4
    assert g.food != (3, 6)


def test_features_shape_and_range():
    g = SnakeGame(seed=1)
    f = extract_features(g, 0)
    assert len(f) == FEATURE_DIM
    assert all(-1.0 <= v <= 1.0 for v in f)
    assert f[6:10] == [0, 0, 0, 1]  # snake 0 faces RIGHT
    assert f[12:16] == [0, 0, 1, 0]  # opponent faces LEFT


def test_heuristic_survives_a_while():
    g = SnakeGame(seed=3, max_steps=100)
    p = HeuristicPolicy()
    while not g.is_over:
        g.step({0: p.choose(g, 0), 1: p.choose(g, 1)})
    assert g.step_count > 10
