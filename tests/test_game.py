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


def test_dagger_episode_labels_with_teacher():
    from snake_ai.data import pack_windows, simulate_dagger_episode, simulate_episode

    # A learner that always goes UP dies fast; every visited state is still
    # labelled by the teacher, so labels are valid actions and lengths match.
    f, a = simulate_dagger_episode(0, lambda game, history: Direction.UP)
    assert f.shape == (len(a), FEATURE_DIM) and 0 < len(a) <= 8
    assert set(a.tolist()) <= {0, 1, 2, 3}

    # Teacher-driven episodes are the same function with the teacher as actor.
    f2, a2 = simulate_episode(0)
    X, Y, M = pack_windows([(f, a), (f2, a2)], seq_len=16)
    assert X.shape[1:] == (16, FEATURE_DIM) and M.sum() == len(a) + len(a2)


def test_minimax_avoids_immediate_traps_and_beats_heuristic():
    from snake_ai.minimax import MinimaxPolicy, evaluate, EvalWeights, voronoi

    g = SnakeGame(seed=0)
    mine, theirs = voronoi(g, 0)
    assert mine == theirs  # symmetric start
    # Territory and length terms are antisymmetric; only the "be near food"
    # bonus is not, so with food removed the scores must mirror.
    g.food = (-1, -1)
    w = EvalWeights(food=0.0)
    assert evaluate(g, 0, w) == -evaluate(g, 1, w)

    wins = 0
    for seed in range(6):
        g = SnakeGame(seed=seed, max_steps=200)
        a, b = MinimaxPolicy(depth=1), HeuristicPolicy(epsilon=0.05, seed=seed)
        while not g.is_over:
            d = a.choose(g, 0)
            assert g.is_safe(0, d) or not any(g.is_safe(0, x) for x in Direction)
            g.step({0: d, 1: b.choose(g, 1)})
        wins += g.winner == 0
    assert wins >= 4
