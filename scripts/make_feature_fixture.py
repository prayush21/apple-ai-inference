"""Dump board states + Python-computed features so the Swift engine can prove parity.

    .venv/bin/python scripts/make_feature_fixture.py
"""
import json
from pathlib import Path

from snake_ai.features import extract_features
from snake_ai.game import SnakeGame
from snake_ai.policy import HeuristicPolicy

OUT = Path(__file__).resolve().parent.parent / "SnakeCoreAI/Tests/SnakeEngineTests/Fixtures/features.json"

cases = []
for seed in range(3):
    game = SnakeGame(seed=seed, max_steps=80)
    p0, p1 = HeuristicPolicy(), HeuristicPolicy(epsilon=0.2, seed=seed)
    while not game.is_over:
        if game.step_count % 7 == 0:
            cases.append({
                "width": game.width, "height": game.height,
                "food": list(game.food),
                "snakes": [
                    {"body": [list(p) for p in s.body], "direction": int(s.direction), "alive": s.alive}
                    for s in game.snakes
                ],
                "features": [extract_features(game, i) if game.snakes[i].alive else None for i in range(2)],
            })
        game.step({0: p0.choose(game, 0), 1: p1.choose(game, 1)})

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(cases))
print(f"wrote {len(cases)} cases to {OUT}")
