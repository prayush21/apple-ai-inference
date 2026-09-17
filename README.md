# Core AI Snake — a clone of the WWDC26 "Meet Core AI" demo

A from-scratch reimplementation of the sample project in
[WWDC26 session 324 · Meet Core AI](https://developer.apple.com/videos/play/wwdc2026/324/):
a two-player snake game where one snake is driven by a small PyTorch transformer
converted to Core AI, first stateless and then with KV-cache **states**.

The goal is to exercise the whole Core AI toolchain end-to-end, not to build a
strong snake bot.

```
PyTorch model ──torch.export──▶ coreai-torch ──▶ .aimodel ──▶ Core AI runtime (Python) ──▶ verify numerics
                                                     │
                                                     └──▶ CoreAI.framework (Swift, macOS 27+) ──▶ ModelPlayer in app
```

## Layout

| Path | What |
|---|---|
| `snake_ai/game.py` | Two-snake game engine (rules, collisions, ASCII render) |
| `snake_ai/features.py` | The 16-dim per-step feature vector from the talk |
| `snake_ai/policy.py` | Flood-fill heuristic — training-data generator and opponent |
| `snake_ai/model.py` | `SnakeTransformer` (stateless) and `SnakeTransformerStateful` (KV-cache buffers → Core AI states) |
| `snake_ai/train.py` | Imitation-learning trainer → `checkpoints/snake.pt` |
| `snake_ai/convert.py` | `torch.export` → `coreai_torch.TorchConverter` → `.aimodel` (both variants) |
| `snake_ai/verify.py` | Loads the assets with `coreai.runtime` and asserts PyTorch ≙ Core AI |
| `snake_ai/play.py` | Python `ModelPlayer`s driving snake 0 through the Core AI runtime, with latency stats |
| `SnakeCoreAI/` | Swift package: `SnakeEngine` (1:1 port of the engine), `SnakeCoreAI` (`ModelPlayer` on `CoreAI.framework`), `snake-cli` |
| `docs/coreai-ecosystem.md` | Notes on the Core AI ecosystem and the gotchas we hit |

## Quick start (Python side — runs today on macOS 26)

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
.venv/bin/python -m snake_ai.train --episodes 1500 --epochs 6     # ~40 s on M-series
```

```bash
.venv/bin/python -m snake_ai.convert          # writes models/SnakeTransformer*.aimodel
```

```bash
.venv/bin/python -m snake_ai.verify --frames 40
```

```bash
.venv/bin/python -m snake_ai.play --player both --games 5    # latency: stateless vs stateful
```

```bash
.venv/bin/python -m snake_ai.play --player stateful --render # watch a game in the terminal
```

Tests: `.venv/bin/python -m pytest`.

### What you should see

`verify` reports `max |diff| = 0.000000` for the stateless asset and ~1e-6 for a
40-step decode through the stateful one. `play --player both` reproduces the
Instruments observation from the talk:

| player | first-5 inference | last-5 inference |
|---|---|---|
| stateless (full history each step) | ~2 ms | ~9 ms, growing with game length |
| stateful (KV caches as states) | ~4.6 ms | ~4.6 ms, flat |

## Swift side

```bash
cd SnakeCoreAI && swift test && swift run snake-cli --games 3 --render
```

`SnakeEngine` builds and tests on the current toolchain, including a fixture
test that proves the Swift `FeatureExtractor` matches the Python one bit-for-bit.
`ModelPlayer` is written against the `CoreAI` Swift API shown in the session
(`AIModel(contentsOf:)`, `loadFunction(named:)`, `NDArray`,
`InferenceFunction.run(inputs:states:)`, `InferenceFunction.MutableViews`,
`AIModelCache`, `AIModel.specialize`) and is compiled only under
`#if canImport(CoreAI)` — i.e. Xcode 27 / macOS 27. On older SDKs the CLI falls
back to the heuristic and says so.

## Status

- [x] Game engine + features (Python and Swift, parity-tested)
- [x] PyTorch model, stateless and KV-cache stateful, equivalence-tested
- [x] Training via imitation of the heuristic
- [x] `coreai-torch` conversion of both variants
- [x] Numerics verification through the Core AI Python runtime
- [x] Python players + latency comparison
- [x] Swift package skeleton with `ModelPlayer` / specialization helpers
- [ ] Build & run `ModelPlayer` against the real `CoreAI.framework` (needs Xcode 27 / macOS 27)
- [ ] SwiftUI app with a human-controlled second snake
- [ ] Profile with the Core AI Instrument, inspect in the Core AI Debugger
- [ ] Ahead-of-time compilation (`.aimodelc`) and `AIModelCache` handling
- [ ] Better model (the local features cap imitation accuracy around 64%)
