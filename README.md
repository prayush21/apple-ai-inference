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
| `snake_ai/policy.py` | Flood-fill heuristic — the original teacher and the evaluation opponent |
| `snake_ai/minimax.py` | Stronger teacher: alpha-beta search over Voronoi territory (94–100% vs the heuristic) |
| `snake_ai/serve.py` | Serves an `.aimodel` over local HTTP through the Core AI Python runtime, so the Swift app can use the model before macOS 27 |
| `snake_ai/model.py` | `SnakeTransformer` (stateless) and `SnakeTransformerStateful` (KV-cache buffers → Core AI states) |
| `snake_ai/train.py` | Behaviour cloning + optional DAgger rounds → `checkpoints/snake.pt` |
| `snake_ai/evaluate.py` | Win rate vs the heuristic (the metric that matters; val accuracy alone misleads) |
| `snake_ai/convert.py` | `torch.export` → `coreai_torch.TorchConverter` → `.aimodel` (stateless, stateful, static-shape decode) |
| `snake_ai/verify.py` | Loads the assets with `coreai.runtime` and asserts PyTorch ≙ Core AI |
| `snake_ai/play.py` | Python `ModelPlayer`s driving snake 0 through the Core AI runtime, with latency stats |
| `snake_ai/debug.py` | Python versions of the Core AI Instrument / Debugger: per-op timings mapped to source lines, op-by-op numerics diff, raw profiler events |
| `snake_ai/specialize.py` | Specialization cache + `SpecializationOptions` demo (cold vs warm load) |
| `SnakeCoreAI/` | Swift package: `SnakeEngine` (1:1 port of engine, heuristic, minimax), `SnakeCoreAI` (`ModelPlayer` on `CoreAI.framework`, `RemoteModelPlayer` via `serve.py`), `snake-cli`, `SnakeApp` (SwiftUI) |
| `docs/coreai-ecosystem.md` | Notes on the Core AI ecosystem and the gotchas we hit |

## Quick start (Python side — runs today on macOS 26)

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
.venv/bin/python -m snake_ai.train --episodes 3000 --epochs 8 --dagger-rounds 3   # ~5 min on M-series
```

```bash
.venv/bin/python -m snake_ai.convert          # writes models/SnakeTransformer{,Stateful,Decode}.aimodel
```

```bash
.venv/bin/python -m snake_ai.verify --frames 40
```

```bash
.venv/bin/python -m snake_ai.play --player all --games 5     # latency: stateless vs stateful vs static decode
```

```bash
.venv/bin/python -m snake_ai.play --render                   # watch a game in the terminal
```

```bash
.venv/bin/python -m snake_ai.debug benchmark                 # per-op timings + source-annotated model.py
```

```bash
.venv/bin/python -m snake_ai.debug compare                   # op-by-op PyTorch vs Core AI numerics
```

```bash
.venv/bin/python -m snake_ai.specialize --clear-cache        # cold vs cached specialization
```

Tests: `.venv/bin/python -m pytest`.

### What you should see

`verify` reports `max |diff| = 0.000000` for the stateless asset and ~1e-6 for a
40-step decode through the stateful one. `play --player all` reproduces the
Instruments observation from the talk, plus the static-shape optimisation:

| player | first-5 inference | last-5 inference |
|---|---|---|
| stateless (full history each step) | ~2 ms | ~9 ms, growing with game length |
| stateful (KV caches as states, dynamic T) | ~4.6 ms | ~4.6 ms, flat |
| decode (states + static `[1,1,16]` shapes) | ~3.8 ms | ~3.8 ms, flat |

The shipped checkpoint (behaviour cloning + 3 DAgger rounds, 118k params) wins
~31% of games against the flood-fill heuristic on 100 fresh seeds; see
`docs/coreai-ecosystem.md` for why bigger models did worse and what the ceiling
is.

`debug benchmark` writes `models/SnakeTransformer.annotated.py.txt` — the
model source with every line annotated by the Core AI ops it produced and
their timings. `specialize --clear-cache` shows cold specialization (~140 ms)
vs cached load (<1 ms) with the artifacts in `~/Library/Caches/coreai-cache`.

### Latency baselines (`docs/bench/`)

Every path that runs the model can write a `snake-bench/1` JSON record (load
ms + per-move inference mean/p50/p95/first-5/last-5, host info) so the numbers
can be compared once `CoreAI.framework` is available in-process:

```bash
.venv/bin/python -m snake_ai.play --player all --games 5 --json docs/bench/python.json
```

```bash
cd SnakeCoreAI && swift run snake-cli --ai model --games 5 --json ../docs/bench/swift-remote.json   # needs serve.py running
```

The app's **Save bench** button writes `docs/bench/app-<model>-<date>.json`
for the game just played (round-trip *and* server-side ms for the remote
player). Current numbers on an M2, decode asset:

| Path | Load | Per move |
|---|---|---|
| Python in-process (`play.py`) | 3 ms warm cache (first in process), ~2 ms after | 3.8 ms |
| Swift → `serve.py` over HTTP (`snake-cli`) | 185 ms first (`URLSession` warm-up + `/info` + `/reset`), ~5 ms after | 6.8 ms round trip, 4.3 ms of it server-side |
| Swift `ModelPlayer` on `CoreAI.framework` | — needs macOS 27 | — |

The HTTP hop costs ~2.5 ms from Swift; the remaining ~4 ms is the runtime
itself, which is why this tiny model will not *feel* different in-process.

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
`#if canImport(CoreAI)` — i.e. Xcode 27 / macOS 27. On older SDKs the CLI and
app fall back to the heuristic and say so.

```bash
.venv/bin/python -m snake_ai.serve                                                          # heuristic-taught model on :8765
```

```bash
.venv/bin/python -m snake_ai.serve --models-dir models/minimax_teacher --port 8766 --tag minimax-taught
```

```bash
cd SnakeCoreAI && swift run SnakeApp        # arrow keys steer snake B; space / Start begins a game
```

The picker at the top switches snake A between **Heuristic**, **Minimax**, and
the two Core AI models (each served by a `snake_ai.serve` instance). A
per-opponent scoreboard and avg ms/move make the differences measurable. On
macOS 27 the model options load in-process via `CoreAI.framework` (it looks for
`models/SnakeTransformerDecode.aimodel` above the working directory, or
`SNAKE_MODEL` / `SNAKE_MODEL_FUNCTION`).

### Teachers and students

| player | vs greedy heuristic, 100 games |
|---|---|
| greedy heuristic (mirror) | 51% |
| minimax depth 1 / depth 2 | 94% / 100% |
| model, heuristic-taught (+3 DAgger) | 31% |
| model, minimax-taught (+2 DAgger) | 30% — dies less, times out more |

The 3x stronger teacher did not move the student's win rate: it decides on
Voronoi territory and body positions, which the talk's 16 distance features
cannot express. The next lever is richer input (board occupancy planes), not
model size or teacher quality. Train the minimax-taught student with
`python -m snake_ai.train --teacher minimax --dagger-rounds 2`.

## Status

- [x] Game engine + features (Python and Swift, parity-tested)
- [x] PyTorch model, stateless and KV-cache stateful, equivalence-tested
- [x] Training: behaviour cloning + DAgger, evaluated by win rate
- [x] `coreai-torch` conversion: stateless, stateful, static-shape decode
- [x] Numerics verification through the Core AI Python runtime
- [x] Python players + latency comparison
- [x] Profiling / numerics debugging from Python (`snake_ai.debug`) — the Python side of the Core AI Instrument & Debugger
- [x] Specialization cache and `SpecializationOptions` from Python (`snake_ai.specialize`)
- [x] Swift package with `ModelPlayer` / specialization helpers
- [x] SwiftUI app with a human-controlled second snake
- [ ] Build & run `ModelPlayer` against the real `CoreAI.framework` (needs Xcode 27 / macOS 27)
- [ ] Xcode-side tooling: Core AI Instrument, Debugger, debug gauge, `.aimodelc` AOT compilation, `AIModelCache` (needs Xcode 27 / macOS 27)
