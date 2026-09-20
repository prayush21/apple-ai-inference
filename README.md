# apple-ai-inference — Core AI experiments

Hands-on ports of the WWDC26 **Core AI** toolchain, one model at a time. Each
project is a self-contained Python package (author · convert · verify · run
· serve) plus a Swift package (in-process `CoreAI.framework` player behind
`#if canImport(CoreAI)`, and an HTTP fallback that works on macOS 26 today).

| Project | Python | Swift | Model | Status |
|---|---|---|---|---|
| 1. Snake | `snake_ai/` | `SnakeCoreAI/` | 118k-param transformer, KV cache as states | complete; latency baselines in `docs/bench/` |
| 2. SmolLM2 | `llm_ai/` | `LLMCoreAI/` | SmolLM2-360M-Instruct, KV cache as states, tokens/sec | scaffolded (branch `llm`) |
| 3. Decide | `decide_ai/` | `DecideCoreAI/` | MiniLM NLI cross-encoder, zero-shot "System One" decisions (text → calibrated yes/no per question) | step 1 done (branch `system-one`): Python pipeline, `/decide` server, holdout, Jev comparison |

Shared: one `.venv` (`pyproject.toml`; `pip install -e '.[dev,llm,decide]'` for all),
`docs/coreai-ecosystem.md` for the converter/runtime gotchas that apply to any
model, and `docs/bench/` for the load / per-step latency records that get a
`CoreAI.framework` column once this machine is on macOS 27.

Why one repo: the 800 MB torch + `coreai-torch` environment and the gotchas
doc are the expensive shared parts; the projects themselves are siblings and
can be split out with `git subtree split` if one ever needs its own life.

---

## Project 1 — Core AI Snake (WWDC26 session 324 clone)

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

### Layout

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

### Quick start (Python side — runs today on macOS 26)

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

#### What you should see

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

#### Latency baselines (`docs/bench/`)

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

### Swift side

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

#### Teachers and students

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

### Status

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

---

## Project 2 — SmolLM2-360M, tokens as the latency gauge

The snake model is too small for any deployment choice to be *felt*: ~4 ms
per move against a 140 ms game tick. This project repeats the same pipeline —
plain-`torch` model with `register_buffer` KV caches → `torch.export` →
`coreai-torch` → `.aimodel` with states → Python runtime / `serve.py` /
`CoreAI.framework` — on a 360M-parameter decoder-only LLM, where stateless vs
stateful is "unusable vs readable", the HTTP hop per token is visible, and
weight loading is long enough for `AIModelCache` to matter.

```bash
.venv/bin/pip install -e '.[dev,llm]'                       # adds huggingface_hub, safetensors, tokenizers
```

```bash
.venv/bin/python -m llm_ai.download                        # models/llm/hf/SmolLM2-360M-Instruct (~725 MB, gitignored)
```

```bash
cd LLMCoreAI && swift build && swift run llm-cli           # reports which generator this build can use
```

Milestones are listed in `llm_ai/__init__.py`; only `download` and the
config loader (`llm_ai.model.LlamaConfig.from_hf`) exist so far. SmolLM2-360M
is a Llama: 32 layers, hidden 960, 15 query heads / 5 KV heads (GQA, `n_rep`
3), head dim 64, vocab 49152, tied embeddings. With `max_seq_len = 1024` each
cache state is `[32, 1, 5, 1024, 64]` fp32 ≈ 42 MB.

`LLMCoreAI` mirrors `SnakeCoreAI`: a `TokenGenerator` protocol,
`RemoteGenerator` (SSE stream from `llm_ai.serve`, port 8770) and
`ModelGenerator` (`CoreAI.framework`, compiled only on Xcode 27+), an
`llm-cli` for benches and a SwiftUI `LLMApp` with a tokens/sec gauge.

---

## Project 3 — Decide: an on-device "System One" model

A *System One* model makes a fast, structured decision in one forward pass:
text `state` in, calibrated probabilities over fixed labels out. No generated
tokens, so nothing to hallucinate — the output space is the label set.
TypeSafe sells this shape as a hosted API ([Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
70–500 ms). The snake model is already a System One model (16 features → 4-way
softmax, 3.8 ms); this project generalises the label set to natural-language
questions and puts the same shape on Core AI, where there is no network floor
and no data egress.

**Mechanism.** Zero-shot NLI with a cross-encoder: each question becomes a
hypothesis, the model scores `(state, hypothesis)` for entailment / neutral /
contradiction, `P(yes) = softmax(logits / T)[entailment]`. N questions are one
batched forward pass `[N, L] → [N, 3]` — Jev's "parallel sampling".

**Wire format.** `/decide` speaks Jev's `/v1/evaluate` JSON, so one client,
one bench harness and one holdout runner hit both backends with zero branching:

```
POST /decide
{"model": "local/nli-minilm2",
 "state": "The pasta was cold and the waiter ignored us.",
 "questions": {"is_complaint": {"type": "boolean", "instructions": "Is the customer complaining?",
                                "hypothesis": "The customer is complaining."},
               "wants_refund": {"type": "boolean", "instructions": "Is the customer asking for a refund?",
                                "hypothesis": "The customer is asking for a refund."}}}
→ {"model": "local/nli-minilm2",
   "answers": {"is_complaint": {"type": "boolean", "probability": 0.97}, "wants_refund": {...}},
   "usage": {"inputTokens": 41, "outputTokens": 0},
   "timing": {"ms_tokenize": 0.4, "ms_infer": 268.0, "ms_total": 268.6, "batch": 2, "padded_len": 64},
   "raw": {"is_complaint": [P(contradiction), P(entailment), P(neutral)], ...}}
```

`instructions` is the question form (what Jev is sent, as in its docs);
`hypothesis` is the declarative form NLI wants (what the local model scores).
Both live in the holdout and bench inputs, so the phrasing difference is
controlled for, not hidden; without `hypothesis` the server uses
`instructions` verbatim. Only `boolean` questions exist in step 1 — `choice`
and `score` return a 400 naming them as step 5 work.

**Model.** `cross-encoder/nli-MiniLM2-L6-H768`: 6 layers, hidden 768, 12
heads, 3-way NLI head. Chosen for plain absolute-position attention that
lowers cleanly (no relative-position gathers — the `index_put` class of op
that broke the snake KV cache). Two things the model card does not make
obvious: it is `model_type: roberta` (MiniLMv2 distilled from RoBERTa-Large),
so the tokenizer is **byte-level BPE** (`vocab.json` + `merges.txt`, 50 265
tokens), not WordPiece, and the 50k-row embedding table makes it 82M params /
328 MB fp32, not ~22M. Label order from `config.json`: `[contradiction,
entailment, neutral]`.

### Layout

| Path | What |
|---|---|
| `decide_ai/download.py` | `snapshot_download` of config + `model.safetensors` + tokenizer into `models/decide/hf/` (gitignored) |
| `decide_ai/model.py` | Self-contained RoBERTa encoder + NLI head loading the HF state dict; explicit softmax attention, additive mask expanded to `[N, H, L, L]` |
| `decide_ai/tokenize.py` | Dependency-free byte-level BPE (GPT-2 regex as a scanner, `bytes_to_unicode`, merges, `<s> A </s></s> B </s>`); the spec for the step-2 Swift port |
| `decide_ai/jev.py` | `urllib` client for Jev via the Vercel AI Gateway: sha256-keyed disk cache, ≤ 25 req/min pacing, `Retry-After`, certifi TLS |
| `decide_ai/convert.py` | `torch.export` → `coreai-torch` → `NLICrossEncoder.aimodel` (dynamic) and `NLICrossEncoderStatic.aimodel` (eight `main_n{N}_l{L}` entrypoints) |
| `decide_ai/verify.py` | PyTorch ≙ Core AI on 20 real padded pairs; batch-of-4 ≙ 4 × batch-of-1; bit-identical reruns; every static entrypoint |
| `decide_ai/decider.py` | `LocalModel` (asset + loop thread + static-shape padding), `LocalDecider` / `JevDecider` with one `evaluate(state, questions)` |
| `decide_ai/holdout.py` | The 150-state triage holdout, its questions, and the human review table |
| `decide_ai/calibrate.py` | Accuracy / ECE / Brier / hedging / repeatability per backend, temperature fit → `models/decide/calibration.json`, `docs/bench/decide-quality.md` |
| `decide_ai/serve.py` | `POST /decide` + `GET /info` on :8770 through the Core AI Python runtime |
| `decide_ai/bench.py` | `decide-bench/1` JSON: in-process, via HTTP, and Jev |
| `DecideCoreAI/` | Swift package: `RemoteDecider` (HTTP), `Decider` on `CoreAI.framework` (macOS 27, pre-tokenized ids), `decide-cli --bench` |
| `data/decide/` | `questions.json`, `holdout.jsonl`, `holdout_review.md`, `bench_questions.json`, Jev response caches |

### Quick start

```bash
.venv/bin/pip install -e '.[dev,decide]'
```

```bash
.venv/bin/python -m decide_ai.download && .venv/bin/python -m decide_ai.convert && .venv/bin/python -m decide_ai.verify
```

```bash
.venv/bin/python -m decide_ai.serve                       # static asset, /decide on :8770
```

```bash
cd DecideCoreAI && swift run decide-cli --state "Order #48213 never showed up."
```

```bash
.venv/bin/python -m decide_ai.bench --backend local --json docs/bench/decide-python.json
```

```bash
AI_GATEWAY_API_KEY=... .venv/bin/python -m decide_ai.calibrate     # needs holdout_review.md marked reviewed: true
```

`verify` prints `max |diff|` ≈ 6e-6 for both assets. `pytest` covers
tokenizer parity with `tokenizers` (200+ strings), model parity with
`transformers` (3e-6), batch-vs-single, and `/decide` end-to-end.

### What we measured

Latency (p50 / p95 ms, Apple M2, macOS 26.3, `coreai-core 1.0.0b2`; full
records and method in [`docs/bench/README.md`](docs/bench/README.md)):

| N × L | Python in-proc, dynamic | Python in-proc, static | Python→HTTP round trip | Swift→HTTP round trip | `CoreAI.framework` | Jev via gateway |
|---|---|---|---|---|---|---|
| 1 × 64 | 293 / 392 | 219 / 269 | 281 / 336 | 218 / 255 | — (macOS 27) | 278 / 381 |
| 1 × 128 | 480 / 730 | 425 / 494 | 533 / 595 | 432 / 489 | — (macOS 27) | — |
| 4 × 64 | 836 / 970 | 838 / 967 | 811 / 876 | 847 / 1026 | — (macOS 27) | 280 / 556 |
| 4 × 128 | 1692 / 2155 | 1711 / 2196 | 1617 / 1718 | 1704 / 1991 | — (macOS 27) | — |
| 8 × 64 | 1641 / 1909 | 1795 / 2517 | 1607 / 1766 | 1726 / 1942 | — (macOS 27) | 255 / 482 |
| 8 × 128 | 3404 / 4134 | 4473 / 5444 | 3240 / 3376 | 3258 / 3519 | — (macOS 27) | — |
| 16 × 64 | 3325 / 3507 | 4290 / 5216 | 3418 / 4491 | 3288 / 3551 | — (macOS 27) | — |
| 16 × 128 | 6797 / 7489 | 8815 / 10321 | 6604 / 6879 | 6630 / 6899 | — (macOS 27) | — |

Jev's post claims 70–500 ms; measured, the warm round trip is p50 255–280 /
p95 380–560 ms, first call 0.9–2.3 s. The HTTP hop from Python or Swift
costs 0.7–4.6 ms. **The local numbers are the interim macOS 26 CPU runtime,
whose matmul runs at ~25 GFLOP/s** (PyTorch does this forward in 29 ms on
the same CPU; gotcha 13 in the ecosystem doc). So on this machine the
on-device path only breaks even with Jev for one short question; the
`CoreAI.framework` column, blank until macOS 27, is the one that decides
the latency thesis. What holds today: no network floor, a p95/p50 of ~1.2
vs Jev's 1.4–2.0, bit-identical outputs across runs, no data egress.

Quality on the 150-state, human-reviewed triage holdout (5 questions each;
`python -m decide_ai.calibrate`, full tables in
[`docs/bench/decide-quality.md`](docs/bench/decide-quality.md)):

| backend | accuracy | ECE | Brier | AUC | repeatability mean / max | latency p50 / p95 ms |
|---|---|---|---|---|---|---|
| local, raw `P = P(entail)` | 0.754 | 0.195 | 0.218 | 0.71 | 0.00 / 0.00 | 529 / 640 |
| local, temperature-scaled (`T = 4.38`) | 0.723 | 0.058 | 0.185 | 0.75 | 0.00 / 0.00 | 529 / 640 |
| local, `P = entail / (entail + contra)` | 0.727 | 0.077 | 0.180 | 0.80 | 0.00 / 0.00 | 529 / 640 |
| Jev via Vercel AI Gateway | 0.898 | 0.039 | 0.070 | 0.97 | 0.01 / 0.06 | 287 / 464 |

Where Jev wins: everywhere quality is concerned, and not by a little. It has
real recall on all five questions (0.74–0.95) with AUC 0.88–0.99. The
zero-shot MiniLM is a different animal: at the 0.5 threshold it **never says
yes** to `is_complaint`, `about_product_quality` or `urgent` — its 0.754
"accuracy" on those is the base rate of *no* — and its ranking is only good
on the two questions whose hypotheses are near-paraphrases of what people
write: `wants_refund` (AUC 1.00, precision 0.94 at recall 0.71) and
`about_shipping` (AUC 0.93). Temperature scaling fixes the calibration
(ECE 0.20 → 0.06 on the held-out split) but not the decision. Where MiniLM
wins: determinism (bit-identical logits across runs; Jev moves individual
probabilities by up to 0.06 between identical calls, so any threshold sits
on a coin flip for ~1 % of Jev's answers), no network, no data egress, and a
p95 within 20 % of its p50. The negation cases the probe worried about are
fine on both ("I don't want a refund, just send a replacement" → refund 0.04
local, 0.02 Jev). This table is the argument for step 4: Jev's probabilities
are a native distillation target, and the on-device model needs them.

Serving note: the 5-question request pads to N=8 / L=64 on the static asset
(2.1 s) but N=5 / L=48 on the dynamic one (0.53 s) — enumerate the shapes
you actually serve, or serve dynamic.

### Status

- [x] Download, self-contained model, dependency-free tokenizer, parity tests
- [x] Jev client with cache, pacing, retry
- [x] Dynamic + static `.aimodel`, verified through `coreai.runtime`
- [x] 150-state human-reviewed holdout; calibration and quality table
- [x] `/decide` server, Python / HTTP / Swift→HTTP / Jev latency records
- [ ] Step 2: Swift BPE port, in-process `Decider` on macOS 27
- [ ] Step 3: app UI with live confidence bars
- [ ] Step 4: distillation from Jev (native probabilities, input-only pricing)
- [ ] Step 5: `choice` / `score` question types
- [ ] Step 6: DeBERTa-v3 upgrade
