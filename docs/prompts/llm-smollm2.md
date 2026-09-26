# Project 2: SmolLM2-360M on Core AI, KV cache as states (`llm` branch)

## Context

This repo (`apple-ai-inference`) ports models through the WWDC26 **Core AI** toolchain, one model per project. Two are done:

- **Snake** (`snake_ai/`, `SnakeCoreAI/`): a 118k-param transformer with its KV cache carried as Core AI **states**, stateless vs stateful vs static decode, in Python, over HTTP and in-process on `CoreAI.framework` (1.3 ms p50 per move).
- **Decide** (`decide_ai/`, `DecideCoreAI/`, branch `system-one`): a 328 MB RoBERTa NLI cross-encoder, Jev/Laya comparisons, `CoreAI.framework` in-process (5.3 ms at 1x64), and a holdout ported to Xcode 27's Evaluations framework.

Read these before writing anything: `README.md`, `docs/coreai-ecosystem.md` (every gotcha applies here, 1–8 especially: `optimize()` for states, `scatter`/`slice_scatter` not `index_put`, explicit broadcast before comparisons, int32 `position_ids`, `set_static_shape_config` renames functions, first load writes a full weight copy to `~/Library/Caches/coreai-cache`), `snake_ai/{model,convert,verify,play,serve}.py`, and `SnakeCoreAI/Sources/SnakeCoreAI/ModelPlayer.swift` (the working `CoreAI.framework` stateful pattern: `stateDescriptor(of:)` → `NDArray(descriptor:)`, `@available(macOS 27, iOS 27, *)` on the type plus `guard #available` at call sites in `snake-cli`/`GameViewModel`, caches in locals around `MutableViews`). Gotchas 11–15 from the decide project matter too: export with a size ≥ 2 example on any `Dim`, several static configs in one `set_static_shape_config` call, and first-load specialization time that grows with the number of functions (94–102 s for decide's eight-function asset).

**Environment:** macOS 27.0, Xcode 27.0, Swift 6.4, `CoreAI.framework` available in-process. Python `coreai-core 1.0.0b2` in `.venv` (3.12) loads the **OS** runtime; `USE_LOCAL_COREAI=1` gives the old macOS 26 CPU runtime for a baseline row. M2 Mac.

**Disk is the constraint.** 8.4 GB free on 2026-09-25, after the decide entries were deleted from `~/Library/Caches/coreai-cache` (it is ~50 MB now, snake only).

How the specialization cache works (seen on disk 2026-09-25): the first load of an asset writes a full specialized copy, **one per loading program**. Python goes under `coreai-cache/<python version>/<program hash>/`. Swift goes under `coreai-cache/<OS build>/<executable name>/` (`llm-cli`, `LLMApp`, and `swiftpm-testing-helper` for `swift test`). The decide asset had a 1.7 GB copy for Python *and* another for `decide-cli`. Entries are never cleaned up when an asset is replaced.

Budget per fp16 asset: HF weights ~725 MB (once), the `.aimodel` ~0.7 GB (~1.4 GB fp32), about the same again **for each program that loads it** (Python, `llm-cli`, `LLMApp`, tests), plus ~1 GB of export temporaries. One fp16 asset loaded from three programs is ~4 GB all in.

Rules:
- Download only the safetensors shard + tokenizer/config files.
- Keep **one** `.aimodel` on disk at a time. Delete the stateless one once its numbers are recorded.
- **When an asset is deleted or re-converted, delete its cache entries too.** For Swift that's `coreai-cache/<OS build>/{llm-cli,LLMApp,swiftpm-testing-helper}`. For Python it's the hash directories whose mtime matches the first load; list `coreai-cache/<python version>/` before and after that load to learn the hash.
- A **cold** load time only counts if that program's cache entry was deleted first. Re-converting changes the asset, so the next load is cold anyway. Record in the bench JSON which it was.
- Prefer `llm-cli` for Swift numbers. Launch `LLMApp` once for the screenshot, then delete its cache entry. Don't run `swift test` against the 360M asset unless a test needs it.
- Run `df -h /System/Volumes/Data` before each conversion and before each first load from a new program. Stop and report if it's under 3 GB. Deleting LLM cache entries is fine without asking. Anything else in `coreai-cache` (snake) needs a question first.

## Branch

`llm` (3e0205a) is a scaffold branched from `main`, before any of the `system-one` work. Merge `system-one` into `llm` first (a merge, not a rebase) so the `CoreAI.framework` code, docs and bench conventions are there. Then work on `llm`. Don't touch `system-one` or `main`.

What the scaffold already has: `llm_ai/download.py` (done), `llm_ai/model.py` (only `LlamaConfig.from_hf`), `LLMCoreAI/` (`TokenGenerator` protocol, `RemoteGenerator` SSE client, `ModelGenerator` stub, `llm-cli`, `LLMApp` with a tok/s gauge). The milestone list is in `llm_ai/__init__.py`. Update milestone 6's docstring, since `CoreAI.framework` is now available.

## The model

SmolLM2-360M-Instruct is a Llama: 32 layers, hidden 960, 15 query heads / 5 KV heads (GQA, n_rep 3), head dim 64, intermediate 2560, vocab 49152, tied embeddings, RoPE θ from `config.json`, RMSNorm. Chat template from `tokenizer_config.json`. `max_seq_len = 1024` for the cache: each of K and V is `[32, 1, 5, 1024, 64]` (≈42 MB fp32, 21 MB fp16).

## Milestones

1. **`download`**: run it (`models/llm/` is empty today). Confirm size and files. `transformers` 5.17 is already in `.venv` via the `decide` extra; add it to the `llm` extra in `pyproject.toml` too, so `.[dev,llm]` alone is enough for the reference check.
2. **`model`**: plain-`torch` Llama, no `transformers` at runtime. Two modules like snake: `SmolLM` (stateless, full sequence) and `SmolLMStateful` (`register_buffer` K/V caches, `position_ids` input, cache writes via `slice_scatter`/`scatter`, causal+cache mask built by explicit expand). Load HF weights from safetensors by name, and assert every tensor is consumed. **Reference check**: load the HF model with `transformers` (already installed) and compare logits on 3 prompts (max abs diff < 1e-3 fp32). Also save a golden greedy continuation to `data/llm/golden.json` so later checks don't need `transformers` loaded. Stateless and stateful must produce the same greedy tokens for 64 steps.
3. **`convert`**: `torch.export` → `coreai_torch` → `AIProgram.optimize()` → `.aimodel`. Target: one asset whose functions share the cache states, `prefill` and `decode`. **Gotcha 7 rules out a dynamic `prefill` beside a static `decode`**: `set_static_shape_config` drops the dynamic `main`. So pass all configs in one call (gotcha 12): `prefill_t16`, `prefill_t128`, `prefill_t512` (prompt padded up to the next size, padding masked out, `position_ids` real) and `decode` `[1,1]`. Functions come out as `main_<config>` (gotcha 8); read `function_names`. **Unproven so far:** snake shipped its stateful and decode graphs as *separate* assets, so nobody has checked that two functions in one asset see the same state buffers. Test that first with a tiny 2-layer config (prefill writes, decode reads, compare against PyTorch) before converting the 360M model. If they don't share, fall back to one static function family that serves both (`decode` = the T=1 config) and say so in the gotchas. Try fp16 weights and keep them if `verify` passes. Record asset size and conversion time. The stateless asset is only for one comparison row.
4. **`verify`**: PyTorch vs Core AI (Python, OS runtime) logits on a real chat prompt: prefill, then 32 decode steps. Report max abs logit diff per step, and require the greedy tokens to match exactly (fp32), or top-1 agreement ≥ 31/32 (fp16, with the diverging step shown). Check that states are mutated in place and reset works.
5. **`play`**: stream a chat reply in the terminal and write `docs/bench/llm-python.json` (`llm-bench/1` schema, below).
6. **Swift `ModelGenerator`** on `CoreAI.framework`: prefill + decode loop, greedy sampling, EOS/`maxTokens` stop, `contextExhausted` at 1024. Tokenizing in Swift: first milestone takes pre-tokenized prompt ids from a Python helper (as `decide_ai.bench_ids` did), and does byte-level BPE **decode** in Swift (vocab + byte map, easy). A Swift BPE **encoder** is a stretch goal. `llm-cli --json` writes `docs/bench/llm-swift-coreai.json`. Launch `LLMApp` once and screenshot the gauge.
7. **`serve`** (lower priority): SSE token stream on :8770 so `RemoteGenerator` works. It exists to measure the per-token HTTP hop (`llm-swift-remote.json`), not to stand in for the framework.
8. **README + `docs/coreai-ecosystem.md`**: numbers table, new gotchas numbered after the existing ones.

## What to measure (`llm-bench/1`)

Each record carries: backend, runtime (OS / local / framework), precision, prompt tokens, generated tokens, **load ms cold and cached** (cold = empty `coreai-cache` entry), **prefill ms** (a.k.a. time-to-first-token), **decode ms/token p50/p95** and **tok/s**, peak RSS if available, machine + OS + date, git sha. Prompt lengths 16 / 128 / 512, 128 generated tokens, 5 runs each, greedy.

Comparison rows the README table should end up with:

| Row | Why |
|---|---|
| stateless (full recompute each token), Python OS runtime | the "unusable" baseline; stop at 64 tokens if it's slow |
| stateful decode, Python OS runtime | the KV-cache win |
| stateful decode, Python `USE_LOCAL_COREAI=1` | macOS 26 baseline (skip if it takes > 10 min) |
| stateful decode, Swift `CoreAI.framework` | the headline |
| Swift → `serve` over HTTP | per-token network-hop cost |
| fp32 vs fp16 (whichever pairs are affordable on disk) | precision/size/speed |

Don't try to beat numbers you've read elsewhere. Record what's measured, and note machine load (`uptime`) with each run; the Laya records were skewed by Chrome.

## Expected outcome (to check against, not to force)

- Greedy output matches the PyTorch reference; the chat reply reads coherently.
- Stateless cost grows with sequence length. Stateful decode is roughly flat, but pays for attention over all 1024 slots every step (gotcha 6), so short replies may favour stateless at very small T. Report the crossover.
- Decode is memory-bandwidth bound: about 0.7 GB of fp16 weights are read per token, so an M2 ceiling is on the order of 100+ tok/s. Anything well under 20 tok/s on the framework is worth a profiler look (`Profiler` events, ANE placement warnings) before calling it done.
- Specialization makes the first load expensive. `AIModelCache` should turn the second load into a small fraction of that.

## Hygiene

- Small commits per milestone, `Co-Authored-By` trailer as usual. Never commit weights, `.aimodel` files or caches (check `.gitignore` covers `models/llm/`).
- `pytest -m "not slow"` stays fast. Anything that loads the 360M model gets `@pytest.mark.slow`.
- If a converter/runtime op fails, reduce it to a ≤ 20-line repro in `scripts/`, record it as a numbered gotcha, and pick the smallest workaround. Don't restructure the model around a guess.
- Stop and ask before: deleting anything outside `models/llm/` and the LLM entries in `coreai-cache`, pushing, or changing `max_seq_len`.

## Done means

All milestones 1–6 and 8 checked off in the README, the bench JSONs in `docs/bench/`, the table filled with measured numbers, and the gotchas documented. Milestone 7 is either done or explicitly deferred.
