# Project 2: SmolLM2-360M on Core AI, KV cache as states (`llm` branch)

## Context

This repo (`apple-ai-inference`) ports models through the WWDC26 **Core AI** toolchain, one model per project. Two are done:

- **Snake** (`snake_ai/`, `SnakeCoreAI/`): a 118k-param transformer with its KV cache carried as Core AI **states**, stateless vs stateful vs static decode, in Python, over HTTP and in-process on `CoreAI.framework` (1.3 ms p50 per move).
- **Decide** (`decide_ai/`, `DecideCoreAI/`, branch `system-one`): a 328 MB RoBERTa NLI cross-encoder, Jev/Laya comparisons, `CoreAI.framework` in-process (5.3 ms at 1x64), and a holdout ported to Xcode 27's Evaluations framework.

Read these before writing anything: `README.md`, `docs/coreai-ecosystem.md` (every gotcha applies here, 1–8 especially: `optimize()` for states, `scatter`/`slice_scatter` not `index_put`, explicit broadcast before comparisons, int32 `position_ids`, `set_static_shape_config` renames functions, first load writes a full weight copy to `~/Library/Caches/coreai-cache`), `snake_ai/{model,convert,verify,play,serve}.py`, and `SnakeCoreAI/Sources/SnakeCoreAI/ModelPlayer.swift` (the working `CoreAI.framework` stateful pattern: `stateDescriptor(of:)` → `NDArray(descriptor:)`, `#available(macOS 27, *)`, caches in locals around `MutableViews`).

**Environment:** macOS 27.0, Xcode 27.0, Swift 6.4, `CoreAI.framework` available in-process. Python `coreai-core 1.0.0b2` in `.venv` (3.12) loads the **OS** runtime; `USE_LOCAL_COREAI=1` gives the old macOS 26 CPU runtime for a baseline row. M2 Mac.

**Disk is the constraint.** About 7.6 GB free on 2026-09-25. Budget: HF weights ~725 MB, one `.aimodel` ~0.7 GB (fp16) or ~1.4 GB (fp32), a specialization copy of the same size in `coreai-cache` on first load, ~1 GB of export temporaries. Rules: download only the safetensors shard + tokenizer/config files; keep **one** `.aimodel` on disk at a time (delete the stateless one once its numbers are recorded); run `df -h /System/Volumes/Data` before each conversion and stop and report if it's under 3 GB. `~/Library/Caches/coreai-cache` (4.4 GB) may be cleared if needed; say so when you do.

## Branch

`llm` (3e0205a) is a scaffold branched from `main`, before any of the `system-one` work. Merge `system-one` into `llm` first (a merge, not a rebase) so the `CoreAI.framework` code, docs and bench conventions are there. Then work on `llm`. Don't touch `system-one` or `main`.

What the scaffold already has: `llm_ai/download.py` (done), `llm_ai/model.py` (only `LlamaConfig.from_hf`), `LLMCoreAI/` (`TokenGenerator` protocol, `RemoteGenerator` SSE client, `ModelGenerator` stub, `llm-cli`, `LLMApp` with a tok/s gauge). The milestone list is in `llm_ai/__init__.py`. Update milestone 6's docstring, since `CoreAI.framework` is now available.

## The model

SmolLM2-360M-Instruct is a Llama: 32 layers, hidden 960, 15 query heads / 5 KV heads (GQA, n_rep 3), head dim 64, intermediate 2560, vocab 49152, tied embeddings, RoPE θ from `config.json`, RMSNorm. Chat template from `tokenizer_config.json`. `max_seq_len = 1024` for the cache: each of K and V is `[32, 1, 5, 1024, 64]` (≈42 MB fp32, 21 MB fp16).

## Milestones

1. **`download`**: run it. Confirm size and files.
2. **`model`**: plain-`torch` Llama, no `transformers` at runtime. Two modules like snake: `SmolLM` (stateless, full sequence) and `SmolLMStateful` (`register_buffer` K/V caches, `position_ids` input, cache writes via `slice_scatter`/`scatter`, causal+cache mask built by explicit expand). Load HF weights from safetensors by name, and assert every tensor is consumed. **Reference check**: if `transformers` is installable without busting disk, compare logits on 3 prompts (max abs diff < 1e-3 fp32). Otherwise compare against a golden greedy continuation generated once and saved to `data/llm/golden.json`. Stateless and stateful must produce the same greedy tokens for 64 steps.
3. **`convert`**: `torch.export` → `coreai_torch` → `AIProgram.optimize()` → `.aimodel`. One asset with two entrypoints sharing the cache states: `prefill` (dynamic T up to 512) and `decode` (static `[1,1]` via `set_static_shape_config`). Try fp16 weights and keep them if `verify` passes. Record asset size and conversion time. The stateless asset is only for one comparison row.
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
- Stop and ask before: deleting anything outside `models/llm/`, clearing `coreai-cache`, pushing, or changing `max_seq_len`.

## Done means

All milestones 1–6 and 8 checked off in the README, the bench JSONs in `docs/bench/`, the table filled with measured numbers, and the gotchas documented. Milestone 7 is either done or explicitly deferred.
