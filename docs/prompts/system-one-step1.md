# Step 1: on-device "System One" decision model on Core AI (`system-one` branch)

## Context

This repo (`apple-ai-inference`) is a set of ports of the WWDC26 **Core AI** toolchain, one model per project. Read `README.md`, `docs/coreai-ecosystem.md` (converter/runtime gotchas — every one of them applies here), and skim `snake_ai/` before writing anything: `convert.py` (torch.export → `coreai_torch.TorchConverter` → `.aimodel`, `set_static_shape_config` for a static decode function), `verify.py` (PyTorch ≙ Core AI numerics on real inputs), `serve.py` (Core AI Python runtime behind a local HTTP server so the Swift app can use the model on macOS 26), `play.py --json` (writes `docs/bench/*.json` in the `snake-bench/1` schema). `llm_ai/download.py` shows the Hugging Face download convention (allow-patterns, gitignored `models/<project>/hf/`).

Environment: macOS 26.3, Xcode 26.6, **no `CoreAI.framework`** (needs macOS 27) — Swift code against the framework goes under `#if canImport(CoreAI)` and is never compiled here. The Python runtime (`coreai.runtime`, CPU only) is what actually runs. One `.venv` at repo root, `pyproject.toml` with optional-dependency groups. Disk is nearly full (~4 GB free): keep downloads to the single safetensors shard + tokenizer files, nothing else.

**Jev is available for real measurements.** The user has Vercel AI Gateway access to `typesafe-ai/jev`. Plain HTTP, no Node needed:

```
POST https://ai-gateway.vercel.sh/v1/evaluate
Authorization: Bearer $AI_GATEWAY_API_KEY
{"model": "typesafe-ai/jev",
 "state": "I was charged twice for my subscription.",
 "questions": {"refund": {"type": "boolean", "instructions": "Is the customer asking for money back?"}}}
→ {"model": "typesafe-ai/jev",
   "answers": {"refund": {"type": "boolean", "probability": 0.98}},
   "usage": {"inputTokens": 275, "outputTokens": 20}}
```

Question types: `boolean` (→ `probability`), `choice` (`criteria: {name: description}` → `choice` + `probabilities` per option), `score` (`criteria: [ordered labels]` → `score` + `probabilities` per rung). Optional `criteria: {true: ..., false: ...}` on boolean. Docs: https://vercel.com/docs/ai-gateway/modalities/evaluation. Pricing is input-tokens-only (~$0.04/M), so running the whole holdout costs cents — but still cache responses to disk (`data/decide/jev_cache.jsonl`, keyed by request hash) so reruns are free and deterministic.

Verified from this machine on 2026-09-19: the call above with two boolean questions on a 15-word state returned `wants_refund: 0.99, about_shipping: 0.03` with `usage: {inputTokens: 304, outputTokens: 40}`. Note the token counts — Jev wraps the request in a ~250-token template and meters ~20 output tokens per question. Record `usage` in `decide-jev.json` and put Jev's `inputTokens` next to our own token count in the bench README; it's part of the cost story. Do not try to explain the numbers, just record them.

**Probe results (2026-09-19, `scripts/probe_jev.py`, 12 states × 6 boolean questions) — design around these:**
- **Rate limit: 30 requests/min per team, per region.** A 429 carries `Retry after Ns`. Every Jev caller must pace at ≤ 25 req/min (sleep ≥ 2.4 s between calls), honor `Retry-After`, and print progress, because the holdout alone is ≥ 5 min of wall-clock. Cache hits skip the sleep.
- Latency across two runs: p50 255–325 ms for 6 questions, min 224, cold 660–920 ms, and **one mid-run call at 877 ms** — 2 of 12 warm-ish calls exceeded 800 ms, so the tail is real, not just cold start. Report p50 *and* p95/max side by side; the p95 is what a user feels. Tokens ≈ 350 in / 110 out for 6 questions (≈ 18 output tokens metered per question).
- **Jev is non-deterministic**: identical requests across the two runs moved individual probabilities by up to ±0.05 (0.51→0.48, 0.05→0.10, 0.44→0.37). The local model is bit-deterministic — that's a real axis where on-device wins for threshold-based branching, so measure it: in `calibrate.py`, call Jev **twice** (`--no-cache` for the second pass) on a fixed 30-state subset of the holdout and report **repeatability** = mean and max |p₁ − p₂| per question; in `verify.py`, assert the local model's logits are bit-identical across two runs of the same batch. Treat Jev quality gaps smaller than its repeatability noise as ties.
- Jev handles the cases zero-shot NLI is weakest on: negation ("I don't want a refund, just a replacement" → refund 0.03), extraction ("Order #48213 never showed up" → has_order_number 0.99 vs 0.01 without the number), and it hedges on arguable states (0.51–0.56 on "is this a complaint" for a replacement request). ~1/3 of its probabilities are extreme, ~1/4 are in [0.3, 0.7]. Expect Jev to win on quality; the project's claim is latency / privacy / cost, and the quality table is what justifies step 4 (distillation from Jev).
- No cross-question interference: `wants_refund` asked alone matched `wants_refund` batched with five others on every state. Say so in the README — it's why a per-row NLI model is a fair comparison.
- Phrasing drift (question vs declarative instructions): median 0.09, max 0.16 — about the size of Jev's own run-to-run noise, so not a weakness to claim. Still send Jev the `instructions` (question) form, since that is what its docs use; the declarative `hypothesis` is for the local model only. Record drift in the quality doc as an observation, not a finding.

Key hygiene: read `AI_GATEWAY_API_KEY` from the environment only. Never write it to a file, a log, a commit, or a bench JSON. If it's missing, every Jev step prints a one-line skip and the rest of the pipeline still runs.

Start from `main`, create branch `system-one`. Do not touch the `llm` branch.

## What we are building and why

A **System One model** makes a fast, structured decision in one forward pass: text `state` in, calibrated probability distributions over fixed labels out. No generated tokens, so it can't hallucinate — the output space *is* the label set. TypeSafe's Jev (https://typesafe.ai/blog/introducing-system-one-models-and-jev) sells this as a hosted API at 70–500 ms. The thesis of this project: the same shape on-device through Core AI has no network floor and no data egress, so it should be an order of magnitude faster. The snake model in this repo is already a System One model (16 features → 4-way softmax, 3.8 ms); this generalizes the label set to natural-language questions.

Mechanism: **zero-shot NLI cross-encoder**. Each question becomes a hypothesis; the model scores `(state, hypothesis)` pairs for entailment / neutral / contradiction. `P(yes) = P(entail)` (or `entail / (entail + contradiction)`). N questions = one batched forward pass `[N, L]` → `[N, 3]` — that is Jev's "parallel sampling."

**Wire format: speak Jev's.** Our `/decide` endpoint accepts and returns the same JSON as `/v1/evaluate`, so one client, one bench harness and one holdout runner hit both backends with zero branching. Step 1 implements `boolean` only; reject `choice`/`score` with a 400 that names them as step 5 work (they map onto NLI too — one hypothesis per option, softmax over entailment scores — but not now).

```
POST /decide
{"model": "local/nli-minilm2",            # ignored, present for symmetry
 "state": "The pasta was cold and the waiter ignored us.",
 "questions": {"is_complaint":  {"type": "boolean", "instructions": "The message is a complaint."},
               "wants_refund":  {"type": "boolean", "instructions": "The customer is asking for a refund."}}}
→ {"model": "local/nli-minilm2",
   "answers": {"is_complaint": {"type": "boolean", "probability": 0.97},
               "wants_refund": {"type": "boolean", "probability": 0.92}},
   "usage": {"inputTokens": 31, "outputTokens": 0},
   "timing": {"ms_tokenize": 0.4, "ms_infer": 3.8, "ms_total": 4.4, "batch": 2, "padded_len": 64},
   "raw": {"is_complaint": [entail, neutral, contradiction], ...}}
```

`instructions` doubles as the NLI hypothesis. Jev's phrasing is a question ("Is a refund requested?"); NLI wants a declarative ("A refund is requested."). Store both forms in the holdout and bench inputs — `instructions` (question form, sent to Jev) and `hypothesis` (declarative, used locally) — and document in the README that this is a difference we control for, not hide. If `hypothesis` is absent, use `instructions` as-is and note it.

## Model choice

`cross-encoder/nli-MiniLM2-L6-H768` (6-layer BERT-style, ~22M transformer params + 30k WordPiece vocab, 3-way NLI head). Chosen over `cross-encoder/nli-deberta-v3-xsmall` **deliberately**: plain BERT attention lowers cleanly (no relative-position gathers — the `index_put`/gather class of op is what broke the snake KV cache), and WordPiece is ~100 lines to port to Swift in step 2 versus a SentencePiece library port. DeBERTa is a later quality upgrade, not step 1. Note the label order from `config.json` `id2label` — do not assume `[entail, neutral, contradiction]`.

## Implementation (Python package `decide_ai/`)

Follow the `snake_ai` conventions: one module per stage, module docstring shows the CLI, `argparse` `main(argv)`, `python -m decide_ai.<stage>` entry points registered in `pyproject.toml`, new optional-dependency group `decide = ["huggingface_hub", "safetensors", "tokenizers", "transformers"]` (transformers only for loading the reference model; do not import it in `serve.py`).

1. **`download.py`** — `snapshot_download` into `models/decide/hf/nli-MiniLM2-L6-H768`, allow-patterns: `config.json`, `model.safetensors` (or `pytorch_model.bin` if that's all the repo has — check first, prefer safetensors), `vocab.txt`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`. Print a summary (layers/hidden/vocab/MB). Gitignore the dir.

2. **`model.py`** — Load the checkpoint into a *self-contained* `nn.Module` (own BERT implementation reading the HF state dict, like `llm_ai/model.py` does for SmolLM2; do not export `transformers`' module directly — its graph has branches and `torch.export` dislikes it). Inputs: `input_ids [N, L]` int32, `attention_mask [N, L]` int32. Output: `logits [N, 3]`. Explicit softmax attention with the additive mask `(1 - mask) * -1e4` (check numerics of `-inf` vs large negative through the converter; the ecosystem doc says comparison kernels want identical operand shapes, so broadcast the mask explicitly to `[N, 1, 1, L]` → `[N, H, L, L]`). Assert against `transformers` `AutoModelForSequenceClassification` on 20 real pairs, atol 1e-4, in a test.

3. **`tokenize.py`** — WordPiece from `vocab.txt`: lowercase, strip accents, split punctuation, greedy longest-match with `##`. Pair encoding `[CLS] state [SEP] hypothesis [SEP]`, `token_type_ids` if the model uses them (MiniLM2 cross-encoders usually don't — check the config). Pad/truncate to a fixed `L`. Test: byte-identical ids to `tokenizers` on 200 diverse strings (unicode, numbers, empty, very long). This module is the spec for the Swift port in step 2, so keep it dependency-free and readable.

4. **`jev.py`** — Minimal `urllib` client for `/v1/evaluate`: `evaluate(state, questions, *, model="typesafe-ai/jev") -> (answers, usage, ms_roundtrip)`. Disk cache keyed by sha256 of the request body (`data/decide/jev_cache.jsonl`). Paces itself to ≤ 25 req/min (module-level timestamp of last call, sleep the remainder); on 429 parses `Retry after Ns` from the error body and sleeps that long before retrying; exponential backoff on 5xx; no retries on other 4xx. This venv's CPython is a python.org build that does not see the macOS keychain, so `urlopen` fails with `CERTIFICATE_VERIFY_FAILED` unless you pass `context=ssl.create_default_context(cafile=certifi.where())` (`certifi` is already installed). `scripts/probe_jev.py` has the working client — start from it. Exposes the same Python signature as the local decider so `calibrate.py` and `bench.py` take a `backend` argument and nothing else changes. The response's `probability` is the number we compare against; ignore `usage` except to record it.

5. **`convert.py`** — `torch.export` with `Dim("batch", 1..32)` and `Dim("seq", 1..256)` → `run_decompositions(coreai_torch.get_decomp_table())` → `TorchConverter().add_exported_program(..., input_names=["input_ids","attention_mask"], output_names=["logits"]).to_coreai()`. Produce two assets, exactly as the snake project did:
   - `models/decide/NLICrossEncoder.aimodel` — dynamic `main`.
   - `models/decide/NLICrossEncoderStatic.aimodel` — `set_static_shape_config` with enumerated shapes for `L ∈ {64, 128}` × `N ∈ {1, 4, 8, 16}` (or whatever `coreai.enumerated_shapes` supports; document what it accepts), then `optimize()`. The snake decode function dropped ~1 ms of per-call type inference this way; measure whether it does here.
   Record every converter error and workaround in `docs/coreai-ecosystem.md` under a new "BERT-style encoders" section — that doc is the deliverable as much as the code.

6. **`verify.py`** — Load both assets with `coreai.runtime.AIModel`, run the same 20 real pairs through PyTorch and Core AI, assert logits match (report max abs diff). Use real tokenized sentences with real padding — padding + mask handling is where conversions silently break. Also verify that a batch of 4 gives the same per-row logits as 4 batches of 1.

7. **`calibrate.py`** — Ship a small labeled set `data/decide/holdout.jsonl` (~150 examples you write: short review/ticket/message texts, 5 fixed questions each with both `instructions` and `hypothesis` forms, human yes/no labels — keep it honest and varied, not templated). The examples and initial labels are drafted by you, so they are not human-labeled until reviewed. Make review cheap: write `data/decide/holdout.jsonl` with one state per line and the five labels on that line, then generate `data/decide/holdout_review.md` — a markdown table, one row per state, columns = state text + the five yes/no labels — and **stop and ask the user to skim it before running `calibrate.py`**. Apply their corrections to the jsonl, add a `reviewed_by: "human"` field to each corrected line and `reviewed: true` at the top of the review file, and only then compute accuracy/ECE. Deliberately include hard cases so the set discriminates: negation ('I don't want a refund, just a replacement'), implicit urgency with no keyword ('third time I've written in'), extraction-shaped questions, and short ambiguous messages where a 0.5 is the right answer. Labels are `true` / `false` / `"unsure"`. Use `"unsure"` for the genuinely 50/50 states (a replacement request as a "complaint", "Still waiting." as "about shipping") — those rows are excluded from accuracy and ECE for that question and instead reported as **hedging**: mean |p − 0.5| over unsure rows, lower is better. Without this, hard labels punish a model for correctly saying 0.5. Labeling rule the probe tripped over: **topic questions are about topic, not sentiment** — "Package arrived a day early, thanks!" is `about_shipping: true`. Composition target: ~100 straightforward, ~50 hard, hard split roughly evenly across negation / implicit intent / extraction-shaped / ambiguous. **Do not report quality numbers on unreviewed labels.** For **each backend** (`local-dynamic`, `local-static`, `jev`) compute accuracy, a 10-bin reliability table, and **ECE**. For the local model additionally fit a single temperature `T` by NLL (split the holdout 100/50 fit/eval so `T` isn't scored on the data it was fit to) and report ECE before/after. Save `T` in `models/decide/calibration.json`; serve applies it. Output one table — rows = backend, columns = accuracy / ECE / Brier / hedging (mean |p−0.5| on unsure rows) / repeatability (mean |p₁−p₂|, 30-state subset; 0.00 for local) / latency p50 / p95 — to `docs/bench/decide-quality.md`. This table is the honest answer to "what do you give up on-device," and it is the number that makes "calibrated confidence" a measurement rather than a claim for *both* models.

8. **`serve.py`** — Same structure as `snake_ai/serve.py` (event loop on a thread, `run_coroutine_threadsafe`, lock, `/info`, quiet logging). Add `POST /decide` per the shape above and `GET /info` returning asset, function, `L`, `T`, host info. `--asset static|dynamic`, `--max-len 128`, `--port 8770`. Per-request timing must separate tokenize / pad, inference, and post-process (`ms_tokenize`, `ms_infer`, `ms_total`) so the bench can attribute cost.

9. **`bench.py`** — Writes `docs/bench/decide-python.json` in a `decide-bench/1` schema modeled on `snake-bench/1` (same `host` block, `recorded_at`, `config`). Rows: `{asset, N, L}` for `N ∈ {1, 4, 8, 16}` × `L ∈ {64, 128}`, each with `load_ms` (first / rest_mean, like snake) and `infer_ms` (count, mean, p50, p95, min, max) over ≥200 calls after 20 warm-up calls, plus tokenization ms separately. Run the same matrix through the local HTTP server → `docs/bench/decide-remote.json` (client round-trip vs server `ms_infer`, so the JSON/HTTP overhead is a number — on the snake it was ~0.5 ms). Then run the same inputs through Jev with `--backend jev --no-cache` → `docs/bench/decide-jev.json`: N ∈ {1, 4, 8} (Jev answers all questions in one call, so N is questions-per-request), ≥30 calls each after 5 warm-ups, record p50/p95/min/max round-trip and `usage`. Fewer calls than local because each one is a network request and the gateway allows 30/min — 90 paced calls is ~4 min; say so in the JSON `config`. Record the first (cold) call separately from the warm distribution, like snake's `load_ms.first`. Redact nothing but the key — hostnames and timings are fine.

## Use case for the demo / holdout data

Customer-message triage: the five fixed questions are `is_complaint`, `wants_refund`, `about_shipping`, `about_product_quality`, `urgent`. This is the "AI-powered workflow / smart conditional logic" use case from the Jev post, it maps 1:1 onto the NLI hypothesis form, and it's the same shape as an on-device intent router (which is what Apple's Siri does with App Intents). Use it for the holdout set, for the bench inputs, and for the README example. Keep the API general — questions are caller-supplied — but every example and test uses this domain so the numbers are comparable across steps.

## Swift (this step: minimal, mirrors the snake split)

New Swift package `DecideCoreAI/` with:
- `RemoteDecider` — `URLSession` client for `/decide`, measures round-trip, exposes server `ms_infer`.
- `Decider` under `#if canImport(CoreAI)`, `@available(macOS 27, *)`, written against the same API surface `SnakeCoreAI/Sources/SnakeCoreAI/ModelPlayer.swift` uses (`AIModel(contentsOf:)`, `loadFunction(named:)`, `NDArray`, `run(inputs:)`). Takes pre-tokenized ids for now — the Swift WordPiece port is step 2.
- `decide-cli` with `--bench --json docs/bench/decide-swift-remote.json`, same schema, so the bench table gets its Swift→HTTP column now and a framework column later.
No app UI this step.

## Definition of done

- `pip install -e '.[dev,decide]'`, then `python -m decide_ai.download && python -m decide_ai.convert && python -m decide_ai.verify` runs clean and prints max abs diffs.
- `pytest` passes: tokenizer parity, model-vs-transformers parity, batch-vs-single parity, `/decide` end-to-end against a live server.
- `docs/bench/decide-python.json`, `decide-remote.json`, `decide-swift-remote.json`, `decide-jev.json` exist with real numbers measured on this machine.
- `docs/bench/README.md` (create if missing) has a latency table: rows = N × L, columns = Python in-proc / Swift→HTTP / `CoreAI.framework` (blank, macOS 27) / **Jev via Vercel gateway (measured p50/p95)**. Jev's post claims 70–500 ms; put the measured number next to the claim.
- `data/decide/holdout_review.md` exists and is marked `reviewed: true` by the user before `decide-quality.md` is produced.
- `docs/bench/decide-quality.md` has the accuracy / ECE / Brier table for local-raw, local-temperature-scaled and Jev on the same 150 human-labeled examples.
- `docs/coreai-ecosystem.md` gained a BERT-encoder section with every gotcha hit.
- `README.md` project table gains row 3 (`decide_ai/`, `DecideCoreAI/`, MiniLM NLI cross-encoder, zero-shot System One decisions).
- The README quotes the quality table and states plainly where Jev wins and where MiniLM does.
- Commit in small steps (download / model+tokenizer / jev client / convert+verify / calibrate / serve+bench / swift), each with a message that says what was measured.

## Out of scope for step 1 (do not start these)

Swift WordPiece port, app UI with live confidence bars, distillation (Jev is now the obvious teacher — native probabilities, input-only pricing — but that's step 4), `choice`/`score` question types, multi-head classifier variant, DeBERTa. Those are steps 2–6.

## Working style

Bash-first for reads and edits. Run every benchmark yourself and paste the numbers in the commit message. If a converter op fails, try the rewrite (explicit broadcast, `slice_update`/`scatter_along_axis` instead of index ops, expand-both-sides before compare) before changing the model. Ask before downloading anything larger than 200 MB. Report what didn't work as plainly as what did.
