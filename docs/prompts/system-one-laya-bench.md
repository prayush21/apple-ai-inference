# Step 1b: add Laya as a third backend to the Decide benchmark (`system-one` branch)

## Context

Read `docs/prompts/system-one-step1.md` first — it is the spec for everything in
`decide_ai/`, and this step extends it without changing its shape. Then read
`README.md` "Project 3", `docs/bench/README.md`, `decide_ai/decider.py`
(`LocalDecider` / `JevDecider` share one `evaluate(state, questions)` signature),
`decide_ai/calibrate.py` and `decide_ai/bench.py` (both take `--backend`).

The comparison today is two backends: our zero-shot NLI cross-encoder
(`cross-encoder/nli-MiniLM2-L6-H768`, ~22M params, on the interim Core AI CPU
runtime) and TypeSafe's hosted Jev via the Vercel AI Gateway. Both are measured
on this machine. This step adds a third: **Laya** (`convaiinnovations/laya`,
Apache 2.0), an open-weight reimplementation of Jev's "System One" design —
ModernBERT-large encoder (395M) + a 2-layer decision head, 421M params, trained
with RL against strictly proper scoring rules, answers every question in one
forward pass with no generated tokens. Its model card benchmarks against Jev
throughout, **but every Jev number in it is quoted from third parties, never
measured** ("no TypeSafe API access"). We have measured Jev. Putting Laya on the
same 150-state holdout and the same latency matrix, on the same machine, makes
ours the only three-way comparison that is actually measured.

Environment is unchanged from step 1: macOS 26.3, no `CoreAI.framework`,
Python 3.12 `.venv` at repo root, `transformers 5.17.0` / `torch 2.12.1`
(ModernBERT imports fine). **Disk is now ~1.9 GB free.** Do not download
anything else and do not convert Laya to an `.aimodel` in this step — the
weights are fp16 on disk (843 MB) and an fp32 `.aimodel` would be ~1.7 GB,
which does not fit. Laya runs in plain PyTorch on CPU; that is the honest
comparison anyway (it's what a user without a T4 gets).

## What is already on disk and verified (2026-09-20)

`models/decide/hf/laya/` (gitignored via `models/decide/hf/`) holds the
English root checkpoint only — `model.safetensors` (843 MB, 205 F16 tensors
+ 1 F32, 421M params), `encoder/config.json` (ModernBERT-large: 28 layers,
hidden 1024, RoPE, `global_attn_every_n_layers: 3`, sliding-window elsewhere),
`tokenizer/` (HF fast tokenizer, 50k vocab), `rl_agent_config.json`,
`rl_agent_api.py`, `rl_common.py`, `email_utils.py`, `eval/results.{json,md}`,
`README.md`. The multilingual and typed-decisions checkpoints were **not**
downloaded (another 1.5 GB).

`rl_agent_api.py` is the whole inference surface and needs no `pip install laya`:

```python
import sys; sys.path.insert(0, "models/decide/hf/laya")
from rl_agent_api import RLAgent
agent = RLAgent("models/decide/hf/laya", device="cpu")          # 63 s load on M2 (fp16 -> fp32)
r = agent.system_one(state, {"wants_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}})
# r == {"model": "rl-agent",
#       "answers": {"wants_refund": {"type": "noul", "noul": 0.88, "rl_agent": {"act_probability": ...}}},
#       "usage": {"input_tokens": 262, "output_tokens": 0}}
```

Facts to design around, all measured here:

- `system_one(state, questions)` takes **Jev's request shape** except the
  boolean type is spelled `noul` (yes / no / unknown; `noul` = P(yes)). So a
  `LayaDecider` is a ~30-line wrapper: map `boolean -> noul` on the way in,
  `noul -> probability` on the way out, and it is interchangeable with
  `JevDecider` / `LocalDecider`. Send it the `instructions` (question) form,
  same as Jev — that is what it was trained on; `hypothesis` is for NLI only.
- `build_model(cfg, encoder_dir=...)` builds ModernBERT from the local config
  with `AutoModel.from_config` — offline, no base-model download. Keep it that
  way.
- It applies its own per-question-type temperature from `rl_agent_config.json`
  (`noul:2 -> 1.98`). That is the model's shipped calibration; report it as-is
  first, then optionally refit a single `T` on our 100/50 split exactly as
  `calibrate.py` does for the local model, and show both. Do not silently
  re-temperature it.
- Warm CPU latency for the 5-question triage request: **1.1–1.4 s per call**
  (PyTorch sdpa, M2). First call 1.4 s. Load 63 s. Compare: Jev ~270 ms p50,
  our MiniLM static at 8×64 ~1.7 s on the CPU runtime.
- Output is deterministic (no sampling); assert bit-identical probabilities
  across two calls of the same request, like `verify.py` does for the local
  model, and put `repeatability = 0.00` in the table with that as the evidence.
- **Quality is not a foregone conclusion.** On the probe state
  `"I don't want a refund, just send a replacement. Order #48213 never showed up."`
  Laya said `wants_refund: 0.88, about_shipping: 0.22, is_complaint: 0.87,
  urgent: 0.83`. Jev's step-1 probe on the same negation gave `wants_refund 0.03`.
  Laya's own `eval/results.md` says zero-shot task families run at 0.65 accuracy
  / ECE 0.20 vs 0.75 / 0.03 in-task, and the card admits "hand-picked examples
  work, real traffic does not". Our holdout is a zero-shot family for it. Let the
  table say whatever it says.
- `usage.input_tokens` (262 for 5 questions on a 15-word state, vs Jev's ~380
  metered and our ~30–60 per question) — record it in the same column as the
  others, no explanation.

## Build

1. **`decide_ai/laya.py`** — `LayaDecider` with the same `evaluate(state,
   questions) -> (answers, usage, timing, raw)` contract as `LocalDecider`.
   Lazy-load the agent on first call (63 s), `sys.path` the model dir rather
   than vendoring `rl_common.py`. `timing` = `ms_total` from a `perf_counter`
   around `system_one` (there is no tokenize/infer split exposed; say so).
   `raw` = the `noul` value and `act_probability`. Reject `choice`/`score`
   with the same 400-style error the local decider uses (Laya supports them,
   but step 1 is boolean-only and the holdout has no labels for them — note it
   as the step-5 opening). `--hf-dir` default `models/decide/hf/laya`.
   Module docstring shows the CLI like every other stage.
2. **`calibrate.py --backend laya`** — same accuracy / ECE / Brier / hedging /
   repeatability rows, on the same reviewed holdout, plus a temperature refit
   row. It's ~150 states × 1.2 s ≈ 3 min; print progress.
3. **`bench.py --backend laya`** — same `N ∈ {1, 4, 8, 16} × L` matrix through
   `LayaDecider` in-process, ≥ 30 timed calls after 5 warm-ups (it is ~1 s a
   call, so the 200-call budget is wrong here — cap at 60 s wall-clock per row
   like the local rows and record `count`). Laya pads internally to its own
   sequence, so L is not ours to set: run N only, put `—` in the L cells, and
   say so in the JSON `config`, exactly as the Jev rows do. Write
   `docs/bench/decide-laya.json` in `decide-bench/1` with a `model` block
   (`repo`, `params`, `dtype_on_disk`, `torch`, `transformers`, `device`).
   Record `load_ms.first` separately — 63 s is part of the story.
4. **`serve.py --backend laya`** (optional, do last) — same `/decide`
   endpoint, so `decide-cli --bench` can hit it and the Swift→HTTP column
   exists for all three. Skip if time is short; say so.
5. **Docs** — `docs/bench/README.md` gets a "Laya (PyTorch, CPU)" column in
   the latency table, `decide-quality.md` gets a Laya row (and the
   temperature-refit row), and the README "What we measured" paragraph
   states in one sentence each where Jev, Laya and MiniLM win. Add a short
   "Laya" subsection under Project 3: what it is, that it is the open
   stand-in for Jev's architecture, that its Jev numbers are unmeasured and
   ours are, and that its 421M fp32 footprint is why there is no `.aimodel`
   column for it yet. `pyproject.toml`: nothing new to install; if you need
   anything, stop and ask (disk).
6. **Tests** — one `pytest` that skips when `models/decide/hf/laya` is
   absent: `boolean -> noul` mapping, the `answers` shape matches
   `JevDecider`'s, two calls bit-identical. Mark it `slow` (load is 63 s).

## Definition of done

- `python -m decide_ai.calibrate --backend laya` and `python -m decide_ai.bench --backend laya` run clean; `pytest -m "not slow"` still passes; the slow test passes once.
- `docs/bench/decide-laya.json` exists with real numbers from this machine.
- `docs/bench/README.md` latency table and `docs/bench/decide-quality.md` quality table each have a Laya column/row, measured on the same inputs as Jev and MiniLM.
- README states plainly, with numbers, where each of the three backends wins.
- Commit in small steps (decider / calibrate / bench / docs), each message carrying the numbers measured.

## Out of scope

Converting Laya to Core AI (disk; also ModernBERT's RoPE + sliding-window
attention + unpadding is a converter project of its own — note it in
`docs/coreai-ecosystem.md` as future work, don't start it), the multilingual
and typed-decisions checkpoints, `choice`/`score`, fine-tuning or distilling
from Laya, any new download.

## Working style

Same as step 1: Bash-first, run every benchmark yourself and paste numbers
into commit messages, report what didn't work as plainly as what did,
`AI_GATEWAY_API_KEY` from the environment only (Jev rows are already recorded;
do not re-run them). If the quality gate (`holdout_review.md` `reviewed: true`)
is still unmet when you get to `calibrate.py`, stop and ask — do not report
quality numbers on unreviewed labels for Laya either.
