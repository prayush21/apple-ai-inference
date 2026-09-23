# Latency records

Every path that runs a model writes a JSON record here so the numbers survive
the terminal and can be diffed. All numbers below were measured on the same
machine, an Apple M2. Everything except the `CoreAI.framework` column ran on
macOS 26.3, Python 3.12.1, `coreai-core 1.0.0b2` (its self-contained CPU
runtime — see `docs/coreai-ecosystem.md` gotcha 13). The `CoreAI.framework`
column ran on macOS 27.0 / Xcode 27.0 (2026-09-22). On macOS 27 `coreai-core`
loads the OS framework instead, so the Python columns are a macOS 26 baseline
and a re-run today would not reproduce them (`USE_LOCAL_COREAI` would).

## Snake (`snake-bench/1`)

| File | Path |
|---|---|
| `python.json` | `python -m snake_ai.play --player all --games 5 --json` — Python in-process |
| `swift-remote.json` | `snake-cli --ai model --json` — Swift → `snake_ai.serve` over HTTP |

Per move, decode asset: 3.8 ms in-process, 6.8 ms round trip from Swift
(4.3 ms of it server-side).

## Decide (`decide-bench/1`)

| File | Path |
|---|---|
| `decide-python.json` | `python -m decide_ai.bench --backend local` — Python in-process, both assets |
| `decide-remote.json` | `python -m decide_ai.bench --backend remote` — Python → `decide_ai.serve` over HTTP |
| `decide-swift-remote.json` | `decide-cli --bench` — Swift `RemoteDecider` → `decide_ai.serve` over HTTP |
| `decide-jev.json` | `python -m decide_ai.bench --backend jev --no-cache` — TypeSafe Jev via the Vercel AI Gateway |
| `decide-laya.json` | `python -m decide_ai.bench --backend laya` — Laya (`convaiinnovations/laya`, 421M) in-process, PyTorch fp32 on the CPU |
| `decide-remote-laya.json` | `python -m decide_ai.bench --backend remote --url :8771` — Python → `decide_ai.serve --backend laya` over HTTP |
| `decide-swift-remote-laya.json` | `decide-cli --bench --url :8771` — Swift `RemoteDecider` → `decide_ai.serve --backend laya` over HTTP |
| `decide-swift-coreai-dynamic.json` | `decide-cli --bench --model ../models/decide/NLICrossEncoder.aimodel` — Swift `Decider` in-process on `CoreAI.framework`, pre-tokenized inputs from `python -m decide_ai.bench_ids` |
| `decide-swift-coreai-static.json` | the same on `NLICrossEncoderStatic.aimodel` (the column in the table) |

Rows are N questions per request × padded length L (real holdout states,
`data/decide/bench_questions.json`). Local / HTTP rows: 20 warm-ups, then
up to 200 timed calls capped at 60 s wall-clock per row (never fewer than
30; `count` in each record says how many ran). Jev rows: 5 warm-ups, 30
timed live calls per N, paced at 25 req/min because the gateway allows 30
(~4 min for the matrix); Jev answers all N questions in one call and has
no L. Laya rows: 5 warm-ups, then up to 200 timed in-process calls capped
at 60 s per row; Laya also answers all N questions in one forward pass
and pads to its own sequence (max 512), so it has no L either. Cells are
**p50 / p95 ms**.

| N × L | Python in-proc, dynamic | Python in-proc, static | Python→HTTP round trip | Swift→HTTP round trip | `CoreAI.framework` (Swift, in-proc, static) | Jev via gateway | Laya (PyTorch, CPU) |
|---|---|---|---|---|---|---|---|
| 1 × 64 | 293 / 392 | 219 / 269 | 281 / 336 | 218 / 255 | **5.3 / 12.0** | 278 / 381 | 123 / 133 |
| 1 × 128 | 480 / 730 | 425 / 494 | 533 / 595 | 432 / 489 | **12.0 / 18.3** | — | — |
| 4 × 64 | 836 / 970 | 838 / 967 | 811 / 876 | 847 / 1026 | **15.7 / 21.7** | 280 / 556 | 243 / 271 |
| 4 × 128 | 1692 / 2155 | 1711 / 2196 | 1617 / 1718 | 1704 / 1991 | **26.2 / 28.8** | — | — |
| 8 × 64 | 1641 / 1909 | 1795 / 2517 | 1607 / 1766 | 1726 / 1942 | **25.4 / 27.1** | 255 / 482 | 422 / 478 |
| 8 × 128 | 3404 / 4134 | 4473 / 5444 | 3240 / 3376 | 3258 / 3519 | **47.8 / 50.4** | — | — |
| 16 × 64 | 3325 / 3507 | 4290 / 5216 | 3418 / 4491 | 3288 / 3551 | **47.3 / 49.4** | — | 854 / 954 |
| 16 × 128 | 6797 / 7489 | 8815 / 10321 | 6604 / 6879 | 6630 / 6899 | **93.1 / 95.5** | — | — |

**`CoreAI.framework` column.** The Swift `Decider` in-process, release build,
default specialization (the framework picks the compute unit; which one was
not recorded). Inputs are the same holdout states × bench questions,
tokenized once by `python -m decide_ai.bench_ids` because there is no Swift
BPE until step 2, so a cell is `Decider.logits`: NDArray build + run +
reading the logits, no tokenize. Before timing, each asset is checked against
the Python Core AI logits for the first state × 16 questions (max |diff|
6e-7 dynamic, 3e-6 static). 20 warm-ups then 200 timed calls per row (every
row hit the cap; the load average was 3.6–4.3). The dynamic asset reads within noise of
static: p50 / p95 4.9 / 9.4, 9.3 / 15.3, 13.3 / 15.1, 27.5 / 29.9, 26.6 /
28.9, 47.5 / 52.1, 50.0 / 51.5, 95.4 / 100.8 in table order. **Load:** the
first `AIModel(contentsOf:)` ever for an asset specializes it — 2.2 s for
dynamic, **94.5 s (Python) and 101.8 s (Swift) for the eight-function static
asset**, each runtime paying it once — and after that it is 10–12 ms from a new
process and ~1 ms within one; `loadFunction(named:)` is 1–14 ms per
function.

Laya over HTTP (same server shape as the local columns, `decide_ai.serve
--backend laya` on :8771), round trip p50 / p95 ms and the HTTP+JSON
overhead at p50: Python→HTTP 130 / 141 (0.8 ms), 255 / 285 (0.8),
448 / 503 (0.9), 902 / 985 (0.9); Swift→HTTP 135 / 142 (1.5 ms),
265 / 303 (1.6), 449 / 533 (1.6), 897 / 1039 (1.6) for N = 1 / 4 / 8 / 16.

**Read every Laya record with its `loadavg_1m` field.** All three Laya
records above were taken on a quiet machine (load average ~3, Chrome
closed) and have a p95/p50 of ~1.1. Earlier the same day, with a load
average of 4–8 from other applications on the same 8-core M2, three
in-process runs gave p50 155 / 918 / 926 / 2194, 289 / 456 / 716 / 1278
and 135 / 322 / 990 / 1225 with p95s up to 6 s, and the HTTP columns
read 630–977 ms at N = 8 — the per-row minimums never moved, so the
421M model's floor is stable but it is far more sensitive to a busy
machine than the 82M one. Load: 35–55 s (fp16 safetensors → fp32
module), first call after load 0.2–1.3 s.

Jev's post claims 70–500 ms; measured from this machine the warm round
trip is p50 255–280 ms with a p95 of 380–560 ms, and the first (cold) call
of each run was 0.9–2.3 s. One of the 105 live calls hit a 30 s read
timeout and was retried (recorded as `transient_errors_retried`).

Reading the table:

- **The HTTP hop is free.** Python→HTTP and Swift→HTTP round trips are
  within 0.7–4.6 ms of the server's own `ms_infer` at every shape; the
  Swift column at 1 × 64 (218 ms) is lower than the Python in-process
  dynamic number (293) only because the server runs the static asset.
- **Cost is linear in N × L**, so pad to the smallest enumerated L that
  fits — 64 covers every message in the holdout — and the 5-question
  triage request lands in the 8 × 64 row (~1.7 s here).
- **Static vs dynamic** is a 25 % win at 1 × 64 (219 vs 293 ms: no
  per-call type inference) and noise elsewhere; the static half of the
  matrix ran second, on a warmer machine, and reads slower at N ≥ 8.
- **On `CoreAI.framework` the on-device path wins at every shape.** Same
  model, same asset: 41× faster than the interim CPU runtime at 1 × 64
  (5.3 vs 219 ms) and 91× at 16 × 64 (47 vs 4290 ms). The 5-question triage
  request (8 × 64) drops from ~1.7 s to 25 ms, 10× under Jev's 255 ms; a
  16-question request at L = 128 (93 ms) is still 3× under one Jev call.
  Against Laya in PyTorch it is 23× at N = 1 and 18× at N = 16. Cost is
  still close to linear in N × L (~3 ms per 64-token row). The macOS 26
  explanation holds: that runtime's matmul ran at ~25 GFLOP/s where PyTorch
  does the forward in 29 ms. Also true on both runtimes: no network floor,
  bit-identical outputs, nothing leaves the machine. p95/p50 is 1.03–1.1 at
  N ≥ 4 except 4 × 64 (1.4); at N = 1 it is 1.5–2.3, where a 5–12 ms call feels any scheduling hiccup.
- **Laya beats the local runtime at every N** (123 / 243 / 422 / 854 ms
  vs the static asset's 219 / 838 / 1795 / 4290 at L = 64), in plain
  PyTorch, with 421M parameters against MiniLM's 82M — which says more
  about the interim runtime's matmul than about Laya. Against Jev it wins
  at N = 1 (123 vs 278 ms p50) and N = 4 (243 vs 280), loses at N = 8
  (422 vs 255), and its p95 is within 10 % of its p50 where Jev's is
  1.4–2×. Cost is close to linear in questions (~50 ms per question past
  the first). Like the local
  path it has no network floor and is bit-deterministic; unlike it, it
  needs ~1.7 GB of RAM as fp32 and 35–55 s to load. Its 5-question triage
  request ran at p50 311–350 ms in `calibrate.py` (busy machine).
- **Tokens.** Jev meters ~250 template tokens plus the state and ~18 output
  tokens per question (284 / 328 / 382 in, 23 / 77 / 150 out for 1 / 4 / 8
  questions); the local request is the raw pair length, 30–60 tokens per
  question, and no output tokens; Laya's `usage.input_tokens` is 43 / 181 /
  362 / 745 for 1 / 4 / 8 / 16 questions (its own `[CLS] type question
  [SEP] [MASK] opt … [SEP] state [SEP]` sequence per question), 0 out.
  Recorded, not explained.

Quality on the human-reviewed holdout is in `decide-quality.md`.
