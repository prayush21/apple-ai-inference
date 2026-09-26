# Triage holdout on the Evaluations framework (Xcode 27)

A port of `python -m decide_ai.calibrate` to Apple's Evaluations framework
(WWDC26 session 299), as a trial run before the Pull app's decision
benchmarks are built on it. Source:
[`DecideCoreAI/Tests/DecideEvalTests/TriageHoldoutEval.swift`](../../../DecideCoreAI/Tests/DecideEvalTests/TriageHoldoutEval.swift).

```bash
.venv/bin/python -m decide_ai.eval_export
```

```bash
cd DecideCoreAI && rm -f ../docs/bench/eval/*.xcevalresult && swift test --filter TriageHoldout --attachments-path ../docs/bench/eval
```

The second command writes one `.xcevalresult` report per subject here
(gitignored, ~1.8 MB each, 21 s for the suite). Open them in Xcode 27's
Evaluations report to compare runs side by side. The whole suite also runs
from Xcode's Test navigator.

## What it runs

- **Dataset:** the 150 human-reviewed holdout states × 5 questions = 750
  samples, one per (state, question) cell. `decide_ai.eval_export` writes the
  states, the reviewed labels as target probabilities (true 1, false 0,
  *unsure* 0.5), the token ids the `local-dynamic` backend used, and each
  state's Jev cache key. Neither model's answers are in the dataset.
- **Subjects:** MiniLM in-process on `CoreAI.framework` (dynamic asset) under
  the three scorings `calibrate.py` reports (raw P(entail); T = 4.38; entail /
  (entail + contra)), and Jev replayed from `data/decide/jev_cache.jsonl`, the
  answers `calibrate.py` recorded on 2026-09-19/20 (no network).
- **Metrics:** per cell `correct`, `brier`, `hedging` (*unsure* cells only);
  aggregated: accuracy, Brier, hedging, ECE, AUC, yes-rate and recall pooled,
  plus accuracy / ECE / AUC / recall / hedging per question.
- **Check:** each test asserts its pooled accuracy, ECE, Brier, hedging and AUC
  are within 0.001 of the matching row in [`../decide-quality.md`](../decide-quality.md).
  All four pass, and the per-question numbers match too.

| subject | accuracy | ECE | Brier | hedging | AUC |
|---|---|---|---|---|---|
| MiniLM, CoreAI.framework, P = entail | 0.754 | 0.195 | 0.218 | 0.452 | 0.712 |
| MiniLM, CoreAI.framework, T = 4.38 | 0.723 | 0.058 | 0.185 | 0.265 | 0.748 |
| MiniLM, CoreAI.framework, entail / (entail + contra) | 0.727 | 0.077 | 0.180 | 0.390 | 0.804 |
| Jev (replayed) | 0.898 | 0.039 | 0.070 | 0.172 | 0.970 |

## What we learned about the framework

- **Any model can be the subject.** `Evaluation.subject(from:)` is plain async
  Swift returning a value, so a Core AI model, an HTTP backend or a replayed
  cache all plug in the same way. The framework is not tied to Apple's
  Foundation Models.
- **The expected value and the subject's output must be the same type.**
  Labels therefore become target probabilities, and the natural unit is one
  cell, not one state with five questions.
- **Metrics are per sample; the only dataset-level hook is
  `MetricsAggregator.custom(of:label:_:)`, and it receives one metric's
  values.** Anything that needs a prediction and its label together (AUC, ECE,
  recall, precision) has to be packed into one number. Here that is
  `2·y + p` in a metric named `pair (2y+p)`, unpacked in the aggregate. It
  works, but it shows up as raw numbers in the per-sample view.
- **`.ignore()` removes a sample from a metric's aggregates**, which gives the
  "exclude *unsure* cells" rule for free.
- **It runs as a Swift Testing trait** (`@Test(.evaluates(...))`), so
  `swift test` works from a Swift package without an Xcode project.
  `Evaluations.framework` ships in Xcode's developer frameworks (next to
  `Testing`), so it is a dev/test dependency, not something an app links.
- **Reports are attachments.** `swift test` discards them unless
  `--attachments-path` points at an existing directory, and it never
  overwrites: a rerun adds `Name-<random>.xcevalresult` beside the old one.
- Not used yet: `SampleGenerator` (synthetic samples), `ModelJudgeEvaluator`
  (LLM-as-judge for text) and `TrajectoryExpectation` (tool-call checks). These
  are what the Pull app's compose and agent layers would need.
