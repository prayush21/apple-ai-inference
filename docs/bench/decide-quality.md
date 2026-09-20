# decide quality — 150 human-reviewed triage states × 5 questions

Recorded 2026-09-20T22:03:58+00:00. States by category: straightforward 100, negation 13, implicit 13, extraction 12, ambiguous 12.
Accuracy / ECE / Brier exclude cells labeled *unsure*; **hedging** is mean |p − 0.5| on those cells (lower is better). **Repeatability** is mean / max |p₁ − p₂| over a fixed 30-state subset scored twice. Latency is the per-state call (5 questions) as measured in this run; Jev's from the first live pass; Laya's is PyTorch fp32 on the CPU. Accuracy is at p ≥ 0.5 — read it against the per-question yes-rate below, because a model that never says yes scores the base rate. AUC is threshold-free (pooled over all 5 questions' cells).

| backend | accuracy | ECE | Brier | hedging | AUC | repeatability mean / max | latency p50 / p95 ms |
|---|---|---|---|---|---|---|---|
| local-dynamic (raw, P=entail) | 0.754 | 0.195 | 0.218 | 0.452 | 0.712 | 0.00 / 0.00 | 524 / 603 |
| local-dynamic (T=4.38, P=entail) | 0.723 | 0.058 | 0.185 | 0.265 | 0.748 | 0.00 / 0.00 | 524 / 603 |
| local-dynamic (raw, P=entail/(entail+contra)) | 0.727 | 0.077 | 0.180 | 0.390 | 0.804 | 0.00 / 0.00 | 524 / 603 |
| local-static (raw, P=entail) | 0.754 | 0.195 | 0.218 | 0.452 | 0.712 | 0.00 / 0.00 | 1658 / 1972 |
| local-static (T=4.38, P=entail) | 0.723 | 0.058 | 0.185 | 0.265 | 0.748 | 0.00 / 0.00 | 1658 / 1972 |
| local-static (raw, P=entail/(entail+contra)) | 0.727 | 0.077 | 0.180 | 0.390 | 0.804 | 0.00 / 0.00 | 1658 / 1972 |
| jev | 0.898 | 0.039 | 0.070 | 0.172 | 0.970 | 0.01 / 0.06 | 287 / 464 |
| laya (shipped T=1.98) | 0.824 | 0.053 | 0.134 | 0.348 | 0.861 | 0.00 / 0.00 | 350 / 662 |
| laya (refit T=3.15) | 0.824 | 0.036 | 0.132 | 0.292 | 0.861 | 0.00 / 0.00 | 350 / 662 |

## Per question (accuracy / ECE / hedging)

| backend | is_complaint | wants_refund | about_shipping | about_product_quality | urgent |
|---|---|---|---|---|---|
| local-dynamic (raw, P=entail) | 0.40 / 0.57 / 0.50 | 0.95 / 0.03 / — | 0.86 / 0.07 / 0.43 | 0.69 / 0.28 / — | 0.87 / 0.11 / 0.48 |
| local-dynamic (T=4.38, P=entail) | 0.40 / 0.40 / 0.39 | 0.93 / 0.12 / — | 0.73 / 0.21 / 0.23 | 0.69 / 0.13 / — | 0.87 / 0.12 / 0.29 |
| local-dynamic (raw, P=entail/(entail+contra)) | 0.58 / 0.21 / 0.50 | 0.97 / 0.05 / — | 0.81 / 0.05 / 0.38 | 0.63 / 0.11 / — | 0.64 / 0.12 / 0.33 |
| local-static (raw, P=entail) | 0.40 / 0.57 / 0.50 | 0.95 / 0.03 / — | 0.86 / 0.07 / 0.43 | 0.69 / 0.28 / — | 0.87 / 0.11 / 0.48 |
| local-static (T=4.38, P=entail) | 0.40 / 0.40 / 0.39 | 0.93 / 0.12 / — | 0.73 / 0.21 / 0.23 | 0.69 / 0.13 / — | 0.87 / 0.12 / 0.29 |
| local-static (raw, P=entail/(entail+contra)) | 0.58 / 0.21 / 0.50 | 0.97 / 0.05 / — | 0.81 / 0.05 / 0.38 | 0.63 / 0.11 / — | 0.64 / 0.12 / 0.33 |
| jev | 0.89 / 0.06 / 0.03 | 0.99 / 0.10 / — | 0.95 / 0.04 / 0.21 | 0.94 / 0.07 / — | 0.73 / 0.07 / 0.17 |
| laya (shipped T=1.98) | 0.73 / 0.12 / 0.50 | 0.94 / 0.06 / — | 0.89 / 0.04 / 0.35 | 0.74 / 0.15 / — | 0.82 / 0.06 / 0.19 |
| laya (refit T=3.15) | 0.73 / 0.11 / 0.50 | 0.94 / 0.11 / — | 0.89 / 0.07 / 0.28 | 0.74 / 0.07 / — | 0.82 / 0.06 / 0.12 |

## Per question, threshold diagnostics (yes-rate · recall · precision · AUC; base rate of *no* in the header)

| backend | is_complaint (no 0.40) | wants_refund (no 0.86) | about_shipping (no 0.68) | about_product_quality (no 0.69) | urgent (no 0.87) |
|---|---|---|---|---|---|
| local-dynamic (raw, P=entail) | 0.00 · 0.00 · — · 0.59 | 0.11 · 0.71 · 0.94 · 1.00 | 0.23 · 0.65 · 0.88 · 0.93 | 0.00 · 0.00 · — · 0.73 | 0.01 · 0.00 · 0.00 · 0.70 |
| local-dynamic (T=4.38, P=entail) | 0.00 · 0.00 · — · 0.67 | 0.07 · 0.48 · 1.00 · 1.00 | 0.04 · 0.13 · 1.00 · 0.93 | 0.00 · 0.00 · — · 0.73 | 0.00 · 0.00 · — · 0.77 |
| local-dynamic (raw, P=entail/(entail+contra)) | 0.32 · 0.42 · 0.77 · 0.72 | 0.14 · 0.90 · 0.90 · 0.99 | 0.45 · 0.91 · 0.64 · 0.92 | 0.47 · 0.66 · 0.44 · 0.72 | 0.46 · 0.89 · 0.25 · 0.81 |
| local-static (raw, P=entail) | 0.00 · 0.00 · — · 0.59 | 0.11 · 0.71 · 0.94 · 1.00 | 0.23 · 0.65 · 0.88 · 0.93 | 0.00 · 0.00 · — · 0.73 | 0.01 · 0.00 · 0.00 · 0.70 |
| local-static (T=4.38, P=entail) | 0.00 · 0.00 · — · 0.67 | 0.07 · 0.48 · 1.00 · 1.00 | 0.04 · 0.13 · 1.00 · 0.93 | 0.00 · 0.00 · — · 0.73 | 0.00 · 0.00 · — · 0.77 |
| local-static (raw, P=entail/(entail+contra)) | 0.32 · 0.42 · 0.77 · 0.72 | 0.14 · 0.90 · 0.90 · 0.99 | 0.45 · 0.91 · 0.64 · 0.92 | 0.47 · 0.66 · 0.44 · 0.72 | 0.46 · 0.89 · 0.25 · 0.81 |
| jev | 0.58 · 0.89 · 0.92 · 0.97 | 0.14 · 0.95 · 0.95 · 0.99 | 0.33 · 0.93 · 0.90 · 0.98 | 0.32 · 0.91 · 0.90 · 0.98 | 0.33 · 0.74 · 0.29 · 0.88 |
| laya (shipped T=1.98) | 0.37 · 0.58 · 0.95 · 0.90 | 0.17 · 0.90 · 0.73 · 0.98 | 0.25 · 0.72 · 0.92 · 0.96 | 0.09 · 0.23 · 0.79 · 0.78 | 0.20 · 0.58 · 0.37 · 0.76 |
| laya (refit T=3.15) | 0.37 · 0.58 · 0.95 · 0.90 | 0.17 · 0.90 · 0.73 · 0.98 | 0.25 · 0.72 · 0.92 · 0.96 | 0.09 · 0.23 · 0.79 · 0.78 | 0.20 · 0.58 · 0.37 · 0.76 |

## Reliability (10 bins over confidence = max(p, 1−p); count · mean confidence · accuracy)

**local-dynamic (raw, P=entail)**  
| bin | 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| n | 0 | 0 | 0 | 0 | 0 | 15 | 20 | 31 | 50 | 628 |
| conf | — | — | — | — | — | 0.56 | 0.65 | 0.75 | 0.86 | 0.99 |
| acc | — | — | — | — | — | 0.20 | 0.60 | 0.65 | 0.76 | 0.78 |

**local-dynamic (T=4.38, P=entail)**  
| bin | 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| n | 0 | 0 | 0 | 0 | 0 | 50 | 58 | 223 | 413 | 0 |
| conf | — | — | — | — | — | 0.55 | 0.66 | 0.76 | 0.84 | — |
| acc | — | — | — | — | — | 0.38 | 0.38 | 0.70 | 0.83 | — |

**local-dynamic (raw, P=entail/(entail+contra))**  
| bin | 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| n | 0 | 0 | 0 | 0 | 0 | 95 | 122 | 119 | 142 | 266 |
| conf | — | — | — | — | — | 0.55 | 0.65 | 0.75 | 0.85 | 0.96 |
| acc | — | — | — | — | — | 0.49 | 0.57 | 0.64 | 0.75 | 0.91 |

**local-static (raw, P=entail)**  
| bin | 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| n | 0 | 0 | 0 | 0 | 0 | 15 | 20 | 31 | 50 | 628 |
| conf | — | — | — | — | — | 0.56 | 0.65 | 0.75 | 0.86 | 0.99 |
| acc | — | — | — | — | — | 0.20 | 0.60 | 0.65 | 0.76 | 0.78 |

**local-static (T=4.38, P=entail)**  
| bin | 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| n | 0 | 0 | 0 | 0 | 0 | 50 | 58 | 223 | 413 | 0 |
| conf | — | — | — | — | — | 0.55 | 0.66 | 0.76 | 0.84 | — |
| acc | — | — | — | — | — | 0.38 | 0.38 | 0.70 | 0.83 | — |

**local-static (raw, P=entail/(entail+contra))**  
| bin | 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| n | 0 | 0 | 0 | 0 | 0 | 95 | 122 | 119 | 142 | 266 |
| conf | — | — | — | — | — | 0.55 | 0.65 | 0.75 | 0.85 | 0.96 |
| acc | — | — | — | — | — | 0.49 | 0.57 | 0.64 | 0.75 | 0.91 |

**jev**  
| bin | 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| n | 0 | 0 | 0 | 0 | 0 | 54 | 57 | 87 | 109 | 437 |
| conf | — | — | — | — | — | 0.54 | 0.65 | 0.74 | 0.85 | 0.96 |
| acc | — | — | — | — | — | 0.50 | 0.68 | 0.77 | 0.92 | 1.00 |

**laya (shipped T=1.98)**  
| bin | 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| n | 0 | 0 | 0 | 0 | 0 | 40 | 46 | 100 | 160 | 398 |
| conf | — | — | — | — | — | 0.55 | 0.65 | 0.75 | 0.86 | 0.96 |
| acc | — | — | — | — | — | 0.62 | 0.63 | 0.66 | 0.82 | 0.91 |

**laya (refit T=3.15)**  
| bin | 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| n | 0 | 0 | 0 | 0 | 0 | 62 | 118 | 167 | 224 | 173 |
| conf | — | — | — | — | — | 0.55 | 0.66 | 0.75 | 0.84 | 0.98 |
| acc | — | — | — | — | — | 0.65 | 0.64 | 0.82 | 0.87 | 0.97 |

## Notes

- local-dynamic: T = 4.38 fit by NLL on 100 states; on the 50 held-out states ECE 0.199 → 0.069, Brier 0.222 → 0.188, accuracy 0.759 → 0.715.
- local-static: T = 4.38 fit by NLL on 100 states; on the 50 held-out states ECE 0.199 → 0.069, Brier 0.222 → 0.188, accuracy 0.759 → 0.715.
- Jev repeatability per question (mean / max |p₁ − p₂|): is_complaint 0.01 / 0.06, wants_refund 0.00 / 0.03, about_shipping 0.01 / 0.04, about_product_quality 0.01 / 0.06, urgent 0.01 / 0.02. Quality gaps smaller than ~0.01 are ties.
- Jev latency: 150 of 150 first-pass calls came from the cache (their ms is the latency recorded when the entry was made).
- laya: shipped noul temperature 1.98 (rl_agent_config.json) is what the first row reports; refit T = 3.15 by NLL on 100 states; on the 50 held-out states ECE 0.056 → 0.066, Brier 0.129 → 0.129, accuracy 0.831 → 0.831. Load 36 s (fp16 safetensors → fp32, PyTorch CPU).
- Phrasing: Jev is sent the question form (`instructions`), the local model the declarative `hypothesis`. The 2026-09-19 probe measured question-vs-declarative drift on Jev at median 0.09, max 0.16 — about its run-to-run noise — so this is recorded as an observation, not a finding.
- No cross-question interference on Jev (a question asked alone matched the same question batched with five others on every probe state), so scoring one (state, hypothesis) row per question locally is a like-for-like comparison.
