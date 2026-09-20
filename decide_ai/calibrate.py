"""Quality and calibration of every backend on the human-reviewed holdout.

    python -m decide_ai.calibrate                       # -> docs/bench/decide-quality.md, models/decide/calibration.json
    python -m decide_ai.calibrate --backends local-static jev
    python -m decide_ai.calibrate --dry-run             # unreviewed labels: print, write nothing
    python -m decide_ai.calibrate --no-cache            # force live Jev calls (both passes)

Refuses to write quality numbers unless ``data/decide/holdout_review.md``
starts with ``reviewed: true``.

For each backend (``local-dynamic``, ``local-static``, ``jev``) every holdout
state is scored on the five questions in one request (the local model runs
the five hypotheses as one batch), then per question and pooled:

* **accuracy** at p >= 0.5 and **Brier** on rows labeled true / false;
* **ECE** with 10 equal-width bins, plus the reliability table;
* **hedging** = mean |p - 0.5| on rows labeled ``"unsure"`` (lower is better:
  those are the states where 0.5 is the right answer);
* **repeatability** = mean / max |p1 - p2| over a fixed 30-state subset
  scored twice. Jev's second pass is live (cached separately in
  ``data/decide/jev_repeat_cache.jsonl`` so reruns are free); the local model
  is asserted bit-identical, 0.00 by construction;
* **latency** p50 / p95 of the per-state call as measured in this run.

Local extras: the holdout is split 100 / 50 (fit / eval, by ``id % 3``), a
single temperature ``T`` is fit by NLL on the fit split and ECE is reported
before / after on the eval split. ``T`` is saved to
``models/decide/calibration.json`` and ``serve.py`` applies it. The
``entail_vs_contra`` scoring (renormalise over entailment + contradiction,
ignoring neutral) is reported as its own row.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .decider import CALIBRATION_PATH, LABELS, make_backend, probabilities
from .holdout import QUESTION_KEYS, load_holdout, load_questions, review_is_approved
from .jev import JevUnavailable

QUALITY_PATH = Path("docs/bench/decide-quality.md")
REPEAT_CACHE = Path("data/decide/jev_repeat_cache.jsonl")
BACKENDS = ["local-dynamic", "local-static", "jev"]
N_BINS = 10


# --------------------------------------------------------------------------- scoring

def repeat_subset(rows: list[dict]) -> list[dict]:
    """Fixed 30-state subset: every fifth id, which samples every category."""
    return [r for r in rows if r["id"] % 5 == 0]


def score_backend(backend, rows: list[dict], questions: dict, *, label: str) -> dict:
    """Score every row. Returns {id: {"p": {q: p}, "raw": {q: [3]}, "ms": float}}."""
    out = {}
    t_start = time.perf_counter()
    for i, r in enumerate(rows):
        answers, _, timing, raw = backend.evaluate(r["state"], questions)
        out[r["id"]] = {"p": {q: answers[q]["probability"] for q in QUESTION_KEYS}, "raw": raw,
                        "ms": timing["ms_total"], "cached": timing.get("cached", False)}
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"  {label}: {i + 1}/{len(rows)} states, {time.perf_counter() - t_start:.0f} s", flush=True)
    return out


def metrics(labels: list, probs: list[float]) -> dict:
    """labels: True / False / 'unsure' aligned with probs."""
    hard = [(p, y) for p, y in zip(probs, labels) if y == "unsure"]
    scored = [(p, 1.0 if y is True else 0.0) for p, y in zip(probs, labels) if y is True or y is False]
    p = np.array([s[0] for s in scored]); y = np.array([s[1] for s in scored])
    pred = (p >= 0.5).astype(float)
    acc = float((pred == y).mean()) if len(p) else float("nan")
    brier = float(((p - y) ** 2).mean()) if len(p) else float("nan")
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    bins = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    table, ece = [], 0.0
    for b in range(N_BINS):
        m = bins == b
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.sum() / len(p) * gap
            table.append((b, int(m.sum()), float(conf[m].mean()), float(correct[m].mean())))
        else:
            table.append((b, 0, float("nan"), float("nan")))
    hedge = float(np.mean([abs(pp - 0.5) for pp, _ in hard])) if hard else float("nan")
    # Threshold-free and threshold diagnostics: a model that never says yes
    # scores the base rate on accuracy, which the table above would hide.
    tp = float(((pred == 1) & (y == 1)).sum()); fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    pos, neg = p[y == 1], p[y == 0]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else float("nan")
    return {"n": int(len(p)), "n_unsure": len(hard), "accuracy": acc, "ece": float(ece), "brier": brier,
            "hedging": hedge, "reliability": table,
            "base_rate_no": float((y == 0).mean()) if len(y) else float("nan"),
            "yes_rate": float(pred.mean()) if len(pred) else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "auc": auc}


def pooled(rows: list[dict], scores: dict, key=lambda s, q: s["p"][q]) -> dict:
    """Metrics pooled over the five questions, plus per question."""
    labels, probs, per_q = [], [], {}
    for q in QUESTION_KEYS:
        lq = [r["labels"][q] for r in rows]
        pq = [key(scores[r["id"]], q) for r in rows]
        per_q[q] = metrics(lq, pq)
        labels += lq; probs += pq
    return {"pooled": metrics(labels, probs), "per_question": per_q}


def fit_temperature(rows: list[dict], scores: dict) -> float:
    """Grid + refine search for T minimising binary NLL of P(entail | logits / T)."""
    logits, y = [], []
    for r in rows:
        for q in QUESTION_KEYS:
            lab = r["labels"][q]
            if lab is True or lab is False:
                logits.append(np.log(np.array(scores[r["id"]]["raw"][q]) + 1e-12))
                y.append(1.0 if lab else 0.0)
    logits, y = np.array(logits), np.array(y)

    def nll(T):
        p = np.clip(probabilities(logits, T), 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 200))
    best = min(grid, key=nll)
    fine = np.exp(np.linspace(np.log(best / 1.5), np.log(best * 1.5), 200))
    return float(min(fine, key=nll))


def rescored(scores: dict, temperature: float, score: str) -> dict:
    """Recompute P(yes) from the cached raw 3-way probabilities."""
    out = {}
    for sid, s in scores.items():
        raw = np.log(np.array([s["raw"][q] for q in QUESTION_KEYS]) + 1e-12)
        p = probabilities(raw, temperature, score)
        out[sid] = {**s, "p": dict(zip(QUESTION_KEYS, map(float, p)))}
    return out


def repeatability(rows: list[dict], first: dict, second: dict) -> dict:
    per_q = {}
    for q in QUESTION_KEYS:
        d = [abs(first[r["id"]]["p"][q] - second[r["id"]]["p"][q]) for r in rows]
        per_q[q] = (float(np.mean(d)), float(np.max(d)))
    allq = [v for r in rows for q in QUESTION_KEYS for v in [abs(first[r["id"]]["p"][q] - second[r["id"]]["p"][q])]]
    return {"mean": float(np.mean(allq)), "max": float(np.max(allq)), "per_question": per_q}


def latency(scores: dict) -> dict:
    ms = np.array([s["ms"] for s in scores.values()])
    cached = sum(s["cached"] for s in scores.values())
    return {"p50": float(np.median(ms)), "p95": float(np.percentile(ms, 95)), "cached": cached, "n": len(ms)}


# --------------------------------------------------------------------------- report

def fmt(x: float, nd: int = 3) -> str:
    return "—" if x != x else f"{x:.{nd}f}"


def write_report(path: Path, results: list[dict], rows: list[dict], notes: list[str]) -> None:
    n_states = len(rows)
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    L = [f"# decide quality — {n_states} human-reviewed triage states × 5 questions", "",
         f"Recorded {datetime.now(timezone.utc).isoformat(timespec='seconds')}. States by category: "
         + ", ".join(f"{k} {v}" for k, v in cats.items()) + ".",
         "Accuracy / ECE / Brier exclude cells labeled *unsure*; **hedging** is mean |p − 0.5| on those cells "
         "(lower is better). **Repeatability** is mean / max |p₁ − p₂| over a fixed 30-state subset scored twice. "
         "Latency is the per-state call (5 questions) as measured in this run; Jev's from the first live pass. "
         "Accuracy is at p ≥ 0.5 — read it against the per-question yes-rate below, because a model that never "
         "says yes scores the base rate. AUC is threshold-free (pooled over all 5 questions' cells).", "",
         "| backend | accuracy | ECE | Brier | hedging | AUC | repeatability mean / max | latency p50 / p95 ms |",
         "|---|---|---|---|---|---|---|---|"]
    for r in results:
        m = r["metrics"]["pooled"]; rep = r["repeatability"]; lat = r["latency"]
        L.append(f"| {r['name']} | {fmt(m['accuracy'])} | {fmt(m['ece'])} | {fmt(m['brier'])} | {fmt(m['hedging'])} | "
                 f"{fmt(m['auc'])} | {fmt(rep['mean'], 2)} / {fmt(rep['max'], 2)} | {lat['p50']:.0f} / {lat['p95']:.0f} |")
    L += ["", "## Per question (accuracy / ECE / hedging)", "",
          "| backend | " + " | ".join(QUESTION_KEYS) + " |", "|---|" + "---|" * len(QUESTION_KEYS)]
    for r in results:
        cells = []
        for q in QUESTION_KEYS:
            m = r["metrics"]["per_question"][q]
            cells.append(f"{fmt(m['accuracy'], 2)} / {fmt(m['ece'], 2)} / {fmt(m['hedging'], 2)}")
        L.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    L += ["", "## Per question, threshold diagnostics (yes-rate · recall · precision · AUC; base rate of *no* in the header)", "",
          "| backend | " + " | ".join(f"{q} (no {results[0]['metrics']['per_question'][q]['base_rate_no']:.2f})" for q in QUESTION_KEYS) + " |",
          "|---|" + "---|" * len(QUESTION_KEYS)]
    for r in results:
        cells = []
        for q in QUESTION_KEYS:
            m = r["metrics"]["per_question"][q]
            cells.append(f"{fmt(m['yes_rate'], 2)} · {fmt(m['recall'], 2)} · {fmt(m['precision'], 2)} · {fmt(m['auc'], 2)}")
        L.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    L += ["", "## Reliability (10 bins over confidence = max(p, 1−p); count · mean confidence · accuracy)", ""]
    for r in results:
        L.append(f"**{r['name']}**  ")
        L.append("| bin | " + " | ".join(f"{b / N_BINS:.1f}–{(b + 1) / N_BINS:.1f}" for b in range(N_BINS)) + " |")
        L.append("|---|" + "---|" * N_BINS)
        tab = r["metrics"]["pooled"]["reliability"]
        L.append("| n | " + " | ".join(str(t[1]) for t in tab) + " |")
        L.append("| conf | " + " | ".join(fmt(t[2], 2) for t in tab) + " |")
        L.append("| acc | " + " | ".join(fmt(t[3], 2) for t in tab) + " |")
        L.append("")
    L += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", nargs="+", default=BACKENDS)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--no-cache", action="store_true", help="live Jev calls for both passes")
    ap.add_argument("--dry-run", action="store_true", help="run on unreviewed labels; print only")
    ap.add_argument("--out", type=Path, default=QUALITY_PATH)
    a = ap.parse_args(argv)

    if not review_is_approved() and not a.dry_run:
        sys.exit("data/decide/holdout_review.md is not marked 'reviewed: true'; "
                 "have a human review the labels first (or --dry-run to print unreviewed numbers)")
    rows = load_holdout()
    questions = {q: {"type": "boolean", **v} for q, v in load_questions().items()}
    subset = repeat_subset(rows)
    print(f"{len(rows)} states, {len(subset)} in the repeatability subset; reviewed={review_is_approved()}")

    results, notes = [], []
    temperature = None
    for name in a.backends:
        try:
            backend = make_backend(name, max_len=a.max_len, temperature=1.0, use_cache=not a.no_cache)
        except JevUnavailable as e:
            print(f"skip {name}: {e}")
            continue
        try:
            scores = score_backend(backend, rows, questions, label=name)
        except JevUnavailable as e:
            print(f"skip {name}: {e}")
            continue
        lat = latency(scores)

        if name == "jev":
            from .decider import JevDecider
            second_backend = JevDecider(use_cache=not a.no_cache, cache_path=REPEAT_CACHE)
            second = score_backend(second_backend, subset, questions, label="jev repeat pass")
            rep = repeatability(subset, scores, second)
            results.append({"name": "jev", "metrics": pooled(rows, scores), "repeatability": rep, "latency": lat})
            notes.append(f"Jev repeatability per question (mean / max |p₁ − p₂|): "
                         + ", ".join(f"{q} {m:.2f} / {x:.2f}" for q, (m, x) in rep["per_question"].items())
                         + f". Quality gaps smaller than ~{rep['mean']:.2f} are ties.")
            notes.append(f"Jev latency: {lat['cached']} of {lat['n']} first-pass calls came from the cache "
                         "(their ms is the latency recorded when the entry was made).")
            continue

        # local: raw, temperature-scaled, entail-vs-contra
        second = score_backend(backend, subset, questions, label=f"{name} repeat pass")
        rep = repeatability(subset, scores, second)
        assert rep["max"] == 0.0, f"local model not deterministic: {rep}"
        results.append({"name": f"{name} (raw, P=entail)", "metrics": pooled(rows, scores),
                        "repeatability": rep, "latency": lat})
        fit_rows = [r for r in rows if r["id"] % 3 != 0]
        eval_rows = [r for r in rows if r["id"] % 3 == 0]
        T = fit_temperature(fit_rows, scores)
        before = pooled(eval_rows, scores)["pooled"]
        after = pooled(eval_rows, rescored(scores, T, "entail"))["pooled"]
        notes.append(f"{name}: T = {T:.2f} fit by NLL on {len(fit_rows)} states; on the {len(eval_rows)} held-out "
                     f"states ECE {before['ece']:.3f} → {after['ece']:.3f}, Brier {before['brier']:.3f} → "
                     f"{after['brier']:.3f}, accuracy {before['accuracy']:.3f} → {after['accuracy']:.3f}.")
        results.append({"name": f"{name} (T={T:.2f}, P=entail)", "metrics": pooled(rows, rescored(scores, T, "entail")),
                        "repeatability": rep, "latency": lat})
        results.append({"name": f"{name} (raw, P=entail/(entail+contra))",
                        "metrics": pooled(rows, rescored(scores, 1.0, "entail_vs_contra")),
                        "repeatability": rep, "latency": lat})
        if temperature is None:
            temperature = T

    notes.append("Phrasing: Jev is sent the question form (`instructions`), the local model the declarative "
                 "`hypothesis`. The 2026-09-19 probe measured question-vs-declarative drift on Jev at median 0.09, "
                 "max 0.16 — about its run-to-run noise — so this is recorded as an observation, not a finding.")
    notes.append("No cross-question interference on Jev (a question asked alone matched the same question batched "
                 "with five others on every probe state), so scoring one (state, hypothesis) row per question "
                 "locally is a like-for-like comparison.")

    # console summary
    print()
    print(f"{'backend':44s} {'acc':>6s} {'ECE':>6s} {'Brier':>6s} {'hedge':>6s} {'AUC':>6s} {'rep':>11s} {'p50/p95 ms':>14s}")
    for r in results:
        m = r["metrics"]["pooled"]; rep = r["repeatability"]; lat = r["latency"]
        print(f"{r['name']:44s} {m['accuracy']:6.3f} {m['ece']:6.3f} {m['brier']:6.3f} {fmt(m['hedging']):>6s} {m['auc']:6.3f} "
              f"{rep['mean']:5.2f}/{rep['max']:5.2f} {lat['p50']:7.0f}/{lat['p95']:6.0f}")
        print("    per question yes-rate/recall/AUC: " + "  ".join(
            f"{q[:9]} {r['metrics']['per_question'][q]['yes_rate']:.2f}/{fmt(r['metrics']['per_question'][q]['recall'], 2)}/"
            f"{fmt(r['metrics']['per_question'][q]['auc'], 2)}" for q in QUESTION_KEYS))
    for n in notes:
        print(f"- {n}")

    if a.dry_run:
        print("\n--dry-run: labels unreviewed, nothing written")
        return
    if temperature is not None:
        CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION_PATH.write_text(json.dumps({"temperature": temperature, "score": "entail", "labels": list(LABELS),
                                                "fit": "NLL on holdout ids with id % 3 != 0 (100 states x 5 questions)",
                                                "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                               indent=2) + "\n")
        print(f"wrote {CALIBRATION_PATH} (T={temperature:.3f})")
    write_report(a.out, results, rows, notes)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
