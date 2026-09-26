"""The labeled holdout set for customer-message triage, and its review table.

    python -m decide_ai.holdout --review      # regenerate data/decide/holdout_review.md
    python -m decide_ai.holdout --stats

``data/decide/questions.json`` holds the five fixed questions in both forms:
``instructions`` (question form, what Jev is sent) and ``hypothesis``
(declarative, what the NLI model scores). ``data/decide/holdout.jsonl`` has
one state per line::

    {"id": 7, "category": "negation", "state": "...",
     "labels": {"is_complaint": true, "wants_refund": false, ..., "urgent": "unsure"},
     "reviewed_by": "human"}          # present once a human has checked the line

Labels are ``true`` / ``false`` / ``"unsure"``. ``"unsure"`` marks states where
0.5 is the right answer; those cells are excluded from accuracy / ECE and
scored as *hedging* (mean |p - 0.5|) instead. Labeling rules:

* topic questions are about topic, not sentiment ("Package arrived a day
  early, thanks!" is ``about_shipping: true``);
* ``wants_refund`` needs an actual request for money back, not a grievance
  that might deserve one;
* ``urgent`` needs time pressure or escalation in the text, keyword or not.

``holdout_review.md`` is the human review surface: a ``reviewed: true`` line
at the top is what ``calibrate.py`` checks before it reports quality numbers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HOLDOUT_PATH = Path("data/decide/holdout.jsonl")
QUESTIONS_PATH = Path("data/decide/questions.json")
REVIEW_PATH = Path("data/decide/holdout_review.md")
QUESTION_KEYS = ["is_complaint", "wants_refund", "about_shipping", "about_product_quality", "urgent"]
SHORT = {"is_complaint": "complaint", "wants_refund": "refund", "about_shipping": "shipping",
         "about_product_quality": "quality", "urgent": "urgent"}


def load_questions(path: Path = QUESTIONS_PATH) -> dict[str, dict]:
    return json.loads(path.read_text())


def load_holdout(path: Path = HOLDOUT_PATH) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in rows:
        assert set(r["labels"]) == set(QUESTION_KEYS), r["id"]
    return rows


def save_holdout(rows: list[dict], path: Path = HOLDOUT_PATH) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def review_is_approved(path: Path = REVIEW_PATH) -> bool:
    return path.exists() and path.read_text().lstrip().startswith("reviewed: true")


def label_cell(v) -> str:
    return "yes" if v is True else "no" if v is False else "?"


def write_review(rows: list[dict], path: Path = REVIEW_PATH, *, reviewed: bool = False) -> None:
    lines = [f"reviewed: {'true' if reviewed else 'false'}", "",
             "# Holdout review", "",
             "One row per state; `yes` / `no` / `?` (unsure — 0.5 is the right answer, scored as hedging).",
             "Edit cells in place or tell Claude the id + column + new value. `cat` = s straightforward, "
             "n negation, i implicit intent, e extraction-shaped, a ambiguous.", "",
             "| id | cat | state | " + " | ".join(SHORT[k] for k in QUESTION_KEYS) + " |",
             "|---|---|---|" + "|".join("---" for _ in QUESTION_KEYS) + "|"]
    for r in rows:
        cells = " | ".join(label_cell(r["labels"][k]) for k in QUESTION_KEYS)
        state = r["state"].replace("|", "\\|")  # not inside the f-string: backslashes there need 3.12
        lines.append(f"| {r['id']} | {r['category'][0]} | {state} | {cells} |")
    path.write_text("\n".join(lines) + "\n")


def stats(rows: list[dict]) -> str:
    cats = Counter(r["category"] for r in rows)
    per_q = {k: Counter(label_cell(r["labels"][k]) for r in rows) for k in QUESTION_KEYS}
    out = [f"{len(rows)} states: " + ", ".join(f"{c} {n}" for c, n in cats.most_common())]
    for k, c in per_q.items():
        out.append(f"  {k:22s} yes {c['yes']:3d}  no {c['no']:3d}  unsure {c['?']:3d}")
    out.append(f"  reviewed_by=human on {sum('reviewed_by' in r for r in rows)} lines")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", action="store_true", help="write the markdown review table")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args(argv)
    rows = load_holdout()
    if a.review:
        write_review(rows, reviewed=review_is_approved())
        print(f"wrote {REVIEW_PATH}")
    if a.stats or not a.review:
        print(stats(rows))


if __name__ == "__main__":
    main()
