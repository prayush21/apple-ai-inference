"""Probe typesafe-ai/jev through the Vercel AI Gateway before writing the holdout.

    export AI_GATEWAY_API_KEY=...          # in the same shell
    .venv/bin/python scripts/probe_jev.py  # 36 calls paced under the 30/min limit, ~90 s, cents

Prints, per probe, the state, each question's probability, round-trip ms and
token usage, then a summary: probability spread (how often Jev says something
other than <0.05 / >0.95), question-vs-declarative phrasing drift, and whether
answers to one question move when other questions are present in the request.
Nothing is written to disk; the key is read from the environment only.
"""

from __future__ import annotations

import json
import os
import ssl
import statistics as st
import sys
import time
import urllib.error
import urllib.request

# python.org builds of CPython on macOS don't see the system keychain; use certifi's bundle.
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # fall back to whatever the interpreter has
    SSL_CTX = ssl.create_default_context()

URL = "https://ai-gateway.vercel.sh/v1/evaluate"
MODEL = "typesafe-ai/jev"

# Question form (what Jev's docs use) and declarative form (what NLI wants).
Q = {
    "is_complaint": ("Is the customer complaining?", "The customer is complaining."),
    "wants_refund": ("Is the customer asking for a refund?", "The customer is asking for a refund."),
    "about_shipping": ("Is this about shipping or delivery?", "This is about shipping or delivery."),
    "about_quality": ("Is this about product quality?", "This is about product quality."),
    "urgent": ("Is this urgent?", "This is urgent."),
    "has_order_number": ("Does the message mention a specific order number?",
                         "The message mentions a specific order number."),
}

# (category, state, expected notes) -- expected is for your eyes, not scored.
PROBES = [
    ("easy", "The pasta was cold and the waiter ignored us, I want my money back.",
     "complaint 1, refund 1, shipping 0"),
    ("easy", "Package arrived a day early, thanks so much!",
     "about_shipping 1 (topic, not sentiment); everything else ~0"),
    ("negation", "I don't want a refund, just send a replacement.",
     "refund should be LOW; NLI models often get this wrong"),
    ("negation", "No complaints about the product, but the box was crushed in transit.",
     "complaint? arguable; shipping 1; quality 0"),
    ("implicit", "This is the third time I've written in about this.",
     "urgent should be high with no keyword; complaint high"),
    ("implicit", "Still waiting.",
     "short, ambiguous: ~0.5 is a fair answer on several"),
    ("extraction", "Order #48213 never showed up.",
     "has_order_number 1, shipping 1"),
    ("extraction", "My order never showed up.",
     "has_order_number 0 -- does Jev separate these two?"),
    ("ambiguous", "Hmm.",
     "everything should be ~0.5 or low, not confidently 0"),
    ("ambiguous", "Can I change the color on my order before it ships?",
     "complaint 0, shipping ~0.5?, refund 0"),
    ("multi", "I can't log in and I also want a refund for last month.",
     "refund 1, complaint 1, shipping 0"),
    ("hard", "The blender works but sounds like a jet engine. Not sure if that's normal.",
     "quality ~0.7?, complaint ~0.5"),
]


# Gateway limit is 30 requests/min per team; stay under it.
MIN_GAP_S = 2.4
_last_call = 0.0


def _pace() -> None:
    global _last_call
    wait = MIN_GAP_S - (time.perf_counter() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.perf_counter()


def call(state: str, questions: dict[str, str]) -> tuple[dict, dict, float]:
    body = json.dumps({
        "model": MODEL,
        "state": state,
        "questions": {k: {"type": "boolean", "instructions": v} for k, v in questions.items()},
    }).encode()
    req = urllib.request.Request(URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {os.environ['AI_GATEWAY_API_KEY']}",
        "Content-Type": "application/json",
    })
    _pace()
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
        out = json.loads(r.read())
    ms = (time.perf_counter() - t0) * 1e3
    probs = {k: v["probability"] for k, v in out["answers"].items()}
    return probs, out.get("usage", {}), ms


def main() -> None:
    if not os.environ.get("AI_GATEWAY_API_KEY"):
        sys.exit("AI_GATEWAY_API_KEY not set in this shell")

    qform = {k: v[0] for k, v in Q.items()}
    dform = {k: v[1] for k, v in Q.items()}
    lat: list[float] = []
    all_probs: list[float] = []
    drift: list[float] = []
    interference: list[float] = []

    print(f"{'cat':10} {'ms':>6}  {'in/out tok':>10}  " + "  ".join(f"{k[:9]:>9}" for k in Q))
    for cat, state, note in PROBES:
        p_q, usage, ms = call(state, qform)
        lat.append(ms)
        all_probs.extend(p_q.values())
        print(f"{cat:10} {ms:6.0f}  {usage.get('inputTokens','?'):>4}/{usage.get('outputTokens','?'):<5}  "
              + "  ".join(f"{p_q[k]:9.2f}" for k in Q))
        print(f"{'':10} {state!r}\n{'':10} expect: {note}")

        # Phrasing drift: same state, declarative hypotheses instead of questions.
        p_d, _, _ = call(state, dform)
        d = max(abs(p_q[k] - p_d[k]) for k in Q)
        drift.append(d)
        if d > 0.15:
            worst = max(Q, key=lambda k: abs(p_q[k] - p_d[k]))
            print(f"{'':10} phrasing drift {d:.2f} on {worst}: question {p_q[worst]:.2f} vs declarative {p_d[worst]:.2f}")

        # Interference: ask wants_refund alone vs with the other five present.
        p_single, _, _ = call(state, {"wants_refund": qform["wants_refund"]})
        diff = abs(p_single["wants_refund"] - p_q["wants_refund"])
        interference.append(diff)
        if diff > 0.1:
            print(f"{'':10} interference: wants_refund alone {p_single['wants_refund']:.2f} vs in batch {p_q['wants_refund']:.2f}")
        print()

    n = len(all_probs)
    extreme = sum(1 for p in all_probs if p < 0.05 or p > 0.95)
    mid = sum(1 for p in all_probs if 0.3 <= p <= 0.7)
    print("=" * 72)
    print(f"latency ms: p50 {st.median(lat):.0f}  min {min(lat):.0f}  max {max(lat):.0f}  (n={len(lat)}, 6 questions each)")
    print(f"probability spread: {extreme}/{n} extreme (<0.05 or >0.95), {mid}/{n} in [0.3, 0.7]")
    print(f"phrasing drift (question vs declarative), max abs diff per state: "
          f"median {st.median(drift):.2f}  max {max(drift):.2f}")
    print(f"interference (wants_refund alone vs batched): median {st.median(interference):.2f}  max {max(interference):.2f}")
    print()
    print("read: high 'extreme' share => confident, maybe not calibrated; low drift => robust to phrasing;")
    print("      low interference => questions are independent (so a per-row NLI model is a fair comparison).")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode()[:300]}")
