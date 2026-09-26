"""Chat with SmolLM2 through the Core AI Python runtime, and record ``llm-bench/1`` numbers.

    python -m llm_ai.play "Why is the sky blue?"            # stream a reply
    python -m llm_ai.play --bench --json docs/bench/llm-python.json --load-cache cached
    python -m llm_ai.play --bench --variant stateless --lengths 16 128 --tokens 64 --json ...
    USE_LOCAL_COREAI=1 python -m llm_ai.play --bench --compute default --json ...   # macOS 26 runtime

Bench: for each prompt length (default 16 / 128 / 512 tokens, the tail of a
long chat prompt so it ends in the assistant header), ``--runs`` greedy runs
of ``--tokens`` generated tokens with EOS ignored, so every run does the same
work. Per run: prefill ms (time to first token, including the argmax) and
every decode step's ms. Records are merged into ``--json`` keyed by
(variant, runtime, compute, precision, prompt tokens), so one file collects
the rows from several invocations.

``--load-cache`` labels the load time: ``cold`` only if this program's
specialization cache entry was deleted first (see docs/prompts/llm-smollm2.md).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from snake_ai.play import host_info

from .runtime import COMPUTE_UNITS, DEFAULT_STATEFUL, DEFAULT_STATELESS, StatefulLM, StatelessLM
from .tokenizer import DEFAULT_MODEL_DIR, ChatTokenizer
from .verify import LONG_PROMPT

SCHEMA = "llm-bench/1"


def runtime_name() -> str:
    return "local" if os.environ.get("USE_LOCAL_COREAI") == "1" else "os"


def git_sha() -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout
    return sha + ("-dirty" if dirty.strip() else "")


def bench_prompt(tok: ChatTokenizer, n: int) -> list[int]:
    """Exactly ``n`` ids: the tail of a long chat prompt (ends with ``<|im_start|>assistant\\n``)."""
    ids = tok.encode_chat(" ".join([LONG_PROMPT] * 8))
    assert len(ids) >= n, f"bench text is only {len(ids)} tokens"
    return ids[-n:]


def ms_stats(xs: list[float]) -> dict:
    a = np.asarray(xs)
    return {"count": len(xs), "mean": round(float(a.mean()), 3), "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "min": round(float(a.min()), 3)}


async def run_stateful(lm: StatefulLM, ids: list[int], tokens: int) -> tuple[float, list[float], list[int]]:
    lm.reset()
    t0 = time.perf_counter()
    logits = await lm.prefill_ids(ids)
    nxt = int(logits.argmax())
    prefill_ms = (time.perf_counter() - t0) * 1e3
    out, steps = [nxt], []
    for _ in range(tokens - 1):
        t0 = time.perf_counter()
        nxt = int((await lm.decode(nxt)).argmax())
        steps.append((time.perf_counter() - t0) * 1e3)
        out.append(nxt)
    return prefill_ms, steps, out


async def run_stateless(lm: StatelessLM, ids: list[int], tokens: int) -> tuple[float, list[float], list[int]]:
    seq = list(ids)
    t0 = time.perf_counter()
    nxt = int((await lm.logits(seq)).argmax())
    prefill_ms = (time.perf_counter() - t0) * 1e3
    out, steps = [nxt], []
    for _ in range(tokens - 1):
        seq.append(nxt)
        t0 = time.perf_counter()
        nxt = int((await lm.logits(seq)).argmax())
        steps.append((time.perf_counter() - t0) * 1e3)
        out.append(nxt)
    return prefill_ms, steps, out


async def bench(a: argparse.Namespace) -> list[dict]:
    tok = ChatTokenizer(a.model_dir)
    stateful = a.variant == "stateful"
    lm = await (StatefulLM.load(a.asset, a.compute) if stateful else StatelessLM.load(a.asset, a.compute))
    fn = lm.decode_fn if stateful else lm.fn
    precision = "fp16" if "float16" in str(fn.desc.output_descriptor("logits")) else "fp32"
    print(f"{a.variant} {precision} runtime={runtime_name()} compute={a.compute}: loaded in {lm.load_ms:.0f} ms ({a.load_cache})")
    run = run_stateful if stateful else run_stateless
    await run(lm, bench_prompt(tok, 16), 4)  # warm-up: first calls pay one-off setup
    records = []
    for n in a.lengths:
        ids = bench_prompt(tok, n)
        prefills, steps, texts = [], [], set()
        for _ in range(a.runs):
            p, s, out = await run(lm, ids, a.tokens)
            prefills.append(p)
            steps.extend(s)
            texts.add(tok.decode(out))
        dec = ms_stats(steps)
        records.append({
            "variant": a.variant,
            "runtime": runtime_name(),
            "compute": a.compute,
            "precision": precision,
            "prompt_tokens": n,
            "generated_tokens": a.tokens,
            "runs": a.runs,
            "load_ms": {"value": round(lm.load_ms, 1), "cache": a.load_cache},
            "prefill_ms": ms_stats(prefills),
            "decode_ms": dec,
            "decode_first_last_5_ms": [round(float(np.mean(steps[: a.tokens - 1][:5])), 3),
                                       round(float(np.mean(steps[: a.tokens - 1][-5:])), 3)],
            "tok_per_s": round(1e3 / dec["mean"], 2),
            "deterministic": len(texts) == 1,
        })
        print(f"  T={n:3d}: prefill p50 {records[-1]['prefill_ms']['p50']:.0f} ms, decode p50 {dec['p50']:.1f} / "
              f"p95 {dec['p95']:.1f} ms, {records[-1]['tok_per_s']} tok/s")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes on macOS
    load = os.getloadavg()
    for r in records:
        r.update({
            "peak_rss_mb": round(peak, 1),
            "load_avg": [round(x, 2) for x in load],
            "host": host_info(),
            "git": git_sha(),
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "asset": str(a.asset),
        })
    return records


def merge(path: Path, records: list[dict]) -> None:
    key = lambda r: (r["variant"], r["runtime"], r["compute"], r["precision"], r["prompt_tokens"])  # noqa: E731
    old = json.loads(path.read_text())["records"] if path.exists() else []
    new_keys = {key(r) for r in records}
    merged = [r for r in old if key(r) not in new_keys] + records
    merged.sort(key=key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": SCHEMA, "records": merged}, indent=1) + "\n")
    print(f"wrote {path} ({len(merged)} records)")


async def chat(a: argparse.Namespace) -> None:
    tok = ChatTokenizer(a.model_dir)
    lm = await StatefulLM.load(a.asset, a.compute)
    ids = tok.encode_chat(a.prompt)
    t0 = time.perf_counter()
    nxt = int((await lm.prefill_ids(ids)).argmax())
    ttft = (time.perf_counter() - t0) * 1e3
    out, t1 = [], time.perf_counter()
    printed = ""
    while nxt != tok.eos_id and len(out) < a.tokens and lm.position < lm.max_seq_len:
        out.append(nxt)
        text = tok.decode(out)  # decode the whole reply so multi-byte characters come out whole
        if not text.endswith("�"):
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text
        nxt = int((await lm.decode(nxt)).argmax())
    secs = time.perf_counter() - t1
    print(f"\n\n[{len(ids)} prompt tokens, first token {ttft:.0f} ms, {len(out)} tokens at "
          f"{len(out) / secs:.1f} tok/s, load {lm.load_ms:.0f} ms]")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default="Give me three tips for writing clear commit messages.")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--asset", type=Path)
    ap.add_argument("--compute", choices=COMPUTE_UNITS, default="gpu")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--variant", choices=["stateful", "stateless"], default="stateful")
    ap.add_argument("--lengths", type=int, nargs="+", default=[16, 128, 512])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--load-cache", choices=["cold", "cached"], default="cached")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    if runtime_name() == "local" and a.compute != "default":
        ap.error("the local runtime has no SpecializationOptions; pass --compute default")
    a.asset = a.asset or (DEFAULT_STATEFUL if a.variant == "stateful" else DEFAULT_STATELESS)
    if not a.bench:
        asyncio.run(chat(a))
        return
    records = asyncio.run(bench(a))
    if a.json:
        merge(a.json, records)


if __name__ == "__main__":
    main()
