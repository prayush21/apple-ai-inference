"""Latency baselines for the decider, as ``decide-bench/1`` JSON records.

    python -m decide_ai.bench --backend local  --json docs/bench/decide-python.json
    python -m decide_ai.bench --backend remote --json docs/bench/decide-remote.json   # needs decide_ai.serve on :8770
    python -m decide_ai.bench --backend jev --no-cache --json docs/bench/decide-jev.json
    python -m decide_ai.bench --backend laya --json docs/bench/decide-laya.json

Rows are ``{asset, N, L}`` for N questions per request x padded length L
(local / remote: N in {1,4,8,16} x L in {64,128}, both assets; Jev: N in
{1,4,8}, one call answers all N questions and there is no L; Laya: N in
{1,4,8,16}, in-process PyTorch on CPU, pads to its own sequence so no L
either). Inputs are real holdout states and the questions in
``data/decide/bench_questions.json``.

Per row: ``load_ms`` (first / rest_mean, three loads), the first call after
load on its own (``first_call_ms``), then ``warmup`` untimed calls and a timed
run of ``calls`` calls capped by ``budget_s`` wall-clock (never fewer than
``min_calls``); ``count`` in the record says how many actually ran, because
the large shapes take seconds per call on the local runtime. Timings are
split into ``tokenize_ms`` and ``infer_ms`` in-process; ``roundtrip_ms`` vs
``server_infer_ms`` for the remote backend (the difference is JSON + HTTP);
``roundtrip_ms`` plus ``usage`` for Jev, whose calls are paced at 25/min;
``total_ms`` plus ``usage`` for Laya (its API has no tokenize / infer
split), 5 warm-ups then >= 30 calls capped at 60 s per row, and a single
``load_ms.first`` because loading the 421M checkpoint is ~35-60 s.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import laya
from .convert import DEFAULT_OUT_DIR, STATIC_BATCHES, STATIC_LENGTHS
from .decider import ASSETS, LocalDecider, LocalModel
from .download import model_dir
from .holdout import load_holdout
from .tokenize import BPETokenizer

BENCH_QUESTIONS_PATH = Path("data/decide/bench_questions.json")
CACHE_DIR = Path.home() / "Library/Caches/coreai-cache"
JEV_BATCHES = (1, 4, 8)


def stats(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    return {"count": int(a.size), "mean": float(a.mean()), "p50": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max())}


def host_info() -> dict:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        chip = None
    try:
        from importlib.metadata import version
        coreai_version = version("coreai-core")
    except Exception:  # noqa: BLE001
        coreai_version = None
    return {"machine": platform.machine(), "chip": chip, "macos": platform.mac_ver()[0],
            "python": platform.python_version(), "coreai_core": coreai_version}


def bench_inputs(n: int) -> tuple[list[str], dict]:
    """Holdout states (all of them, cycled) and the first ``n`` bench questions."""
    qs = json.loads(BENCH_QUESTIONS_PATH.read_text())
    names = list(qs)[:n]
    questions = {k: {"type": "boolean", **qs[k]} for k in names}
    return [r["state"] for r in load_holdout()], questions


def timed_loop(call, *, warmup: int, calls: int, budget_s: float, min_calls: int, label: str):
    """Run ``call(i)`` (returns a dict of ms measurements) and collect them."""
    for i in range(warmup):
        call(i)
    samples: dict[str, list[float]] = {}
    t_start = time.perf_counter()
    n = 0
    while n < calls and (n < min_calls or time.perf_counter() - t_start < budget_s):
        for k, v in call(n).items():
            samples.setdefault(k, []).append(v)
        n += 1
        if n % 20 == 0:
            print(f"    {label}: {n} calls, {time.perf_counter() - t_start:.0f} s", flush=True)
    return {k: stats(v) for k, v in samples.items()}


# --------------------------------------------------------------------------- local

def run_local(a: argparse.Namespace) -> tuple[str, list[dict]]:
    tok = BPETokenizer.from_hf(model_dir())
    states, _ = bench_inputs(1)
    rows = []
    for asset in a.assets:
        loads = []
        model = None
        for _ in range(3):
            model = LocalModel.load(a.models_dir, asset)
            loads.append(model.load_ms)
        print(f"{asset}: load ms first {loads[0]:.0f}, rest {np.mean(loads[1:]):.0f}")
        for n in a.batches:
            for length in a.lengths:
                _, questions = bench_inputs(n)
                decider = LocalDecider(model, tok, max_len=length)
                fn_name, _, _ = model.function_for(n, length)

                def call(i, d=decider, q=questions, L=length):
                    _, _, timing, _ = d.evaluate(states[i % len(states)], q, length=L)
                    return {"tokenize_ms": timing["ms_tokenize"], "infer_ms": timing["ms_infer"],
                            "total_ms": timing["ms_total"]}

                t0 = time.perf_counter()
                call(0)
                first_call = (time.perf_counter() - t0) * 1e3
                res = timed_loop(call, warmup=a.warmup, calls=a.calls, budget_s=a.budget_s,
                                 min_calls=a.min_calls, label=f"{asset} N={n} L={length}")
                row = {"asset": ASSETS[asset], "variant": asset, "function": fn_name, "N": n, "L": length,
                       "load_ms": {"first": loads[0], "rest_mean": float(np.mean(loads[1:])), "all": loads},
                       "first_call_ms": first_call, **res}
                rows.append(row)
                print(f"  {asset:8s} N={n:2d} L={length:3d} {fn_name:14s} infer p50 {res['infer_ms']['p50']:8.1f} "
                      f"p95 {res['infer_ms']['p95']:8.1f} ms  tokenize p50 {res['tokenize_ms']['p50']:.2f} ms "
                      f"(n={res['infer_ms']['count']})", flush=True)
    return "coreai.runtime (Python, in-process, CPU)", rows


# --------------------------------------------------------------------------- remote

def _post(url: str, payload: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_remote(a: argparse.Namespace) -> tuple[str, list[dict]]:
    with urllib.request.urlopen(a.url + "/info", timeout=5) as r:
        info = json.loads(r.read())
    print(f"server: {info['asset']} max_len={info['max_len']} T={info['temperature']}")
    states, _ = bench_inputs(1)
    rows = []
    for n in a.batches:
        for length in a.lengths:
            if length > info["max_len"]:
                print(f"  skip L={length}: server --max-len is {info['max_len']}")
                continue
            _, questions = bench_inputs(n)

            def call(i, q=questions, L=length):
                t0 = time.perf_counter()
                out = _post(a.url + "/decide", {"model": "local", "state": states[i % len(states)],
                                                "questions": q, "padded_len": L})
                rt = (time.perf_counter() - t0) * 1e3
                return {"roundtrip_ms": rt, "server_infer_ms": out["timing"]["ms_infer"],
                        "server_total_ms": out["timing"]["ms_total"]}

            t0 = time.perf_counter()
            call(0)
            first_call = (time.perf_counter() - t0) * 1e3
            res = timed_loop(call, warmup=a.warmup, calls=a.calls, budget_s=a.budget_s,
                             min_calls=a.min_calls, label=f"remote N={n} L={length}")
            overhead = res["roundtrip_ms"]["p50"] - res["server_total_ms"]["p50"]
            rows.append({"asset": info["asset"], "N": n, "L": length, "first_call_ms": first_call,
                         "http_overhead_p50_ms": overhead, **res})
            print(f"  N={n:2d} L={length:3d} roundtrip p50 {res['roundtrip_ms']['p50']:8.1f} ms, server infer p50 "
                  f"{res['server_infer_ms']['p50']:8.1f}, HTTP+JSON overhead {overhead:.1f} ms (n={res['roundtrip_ms']['count']})",
                  flush=True)
    return "coreai.runtime via decide_ai.serve (HTTP from Python)", rows


# --------------------------------------------------------------------------- jev

def run_jev(a: argparse.Namespace) -> tuple[str, list[dict]]:
    from . import jev

    if not jev.available():
        print("AI_GATEWAY_API_KEY is not set; skipping the Jev bench")
        return "typesafe-ai/jev via Vercel AI Gateway", []
    states, _ = bench_inputs(1)
    rows = []
    seen_errors = len(jev.transient_errors)
    for n in a.batches:
        _, questions = bench_inputs(n)
        qs = {k: {"type": "boolean", "instructions": v["instructions"]} for k, v in questions.items()}
        usages: list[dict] = []

        def call(i, q=qs):
            _, usage, ms = jev.evaluate(states[i % len(states)], q, use_cache=not a.no_cache)
            usages.append(usage)
            return {"roundtrip_ms": ms}

        t0 = time.perf_counter()
        call(0)
        first_call = (time.perf_counter() - t0) * 1e3
        res = timed_loop(call, warmup=a.warmup, calls=a.calls, budget_s=a.budget_s, min_calls=a.min_calls,
                         label=f"jev N={n}")
        usage = {k: float(np.mean([u.get(k, 0) for u in usages])) for k in ("inputTokens", "outputTokens")}
        errors = len(jev.transient_errors) - seen_errors
        seen_errors = len(jev.transient_errors)
        rows.append({"model": jev.MODEL, "N": n, "first_call_ms": first_call, "usage_mean": usage,
                     "transient_errors_retried": errors, **res})
        print(f"  N={n:2d} roundtrip p50 {res['roundtrip_ms']['p50']:6.0f} p95 {res['roundtrip_ms']['p95']:6.0f} "
              f"min {res['roundtrip_ms']['min']:6.0f} max {res['roundtrip_ms']['max']:6.0f} ms, first {first_call:.0f} ms, "
              f"tokens {usage['inputTokens']:.0f} in / {usage['outputTokens']:.0f} out (n={res['roundtrip_ms']['count']})",
              flush=True)
    return "typesafe-ai/jev via Vercel AI Gateway", rows


# --------------------------------------------------------------------------- laya

def run_laya(a: argparse.Namespace) -> tuple[str, list[dict]]:
    if not laya.available(a.hf_dir):
        print(f"no Laya checkpoint under {a.hf_dir}; skipping")
        return "convaiinnovations/laya (PyTorch, in-process, CPU)", []
    states, _ = bench_inputs(1)
    decider = laya.LayaDecider(a.hf_dir)
    print(f"loading {a.hf_dir} ...", flush=True)
    load_ms = decider.load()
    print(f"laya: load ms first {load_ms:.0f} (one load; the checkpoint is 421M params)")
    rows = []
    for n in a.batches:
        _, questions = bench_inputs(n)
        usages: list[dict] = []

        def call(i, q=questions):
            _, usage, timing, _ = decider.evaluate(states[i % len(states)], q)
            usages.append(usage)
            return {"total_ms": timing["ms_total"]}

        t0 = time.perf_counter()
        call(0)
        first_call = (time.perf_counter() - t0) * 1e3
        res = timed_loop(call, warmup=a.warmup, calls=a.calls, budget_s=a.budget_s, min_calls=a.min_calls,
                         label=f"laya N={n}")
        usage = {k: float(np.mean([u.get(k, 0) for u in usages])) for k in ("inputTokens", "outputTokens")}
        rows.append({"model": "laya", "N": n, "L": None, "load_ms": {"first": load_ms},
                     "first_call_ms": first_call, "usage_mean": usage, **res})
        print(f"  N={n:2d} total p50 {res['total_ms']['p50']:6.0f} p95 {res['total_ms']['p95']:6.0f} "
              f"min {res['total_ms']['min']:6.0f} max {res['total_ms']['max']:6.0f} ms, first {first_call:.0f} ms, "
              f"tokens {usage['inputTokens']:.0f} in (n={res['total_ms']['count']})", flush=True)
    return "convaiinnovations/laya (PyTorch, in-process, CPU)", rows


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["local", "remote", "jev", "laya"], default="local")
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--assets", nargs="+", choices=list(ASSETS), default=["dynamic", "static"])
    ap.add_argument("--batches", nargs="+", type=int)
    ap.add_argument("--lengths", nargs="+", type=int, default=list(STATIC_LENGTHS))
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--calls", type=int)
    ap.add_argument("--min-calls", type=int)
    ap.add_argument("--budget-s", type=float, default=60.0, help="wall-clock cap per row (local/remote)")
    ap.add_argument("--url", default="http://127.0.0.1:8770")
    ap.add_argument("--hf-dir", type=Path, default=laya.DEFAULT_HF_DIR, help="laya checkpoint dir")
    ap.add_argument("--no-cache", action="store_true", help="jev: bypass the response cache (live calls)")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    if a.backend == "jev":
        a.batches = a.batches or list(JEV_BATCHES)
        a.warmup = 5 if a.warmup is None else a.warmup
        a.calls = 30 if a.calls is None else a.calls
        a.min_calls = a.calls if a.min_calls is None else a.min_calls
        a.budget_s = 1e9
    elif a.backend == "laya":
        a.batches = a.batches or list(STATIC_BATCHES)
        a.warmup = 5 if a.warmup is None else a.warmup
        a.calls = 200 if a.calls is None else a.calls
        a.min_calls = 30 if a.min_calls is None else a.min_calls
    else:
        a.batches = a.batches or list(STATIC_BATCHES)
        a.warmup = 20 if a.warmup is None else a.warmup
        a.calls = 200 if a.calls is None else a.calls
        a.min_calls = 30 if a.min_calls is None else a.min_calls

    cache_was_warm = CACHE_DIR.exists()
    load_before = os.getloadavg()[0]  # other processes compete for the same cores; recorded, not controlled
    runtime, rows = {"local": run_local, "remote": run_remote, "jev": run_jev, "laya": run_laya}[a.backend](a)
    if a.json:
        record = {
            "schema": "decide-bench/1",
            "runtime": runtime,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": host_info(),
            "model_cache_warm": cache_was_warm,
            "loadavg_1m": {"before": load_before, "after": os.getloadavg()[0]},
            **({"model": laya.model_info(a.hf_dir)} if a.backend == "laya" and rows else {}),
            "config": {
                "backend": a.backend, "batches": a.batches,
                "lengths": None if a.backend in ("jev", "laya") else a.lengths,
                "warmup": a.warmup, "calls": a.calls, "min_calls": a.min_calls,
                "budget_s": None if a.backend == "jev" else a.budget_s,
                "inputs": "data/decide/holdout.jsonl states x data/decide/bench_questions.json",
                "note": ("Jev: N is questions per request; 30 timed calls after 5 warm-ups per N because the gateway "
                         "allows 30 req/min and calls are paced at 25/min (~4 min for the matrix); --no-cache means "
                         "every call was live" if a.backend == "jev" else
                         "Laya: N is questions per request, one forward pass in PyTorch fp32 on the CPU; it pads to "
                         "its own sequence so there is no L; 5 warm-ups then timed calls capped by budget_s; load_ms.first "
                         "is the single load of the 421M checkpoint (fp16 safetensors -> fp32)" if a.backend == "laya" else
                         "count per row is capped by budget_s; the local runtime takes seconds per call at large N x L"),
                **({"url": a.url} if a.backend == "remote" else {}),
                **({"no_cache": a.no_cache} if a.backend == "jev" else {}),
            },
            "rows": rows,
        }
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
