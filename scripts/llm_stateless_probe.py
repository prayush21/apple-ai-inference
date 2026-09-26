"""Stateless SmolLM2 row, measured the only way that fits on disk.

``SmolLM2.aimodel`` has a dynamic sequence length. On the macOS 27.0 OS
runtime every call with a new length compiles a new MPSGraph (~12 s, ~450 MB
in $TMPDIR/com.apple.MetalPerformanceShadersGraph until the process exits),
so a 64-token generation would be ~64 compiles. This probe times a repeated
length (compile vs steady state) and two new lengths, and merges one
``llm-bench/1`` record into docs/bench/llm-python.json.

    .venv/bin/python scripts/llm_stateless_probe.py --load-cache cold
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_ai.play import bench_prompt, git_sha, merge, runtime_name  # noqa: E402
from llm_ai.runtime import StatelessLM  # noqa: E402
from llm_ai.tokenizer import ChatTokenizer  # noqa: E402
from snake_ai.play import host_info  # noqa: E402


async def main(a):
    tok = ChatTokenizer()
    lm = await StatelessLM.load(compute="gpu")
    ids = bench_prompt(tok, 16 + 2)
    calls = []
    for n in (16, 16, 16, 17, 18, 18):
        t0 = time.perf_counter()
        await lm.logits(ids[:n])
        calls.append({"T": n, "ms": round((time.perf_counter() - t0) * 1e3, 1)})
        print(calls[-1], flush=True)
    # Repeats of an already-seen length only; the first call at each length compiles.
    steady = [c["ms"] for i, c in enumerate(calls) if any(p["T"] == c["T"] for p in calls[:i])]
    record = {
        "variant": "stateless", "runtime": runtime_name(), "compute": "gpu", "precision": "fp16",
        "prompt_tokens": 16, "generated_tokens": 0, "runs": 1,
        "load_ms": {"value": round(lm.load_ms, 1), "cache": a.load_cache},
        "calls": calls,
        "first_call_new_T_ms": [c["ms"] for c in calls if c["T"] in (17,)] + [calls[0]["ms"]],
        "steady_same_T_ms": min(steady),
        "note": "each new sequence length recompiles (~450 MB temp each); full 64-token generation not run",
        "host": host_info(), "git": git_sha(), "load_avg": [round(x, 2) for x in os.getloadavg()],
    }
    merge(Path("docs/bench/llm-python.json"), [record])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--load-cache", choices=["cold", "cached"], default="cached")
    asyncio.run(main(ap.parse_args()))
