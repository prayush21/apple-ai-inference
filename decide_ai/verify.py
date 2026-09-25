"""Verify the converted ``.aimodel`` assets match the PyTorch model numerically.

    python -m decide_ai.verify                 # both assets
    python -m decide_ai.verify --asset static --tol 1e-4

For each asset, 20 real tokenized triage pairs with real right-padding go
through PyTorch and ``coreai.runtime``; the max |logit diff| is printed and
asserted. Then:

* batch-of-4 vs four batches-of-1 (per-row independence through the mask),
* two runs of the same batch must be bit-identical (the local model is
  deterministic; Jev is not — that asymmetry is measured in calibrate.py),
* static asset only: every enumerated ``main_n{N}_l{L}`` function agrees
  with the dynamic PyTorch reference on padded copies of the same pairs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .convert import DEFAULT_OUT_DIR
from .decider import LocalModel, _pad_rows
from .download import DEFAULT_HF_DIR, model_dir
from .model import NLICrossEncoder
from .tokenize import BPETokenizer

# Same 20 pairs as tests/test_decide_model.py: mixed lengths so padding is real.
PAIRS = [
    ("The pasta was cold and the waiter ignored us, I want my money back.", "The customer is complaining."),
    ("The pasta was cold and the waiter ignored us, I want my money back.", "The customer is asking for a refund."),
    ("Package arrived a day early, thanks so much!", "This is about shipping or delivery."),
    ("Package arrived a day early, thanks so much!", "The customer is complaining."),
    ("I don't want a refund, just send a replacement.", "The customer is asking for a refund."),
    ("No complaints about the product, but the box was crushed in transit.", "This is about product quality."),
    ("This is the third time I've written in about this.", "This is urgent."),
    ("Still waiting.", "This is about shipping or delivery."),
    ("Order #48213 never showed up.", "The message mentions a specific order number."),
    ("My order never showed up.", "The message mentions a specific order number."),
    ("Hmm.", "The customer is complaining."),
    ("Can I change the color on my order before it ships?", "The customer is asking for a refund."),
    ("I can't log in and I also want a refund for last month.", "The customer is asking for a refund."),
    ("The blender works but sounds like a jet engine. Not sure if that's normal.", "This is about product quality."),
    ("Charged twice for the same subscription, please fix it today.", "This is urgent."),
    ("Love the new colour, exactly as pictured.", "The customer is complaining."),
    ("Where is my parcel? Tracking hasn't moved in nine days.", "This is about shipping or delivery."),
    ("The zipper broke on day two.", "This is about product quality."),
    ("Please cancel and refund, I ordered by mistake.", "The customer is asking for a refund."),
    ("Thanks for the quick reply yesterday.", "This is urgent."),
]


def reference(model: NLICrossEncoder, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(ids), torch.from_numpy(mask)).numpy()


def verify_asset(asset: str, models_dir: Path, hf_dir: Path, length: int, tol: float) -> float:
    pt = NLICrossEncoder.from_pretrained(hf_dir)
    tok = BPETokenizer.from_hf(hf_dir)
    local = LocalModel.load(models_dir, asset)
    print(f"loaded {local.path.name} in {local.load_ms:.0f} ms: functions={list(local.functions)}")
    fn = next(iter(local.functions.values()))
    for n in fn.desc.input_names:
        print(f"  input  {n:15s} {fn.desc.input_descriptor(n)}")
    for n in fn.desc.output_names:
        print(f"  output {n:15s} {fn.desc.output_descriptor(n)}")

    ids, mask = tok.encode_batch(PAIRS, length)
    want = reference(pt, ids, mask)

    # 1. 20 real pairs in chunks of 16 (the largest enumerated static N).
    got = np.concatenate([local.run(ids[i : i + 16], mask[i : i + 16]) for i in range(0, len(PAIRS), 16)])
    worst = float(np.abs(got - want).max())
    print(f"  20 pairs @ L={length}: max |diff| = {worst:.2e}  ({'OK' if worst < tol else 'FAIL'} tol {tol})")
    for i in (4, 8, 10):  # negation, extraction, one-word state
        print(f"    {PAIRS[i][0][:40]!r:44} pytorch {np.round(want[i], 3)}  coreai {np.round(got[i], 3)}")
    assert worst < tol

    # 2. batch of 4 vs 4 x batch of 1
    b4 = local.run(ids[:4], mask[:4])
    b1 = np.concatenate([local.run(ids[i : i + 1], mask[i : i + 1]) for i in range(4)])
    d = float(np.abs(b4 - b1).max())
    print(f"  batch-of-4 vs 4 x batch-of-1: max |diff| = {d:.2e}")
    assert d < tol

    # 3. determinism: bit-identical across two runs
    again = np.concatenate([local.run(ids[i : i + 16], mask[i : i + 16]) for i in range(0, len(PAIRS), 16)])
    identical = np.array_equal(got, again)
    print(f"  two runs of the same batch bit-identical: {identical}")
    assert identical

    # 4. static: every enumerated function
    for (n, ll), name in sorted(local.static_shapes.items()):
        sub_ids, sub_mask = tok.encode_batch(PAIRS[:n], ll)
        out = local.run(_pad_rows(sub_ids, n, ll, fill=1), _pad_rows(sub_mask, n, ll, fill=0), function=name)
        ref = reference(pt, sub_ids, sub_mask)
        dd = float(np.abs(out[: len(sub_ids)] - ref).max())
        print(f"  {name:14s} max |diff| = {dd:.2e}")
        assert dd < tol
    return worst


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", type=Path, default=DEFAULT_HF_DIR)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--asset", choices=["dynamic", "static", "both"], default="both")
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args(argv)
    hf_dir = model_dir(a.hf_dir)
    for asset in (["dynamic", "static"] if a.asset == "both" else [a.asset]):
        verify_asset(asset, a.models_dir, hf_dir, a.length, a.tol)


if __name__ == "__main__":
    main()
