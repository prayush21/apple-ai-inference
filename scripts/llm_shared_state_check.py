"""Do two static functions of one asset share their Core AI states?

A tiny random 2-layer SmolLM is converted exactly like the real one
(``llm_ai.convert.build_stateful``); ``main_prefill_t16`` then fills the
caches from a prompt and ``main_decode`` continues from them. If the
functions saw separate buffers, decode would attend to an empty cache and
its logits would not match PyTorch.

    .venv/bin/python scripts/llm_shared_state_check.py [--precision fp16]
"""

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModel, NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_ai.convert import DTYPES, build_stateful, save  # noqa: E402
from llm_ai.model import SmolLM, SmolLMStateful, left_pad  # noqa: E402
from tests.test_llm_model import TINY  # noqa: E402


async def main(precision: str) -> None:
    torch.manual_seed(0)
    ref = SmolLM(TINY).eval()
    st = SmolLMStateful(TINY).eval()
    st.load_state_dict(ref.state_dict(), strict=False)
    ids = torch.randint(0, TINY.vocab_size, (1, 20))
    with torch.no_grad():
        expected = ref(ids)[0, 12:20].numpy()

    with tempfile.TemporaryDirectory() as d:
        asset = Path(d) / "tiny.aimodel"
        save(build_stateful(st.to(DTYPES[precision]), prefill_lengths=(16, 32)), asset)
        model = await AIModel.load(asset)
        print("functions:", model.function_names)
        prefill = model.load_function("main_prefill_t16")
        decode = model.load_function("main_decode")
        desc = decode.desc
        for n in desc.input_names:
            print(f"  input  {n:12s} {desc.input_descriptor(n)}")
        for n in desc.state_names:
            print(f"  state  {n:12s} {desc.state_descriptor(n)}")
        sd = desc.state_descriptor("keyCache")
        np_dtype = np.float16 if "16" in str(sd) else np.float32
        k = NDArray(data=np.zeros(TINY.kv_cache_shape, dtype=np_dtype))
        v = NDArray(data=np.zeros(TINY.kv_cache_shape, dtype=np_dtype))
        state = {"keyCache": k, "valueCache": v}

        x, p = left_pad(ids[0, :13].tolist(), 0, 16, TINY.max_seq_len)
        got = [(await prefill({"input_ids": NDArray(data=x.numpy()), "position_ids": NDArray(data=p.numpy())},
                              state=state))["logits"].numpy()[0, -1]]
        after_prefill = np.abs(k.numpy()).sum()
        for t in range(13, 20):
            out = await decode({"input_ids": NDArray(data=ids[:, t : t + 1].numpy().astype(np.int32)),
                                "position_ids": NDArray(data=np.array([[t]], dtype=np.int32))}, state=state)
            got.append(out["logits"].numpy()[0, -1])
        diff = np.abs(np.stack(got).astype(np.float32) - expected).max(axis=-1)
        print(f"cache |sum| after prefill {after_prefill:.2f}; per-step max |logit diff| {np.round(diff, 5).tolist()}")
        # Decode against an *empty* cache for contrast: this is what separate buffers would give.
        k0 = NDArray(data=np.zeros(TINY.kv_cache_shape, dtype=np_dtype))
        v0 = NDArray(data=np.zeros(TINY.kv_cache_shape, dtype=np_dtype))
        out = await decode({"input_ids": NDArray(data=ids[:, 13:14].numpy().astype(np.int32)),
                            "position_ids": NDArray(data=np.array([[13]], dtype=np.int32))},
                           state={"keyCache": k0, "valueCache": v0})
        cold = np.abs(out["logits"].numpy()[0, -1].astype(np.float32) - expected[1]).max()
        print(f"contrast, decode on an empty cache: max |logit diff| {cold:.4f}")
        tol = 5e-2 if precision == "fp16" else 1e-4
        print("SHARED" if diff.max() < tol else "NOT SHARED / MISMATCH")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=list(DTYPES), default="fp32")
    asyncio.run(main(ap.parse_args().precision))
