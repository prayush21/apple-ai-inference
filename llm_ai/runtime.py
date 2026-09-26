"""Run the converted SmolLM2 assets through ``coreai.runtime`` (Python).

``StatefulLM`` wraps ``SmolLM2Stateful.aimodel``: it owns the
``keyCache`` / ``valueCache`` NDArrays, picks the smallest
``main_prefill_t*`` function a prompt chunk fits (left-padded, see
``model.left_pad``) and steps ``main_decode`` one token at a time.
``StatelessLM`` wraps ``SmolLM2.aimodel`` (full sequence every call).

Shared by ``verify``, ``play`` and ``serve``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

from .model import left_pad

DEFAULT_STATEFUL = Path("models/llm/SmolLM2Stateful.aimodel")
DEFAULT_STATELESS = Path("models/llm/SmolLM2.aimodel")
# Default specialization places the fp16 graph on the Neural Engine, which
# fails to load it on this 8 GB M2 (gotcha 17); "gpu" prefers the GPU.
COMPUTE_UNITS = ("gpu", "default", "cpu")


def specialization(compute: str) -> SpecializationOptions | None:
    if compute == "default":
        return None
    return SpecializationOptions.from_preferred_compute_unit_kind(getattr(ComputeUnitKind, compute)())


async def load_model(path: Path, compute: str) -> tuple[AIModel, float]:
    t0 = time.perf_counter()
    model = await AIModel.load(path, specialization(compute))
    return model, (time.perf_counter() - t0) * 1e3


def _np_dtype(descriptor) -> type:
    return np.float16 if "float16" in str(descriptor) else np.float32


def _ids(a: np.ndarray) -> NDArray:
    # The runtime does not coerce dtypes; coreai-torch made the int inputs si32 (gotcha 4).
    return NDArray(data=np.ascontiguousarray(a, dtype=np.int32))


class StatefulLM:
    def __init__(self, model: AIModel, load_ms: float) -> None:
        self.model = model
        self.load_ms = load_ms
        names = model.function_names
        self.prefill = {
            int(m.group(1)): model.load_function(n) for n in names if (m := re.fullmatch(r"main_prefill_t(\d+)", n))
        }
        if "main_decode" not in names or not self.prefill:
            raise KeyError(f"expected main_prefill_t* and main_decode, asset has {names}")
        self.decode_fn = model.load_function("main_decode")
        desc = self.decode_fn.desc.state_descriptor("keyCache")
        self.cache_shape = tuple(desc.shape)
        self.cache_dtype = _np_dtype(desc)
        self.max_seq_len = self.cache_shape[3]
        self.position = 0
        self.reset()

    @classmethod
    async def load(cls, path: Path = DEFAULT_STATEFUL, compute: str = "gpu") -> "StatefulLM":
        return cls(*await load_model(path, compute))

    def reset(self) -> None:
        """Fresh zeroed caches. Zeroed matters: masked slots get softmax weight
        0, but 0 * NaN from uninitialised memory would still poison attn @ v."""
        self.key_cache = NDArray(data=np.zeros(self.cache_shape, dtype=self.cache_dtype))
        self.value_cache = NDArray(data=np.zeros(self.cache_shape, dtype=self.cache_dtype))
        self.position = 0

    @property
    def state(self) -> dict[str, NDArray]:
        return {"keyCache": self.key_cache, "valueCache": self.value_cache}

    def bucket(self, n: int) -> int:
        fits = [t for t in sorted(self.prefill) if t >= n]
        if not fits:
            raise ValueError(f"{n} tokens exceed the largest prefill function ({max(self.prefill)})")
        return fits[0]

    async def prefill_ids(self, ids: list[int]) -> np.ndarray:
        """Append ``ids`` at the current position -> logits ``[vocab]`` of the last one.
        Chunks longer than the largest prefill function are split."""
        largest = max(self.prefill)
        logits = None
        for i in range(0, len(ids), largest):
            chunk = ids[i : i + largest]
            t = self.bucket(len(chunk))
            x, p = left_pad(chunk, self.position, t, self.max_seq_len)
            out = await self.prefill[t]({"input_ids": _ids(x.numpy()), "position_ids": _ids(p.numpy())}, state=self.state)
            self.position += len(chunk)
            logits = out["logits"].numpy()[0, -1]
        return logits

    async def decode(self, token: int) -> np.ndarray:
        if self.position >= self.max_seq_len:
            raise RuntimeError("KV cache is full; call reset()")
        out = await self.decode_fn(
            {"input_ids": _ids(np.array([[token]])), "position_ids": _ids(np.array([[self.position]]))},
            state=self.state,
        )
        self.position += 1
        return out["logits"].numpy()[0, -1]


class StatelessLM:
    def __init__(self, model: AIModel, load_ms: float) -> None:
        self.model = model
        self.load_ms = load_ms
        self.fn = model.load_function(model.function_names[0])

    @classmethod
    async def load(cls, path: Path = DEFAULT_STATELESS, compute: str = "gpu") -> "StatelessLM":
        return cls(*await load_model(path, compute))

    async def logits(self, ids: list[int]) -> np.ndarray:
        out = await self.fn({"input_ids": _ids(np.array([ids]))})
        return out["logits"].numpy()[0, -1]
