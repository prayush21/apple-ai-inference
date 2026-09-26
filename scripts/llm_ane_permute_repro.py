"""Gotcha 17 repro: 5-D grouped-query attention output -> ANE compile failure (macOS 27.0 OS runtime).
Usage: [COMPUTE=gpu|cpu] [DTYPE=fp32] .venv/bin/python scripts/llm_ane_permute_repro.py {permute5d|repeat4d} [T]"""
from pathlib import Path
import asyncio, os, sys, tempfile, numpy as np, torch, coreai_torch
from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions as SO
T = int(sys.argv[2]) if len(sys.argv) > 2 else 128
class M(torch.nn.Module):
    def forward(self, x, k, v):  # x [1, T, 960]; k, v [1, 5, 256, 64]
        if sys.argv[1] == "permute5d":
            q = x.view(1, T, 15, 64).transpose(1, 2).reshape(1, 5, 3, T, 64)
            a = (q @ k[:, :, None].transpose(-1, -2)).softmax(-1)
            return x + (a @ v[:, :, None]).permute(0, 3, 1, 2, 4).reshape(1, T, 960)
        q = x.view(1, T, 15, 64).transpose(1, 2)
        k, v = (t[:, :, None].expand(1, 5, 3, 256, 64).reshape(1, 15, 256, 64) for t in (k, v))
        return x + ((q @ k.transpose(-1, -2)).softmax(-1) @ v).transpose(1, 2).reshape(1, T, 960)
dt = torch.float32 if os.environ.get("DTYPE") == "fp32" else torch.float16
args = tuple(torch.randn(*s).to(dt) for s in [(1, T, 960), (1, 5, 256, 64), (1, 5, 256, 64)])
ep = torch.export.export(M(), args).run_decompositions(coreai_torch.get_decomp_table())
prog = coreai_torch.TorchConverter().add_exported_program(ep, input_names=["x", "k", "v"], output_names=["y"]).to_coreai()
prog.optimize(); d = tempfile.mkdtemp(); prog.save_asset(Path(d) / "m.aimodel")
async def run():
    cu = os.environ.get("COMPUTE"); opts = SO.from_preferred_compute_unit_kind(getattr(ComputeUnitKind, cu)()) if cu else None
    fn = (await AIModel.load(f"{d}/m.aimodel", opts)).load_function("main")
    y = (await fn({n: NDArray(data=a.numpy()) for n, a in zip("xkv", args)}))["y"].numpy()
    print(sys.argv[1], T, "max |diff|", float(np.abs(y.astype(np.float32) - M()(*[a.float() for a in args]).numpy()).max()))
asyncio.run(run())
