# Core AI ecosystem notes

Working notes from building the snake clone. Everything here was observed on
macOS 26.3 with `coreai-torch 0.4.2` / `coreai-core 1.0.0b2` unless stated.

## The pieces

| Piece | Form | Role |
|---|---|---|
| **coreai-core** (`import coreai`) | PyPI wheel, macOS arm64, py3.10–3.13 | `coreai.authoring` (build/optimize `AIProgram`, MLIR-based), `coreai.runtime` (`AIModel`, `InferenceFunction`, `NDArray`, profiler, intermediate logger, `SpecializationOptions`) |
| **coreai-torch** (`import coreai_torch`) | PyPI wheel, pure Python, needs `torch>=2.8` | `TorchConverter` from `torch.export` programs; `get_decomp_table()`; custom Metal kernels (`TorchMetalKernel`); debugging (validator, comparator, benchmarker, source annotator) |
| **coreai-optimization** | separate package (not needed here) | quantization / pruning / palettization for Apple Silicon |
| **CoreAI.framework** | OS framework, macOS 27 / iOS 27+ | Swift API: `AIModel`, `InferenceFunction`, `NDArray`, `AIModelCache`, states via `MutableViews` |
| **Core AI Models repo** | GitHub | ready-to-convert popular models, Swift package with model-family helpers, `LanguageModel` bridge into Foundation Models |
| Xcode 27 tooling | — | `.aimodel` inspector (Functions tab), AOT compilation to `.aimodelc`, Core AI Instrument, Core AI Debugger, debug gauge |

### Two runtimes inside one wheel

`coreai/runtime/__init__.py` picks a backend at import:

- `_coreai_runtime_os.so` — thin binding that links `/System/Library/Frameworks/CoreAI.framework`. Used when `macOS >= 27` and installed from a wheel (`USE_OS_COREAI` / `USE_LOCAL_COREAI` override).
- `_coreai_runtime.so` — a self-contained "StaticCoreAIRuntime" (kernel errors are reported as `StaticCoreAIRuntime.KernelError`). Used everywhere else, including macOS 26 and Linux.

So the Python half of the pipeline — author, convert, run, verify, profile — works today without the OS framework. Numerics matched PyTorch exactly for our model. Expect the OS runtime to differ in compute-unit dispatch (CPU/GPU/ANE) and in what specialization does, not in semantics.

## `.aimodel` on disk

```
SnakeTransformer.aimodel/
├── main.mlirb       # MLIR bytecode of the coreai dialect program (weights inline / as dense resources)
├── main.hash
└── metadata.json
```

`AIModelAsset.load(path).program` gives you the `AIProgram` back; `str(program)` prints readable MLIR. This is the fastest way to see what the converter produced (op names, dynamic dims as `?`, attributes on graph args).

## Conversion pipeline, and what each step does

```python
exported = torch.export.export(model, args=(example,), dynamic_shapes={...})
exported = exported.run_decompositions(coreai_torch.get_decomp_table())
program  = TorchConverter().add_exported_program(exported, input_names=[...], output_names=[...], state_names=[...]).to_coreai()
program.optimize()          # <- see gotcha 1
program.save_asset(Path("X.aimodel"))
```

- `torch.export.Dim` marks dynamic dims; they show up as `?` in the asset and in Xcode's Functions tab.
- `get_decomp_table()` lowers composite aten ops to the subset `_aten_to_core.py` knows how to emit. The converter has explicit lowerings for ~30 aten ops (`index`, `index_put`, `gather`, `scatter`, `slice_scatter`, `scaled_dot_product_attention`, `embedding`, `cumsum`, pooling/upsample, …) plus generic elementwise/matmul handling.
- `state_names` are applied in order to the exported program's mutated buffers (registration order), then to mutated user inputs.
- `TorchConverter(mode=Mode.DEBUG)` (the default) records source locations so the Core AI Debugger can map ops back to Python lines.

## Gotchas we hit (not covered in the session)

1. **States only exist after `AIProgram.optimize()`.** `to_coreai()` tags mutated inputs with `MutableBuffers.buffer_mutation = "keyCache"` but leaves them as ordinary inputs+outputs; the runtime reports `States: []` and `state=` calls fail with `RuntimeError: keyCache`. `optimize()` runs the `coreai-pre-compilation-rewrite` pass which rewrites them into `!coreai.handle<tensor<…>>` and the function then advertises `States: ["keyCache", "valueCache"]`. Presumably the OS runtime does this during specialization; doing it at conversion time makes the asset self-describing everywhere.

2. **`index_put` needs static shapes beyond dim 0.** `torch.index_copy` / `x[:, pos] = new` decompose to `aten.index_put`, whose lowering raises `index_put currently only supports dynamic batch dimension (dim 0)`. For KV-cache writes with a dynamic number of new tokens use `torch.scatter(cache, 1, index.expand(...), new)` (→ `coreai.scatter_along_axis`) or `slice_scatter` for contiguous positions (→ `coreai.slice_update`).

3. **Comparison kernels want identical operand shapes.** `arange(MAX)[None, :] <= pos[:, None]` converts fine but fails at runtime with `coreai.greater … Expected inputs to have the same type`. Expanding both sides to `[T, MAX]` explicitly before comparing fixes it (`broadcast_in_dims` is emitted for each side).

4. **The runtime does not coerce input dtypes.** `coreai-torch` maps int64 indices to `si32`, so `position_ids` must be passed as `int32` NDArrays or the same "same type" kernel error appears. Always read `function.desc.input_descriptor(name)` rather than assuming the PyTorch dtype.

5. **States are mutated in place.** After a stateful call, `key_cache.numpy()` reflects the update — no output to copy back. Reset by allocating fresh zero arrays (Python) or new `NDArray`s (Swift).

6. **Stateful is not free.** Attention always runs over the full fixed-size cache (256 slots here), so the per-call cost is a constant ~2x the stateless cost at T=1 — but it is flat, whereas stateless grows with T. For a 256-move game the crossover is around T≈20.

7. **Dynamic shapes cost type inference on every call.** The raw `Profiler` events for one call show `Function Type Inference` at ~3.8 ms of ~20 ms total. `AIProgram.set_static_shape_config("main", {"decode": {"features": (1,1,16), "position_ids": (1,1)}})` before `optimize()` attaches `coreai.enumerated_shapes` and emits a fully static `main_decode` function (the dynamic `main` is dropped). Per-step decode went 4.65 → 3.83 ms. This is the Python side of the talk's "check the optimal memory layout / pre-allocate outputs" tight-loop advice.

8. **`set_static_shape_config` renames the function.** The new entrypoint is `<graph>_<config name>`; `load_function("main")` then fails with `KeyError`. Always read `model.function_names`.

## BERT-style encoders (`decide_ai`, RoBERTa NLI cross-encoder)

Observed converting `cross-encoder/nli-MiniLM2-L6-H768` (6 layers, H=768,
50k-vocab byte-level BPE, 82M params) with `Dim("batch", 1..32)` and
`Dim("seq", 1..256)` on both inputs.

9. **The plain encoder converts first time.** Explicit softmax attention with
   an additive mask, `arange(L) + 2` position ids, `F.gelu`, LayerNorm and a
   `tanh` head all have lowerings; no rewrite was needed. Ops emitted:
   `broadcast_in_dims / broadcast_to / cast / concat / gather_nd / gelu /
   range / reduce_mean / reshape / rsqrt / slice / softmax / tanh / transpose`
   plus `coreai.decomposable.broadcasting_{batch_matmul,add,mul,sub}`. Max
   |logit diff| vs PyTorch on 20 real padded pairs: 8e-6 (dynamic), 6e-6
   (static). Batch-of-4 equals four batches-of-1 to 2e-6, so the mask path
   is right, and two runs of one batch are bit-identical.

10. **Expand the additive mask to `[N, H, L, L]` yourself.** Done
    pre-emptively because of gotcha 3; `(1 - mask) * -1e4` with an explicit
    `.view(N,1,1,L).expand(N,H,L,L)` produced one `broadcast_to` per layer
    and no "same type" errors. `-1e4` (not `-inf`) keeps every intermediate
    finite; padded rows with an all-zero mask softmax to uniform instead of
    NaN, which is what lets the static asset pad the batch dimension with
    dummy rows.

11. **`torch.export` refuses a size-1 example on a `Dim`.** Tracing with a
    batch of 1 specialises the dim to a constant and then raises
    "Constraints violated (batch) ... specialized it to be a constant (1)".
    Export with an example batch of 2 (and any L ≥ 2).

12. **`set_static_shape_config` takes several entrypoints at once.**
    `shapes_config` is `{config name: {input: shape}}`; every input mentioned
    gets one `coreai.enumerated_shapes` attribute listing all of its
    specialisations and `optimize()` emits one fully typed function per
    config. Eight configs (N ∈ {1,4,8,16} × L ∈ {64,128}) converted in 12 s
    and the asset is the same 313 MB as the dynamic one — weights are shared,
    only the graphs are duplicated. Gotcha 8 applies to every config: the
    function is `main_<config>`, so a config called `main_n1_l64` becomes
    `main_main_n1_l64`.

13. **The local (macOS 26) runtime's matmul is ~25 GFLOP/s.** One inference
    at N=1, L=64 is ~330 ms, of which 50 `broadcasting_batch_matmul` calls
    are ~240 ms (4.8 ms each for a `[64,768]·[768,768]` product) and six
    `gelu` calls ~17 ms; the same forward is 29 ms in PyTorch on the CPU.
    `nn.Linear` on `[N,L,D]`, a flattened `[N·L,D]` linear and an explicit
    `torch.mm` all lower to the same `broadcasting_batch_matmul` kernel and
    time the same, and `COREAI_FAST_KERNELS=1` changes nothing. The binary
    contains BNNS and Metal-stream symbols, but `SpecializationOptions`
    is unsupported on this runtime (see above), so there is no way to reach
    them from Python. This is the interim CPU runtime, not the model: at
    5.4 GFLOP per row the OS runtime on macOS 27 should be tens of ms.
    Measured on macOS 27.0 (2026-09-22): `CoreAI.framework` from Swift runs
    1 × 64 in 5.3 ms and 16 × 64 in 47 ms (41–91× this runtime), same
    asset, logits within 3e-6 — see docs/bench/README.md. Two things
    change on macOS 27: `coreai-core` in Python switches to the OS framework
    (`_coreai_runtime_os`) on its own, and the first load ever of an asset
    specializes it — 2.2 s dynamic, 94–102 s for the eight-function static
    asset, paid once by Python and once by Swift — then 10–12 ms.
    Consequence for the bench: cost scales with N·L, so pad to the
    smallest enumerated L that fits (64 covers every triage message in the
    holdout) rather than always using the ceiling.

14. **Static shapes save 3–20 % here, not the 18 % flat the snake decode saw.**
    "Function Type Inference" is ~14 ms per dynamic call. Static vs dynamic
    p50 over 20 calls: N=1 L=64 268 vs 276 ms; N=4 L=64 1040 vs 1311 ms;
    N=1 L=128 538 vs 533 ms; N=8 L=128 4371 vs 4377 ms. Once the matmuls
    are seconds, the fixed per-call overhead disappears in the noise.
    Full matrix in `docs/bench/decide-python.json`. The flip side: a request
    that does not match an enumerated shape is padded up to the next one,
    and cost is linear in N·L — the 5-question triage call runs as N=8/L=64
    on the static asset (2.1 s) but N=5/L=48 on the dynamic one (0.53 s).
    Enumerate the shapes you actually serve.

15. **First load writes a full copy of the weights to the specialization
    cache.** `~/Library/Caches/coreai-cache/<python>/<hash>/` gained 316 MB
    per asset (first load ~5–8 s, later loads ~250 ms). With two assets and
    the 328 MB checkpoint that is ~1.3 GB on disk for one 82M-parameter
    model; delete stale hashes when re-converting.

### Future work: ModernBERT (Laya) is a converter project of its own

`convaiinnovations/laya` (ModernBERT-large + decision head, 421M) runs in
`decide_ai` as a plain-PyTorch backend only; there is no `.aimodel` for it.
Two reasons, recorded so nobody restarts this by accident:

- **Disk.** The weights are fp16 on disk (843 MB); an fp32 `.aimodel` would
  be ~1.7 GB plus a ~1.6 GB runtime cache entry (gotcha 15 above scales with
  the checkpoint), and this machine has ~2 GB free.
- **Ops.** ModernBERT is not the plain encoder of gotcha 9: rotary position
  embeddings (a `cos`/`sin` gather + rotate-half per layer), sliding-window
  local attention on two of every three layers (a banded mask the converter
  would have to see as a static `[L, L]` tensor, or a windowed kernel),
  `global_attn_every_n_layers: 3` (two attention flavours in one graph), and
  `transformers`' unpadding path (`index_put` / gather to pack the batch —
  the exact op class that broke the snake KV cache, gotcha 2). Each of those
  is a rewrite before `torch.export` sees a clean graph, and the decision
  head adds a 2-layer `nn.TransformerEncoder` with a key-padding mask on top.

The honest comparison today is PyTorch on the CPU (what a user without a
GPU gets): ~0.3 s for a 5-question request on a quiet M2, ~35–60 s to load. See
`docs/bench/decide-laya.json`.

## Profiling and debugging from Python

| Xcode 27 tool | Python equivalent | Where |
|---|---|---|
| Core AI Instrument (per-inference / per-op intervals) | `coreai.runtime.Profiler(on_log_event_begin/end)` passed to `load_function(..., profiler=)` | `snake_ai.debug events` |
| Instrument's per-op table | `coreai_torch.debugging.benchmarker.benchmark_coreai_program(program, inputs, num_runs)` → `BenchmarkResult.write_summary`, `get_module_timings()` | `snake_ai.debug benchmark` |
| Debugger's "trace back to Python source" | `ModuleTiming.annotate_dominant_source(file)` writes the authoring `.py` with each line annotated by the ops it produced and their timings (needs `TorchConverter(mode=DEBUG)`, the default) | `snake_ai.debug benchmark` → `models/SnakeTransformer.annotated.py.txt` |
| Debugger's intermediate tensor inspection | `coreai.runtime.IntermediateLogger` + `coreai_torch.debugging.comparator.create_comparator_for_programs(exported, program, "main")` → `compare_with_tolerance(inputs)` bisects op-by-op | `snake_ai.debug compare` |
| Debug gauge | — (`Profiler` events are the same data) | |

Observed: the stateless snake model compares 108 op pairs pass / 0 fail (55 "unknown" are torch views/permutes with no Core AI op). Per-op time is dominated by `reshape`/`transpose`/`concat` glue (~50%), not matmuls — the model is too small for compute to matter.

## Specialization and caching from Python

- First `AIModel.load` of an asset specializes it (~140 ms for this model) and writes the artifacts to `~/Library/Caches/coreai-cache/<OS build>/<program hash>`; later loads take <1 ms. That directory is the Python-side analogue of Swift's `AIModelCache.default`.
- `SpecializationOptions.cpu_only()/default()/from_preferred_compute_unit_kind(ComputeUnitKind.gpu())/.with_debug(enabled=True)` are accepted, but `SpecializationOptions.is_supported()` is False on the in-package runtime: compute-unit delegation (GPU / Neural Engine) needs the OS `CoreAI.framework` (`USE_OS_COREAI` on macOS 27).
- `.aimodelc` ahead-of-time compilation is an Xcode 27 toolchain feature; nothing in the Python packages produces it.

## Model-side lessons

- Validation accuracy is a poor proxy for the snake actually surviving. Measured on 100 fresh games vs the heuristic (±5%): baseline cloning 27%; 2.7x more data 25%; a 5x bigger model (73% val acc vs 66%) **17%**; three DAgger rounds 31%. The bigger model imitates the teacher better on teacher-visited states and fails harder on its own (covariate shift); DAgger (`train.py --dagger-rounds`) helps modestly. Small-sample evals mislead — 30-game runs of the same checkpoints read 47–50%.
- The remaining gap is information, not capacity: the teacher's flood-fill sees the whole board, while the talk's 16 features carry no body occupancy. Closing it means adding features (a deliberate departure from the session's model), not a bigger transformer.

- Author with plain tensor ops (explicit softmax attention, `register_buffer` caches, `copy_` at the end of `forward`). The graph is easy to read in MLIR and every op has a lowering.
- Keep the stateless and stateful modules sharing one parameter set (subclass + `load_state_dict(strict=False)`) and unit-test their equivalence token-by-token *before* converting; it separates authoring bugs from conversion bugs.
- Verify numerics on real game inputs, not `randn`, and thread a real multi-step decode through the states — a single-call check would not have caught a broken cache write.

## Runtime API cheat-sheet (Python ↔ Swift)

| Python (`coreai.runtime`) | Swift (`CoreAI`) |
|---|---|
| `await AIModel.load(path, specialization_options=None)` | `try await AIModel(contentsOf: url)` |
| `model.function_names` / `model.load_function("main")` | `try model.loadFunction(named: "main")` |
| `fn.desc` (`input_names`, `state_names`, `output_names`, `*_descriptor(name)` → shape/dtype/storage) | Functions tab in Xcode; `stateDescriptor(named:)` |
| `NDArray(data=np_array)` / `.numpy()` | `NDArray(shape:scalarType:)`, `.mutableView()` / `.view()` |
| `await fn(inputs={...}, state={...})` → `dict[str, NDArray]` | `try await fn.run(inputs: [...], states: MutableViews)` → `outputs.remove("logits")?.ndArray` |
| `AIModel.load(path, SpecializationOptions(...))` | `AIModel.specialize(contentsOf:)`, `AIModelCache.default.model(for:options:)` |
| `model.load_function(name, profiler=Profiler())` | Core AI Instrument / debug gauge |
| `model.load_function(name, intermediate_logger=IntermediateLogger())` | Core AI Debugger |
