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

## Model-side lessons

- Author with plain tensor ops (explicit softmax attention, `register_buffer` caches, `copy_` at the end of `forward`). The graph is easy to read in MLIR and every op has a lowering.
- Keep the stateless and stateful modules sharing one parameter set (subclass + `load_state_dict(strict=False)`) and unit-test their equivalence token-by-token *before* converting; it separates authoring bugs from conversion bugs.
- Verify numerics on real game inputs, not `randn`, and thread a real multi-step decode through the states — a single-call check would not have caught a broken cache write.
