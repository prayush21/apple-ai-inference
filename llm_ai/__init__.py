"""SmolLM2-360M through the Core AI toolchain.

The snake project (``snake_ai``) proved the pipeline on a 118k-parameter
transformer. This package repeats it at a scale where latency is felt: a
decoder-only LLM whose KV cache is carried as Core AI *states* and whose
output is a tokens/sec counter rather than a snake move.

Milestones (each gets its own module, mirroring ``snake_ai``):

1. ``download``   fetch weights + tokenizer from Hugging Face       (done)
2. ``model``      plain-``torch`` Llama re-implementation with
                  ``register_buffer`` KV caches, loading HF weights
3. ``convert``    ``torch.export`` → ``coreai_torch`` → ``.aimodel``
                  with ``prefill`` (dynamic T) and ``decode`` (static
                  ``[1, 1]``) functions sharing the cache states
4. ``verify``     PyTorch ≙ Core AI logits over a real prompt + decode
5. ``play``       stream tokens in the terminal, record tok/s
6. ``serve``      HTTP bridge (SSE token stream) for the Swift app until
                  ``CoreAI.framework`` is available on macOS 27
7. ``bench``      ``llm-bench/1`` JSON records next to ``docs/bench/snake*``
"""
