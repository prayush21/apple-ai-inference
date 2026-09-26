"""SmolLM2-360M through the Core AI toolchain.

The snake project (``snake_ai``) proved the pipeline on a 118k-parameter
transformer. This package repeats it at a scale where latency is felt: a
decoder-only LLM whose KV cache is carried as Core AI *states* and whose
output is a tokens/sec counter rather than a snake move.

Milestones (each gets its own module, mirroring ``snake_ai``):

1. ``download``   fetch weights + tokenizer from Hugging Face            (done)
2. ``model``      plain-``torch`` Llama with ``register_buffer`` KV
                  caches; ``reference`` checks it against transformers  (done)
3. ``convert``    ``torch.export`` → ``coreai_torch`` → ``.aimodel`` with
                  static ``main_prefill_t64`` and ``main_decode`` sharing
                  the cache states                                      (done)
4. ``verify``     PyTorch ≙ Core AI logits over a real prompt + decode   (done)
5. ``play``       stream tokens in the terminal; ``--bench`` writes
                  ``llm-bench/1`` records                               (done)
6. Swift          ``LLMCoreAI.ModelGenerator`` in-process on
                  ``CoreAI.framework`` (macOS 27), prompts pre-tokenized by
                  ``prompt_ids``; ``llm-cli`` bench, ``LLMApp`` gauge     (done)
7. ``serve``      HTTP bridge (SSE token stream) to measure the
                  per-token network hop                             (deferred)
"""
