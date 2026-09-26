"""llm_ai.model on a tiny random Llama (fast) and the real checkpoint (slow)."""

import json
from pathlib import Path

import pytest
import torch

from llm_ai.model import LlamaConfig, SmolLM, SmolLMStateful, left_pad
from llm_ai.tokenizer import DEFAULT_MODEL_DIR, chat_prompt

TINY = LlamaConfig(
    vocab_size=97,
    hidden_size=48,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=6,
    num_key_value_heads=2,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    max_position_embeddings=256,
    max_seq_len=64,
)


def tiny_pair():
    torch.manual_seed(0)
    ref = SmolLM(TINY).eval()
    st = SmolLMStateful(TINY).eval()
    st.load_state_dict(ref.state_dict(), strict=False)
    return ref, st


@torch.no_grad()
def test_stateful_prefill_then_decode_matches_stateless():
    ref, st = tiny_pair()
    ids = torch.randint(0, TINY.vocab_size, (1, 20))
    full = ref(ids)[0]
    # Prefill 13 tokens left-padded to 16, then decode the remaining 7 one by one.
    x, p = left_pad(ids[0, :13].tolist(), 0, 16, TINY.max_seq_len)
    out = [st(x.long(), p.long())[0, -1]]
    for t in range(13, 20):
        out.append(st(ids[:, t : t + 1], torch.tensor([[t]]))[0, -1])
    assert torch.allclose(torch.stack(out), full[12:20], atol=1e-5)


@torch.no_grad()
def test_chunked_prefill_continues_from_start():
    ref, st = tiny_pair()
    ids = torch.randint(0, TINY.vocab_size, (1, 30))
    x, p = left_pad(ids[0, :10].tolist(), 0, 16, TINY.max_seq_len)
    st(x.long(), p.long())
    x, p = left_pad(ids[0, 10:30].tolist(), 10, 32, TINY.max_seq_len)
    last = st(x.long(), p.long())[0, -1]
    assert torch.allclose(last, ref(ids)[0, -1], atol=1e-5)


def test_left_pad_refuses_the_last_slot():
    with pytest.raises(ValueError):
        left_pad([1, 2, 3], 61, 4, 64)
    x, p = left_pad([1, 2, 3], 60, 4, 64)
    assert p.tolist() == [[63, 60, 61, 62]] and x.dtype == torch.int32


def test_chat_prompt_default_system():
    s = chat_prompt([{"role": "user", "content": "hi"}])
    assert s.startswith("<|im_start|>system\nYou are a helpful AI assistant named SmolLM")
    assert s.endswith("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")


@pytest.mark.slow
@pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "model.safetensors").exists(), reason="run python -m llm_ai.download")
@torch.no_grad()
def test_golden_greedy_run():
    golden = json.loads(Path("data/llm/golden.json").read_text())
    model = SmolLM.from_hf(DEFAULT_MODEL_DIR)
    seq = list(golden["prompt_ids"])
    for expected in golden["greedy_ids"][:16]:
        nxt = int(model(torch.tensor([seq]))[0, -1].argmax())
        assert nxt == expected
        seq.append(nxt)
