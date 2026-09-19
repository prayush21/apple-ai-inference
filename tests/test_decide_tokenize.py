"""Byte-identical parity between decide_ai.tokenize and Hugging Face ``tokenizers``."""

import pytest

from decide_ai.download import model_dir
from decide_ai.tokenize import BPETokenizer, pre_tokenize

HF_DIR = model_dir()
pytestmark = pytest.mark.skipif(not (HF_DIR / "tokenizer.json").exists(), reason="run decide_ai.download first")

STRINGS = [
    "", " ", "  ", "\t", "\n\n", "a", " a", "a ", "  a  b   c ",
    "The pasta was cold and the waiter ignored us.",
    "I don't want a refund, just send a replacement.",
    "Order #48213 never showed up.", "Still waiting.", "Hmm.",
    "It's 3:45pm and I'VE had it. Don't.", "we're we've I'm they'll you'd 'S 'T",
    "email me at foo.bar+baz@example.com or call +1 (555) 010-2030",
    "Price: $1,299.99 -- 20% off!!! ¿Qué? ¡Sí!", "naïve café résumé Zürich",
    "日本語のテキストです。", "中文 测试 123", "한국어 문장", "مرحبا بالعالم", "Привет, мир!",
    "emoji 🚚📦 arrived 🎉🎉", "👨‍👩‍👧‍👦 family", "tabs\tand\nnewlines\r\nmixed",
    "non-breaking space and em space and　ideographic", "zero​width",
    "NELchar", "ctrlchars", "trailing spaces   ", "   leading spaces",
    "á combining", "ﬁ ligature ½ ² ³ Ⅷ ٣", "snake_case camelCase kebab-case",
    "http://example.com/path?q=1&r=2#frag", "<html><body>tag</body></html>",
    "quotes “smart” ‘single’ \"straight\"", "math ∑ ∫ ≠ ≤ ≥ ± × ÷",
    "very " * 200 + "long", "x" * 500, "1234567890" * 30,
    "Package arrived a day early, thanks so much!",
    "This is the third time I've written in about this.",
    "The blender works but sounds like a jet engine. Not sure if that's normal.",
]
# Pad out to 200+ with generated variants: case flips, trailing spaces, a numeric tail.
for _i in range(160):
    _base = STRINGS[_i % 40]
    STRINGS.append((_base.upper() if _i % 3 == 0 else _base.lower()) + " " * (_i % 4) + f"#{_i}")


@pytest.fixture(scope="module")
def ours():
    return BPETokenizer.from_hf(HF_DIR)


@pytest.fixture(scope="module")
def theirs():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))


def test_pre_tokenize_matches_gpt2_pattern():
    from tokenizers import decoders
    from tokenizers.pre_tokenizers import ByteLevel

    bl = ByteLevel(add_prefix_space=False)
    dec = decoders.ByteLevel()
    for s in STRINGS:
        want = [dec.decode([p]) for p, _ in bl.pre_tokenize_str(s)]
        assert pre_tokenize(s) == want, repr(s)


def test_encode_matches_tokenizers(ours, theirs):
    assert len(STRINGS) >= 200
    for s in STRINGS:
        want = theirs.encode(s, add_special_tokens=False).ids
        assert ours.encode(s) == want, repr(s)


@pytest.mark.parametrize("max_len", [16, 64, 128])
def test_encode_pair_matches_tokenizers(ours, theirs, max_len):
    theirs.enable_truncation(max_len, strategy="longest_first")
    theirs.enable_padding(length=max_len, pad_id=ours.pad_id, pad_token="<pad>")
    try:
        hyps = ["The customer is complaining.", "A refund is requested.", "x" * 400]
        for i, s in enumerate(STRINGS):
            h = hyps[i % len(hyps)]
            enc = theirs.encode(s, h)
            ids, mask = ours.encode_pair(s, h, max_len)
            assert ids == enc.ids, (repr(s), h)
            assert mask == enc.attention_mask, (repr(s), h)
    finally:
        theirs.no_truncation()
        theirs.no_padding()


def test_pair_layout(ours):
    ids, mask = ours.encode_pair("hi", "yo", 16)
    n = sum(mask)
    assert ids[0] == ours.bos_id and ids[n - 1] == ours.eos_id
    assert ids.count(ours.eos_id) == 3 and ids[n:] == [ours.pad_id] * (16 - n)
    assert n == ours.count_tokens("hi", "yo")
