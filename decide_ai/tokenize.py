"""Byte-level BPE tokenizer for the RoBERTa-family cross-encoder, dependency-free.

This is the reference for the Swift port (step 2), so it favours a literal,
readable description of what Hugging Face ``tokenizers`` does for this
checkpoint over speed. ``tests/test_decide_tokenize.py`` checks it produces
byte-identical ids to ``tokenizers`` on a few hundred diverse strings.

Pipeline (``tokenizer.json``: no normalizer, ``ByteLevel`` pre-tokenizer with
``add_prefix_space=false``, ``BPE`` model, ``RobertaProcessing`` post-processor):

1. **Pre-tokenize** with the GPT-2 pattern, spelled out by hand because
   Python's ``re`` lacks ``\\p{L}`` / ``\\p{N}``::

       's|'t|'re|'ve|'m|'ll|'d | ?\\p{L}+ | ?\\p{N}+ | ?[^\\s\\p{L}\\p{N}]+ | \\s+(?!\\S) | \\s+

   Letters / numbers are Unicode general categories ``L*`` / ``N*``;
   whitespace is the Unicode ``White_Space`` property (``WHITESPACE`` below);
   the optional leading space is U+0020 only. ``\\s+(?!\\S)`` means: a
   whitespace run keeps its last character for the next piece when a
   non-space follows (so ``"a   b"`` -> ``"a"``, ``"  "``, ``" b"``).
2. **Bytes -> printable unicode**: each UTF-8 byte of a piece maps to a
   single character through GPT-2's ``bytes_to_unicode`` table, so every
   byte string is representable and there is no ``<unk>``.
3. **BPE**: repeatedly merge the adjacent pair with the lowest rank in
   ``merges.txt`` until none applies; look each symbol up in ``vocab.json``.
4. **Pair layout** (RoBERTa): ``<s> A </s></s> B </s>`` — four special tokens
   and *two* separators between the sequences. No token-type ids.
5. **Truncate longest-first** to ``max_len`` (drop tokens from the right of
   whichever sequence is longer; ties drop from B), then right-pad with
   ``<pad>`` (id 1) and build ``attention_mask``.

    tok = BPETokenizer.from_hf(Path("models/decide/hf/nli-MiniLM2-L6-H768"))
    ids, mask = tok.encode_pair("Order #48213 never showed up.", "This is about shipping.", max_len=64)
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

# Unicode White_Space property (what \s means in the tokenizer's regex engine).
WHITESPACE = frozenset(
    [chr(c) for c in range(0x09, 0x0E)]
    + [chr(c) for c in (0x20, 0x85, 0xA0, 0x1680)]
    + [chr(c) for c in range(0x2000, 0x200B)]
    + [chr(c) for c in (0x2028, 0x2029, 0x202F, 0x205F, 0x3000)]
)
CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")  # case-sensitive, as in the pattern


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


def _is_other(ch: str) -> bool:
    """``[^\\s\\p{L}\\p{N}]``: punctuation, symbols, marks, controls not in White_Space."""
    return ch not in WHITESPACE and not _is_letter(ch) and not _is_number(ch)


def pre_tokenize(text: str) -> list[str]:
    """Split ``text`` into the pieces the GPT-2 regex would produce."""
    pieces: list[str] = []
    i, n = 0, len(text)
    while i < n:
        # 's|'t|'re|'ve|'m|'ll|'d
        if text[i] == "'":
            hit = next((c for c in CONTRACTIONS if text.startswith(c, i)), None)
            if hit is not None:
                pieces.append(hit)
                i += len(hit)
                continue
        # ' ?\p{L}+' | ' ?\p{N}+' | ' ?[^\s\p{L}\p{N}]+'
        j = i + 1 if text[i] == " " and i + 1 < n else i
        for pred in (_is_letter, _is_number, _is_other):
            if pred(text[j]):
                k = j + 1
                while k < n and pred(text[k]):
                    k += 1
                pieces.append(text[i:k])
                i = k
                break
        else:
            # whitespace run: \s+(?!\S) then \s+
            k = i + 1
            while k < n and text[k] in WHITESPACE:
                k += 1
            if k < n and k - i > 1:  # a non-space follows: leave the last space to it
                k -= 1
            pieces.append(text[i:k])
            i = k
    return pieces


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable character table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class BPETokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]], *,
                 bos_id: int = 0, pad_id: int = 1, eos_id: int = 2) -> None:
        self.vocab = vocab
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.bos_id, self.pad_id, self.eos_id = bos_id, pad_id, eos_id
        self._byte_map = bytes_to_unicode()
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def from_hf(cls, hf_dir: Path) -> "BPETokenizer":
        hf_dir = Path(hf_dir)
        vocab = json.loads((hf_dir / "vocab.json").read_text(encoding="utf-8"))
        merges = []
        for line in (hf_dir / "merges.txt").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#version"):
                continue
            a, b = line.split(" ")
            merges.append((a, b))
        specials = json.loads((hf_dir / "special_tokens_map.json").read_text())

        def tok(k: str) -> str:
            v = specials[k]
            return v if isinstance(v, str) else v["content"]

        return cls(vocab, merges, bos_id=vocab[tok("cls_token")], pad_id=vocab[tok("pad_token")],
                   eos_id=vocab[tok("sep_token")])

    # ------------------------------------------------------------------ core

    def _bpe(self, piece: str) -> list[int]:
        if piece in self._cache:
            return self._cache[piece]
        word = [self._byte_map[b] for b in piece.encode("utf-8")]
        while len(word) > 1:
            best = None
            for a, b in zip(word, word[1:]):
                r = self.ranks.get((a, b))
                if r is not None and (best is None or r < best[0]):
                    best = (r, a, b)
            if best is None:
                break
            _, a, b = best
            merged: list[str] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        ids = [self.vocab[w] for w in word]
        self._cache[piece] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Token ids for ``text`` with no special tokens."""
        out: list[int] = []
        for piece in pre_tokenize(text):
            out.extend(self._bpe(piece))
        return out

    def encode_pair(self, a: str, b: str, max_len: int) -> tuple[list[int], list[int]]:
        """``<s> a </s></s> b </s>`` truncated longest-first and right-padded to ``max_len``.
        Returns ``(input_ids, attention_mask)``."""
        ia, ib = self.encode(a), self.encode(b)
        budget = max_len - 4
        if budget < 0:
            raise ValueError(f"max_len={max_len} cannot hold the 4 special tokens")
        while len(ia) + len(ib) > budget:
            if len(ia) > len(ib):
                ia.pop()
            else:
                ib.pop()
        ids = [self.bos_id, *ia, self.eos_id, self.eos_id, *ib, self.eos_id]
        mask = [1] * len(ids)
        pad = max_len - len(ids)
        return ids + [self.pad_id] * pad, mask + [0] * pad

    def encode_batch(self, pairs: list[tuple[str, str]], max_len: int) -> tuple[np.ndarray, np.ndarray]:
        """-> ``input_ids [N, max_len]``, ``attention_mask [N, max_len]``, both int32."""
        rows = [self.encode_pair(a, b, max_len) for a, b in pairs]
        ids = np.asarray([r[0] for r in rows], dtype=np.int32)
        mask = np.asarray([r[1] for r in rows], dtype=np.int32)
        return ids, mask

    def count_tokens(self, a: str, b: str) -> int:
        """Unpadded pair length (for ``usage.inputTokens``)."""
        return len(self.encode(a)) + len(self.encode(b)) + 4
