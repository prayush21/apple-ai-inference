"""Write the Swift BPEEncoder test fixture: diverse strings and their ids from Hugging Face ``tokenizers``.

    .venv/bin/python scripts/llm_tokenizer_fixture.py
    # -> LLMCoreAI/Tests/LLMCoreAITests/Fixtures/encode.json
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_ai.play import LONG_PROMPT  # noqa: E402
from llm_ai.prompt_ids import CHAT_PROMPTS  # noqa: E402
from llm_ai.tokenizer import ChatTokenizer, chat_prompt  # noqa: E402

OUT = Path("LLMCoreAI/Tests/LLMCoreAITests/Fixtures/encode.json")

HAND = [
    "", " ", "  ", "\n", "\n\n", " \n ", "a", " a", "a ", "a  b", "a   b", "a\tb", "a\n\nb", "  leading", "trailing  ",
    "Hello, world!", "I'm here; you're there. We'll see, they've gone, she'd go, it's ok.", "DON'T 'quote' 'S",
    "1234567890", "Pi is 3.14159, e is 2.71828.", "x1y22z333", "Room 101 at 9:30am on 2026-09-26", "½ ¾ ⅓ Ⅻ ٣ ४",
    "naïve café résumé", "Straße über Größe", "日本語のテキスト", "中文，标点。", "한국어 문장", "Привет, мир!",
    "مرحبا بالعالم", "emoji 😀👍🏽 family 👨‍👩‍👧 flags 🇺🇸", "tabs\tand nbsp em space", "zero​width",
    "def f(x):\n    return x ** 2  # square\n", "if (a && b) { c = d[i++]; }", "https://example.com/a?b=c&d=e#f",
    "snake_case camelCase kebab-case", "...!!!???", "--- *** ___", "$100 and €50 or £20", "<|im_start|>user\nhi<|im_end|>\n",
    "<|im_end|><|im_start|><|endoftext|>", "text<|im_end|>more", "<|im_start", "a <repo_name> b",
]


def main() -> None:
    tok = ChatTokenizer()
    rng = random.Random(0)
    pool = HAND + CHAT_PROMPTS + [LONG_PROMPT]
    pool += [chat_prompt([{"role": "user", "content": p}]) for p in CHAT_PROMPTS]
    pool += [chat_prompt([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"},
                          {"role": "assistant", "content": "4"}, {"role": "user", "content": "And 3×3?"}])]
    words = (LONG_PROMPT + " " + " ".join(HAND)).split(" ")
    seps = [" ", "  ", "\n", " \n", "\t", "", "'s ", ", ", "123"]
    for _ in range(300):
        pool.append("".join(rng.choice(words) + rng.choice(seps) for _ in range(rng.randint(1, 12))))
    cases = [{"text": t, "ids": tok.encode(t)} for t in pool]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(cases)} cases")


if __name__ == "__main__":
    main()
