"""LayaDecider: boolean -> noul mapping, Jev-shaped answers, bit-identical repeat calls.

Marked ``slow`` (the 421M checkpoint takes 35-60 s to load); skipped when the
checkpoint is not under ``models/decide/hf/laya``.
"""

import pytest

from decide_ai.laya import DEFAULT_HF_DIR, LayaDecider, available

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not available(DEFAULT_HF_DIR), reason="Laya checkpoint not downloaded")]

STATE = "I don't want a refund, just send a replacement. Order #48213 never showed up."
QUESTIONS = {
    "wants_refund": {"type": "boolean", "instructions": "Is the customer asking for a refund?",
                     "hypothesis": "The customer is asking for a refund."},
    "about_shipping": {"type": "boolean", "instructions": "Is this message about shipping or delivery?"},
}


@pytest.fixture(scope="module")
def decider():
    d = LayaDecider(DEFAULT_HF_DIR)
    d.load()
    return d


def test_boolean_maps_to_noul(decider, monkeypatch):
    seen = {}
    real = decider.agent.system_one

    def spy(state, questions):
        seen["questions"] = questions
        return real(state, questions)

    monkeypatch.setattr(decider.agent, "system_one", spy)
    decider.evaluate(STATE, QUESTIONS)
    assert set(seen["questions"]) == set(QUESTIONS)
    for q in seen["questions"].values():
        assert q["type"] == "noul"
        assert "hypothesis" not in q  # question form only, like Jev


def test_answers_match_jev_shape(decider):
    answers, usage, timing, raw = decider.evaluate(STATE, QUESTIONS)
    assert set(answers) == set(QUESTIONS)
    for name, a in answers.items():
        assert set(a) == {"type", "probability"} and a["type"] == "boolean"
        assert 0.0 <= a["probability"] <= 1.0
        assert set(raw[name]) == {"noul", "act_probability", "temperature"}
    assert set(usage) == {"inputTokens", "outputTokens"} and usage["outputTokens"] == 0 and usage["inputTokens"] > 0
    assert timing["batch"] == 2 and timing["ms_total"] > 0
    assert decider.temperature == pytest.approx(1.9834, abs=1e-3)  # rl_agent_config.json noul:2


def test_repeat_is_bit_identical(decider):
    a1, _, _, _ = decider.evaluate(STATE, QUESTIONS)
    a2, _, _, _ = decider.evaluate(STATE, QUESTIONS)
    assert [a1[q]["probability"] for q in QUESTIONS] == [a2[q]["probability"] for q in QUESTIONS]


def test_choice_and_score_rejected(decider):
    with pytest.raises(ValueError, match="step 5"):
        decider.evaluate(STATE, {"kind": {"type": "choice", "instructions": "x", "criteria": {"a": "", "b": ""}}})
    with pytest.raises(ValueError, match="step 5"):
        decider.evaluate(STATE, {"tone": {"type": "score", "instructions": "x", "criteria": ["low", "high"]}})
