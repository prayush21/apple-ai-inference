"""decide_ai.model.NLICrossEncoder must match transformers' RobertaForSequenceClassification."""

import numpy as np
import pytest
import torch

from decide_ai.download import model_dir
from decide_ai.model import NLICrossEncoder
from decide_ai.tokenize import BPETokenizer

HF_DIR = model_dir()
pytestmark = pytest.mark.skipif(not (HF_DIR / "model.safetensors").exists(), reason="run decide_ai.download first")

# 20 real (state, hypothesis) pairs from the triage domain, mixed lengths so padding is exercised.
PAIRS = [
    ("The pasta was cold and the waiter ignored us, I want my money back.", "The customer is complaining."),
    ("The pasta was cold and the waiter ignored us, I want my money back.", "The customer is asking for a refund."),
    ("Package arrived a day early, thanks so much!", "This is about shipping or delivery."),
    ("Package arrived a day early, thanks so much!", "The customer is complaining."),
    ("I don't want a refund, just send a replacement.", "The customer is asking for a refund."),
    ("No complaints about the product, but the box was crushed in transit.", "This is about product quality."),
    ("This is the third time I've written in about this.", "This is urgent."),
    ("Still waiting.", "This is about shipping or delivery."),
    ("Order #48213 never showed up.", "The message mentions a specific order number."),
    ("My order never showed up.", "The message mentions a specific order number."),
    ("Hmm.", "The customer is complaining."),
    ("Can I change the color on my order before it ships?", "The customer is asking for a refund."),
    ("I can't log in and I also want a refund for last month.", "The customer is asking for a refund."),
    ("The blender works but sounds like a jet engine. Not sure if that's normal.", "This is about product quality."),
    ("Charged twice for the same subscription, please fix it today.", "This is urgent."),
    ("Love the new colour, exactly as pictured.", "The customer is complaining."),
    ("Where is my parcel? Tracking hasn't moved in nine days.", "This is about shipping or delivery."),
    ("The zipper broke on day two.", "This is about product quality."),
    ("Please cancel and refund, I ordered by mistake.", "The customer is asking for a refund."),
    ("Thanks for the quick reply yesterday.", "This is urgent."),
]


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.from_hf(HF_DIR)


@pytest.fixture(scope="module")
def ours():
    return NLICrossEncoder.from_pretrained(HF_DIR)


@pytest.fixture(scope="module")
def theirs():
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(str(HF_DIR)).eval()


def test_label_order_from_config(ours):
    assert ours.cfg.labels == ("contradiction", "entailment", "neutral")


def test_matches_transformers(tok, ours, theirs):
    ids, mask = tok.encode_batch(PAIRS, max_len=64)
    with torch.no_grad():
        got = ours(torch.from_numpy(ids), torch.from_numpy(mask)).numpy()
        want = theirs(input_ids=torch.from_numpy(ids).long(), attention_mask=torch.from_numpy(mask).long()).logits.numpy()
    diff = np.abs(got - want).max()
    assert diff < 1e-4, diff


def test_batch_matches_single_rows(tok, ours):
    ids, mask = tok.encode_batch(PAIRS[:4], max_len=64)
    with torch.no_grad():
        batched = ours(torch.from_numpy(ids), torch.from_numpy(mask)).numpy()
        singles = np.concatenate([
            ours(torch.from_numpy(ids[i : i + 1]), torch.from_numpy(mask[i : i + 1])).numpy() for i in range(4)
        ])
    assert np.abs(batched - singles).max() < 1e-4


def test_padding_length_does_not_change_logits(tok, ours):
    """Padded positions are masked, so L=32 and L=64 must agree for a short pair."""
    with torch.no_grad():
        outs = []
        for L in (32, 64):
            ids, mask = tok.encode_batch(PAIRS[:2], max_len=L)
            outs.append(ours(torch.from_numpy(ids), torch.from_numpy(mask)).numpy())
    assert np.abs(outs[0] - outs[1]).max() < 1e-4
