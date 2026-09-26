"""POST /decide end-to-end against a live decide_ai.serve process."""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from decide_ai.convert import DEFAULT_OUT_DIR, STATIC_ASSET

pytestmark = pytest.mark.skipif(not (DEFAULT_OUT_DIR / STATIC_ASSET).exists(), reason="run decide_ai.convert first")
PORT = 8779
BASE = f"http://127.0.0.1:{PORT}"


def _post(path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture(scope="module")
def server():
    proc = subprocess.Popen([sys.executable, "-m", "decide_ai.serve", "--port", str(PORT), "--max-len", "64"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(BASE + "/info", timeout=1) as r:
                    info = json.loads(r.read())
                    break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not come up")
        yield info
    finally:
        proc.kill()
        proc.wait()


def test_info(server):
    assert server["asset"] == STATIC_ASSET
    assert server["labels"] == ["contradiction", "entailment", "neutral"]
    assert [1, 64] in server["static_shapes"]


def test_decide_jev_shape(server):
    code, out = _post("/decide", {
        "model": "local/nli-minilm2",
        "state": "Please refund my order, the jacket doesn't fit.",
        "questions": {
            "wants_refund": {"type": "boolean", "instructions": "Is the customer asking for a refund?",
                             "hypothesis": "The customer is asking for a refund."},
            "about_shipping": {"type": "boolean", "instructions": "Is this about shipping or delivery?",
                               "hypothesis": "This message is about shipping or delivery."},
        },
    })
    assert code == 200, out
    assert out["model"] == "local/nli-minilm2"
    assert set(out["answers"]) == {"wants_refund", "about_shipping"}
    for a in out["answers"].values():
        assert a["type"] == "boolean" and 0.0 <= a["probability"] <= 1.0
    assert out["answers"]["wants_refund"]["probability"] > 0.5 > out["answers"]["about_shipping"]["probability"]
    assert out["usage"]["outputTokens"] == 0 and out["usage"]["inputTokens"] > 0
    t = out["timing"]
    assert t["batch"] == 2 and t["padded_len"] == 64
    assert t["ms_tokenize"] + t["ms_infer"] <= t["ms_total"] + 0.01
    assert len(out["raw"]["wants_refund"]) == 3 and abs(sum(out["raw"]["wants_refund"]) - 1) < 1e-4


def test_instructions_used_when_no_hypothesis(server):
    code, out = _post("/decide", {"state": "Where is my parcel?",
                                  "questions": {"s": {"type": "boolean", "instructions": "This is about delivery."}}})
    assert code == 200 and "s" in out["answers"]


@pytest.mark.parametrize("qtype", ["choice", "score"])
def test_choice_and_score_rejected(server, qtype):
    code, out = _post("/decide", {"state": "x", "questions": {"q": {"type": qtype, "instructions": "?"}}})
    assert code == 400 and "step 5" in out["error"] and qtype in out["error"]


def test_bad_requests(server):
    assert _post("/decide", {"questions": {}})[0] == 400
    assert _post("/decide", {"state": "x", "questions": {}})[0] == 400
    assert _post("/decide", {"state": "x", "questions": {"q": {"type": "boolean"}}})[0] == 400
