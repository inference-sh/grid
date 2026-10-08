"""No-network tests for the request mapping and answer typing. Run: pytest test_jev_27b_vl.py"""

import asyncio
import json
import math
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev-27b-vl"))
import inference as app_module  # noqa: E402
from inference import App, AppInput, build_output, build_requests, build_state, confidence  # noqa: E402

LEVELS_6 = ["none", "slight", "some", "clear", "strong", "extreme"]


def make_input(**overrides):
    data = {
        "state": {"ticket": "charged twice"},
        "choices": [{"id": "team", "instructions": "Which team?", "options": [
            {"name": "billing", "description": "Charges\nand refunds"}, {"name": "orders"}, {"name": "other"}]}],
        "scores": [
            {"id": "urgency", "instructions": "How urgent?", "levels": LEVELS_6},
            {"id": "tone", "instructions": "How angry?", "levels": ["calm", "annoyed", "furious"]},
        ],
        "nouls": [
            {"id": "refund", "instructions": "Is a refund requested?"},
            {"id": "legal", "instructions": "Is a lawyer mentioned?", "criteria": {"true": "explicit mention", "false": "no mention"}},
        ],
    }
    data.update(overrides)
    return AppInput(**data)


def test_requests_map_each_type_to_decide():
    bodies = {qid: (kind, body) for qid, kind, body in build_requests(make_input())}
    assert bodies["team"] == ("choice", {"kind": "choice", "question": "Which team?", "options": ["billing: Charges and refunds", "orders", "other"]})
    assert bodies["urgency"][1]["kind"] == "score"
    assert bodies["urgency"][1]["question"] == "How urgent? Levels: 0 = none; 1 = slight; 2 = some; 3 = clear; 4 = strong; 5 = extreme"
    assert bodies["tone"][1] == {"kind": "choice", "question": "How angry?", "options": ["0: calm", "1: annoyed", "2: furious"]}
    assert bodies["refund"][1] == {"kind": "noul", "question": "Is a refund requested?"}
    assert bodies["legal"][1]["question"] == "Is a lawyer mentioned? (true: explicit mention; false: no mention)"


def test_state_is_text_without_images_and_parts_with_them(tmp_path):
    assert build_state(make_input()) == json.dumps({"ticket": "charged twice"})
    path = tmp_path / "a.png"
    path.write_bytes(b"\x89PNG")
    state = build_state(make_input(state="Seller title: earbuds", images=[str(path)]))
    assert state[0]["image"].startswith("data:image/png;base64,")
    assert state[1] == "\nSeller title: earbuds"
    assert len(build_state(make_input(state="", images=[str(path)]))) == 1


def test_validation():
    with pytest.raises(ValueError, match="at least one question"):
        make_input(choices=[], scores=[], nouls=[])
    with pytest.raises(ValueError, match="unique"):
        make_input(nouls=[{"id": "team", "instructions": "x?"}])
    with pytest.raises(ValueError, match="`state`, `images`, or both"):
        make_input(state="")


def test_answers_are_typed_and_metered():
    results = {
        "team": {"probabilities": [0.7, 0.2, 0.1], "usage": {"prompt_tokens": 50}},
        "urgency": {"probabilities": [0, 0, 0, 0.5, 0.5, 0], "usage": {"prompt_tokens": 60}},
        "tone": {"probabilities": [0.0, 0.5, 0.5], "usage": {"prompt_tokens": 40}},
        "refund": {"probabilities": [0.1, 0.9], "usage": {"prompt_tokens": 30}},
        "legal": {"probabilities": [0.99, 0.01], "usage": {"prompt_tokens": 30}},
    }
    out = build_output(make_input(), results)
    assert out.choices["team"].choice == "billing"
    assert out.choices["team"].probabilities == {"billing": 0.7, "orders": 0.2, "other": 0.1}
    assert out.scores["urgency"].score == pytest.approx(3.5)
    assert out.scores["urgency"].normalized == pytest.approx(0.7)
    assert out.scores["tone"].score == pytest.approx(1.5)
    assert out.scores["tone"].normalized == pytest.approx(0.75)
    assert out.scores["tone"].legend == {"0": "calm", "1": "annoyed", "2": "furious"}
    assert out.nouls["refund"].noul == 0.9
    assert out.input_tokens == 210
    assert out.output_meta.inputs[0].tokens == 210


def test_confidence_bounds():
    assert confidence([1.0, 0.0]) == 1.0
    assert confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert 0 < confidence([0.7, 0.2, 0.1]) < 1
    assert not math.isnan(confidence([1.0, 0.0, 0.0]))


class FakeServer:
    returncode = None


def make_app(handler):
    app = App()
    app.logger = app_module.logging.getLogger("test")
    app.server = FakeServer()
    app.server_log = []
    app.client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    return app


def test_run_sends_one_call_per_question_with_the_shared_state():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        n = {"noul": 2, "score": 6}.get(body["kind"]) or len(body["options"])
        return httpx.Response(200, json={"probabilities": [1 / n] * n, "usage": {"prompt_tokens": 10}})

    out = asyncio.run(make_app(handler).run(make_input()))
    assert len(seen) == 5
    assert {body["state"] for body in seen} == {json.dumps({"ticket": "charged twice"})}
    assert out.input_tokens == 50
    assert set(out.scores) == {"urgency", "tone"}


def test_run_names_the_failing_question():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "maximum context length is 32768 tokens"}})

    with pytest.raises(RuntimeError, match=r"question '\w+' could not be answered \(400\): maximum context length"):
        asyncio.run(make_app(handler).run(make_input()))


def test_run_rejects_a_wrong_number_of_probabilities():
    def handler(request):
        return httpx.Response(200, json={"probabilities": [0.5, 0.5]})

    with pytest.raises(RuntimeError, match="probabilities for"):
        asyncio.run(make_app(handler).run(make_input()))
