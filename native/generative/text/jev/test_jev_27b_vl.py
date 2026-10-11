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
from inference import (  # noqa: E402
    App, AppInput, AppOutput, ImageGrid, build_answers, build_requests, confidence, decide_warmups, encode_state,
    estimate_prefix_tokens, saved_tokens, warm_request,
)

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
    bodies = {qid: (body, expected) for qid, body, expected in build_requests(make_input())}
    assert bodies["team"] == ({"kind": "choice", "question": "Which team?", "options": ["billing: Charges and refunds", "orders", "other"]}, 3)
    assert bodies["urgency"][0]["kind"] == "score"
    assert bodies["urgency"][0]["question"] == "How urgent? Levels: 0 = none; 1 = slight; 2 = some; 3 = clear; 4 = strong; 5 = extreme"
    assert bodies["urgency"][1] == 6
    assert bodies["tone"] == ({"kind": "choice", "question": "How angry?", "options": ["0: calm", "1: annoyed", "2: furious"]}, 3)
    assert bodies["refund"] == ({"kind": "noul", "question": "Is a refund requested?"}, 2)
    assert bodies["legal"][0]["question"] == "Is a lawyer mentioned? (true: explicit mention; false: no mention)"


def test_state_is_text_without_images_and_parts_with_them():
    assert json.loads(encode_state([], '{"ticket": "charged twice"}')) == '{"ticket": "charged twice"}'
    state = json.loads(encode_state(["data:image/png;base64,AAAA"], "Seller title: earbuds"))
    assert state == [{"image": "data:image/png;base64,AAAA"}, "\nSeller title: earbuds"]
    assert json.loads(encode_state(["data:image/png;base64,AAAA"], "")) == [{"image": "data:image/png;base64,AAAA"}]


def test_validation():
    with pytest.raises(ValueError, match="at least one question"):
        make_input(choices=[], scores=[], nouls=[])
    with pytest.raises(ValueError, match="unique"):
        make_input(nouls=[{"id": "team", "instructions": "x?"}])
    with pytest.raises(ValueError, match="`state`, `images`, or both"):
        make_input(state="")


def test_answers_are_typed_and_metered():
    probabilities = {
        "team": [0.7, 0.2, 0.1],
        "urgency": [0, 0, 0, 0.5, 0.5, 0],
        "tone": [0.0, 0.5, 0.5],
        "refund": [0.1, 0.9],
        "legal": [0.99, 0.01],
    }
    data = make_input()
    out = AppOutput.from_answers(build_answers(data, probabilities), data, model="m", input_tokens=210)
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
    app.image_grid = ImageGrid()
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


# ── Warm-up gate ─────────────────────────────────────────────────────────────

GRID = ImageGrid()
LISTING = "Seller title: wireless earbuds, barely used. Includes the charging case and cable."
LONG_TEXT = "The seller listed a pair of wireless earbuds with the case, a charging cable and the original box. " * 60


def test_image_tokens_follow_the_processor_resize():
    # One token per 32 x 32 cell after rounding to the grid; vLLM counted 137 prompt tokens for the
    # 448 x 301 warm-up (126 image + markers + template).
    assert GRID.tokens(448, 301) == 126
    assert GRID.tokens(768, 516) == 384
    assert GRID.tokens(10, 10) == 64          # scaled up to min_pixels
    assert GRID.tokens(8000, 8000) <= GRID.max_pixels // 32 ** 2  # scaled down to max_pixels


def test_image_grid_reads_the_preprocessor_config(tmp_path):
    path = tmp_path / "preprocessor_config.json"
    path.write_text(json.dumps({"patch_size": 14, "merge_size": 2, "size": {"shortest_edge": 1000, "longest_edge": 2000}}))
    assert ImageGrid.from_preprocessor_config(str(path)) == ImageGrid(factor=28, min_pixels=1000, max_pixels=2000)
    assert ImageGrid.from_preprocessor_config(str(tmp_path / "missing.json")) == ImageGrid()


def test_saved_tokens_resume_at_the_last_match_boundary():
    assert saved_tokens(1, 137) == 0
    assert saved_tokens(2, 137) == 128
    assert saved_tokens(16, 26) == 15 * 16


# Each request shape the bench app timed with and without a warm-up (A100, cache reset per call):
# (image sizes, text, question kinds, faster path).
MEASURED = {
    "image_1q": ([(448, 301)], "", ["noul"], False),
    "image_2q": ([(448, 301)], "", ["noul"] * 2, False),
    "image_4q": ([(448, 301)], "", ["noul"] * 4, False),
    "image_6q": ([(448, 301)], "", ["noul"] * 6, True),
    "image_8q": ([(448, 301)], "", ["noul"] * 8, True),
    "image_16q": ([(448, 301)], "", ["noul"] * 16, True),
    "image_768px_3q": ([(768, 516)], "", ["noul"] * 3, True),
    "two_images_8q": ([(448, 301)] * 2, "", ["noul"] * 8, True),
    "text_short_2q": ([], LISTING, ["noul"] * 2, False),
    "text_short_16q": ([], LISTING, ["noul"] * 16, False),
    "text_long_2q": ([], LONG_TEXT, ["noul"] * 2, False),
    "text_long_16q": ([], LONG_TEXT, ["noul"] * 16, True),
}


@pytest.mark.parametrize("name", MEASURED)
def test_gate_picks_the_measured_faster_path(name):
    sizes, text, kinds, warm = MEASURED[name]
    (decision,) = decide_warmups(sizes, text, kinds, GRID)
    assert decision.warm is warm


def test_gate_decides_per_kind():
    decisions = {d.kind: d for d in decide_warmups([(448, 301)], "", ["choice"] + ["noul"] * 15, GRID)}
    assert decisions["noul"].warm and not decisions["choice"].warm
    assert decisions["noul"].prefix_tokens == estimate_prefix_tokens([(448, 301)], "", GRID)


def test_warm_up_is_the_decide_prompt_up_to_the_question():
    path, body = warm_request("noul", ["data:image/png;base64,AAAA"], "Seller title: earbuds")
    assert path == "/v1/chat/completions"
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "[kind] noul\n[state] "},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": "\nSeller title: earbuds"},
    ]
    assert body["chat_template"] == app_module.RAW_TEMPLATE and body["add_generation_prompt"] is False
    assert body["max_tokens"] == 1 and body["add_special_tokens"] is False
    path, body = warm_request("choice", [], "Seller title: earbuds")
    assert (path, body["prompt"]) == ("/v1/completions", "[kind] choice\n[state] Seller title: earbuds")


def write_image(tmp_path, size=(448, 301)):
    from PIL import Image
    path = tmp_path / "face.png"
    Image.new("RGB", size).save(path)
    return str(path)


def decide_handler(seen, warm_status=200):
    def handler(request):
        seen.append(request.url.path)
        if request.url.path != "/v1/decide":
            return httpx.Response(warm_status, json={"usage": {"prompt_tokens": 137}})
        body = json.loads(request.content)
        n = {"noul": 2, "score": 6}.get(body["kind"]) or len(body["options"])
        return httpx.Response(200, json={"probabilities": [1 / n] * n, "usage": {"prompt_tokens": 10}})
    return handler


def nouls(n):
    return [{"id": f"q{i}", "instructions": f"Is the person in the image doing thing {i}?"} for i in range(n)]


def test_run_warms_up_before_the_questions_when_they_share_enough(tmp_path):
    seen = []
    image = write_image(tmp_path)
    asyncio.run(make_app(decide_handler(seen)).run(make_input(state="", images=[image], choices=[], scores=[], nouls=nouls(8))))
    assert seen[0] == "/v1/chat/completions"
    assert seen[1:] == ["/v1/decide"] * 8


def test_run_skips_the_warm_up_when_it_would_not_pay(tmp_path):
    seen = []
    image = write_image(tmp_path)
    asyncio.run(make_app(decide_handler(seen)).run(make_input(state="", images=[image], choices=[], scores=[], nouls=nouls(2))))
    assert seen == ["/v1/decide"] * 2


def test_run_answers_when_the_warm_up_fails(tmp_path):
    seen = []
    image = write_image(tmp_path)
    out = asyncio.run(make_app(decide_handler(seen, warm_status=500)).run(
        make_input(state="", images=[image], choices=[], scores=[], nouls=nouls(8))))
    assert seen.count("/v1/decide") == 8
    assert len(out.nouls) == 8
