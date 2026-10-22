"""No-network tests for the Flint app: request building, stream folding, worker checks.

    PYTHONPATH=<sdk-py>/src python3 -m pytest test_flint.py -q
"""

import importlib
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
pkg = types.ModuleType("flint_app")
pkg.__path__ = [os.path.join(HERE, "flint")]
sys.modules["flint_app"] = pkg
flint = importlib.import_module("flint_app.inference")

from inferencesh.models.llm import ContextMessage  # noqa: E402


def test_defaults_follow_the_card():
    body = flint.build_request_body(flint.AppInput(text="hi"))
    assert body["model"] == "flint"
    assert body["temperature"] == 1.0 and body["top_p"] == 0.95 and body["top_k"] == 20
    assert body["max_tokens"] == 32768
    assert "reasoning_effort" not in body
    # No default system prompt: an empty one is not sent as a message with content.
    assert all(m.get("content") for m in body["messages"] if m["role"] == "system")


def test_content_is_a_string_and_reasoning_is_replayed():
    inp = flint.AppInput(
        text="and now?",
        system_prompt="be brief",
        context=[
            ContextMessage(role="user", text="q1"),
            ContextMessage(role="assistant", text="a1", reasoning="thought about q1"),
        ],
    )
    body = flint.build_request_body(inp)
    assert all(isinstance(m.get("content", ""), str) for m in body["messages"])
    assistant = [m for m in body["messages"] if m["role"] == "assistant"][0]
    assert assistant["reasoning_content"] == "thought about q1"


def test_tools_are_sent():
    tool = {"type": "function", "function": {"name": "get_weather", "description": "w",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
    body = flint.build_request_body(flint.AppInput(text="weather in Paris?", tools=[tool]))
    assert body["tools"][0]["function"]["name"] == "get_weather"
    assert "tool_choice" in body


def test_media_is_refused():
    inp = flint.AppInput(text="x")
    inp.images = [types.SimpleNamespace(path="/tmp/a.png", uri=None, content_type="image/png", filename="a.png")]
    with pytest.raises(ValueError, match="text only"):
        flint.build_request_body(inp)


def test_flatten_refuses_non_text_parts():
    assert flint._flatten_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    with pytest.raises(ValueError):
        flint._flatten_content([{"type": "image_url", "image_url": {"url": "x"}}])


def chunk(delta=None, finish=None, usage=None):
    data = {"choices": [{"delta": delta or {}, "finish_reason": finish}]} if delta is not None or finish else {"choices": []}
    if usage:
        data["usage"] = usage
    return data


def test_stream_folds_reasoning_content_and_tool_calls():
    state = flint.new_state()
    deltas = [
        flint.handle_chunk(chunk({"reasoning_content": "think "}), state),
        flint.handle_chunk(chunk({"reasoning_content": "more"}), state),
        flint.handle_chunk(chunk({"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": None}}]}), state),
        flint.handle_chunk(chunk({"tool_calls": [{"index": 0, "function": {"name": None, "arguments": "{\"city\": \"Paris\""}}]}), state),
        flint.handle_chunk(chunk({"tool_calls": [{"index": 1, "id": "c2", "type": "function", "function": {"name": "f", "arguments": "{\"a\": {\"b\": 1}"}}]}), state),
        flint.handle_chunk(chunk({}, finish="tool_calls"), state),
        flint.handle_chunk({"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 20,
                            "completion_tokens_details": {"reasoning_tokens": 5}}}, state),
    ]
    assert deltas[0] == {"reasoning": "think "}
    assert "name" not in deltas[3]["tool_calls"][0].get("function", {})
    out = flint.build_output(state, final=True)
    assert out["reasoning"] == "think more"
    # Both calls lost their final brace; the final output repairs them.
    assert json.loads(out["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}
    assert json.loads(out["tool_calls"][1]["function"]["arguments"]) == {"a": {"b": 1}}
    assert out["usage"].prompt_tokens == 50 and out["usage"].reasoning_tokens == 5
    assert out["usage"].stop_reason == "tool_calls"
    flint.AppOutput(**out)


def test_repair_leaves_valid_and_hopeless_arguments_alone():
    assert flint.repair_arguments('{"a": 1}') == '{"a": 1}'
    assert flint.repair_arguments('{"a": [1') == '{"a": [1'
    assert flint.repair_arguments("") == ""


def test_stream_error_chunk_raises():
    with pytest.raises(RuntimeError, match="boom"):
        flint.handle_chunk({"error": {"message": "boom"}}, flint.new_state())


def test_gpu_checks():
    gpus = flint.parse_gpus("NVIDIA H200, 143771, 570.133.20\nNVIDIA H200, 143771, 570.133.20\n")
    assert len(gpus) == 2 and gpus[0]["memory_gb"] > 140
    assert flint.pick_serve_script(gpus[0]) == "serve_h200.sh"
    assert flint.pick_serve_script({"name": "NVIDIA B300 SXM6 AC", "memory_gb": 275}) == "serve_b300.sh"
    with pytest.raises(RuntimeError, match="80 GB"):
        flint.pick_serve_script({"name": "NVIDIA A100-SXM4-80GB", "memory_gb": 80})
    assert flint.driver_major("570.133.20") == 570 and flint.driver_major("580.126.20") >= flint.MIN_DRIVER_MAJOR
