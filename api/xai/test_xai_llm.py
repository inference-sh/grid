"""No-network tests for xai_llm: request mapping, stream folding, metering."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from inferencesh.models.llm import LLMDelta, LLMInput

_spec = importlib.util.spec_from_file_location("xai_llm", Path(__file__).with_name("xai_llm.py"))
xai_llm = importlib.util.module_from_spec(_spec)
sys.modules["xai_llm"] = xai_llm
_spec.loader.exec_module(xai_llm)

GROK_47 = ("low", "medium", "high", "xhigh")
GROK_43 = ("none", "low", "medium", "high", "xhigh")


def body(efforts=GROK_43, **kw):
    return xai_llm.build_request_body(LLMInput(text="hi", **kw), "m", max_output_tokens=100, efforts=efforts)


def fold(events):
    state = xai_llm._create_initial_state()
    deltas = [d for d in (xai_llm._handle_event(e, state) for e in events) if d is not None]
    for d in deltas:
        LLMDelta(**d)
    return state, deltas


def completed(**usage):
    return {"type": "response.completed", "response": {"id": "r", "status": "completed", "usage": usage}}


class TestReasoning:
    def test_none_maps_to_low_on_grok_47(self):
        assert body(GROK_47, reasoning_effort="none")["reasoning"] == {"effort": "low"}

    def test_none_passes_on_grok_43(self):
        assert body(GROK_43, reasoning_effort="none")["reasoning"] == {"effort": "none"}

    def test_supported_value_passes_through(self):
        assert body(GROK_47, reasoning_effort="medium")["reasoning"] == {"effort": "medium"}

    def test_no_effort_control_sends_nothing(self):
        assert "reasoning" not in body(None, reasoning_effort="high")

    def test_no_summary_param(self):
        assert "summary" not in body(GROK_47, reasoning_effort="high")["reasoning"]


class TestInput:
    def test_current_turn_files_rejected(self):
        with pytest.raises(ValueError, match="File input is not supported"):
            xai_llm._user_content("hi", None, [SimpleNamespace(uri="https://example.com/a.pdf", path=None)])

    def test_image_url_passed(self):
        content = xai_llm._user_content("hi", [SimpleNamespace(uri="https://example.com/a.png", path=None)], None)
        assert {"type": "input_image", "image_url": "https://example.com/a.png", "detail": "auto"} in content


class TestStream:
    def test_reasoning_not_duplicated_across_event_types(self):
        state, _ = fold([
            {"type": "response.reasoning_text.delta", "delta": "think"},
            {"type": "response.reasoning_summary_text.delta", "delta": "think"},
        ])
        assert state["reasoning"] == "think"

    def test_whole_function_call_in_added_event(self):
        state, deltas = fold([{"type": "response.output_item.added", "item": {
            "type": "function_call", "id": "fc1", "call_id": "c1", "name": "f", "arguments": '{"a":1}'}}])
        assert deltas[0]["tool_calls"][0]["function"]["arguments"] == '{"a":1}'
        assert state["tool_calls"][0]["function"]["arguments"] == '{"a":1}'

    def test_function_call_only_in_done_event(self):
        state, deltas = fold([{"type": "response.output_item.done", "item": {
            "type": "function_call", "id": "fc1", "call_id": "c1", "name": "f", "arguments": "{}"}}])
        assert len(state["tool_calls"]) == 1 and deltas[0]["tool_calls"][0]["id"] == "c1"

    def test_added_then_done_is_one_call(self):
        item = {"type": "function_call", "id": "fc1", "call_id": "c1", "name": "f", "arguments": "{}"}
        state, _ = fold([
            {"type": "response.output_item.added", "item": item},
            {"type": "response.output_item.done", "item": item},
        ])
        assert len(state["tool_calls"]) == 1


class TestMetering:
    def test_reasoning_reported_separately_is_added(self):
        state, _ = fold([completed(input_tokens=32, output_tokens=9, total_tokens=151,
                                   output_tokens_details={"reasoning_tokens": 110})])
        assert xai_llm.billable_output_tokens(state) == 119

    def test_reasoning_already_inside_output_is_not_double_counted(self):
        state, _ = fold([completed(input_tokens=12000, output_tokens=800, total_tokens=12800,
                                   output_tokens_details={"reasoning_tokens": 240})])
        assert xai_llm.billable_output_tokens(state) == 800

    def test_output_meta_carries_cache_and_cost(self):
        state, _ = fold([completed(input_tokens=100, output_tokens=10, total_tokens=110,
                                   input_tokens_details={"cached_tokens": 40}, cost_in_usd_ticks=37756000)])
        out = xai_llm._build_output(state, final=True)
        meta = out["output_meta"]
        assert meta.inputs[0].tokens == 100 and meta.inputs[0].extra["cache_read_tokens"] == 40
        assert meta.outputs[0].extra["upstream_cost_usd"] == pytest.approx(0.0037756)
        assert meta.outputs[0].extra["usage_violation"] == 0


class TestUsageViolation:
    def test_http_rejection_detected(self):
        err = {"error": {"code": "invalid_request", "message": "Request violates our usage guidelines"}}
        assert xai_llm.is_usage_violation(400, err)

    def test_schema_error_is_not_a_violation(self):
        err = {"error": {"message": "tool parameters violate JSON schema", "param": "safety_identifier"}}
        assert not xai_llm.is_usage_violation(400, err)

    def test_rate_limit_is_not_a_violation(self):
        assert not xai_llm.is_usage_violation(429, {"error": {"message": "content policy"}})

    def test_violation_output_is_billable_and_empty(self):
        state = xai_llm._create_initial_state()
        state["usage_violation"] = True
        out = xai_llm._build_output(state, final=True)
        assert out["response"] == "" and out["notice"]
        assert out["output_meta"].inputs == []
        assert out["output_meta"].outputs[0].extra["usage_violation"] == 1
        assert out["usage"].stop_reason == "usage_violation"

    def test_stream_failure_with_violation_before_output(self):
        state, _ = fold([{"type": "response.failed", "response": {"error": {
            "code": "content_moderation", "message": "blocked by moderation"}}}])
        assert state["usage_violation"]

    def test_stream_failure_after_output_still_raises(self):
        with pytest.raises(RuntimeError):
            fold([
                {"type": "response.output_text.delta", "delta": "partial"},
                {"type": "response.failed", "response": {"error": {"message": "blocked by moderation"}}},
            ])
