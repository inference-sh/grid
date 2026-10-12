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
        assert meta.inputs[0].tokens == 100 and meta.inputs[0].cached_tokens == 40
        assert out["usage"].prompt_tokens == 100 and out["usage"].cached_tokens == 40
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


class TestToolResultImages:
    def test_tool_images_ride_in_function_call_output(self, tmp_path):
        from inferencesh import File
        from inferencesh.models.llm import ContextMessage, ContextMessageRole

        png = tmp_path / "shot.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        inp = LLMInput(
            context=[ContextMessage(role=ContextMessageRole.TOOL, tool_call_id="t0", text="captured", images=[File(path=str(png))])],
            text="done",
            role=ContextMessageRole.TOOL,
            tool_call_id="t1",
            images=[File(path=str(png))],
        )
        _, items = xai_llm.build_input(inp)
        assert [i["type"] for i in items] == ["function_call_output", "function_call_output"]
        output = items[0]["output"]
        assert output[0] == {"type": "input_text", "text": "captured"}
        assert output[1]["type"] == "input_image" and output[1]["image_url"].startswith("data:image/png;base64,")

    def test_text_only_tool_result_unchanged(self):
        from inferencesh.models.llm import ContextMessageRole

        _, items = xai_llm.build_input(LLMInput(text="42", role=ContextMessageRole.TOOL, tool_call_id="t1"))
        assert items == [{"type": "function_call_output", "call_id": "t1", "output": "42"}]


class TestImageFitting:
    def test_webp_is_sent_as_png(self, tmp_path):
        from PIL import Image
        from inferencesh import File
        from inferencesh.models.llm import ContextMessageRole

        webp = tmp_path / "shot.webp"
        Image.new("RGB", (64, 64), (0, 160, 0)).save(webp, format="WEBP")
        _, items = xai_llm.build_input(LLMInput(text="captured", role=ContextMessageRole.TOOL, tool_call_id="t1", images=[File(path=str(webp))]))
        assert items[0]["output"][1]["image_url"].startswith("data:image/png;base64,")


class TestReasoningState:
    """The encrypted reasoning items a response returns ride on the output's
    reasoning_state under this helper's key, and go back before the turn they
    produced."""

    ITEM = {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "think"}],
            "encrypted_content": "gAAA-enc"}

    def _req(self, **kw):
        return xai_llm.build_request_body(LLMInput(text="hi", **kw), "m", max_output_tokens=100, efforts=GROK_43)

    def test_requests_encrypted_reasoning_unless_reasoning_is_off(self):
        assert self._req(reasoning_effort="high")["include"] == ["reasoning.encrypted_content"]
        assert "include" not in self._req(reasoning_effort="none")  # the LLMInput default
        # grok-4.7 always reasons: "none" is sent as "low", and grok-build takes no effort at all.
        assert body(GROK_47, reasoning_effort="none")["include"] == ["reasoning.encrypted_content"]
        assert body(None)["include"] == ["reasoning.encrypted_content"]

    def test_reasoning_items_are_captured_in_order_without_status(self):
        state = xai_llm._create_initial_state()
        second = dict(self.ITEM, id="rs_2")
        for item in (dict(self.ITEM, status="completed"), second):
            assert xai_llm._handle_event({"type": "response.output_item.done", "item": item}, state) is None
        out = xai_llm._build_output(state, final=True)
        assert out["reasoning_state"] == {xai_llm.REASONING_STATE_KEY: {"items": [self.ITEM, second]}}
        assert "reasoning_state" not in xai_llm._build_output(state)  # whole only on the final output

    def test_no_reasoning_no_state(self):
        state = xai_llm._create_initial_state()
        assert "reasoning_state" not in xai_llm._build_output(state, final=True)

    def test_replayed_before_the_turns_text_and_calls(self):
        from inferencesh.models.llm import ContextMessage, ContextMessageRole

        call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        inp = LLMInput(
            context=[
                ContextMessage(role=ContextMessageRole.USER, text="go"),
                ContextMessage(role=ContextMessageRole.ASSISTANT, text="calling", tool_calls=[call],
                               reasoning_state={xai_llm.REASONING_STATE_KEY: {"items": [self.ITEM]}}),
            ],
            text="42", role=ContextMessageRole.TOOL, tool_call_id="c1",
        )
        _, items = xai_llm.build_input(inp)
        assert [i.get("type") or i.get("role") for i in items] == [
            "user", "reasoning", "assistant", "function_call", "function_call_output"]
        assert items[1] == self.ITEM

    def test_another_providers_entry_is_not_sent(self):
        from inferencesh.models.llm import ContextMessage, ContextMessageRole

        inp = LLMInput(
            context=[ContextMessage(role=ContextMessageRole.ASSISTANT, text="hi", reasoning_state={
                "openai": {"items": [self.ITEM]},
                "anthropic": [{"type": "thinking", "thinking": "t", "signature": "s"}],
            })],
            text="next",
        )
        _, items = xai_llm.build_input(inp)
        assert all(i.get("type") != "reasoning" for i in items)
