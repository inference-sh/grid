"""No-network tests for openai_llm: Responses API param mapping and delta extraction."""

import importlib.util
import sys
from pathlib import Path

import pytest

from inferencesh.llm_types_gen import ResponseFormat, ResponseFormatType, ToolChoice, ToolChoiceMode
from inferencesh.models.llm import LLMInput

_spec = importlib.util.spec_from_file_location("openai_llm", Path(__file__).with_name("openai_llm.py"))
openai_llm = importlib.util.module_from_spec(_spec)
sys.modules["openai_llm"] = openai_llm
_spec.loader.exec_module(openai_llm)


def body(**kw):
    i = LLMInput(text="hi", **kw)
    return openai_llm.build_request_body(i, "m", max_output_tokens=100, supports_none_reasoning=True, with_summary=False)


TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]


class TestParams:
    def test_default_tool_choice_auto(self):
        assert body(tools=TOOLS)["tool_choice"] == "auto"

    def test_tool_choice_required_and_named(self):
        assert body(tools=TOOLS, tool_choice=ToolChoice(mode=ToolChoiceMode.REQUIRED))["tool_choice"] == "required"
        named = body(tools=TOOLS, tool_choice=ToolChoice(mode=ToolChoiceMode.FUNCTION, name="f"))["tool_choice"]
        assert named == {"type": "function", "name": "f"}  # Responses API: flat, not nested under "function"

    def test_no_text_format_for_text(self):
        assert "text" not in body()
        assert "text" not in body(response_format=ResponseFormat(type=ResponseFormatType.TEXT))

    def test_json_object(self):
        assert body(response_format=ResponseFormat(type=ResponseFormatType.JSON_OBJECT))["text"] == {"format": {"type": "json_object"}}

    def test_json_schema_flattened(self):
        rf = ResponseFormat(type=ResponseFormatType.JSON_SCHEMA, name="ans", json_schema={"type": "object"}, strict=True)
        assert body(response_format=rf)["text"] == {"format": {
            "type": "json_schema", "name": "ans", "schema": {"type": "object"}, "strict": True}}


class TestDeltaExtraction:
    def test_text_and_tool_call_deltas(self):
        state = openai_llm._create_initial_state()
        assert openai_llm._handle_event({"type": "response.output_text.delta", "delta": "Hel"}, state) == {"response": "Hel"}
        added = openai_llm._handle_event({"type": "response.output_item.added", "item": {
            "type": "function_call", "id": "it1", "call_id": "c1", "name": "f"}}, state)
        assert added["tool_calls"][0]["index"] == 0 and added["tool_calls"][0]["id"] == "c1"
        frag = openai_llm._handle_event({"type": "response.function_call_arguments.delta", "item_id": "it1", "delta": '{"a":1}'}, state)
        assert frag == {"tool_calls": [{"index": 0, "function": {"arguments": '{"a":1}'}}]}
        assert state["tool_calls"][0]["function"]["arguments"] == '{"a":1}'


class TestUsage:
    def test_cache_reads_and_writes_reach_usage_and_token_report(self):
        state = openai_llm._create_initial_state()
        openai_llm._handle_event({"type": "response.completed", "response": {"id": "r1", "status": "completed", "usage": {
            "input_tokens": 1000, "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 300},
            "output_tokens_details": {"reasoning_tokens": 5}}}}, state)
        out = openai_llm._build_output(state, final=True)
        u = out["usage"]
        assert (u.prompt_tokens, u.cached_tokens, u.cache_write_tokens, u.completion_tokens) == (1000, 600, 300, 20)
        item = out["output_meta"].inputs[0]
        assert (item.tokens, item.cached_tokens, item.cache_write_tokens) == (1000, 600, 300)
        assert out["output_meta"].outputs[0].tokens == 20


class TestToolResultImages:
    def test_tool_images_ride_in_function_call_output(self, tmp_path):
        from inferencesh import File
        from inferencesh.models.llm import ContextMessage, ContextMessageRole

        png = tmp_path / "shot.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        inp = LLMInput(
            context=[ContextMessage(role=ContextMessageRole.TOOL, tool_call_id="t0", text="captured",
                                    images=[File(path=str(png))])],
            text="done",
            role=ContextMessageRole.TOOL,
            tool_call_id="t1",
            images=[File(path=str(png))],
        )
        _, items = openai_llm.build_input(inp)
        assert [i["type"] for i in items] == ["function_call_output", "function_call_output"]
        out = items[0]["output"]
        assert out[0] == {"type": "input_text", "text": "captured"}
        assert out[1]["type"] == "input_image" and out[1]["image_url"].startswith("data:image/png;base64,")
        assert items[1]["call_id"] == "t1"
        assert items[1]["output"][1]["image_url"].startswith("data:image/png;base64,")

    def test_text_only_tool_result_unchanged(self):
        from inferencesh.models.llm import ContextMessageRole

        _, items = openai_llm.build_input(LLMInput(text="42", role=ContextMessageRole.TOOL, tool_call_id="t1"))
        assert items == [{"type": "function_call_output", "call_id": "t1", "output": "42"}]


class TestReasoningState:
    """The encrypted reasoning items a response returns ride on the output's
    reasoning_state under this helper's key, and go back before the turn they
    produced."""

    ITEM = {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "think"}],
            "encrypted_content": "gAAA-enc"}

    def _req(self, **kw):
        return openai_llm.build_request_body(LLMInput(text="hi", **kw), "m", max_output_tokens=100, supports_none_reasoning=True, with_summary=False)

    def test_requests_encrypted_reasoning_unless_reasoning_is_off(self):
        assert self._req(reasoning_effort="high")["include"] == ["reasoning.encrypted_content"]
        assert "include" not in self._req(reasoning_effort="none")  # the LLMInput default

    def test_reasoning_items_are_captured_in_order_without_status(self):
        state = openai_llm._create_initial_state()
        second = dict(self.ITEM, id="rs_2")
        for item in (dict(self.ITEM, status="completed"), second):
            assert openai_llm._handle_event({"type": "response.output_item.done", "item": item}, state) is None
        out = openai_llm._build_output(state, final=True)
        assert out["reasoning_state"] == {openai_llm.REASONING_STATE_KEY: {"items": [self.ITEM, second]}}
        assert "reasoning_state" not in openai_llm._build_output(state)  # whole only on the final output

    def test_no_reasoning_no_state(self):
        state = openai_llm._create_initial_state()
        assert "reasoning_state" not in openai_llm._build_output(state, final=True)

    def test_replayed_before_the_turns_text_and_calls(self):
        from inferencesh.models.llm import ContextMessage, ContextMessageRole

        call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        inp = LLMInput(
            context=[
                ContextMessage(role=ContextMessageRole.USER, text="go"),
                ContextMessage(role=ContextMessageRole.ASSISTANT, text="calling", tool_calls=[call],
                               reasoning_state={openai_llm.REASONING_STATE_KEY: {"items": [self.ITEM]}}),
            ],
            text="42", role=ContextMessageRole.TOOL, tool_call_id="c1",
        )
        _, items = openai_llm.build_input(inp)
        assert [i.get("type") or i.get("role") for i in items] == [
            "user", "reasoning", "assistant", "function_call", "function_call_output"]
        assert items[1] == self.ITEM

    def test_another_providers_entry_is_not_sent(self):
        from inferencesh.models.llm import ContextMessage, ContextMessageRole

        inp = LLMInput(
            context=[ContextMessage(role=ContextMessageRole.ASSISTANT, text="hi", reasoning_state={
                "xai": {"items": [self.ITEM]},
                "anthropic": [{"type": "thinking", "thinking": "t", "signature": "s"}],
            })],
            text="next",
        )
        _, items = openai_llm.build_input(inp)
        assert all(i.get("type") != "reasoning" for i in items)
