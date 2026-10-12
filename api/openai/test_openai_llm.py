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
