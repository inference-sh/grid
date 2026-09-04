"""No-network tests for minimax_llm: delta extraction and param policy."""

import importlib.util
import sys
from pathlib import Path

import pytest

from inferencesh.llm_types_gen import ResponseFormat, ResponseFormatType, ToolChoice, ToolChoiceMode
from inferencesh.models.llm import LLMInput

_spec = importlib.util.spec_from_file_location("minimax_llm", Path(__file__).with_name("minimax_llm.py"))
minimax_llm = importlib.util.module_from_spec(_spec)
sys.modules["minimax_llm"] = minimax_llm
_spec.loader.exec_module(minimax_llm)

TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]


def body(**kw):
    return minimax_llm._build_request_body(LLMInput(text="hi", **kw), "MiniMax-M3")


class TestParams:
    def test_tools_default_auto(self):
        b = body(tools=TOOLS)
        assert b["tool_choice"] == "auto" and b["tools"]

    def test_tool_choice_none_drops_tools(self):
        b = body(tools=TOOLS, tool_choice=ToolChoice(mode=ToolChoiceMode.NONE))
        assert "tools" not in b and "tool_choice" not in b

    @pytest.mark.parametrize("choice", [
        ToolChoice(mode=ToolChoiceMode.REQUIRED),
        ToolChoice(mode=ToolChoiceMode.FUNCTION, name="f"),
    ])
    def test_undocumented_tool_choice_rejected(self, choice):
        with pytest.raises(ValueError, match="tool_choice"):
            body(tools=TOOLS, tool_choice=choice)

    def test_text_response_format_ok_json_rejected(self):
        assert "response_format" not in body(response_format=ResponseFormat(type=ResponseFormatType.TEXT))
        with pytest.raises(ValueError, match="response_format"):
            body(response_format=ResponseFormat(type=ResponseFormatType.JSON_OBJECT))


class TestDeltaExtraction:
    def test_content_reasoning_and_tool_call_deltas(self):
        state = minimax_llm._create_initial_state()
        fr, d = minimax_llm._parse_sse_chunk({"choices": [{"delta": {"content": "Hel", "reasoning_content": "why"}}]}, state)
        assert fr is None and d == {"response": "Hel", "reasoning": "why"}
        tc = [{"index": 0, "id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a"'}}]
        _, d = minimax_llm._parse_sse_chunk({"choices": [{"delta": {"tool_calls": tc}}]}, state)
        assert d == {"tool_calls": tc}
        _, d = minimax_llm._parse_sse_chunk({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ':1}'}}]}, "finish_reason": "tool_calls"}]}, state)
        assert state["tool_calls"][0]["function"]["arguments"] == '{"a":1}'
        assert minimax_llm._build_output(state)["response"] == "Hel"

    def test_usage_only_chunk_yields_no_delta(self):
        state = minimax_llm._create_initial_state()
        fr, d = minimax_llm._parse_sse_chunk({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}, state)
        assert (fr, d) == (None, None) and state["input_tokens"] == 3
