"""No-network tests for opper helper: param mapping and delta extraction."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

from inferencesh.llm_types_gen import ResponseFormat, ResponseFormatType, ToolChoice, ToolChoiceMode
from inferencesh.models.llm import LLMInput

_spec = importlib.util.spec_from_file_location("opper", Path(__file__).with_name("opper.py"))
opper = importlib.util.module_from_spec(_spec)
sys.modules["opper"] = opper
_spec.loader.exec_module(opper)

TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]


def params(**kw):
    return opper._build_params(LLMInput(text="hi", **kw), "anthropic/claude-sonnet-4-5", stream=True)


class TestParams:
    def test_tool_choice_passthrough(self):
        assert params(tools=TOOLS)["tool_choice"] == "auto"
        assert params(tools=TOOLS, tool_choice=ToolChoice(mode=ToolChoiceMode.REQUIRED))["tool_choice"] == "required"
        assert params(tools=TOOLS, tool_choice=ToolChoice(mode=ToolChoiceMode.FUNCTION, name="f"))["tool_choice"] == {
            "type": "function", "function": {"name": "f"}}

    def test_response_format_passthrough(self):
        assert "response_format" not in params()
        rf = ResponseFormat(type=ResponseFormatType.JSON_SCHEMA, name="a", json_schema={"type": "object"})
        assert params(response_format=rf)["response_format"] == {
            "type": "json_schema", "json_schema": {"name": "a", "schema": {"type": "object"}}}


def chunk(content=None, reasoning=None, tool_calls=None, finish=None, usage=None):
    delta = NS(content=content, reasoning=reasoning, reasoning_details=None, tool_calls=tool_calls)
    return NS(choices=[NS(delta=delta, finish_reason=finish)], usage=usage, error=None)


class TestDeltaExtraction:
    def test_text_and_tool_calls(self):
        state = opper.create_initial_state()
        fr, d = opper.process_chunk(chunk(content="Hel", reasoning="why"), state)
        assert (fr, d) == (None, {"response": "Hel", "reasoning": "why"})

        tc = NS(index=0, id="c1", type="function", function=NS(name="f", arguments='{"a"'))
        _, d = opper.process_chunk(chunk(tool_calls=[tc]), state)
        assert d["tool_calls"][0] is tc  # plain objects pass through; SDK models are dumped
        tc2 = NS(index=0, id=None, type=None, function=NS(name=None, arguments=':1}'))
        fr, _ = opper.process_chunk(chunk(tool_calls=[tc2], finish="tool_calls"), state)
        assert fr == "tool_calls"
        assert state["tool_calls"][0]["function"]["arguments"] == '{"a":1}'

    def test_usage_only_chunk(self):
        state = opper.create_initial_state()
        fr, d = opper.process_chunk(NS(choices=[], usage=NS(prompt_tokens=3, completion_tokens=1), error=None), state)
        assert (fr, d) == (None, None) and state["input_tokens"] == 3
