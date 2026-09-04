"""anthropic_helper: delta extraction and tool_choice / response_format mapping.

Run with the sdk-py venv:
  cd grid/api/anthropic && /home/ok/inference/sdk-py/.venv/bin/python -m pytest test_anthropic_helper.py -q
"""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import anthropic_helper as h  # noqa: E402
from inferencesh.llm_types_gen import ResponseFormat, ResponseFormatType, ToolChoice, ToolChoiceMode  # noqa: E402
from inferencesh.models.llm import LLMInput  # noqa: E402


# ── stream event folding ──────────────────────────────────────────────

def ev_text(text):
    return NS(type="content_block_delta", delta=NS(type="text_delta", text=text))


def ev_think(text):
    return NS(type="content_block_delta", delta=NS(type="thinking_delta", thinking=text))


def ev_tool_start(id_, name):
    return NS(type="content_block_start", content_block=NS(type="tool_use", id=id_, name=name))


def ev_tool_json(fragment):
    return NS(type="content_block_delta", delta=NS(type="input_json_delta", partial_json=fragment))


def ev_block_stop():
    return NS(type="content_block_stop")


def fold(events):
    state = h.create_initial_state()
    deltas = [h.apply_stream_event(e, state) for e in events]
    return state, deltas


class TestApplyStreamEvent:
    def test_text_and_thinking_deltas(self):
        state, deltas = fold([ev_think("plan"), ev_text("Hel"), ev_text("lo")])
        assert deltas == [{"reasoning": "plan"}, {"response": "Hel"}, {"response": "lo"}]
        assert state["response"] == "Hello" and state["thinking"] == "plan"

    def test_tool_call_fragments_carry_index(self):
        state, deltas = fold([
            ev_tool_start("t1", "get_weather"), ev_tool_json('{"city"'), ev_tool_json(': "Paris"}'), ev_block_stop(),
            ev_tool_start("t2", "other"), ev_tool_json("{}"), ev_block_stop(),
        ])
        assert deltas[0] == {"tool_calls": [{"index": 0, "id": "t1", "type": "function",
                                             "function": {"name": "get_weather", "arguments": ""}}]}
        assert deltas[1] == {"tool_calls": [{"index": 0, "function": {"arguments": '{"city"'}}]}
        assert deltas[3] is None
        assert deltas[4]["tool_calls"][0]["index"] == 1
        assert state["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'
        assert state["tool_calls"][1]["id"] == "t2"

    def test_usage_events_produce_no_delta(self):
        start = NS(type="message_start", message=NS(usage=NS(input_tokens=7, cache_read_input_tokens=0, cache_creation_input_tokens=0)))
        end = NS(type="message_delta", usage=NS(output_tokens=3))
        state, deltas = fold([start, end])
        assert deltas == [None, None]
        assert (state["input_tokens"], state["output_tokens"]) == (7, 3)
        out = h.build_output(state)
        assert out["output_meta"].inputs[0].tokens == 7 and out["output_meta"].outputs[0].tokens == 3


# ── tool_choice / response_format mapping ─────────────────────────────

class TestToolChoice:
    def test_modes(self):
        assert h.anthropic_tool_choice(None) is None
        assert h.anthropic_tool_choice(ToolChoice(mode=ToolChoiceMode.AUTO)) is None
        assert h.anthropic_tool_choice(ToolChoice(mode=ToolChoiceMode.NONE)) == {"type": "none"}
        assert h.anthropic_tool_choice(ToolChoice(mode=ToolChoiceMode.REQUIRED)) == {"type": "any"}
        assert h.anthropic_tool_choice(ToolChoice(mode=ToolChoiceMode.FUNCTION, name="f")) == {"type": "tool", "name": "f"}

    def test_function_mode_needs_name(self):
        with pytest.raises(ValueError, match="requires a tool name"):
            h.anthropic_tool_choice(ToolChoice(mode=ToolChoiceMode.FUNCTION))


class TestOutputFormat:
    def test_text_and_none(self):
        assert h.anthropic_output_format(None) is None
        assert h.anthropic_output_format(ResponseFormat(type=ResponseFormatType.TEXT)) is None

    def test_json_schema_native(self):
        rf = ResponseFormat(type=ResponseFormatType.JSON_SCHEMA, name="x", json_schema={"type": "object"})
        assert h.anthropic_output_format(rf) == {"type": "json_schema", "schema": {"type": "object"}}

    def test_json_object_rejected(self):
        with pytest.raises(ValueError, match="json_object"):
            h.anthropic_output_format(ResponseFormat(type=ResponseFormatType.JSON_OBJECT))


TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]


class TestBuildParams:
    def test_tool_choice_and_format_land_in_request(self):
        i = LLMInput(text="hi", tools=TOOLS,
                     tool_choice=ToolChoice(mode=ToolChoiceMode.REQUIRED),
                     response_format=ResponseFormat(type=ResponseFormatType.JSON_SCHEMA, json_schema={"type": "object"}))
        p = h.build_params(i, "claude-sonnet-5")
        assert p["tool_choice"] == {"type": "any"}
        assert p["output_config"]["format"] == {"type": "json_schema", "schema": {"type": "object"}}
        assert p["tools"][0]["name"] == "f"

    def test_format_merges_with_effort(self):
        i = LLMInput(text="hi", reasoning_effort="high", reasoning_max_tokens=4096,
                     response_format=ResponseFormat(type=ResponseFormatType.JSON_SCHEMA, json_schema={"type": "object"}))
        p = h.build_params(i, "claude-opus-5")  # effort model
        assert p["output_config"]["effort"] == "high"
        assert p["output_config"]["format"]["type"] == "json_schema"

    def test_forced_tool_choice_without_tools_rejected(self):
        with pytest.raises(ValueError, match="requires tools"):
            h.build_params(LLMInput(text="hi", tool_choice=ToolChoice(mode=ToolChoiceMode.REQUIRED)), "claude-sonnet-5")

    def test_none_without_tools_is_fine(self):
        p = h.build_params(LLMInput(text="hi", tool_choice=ToolChoice(mode=ToolChoiceMode.NONE)), "claude-sonnet-5")
        assert "tool_choice" not in p and "tools" not in p

    def test_defaults_unchanged(self):
        p = h.build_params(LLMInput(text="hi"), "claude-sonnet-4-5")
        assert "tool_choice" not in p and "output_config" not in p


# ── with_deltas streaming through a fake client ───────────────────────

class FakeStream:
    def __init__(self, events): self._events = events
    def __aiter__(self):
        async def gen():
            for e in self._events: yield e
        return gen()


class FakeClient:
    def __init__(self, events): self.events = events; self.params = None
    class _Messages:
        def __init__(self, outer): self.outer = outer
        async def create(self, **params):
            self.outer.params = params
            return FakeStream(self.outer.events)
    @property
    def messages(self): return self._Messages(self)


@pytest.mark.asyncio
async def test_stream_completion_with_deltas():
    client = FakeClient([ev_think("t"), ev_text("A"), ev_text("B")])
    pairs = [x async for x in h.stream_completion(client, LLMInput(text="hi"), "claude-sonnet-5", with_deltas=True)]
    assert [d for _, d in pairs] == [{"reasoning": "t"}, {"response": "A"}, {"response": "B"}]
    assert pairs[-1][0]["response"] == "AB" and pairs[-1][0]["reasoning"] == "t"


@pytest.mark.asyncio
async def test_stream_completion_default_shape_unchanged():
    client = FakeClient([ev_text("A")])
    outs = [x async for x in h.stream_completion(client, LLMInput(text="hi"), "claude-sonnet-5")]
    assert outs == [{"response": "A"}]
