"""anthropic_helper: delta extraction and tool_choice / response_format mapping.

Run with the sdk-py venv:
  cd grid/api/anthropic && /home/ok/inference/sdk-py/.venv/bin/python -m pytest test_anthropic_helper.py -q
"""

import base64
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

    def test_cache_reads_and_writes_count_in_the_input(self):
        # Anthropic's input_tokens excludes the cache; the usage and the token
        # report count every input token, with the cache as parts of it.
        start = NS(type="message_start", message=NS(usage=NS(input_tokens=100, cache_read_input_tokens=6000, cache_creation_input_tokens=900)))
        end = NS(type="message_delta", usage=NS(output_tokens=50))
        state, _ = fold([start, end])
        out = h.build_output(state)
        assert (out["usage"].prompt_tokens, out["usage"].cached_tokens, out["usage"].cache_write_tokens) == (7000, 6000, 900)
        item = out["output_meta"].inputs[0]
        assert (item.tokens, item.cached_tokens, item.cache_write_tokens) == (7000, 6000, 900)
        assert out["output_meta"].outputs[0].tokens == 50


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


# ── tool input_schema normalization ───────────────────────────────────

class TestNormalizeInputSchema:
    def test_missing_or_invalid_schema_becomes_empty_object(self):
        for bad in (None, {}, "x", []):
            assert h.normalize_input_schema(bad) == {"type": "object", "properties": {}}

    def test_missing_type_is_set_to_object(self):
        out = h.normalize_input_schema({"properties": {"q": {"type": "string"}}, "required": ["q"]})
        assert out == {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}

    def test_top_level_anyof_is_flattened(self):
        out = h.normalize_input_schema({
            "anyOf": [
                {"type": "object", "properties": {"a": {"type": "string"}, "id": {"type": "string"}}, "required": ["id", "a"]},
                {"type": "object", "properties": {"b": {"type": "integer"}, "id": {"type": "string"}}, "required": ["id", "b"]},
                {"type": "null"},
            ],
            "$defs": {"X": {"type": "string"}},
        })
        assert not any(k in out for k in ("anyOf", "oneOf", "allOf"))
        assert out["type"] == "object"
        assert set(out["properties"]) == {"a", "b", "id"}
        assert out["required"] == ["id"]
        assert out["$defs"] == {"X": {"type": "string"}}

    def test_top_level_allof_unions_required(self):
        out = h.normalize_input_schema({
            "allOf": [
                {"properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"properties": {"b": {"type": "string"}}, "required": ["b"]},
            ],
        })
        assert out["required"] == ["a", "b"]
        assert out["type"] == "object"

    def test_valid_schema_is_untouched_and_not_mutated(self):
        schema = {"type": "object", "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}
        assert h.normalize_input_schema(schema) == schema
        anyof = {"anyOf": [{"type": "object", "properties": {}}]}
        h.normalize_input_schema(anyof)
        assert "anyOf" in anyof

    def test_convert_tools_applies_normalization(self):
        tools = h.convert_tools_to_anthropic([
            {"type": "function", "function": {"name": "f", "parameters": {"properties": {}}}},
            {"type": "function", "function": {"name": "g"}},
        ])
        assert all(t["input_schema"]["type"] == "object" for t in tools)


# ── tool results with images ──────────────────────────────────────────

class TestToolResultImages:
    @pytest.fixture
    def png(self, tmp_path):
        p = tmp_path / "shot.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        return str(p)

    def _input(self, png, **kw):
        from inferencesh import File
        from inferencesh.models.llm import ContextMessage, ContextMessageRole

        ctx = [
            ContextMessage(role=ContextMessageRole.USER, text="look at the page"),
            ContextMessage(
                role=ContextMessageRole.ASSISTANT,
                text="",
                tool_calls=[{"id": "t0", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}],
            ),
            ContextMessage(role=ContextMessageRole.TOOL, tool_call_id="t0", text="captured", images=[File(path=png)]),
        ]
        return LLMInput(context=ctx, **kw)

    def test_context_tool_message_carries_image_blocks(self, png):
        _, msgs = h.convert_messages_to_anthropic(self._input(png, text="what do you see?"))
        block = msgs[2]["content"][0]
        assert block["type"] == "tool_result" and block["tool_use_id"] == "t0"
        assert block["content"][0] == {"type": "text", "text": "captured"}
        img = block["content"][1]
        assert img["type"] == "image" and img["source"]["type"] == "base64"
        assert img["source"]["media_type"] == "image/png"

    def test_current_tool_input_carries_image_blocks(self, png):
        from inferencesh import File
        from inferencesh.models.llm import ContextMessageRole

        inp = self._input(png, text="", role=ContextMessageRole.TOOL, tool_call_id="t1", images=[File(path=png)])
        _, msgs = h.convert_messages_to_anthropic(inp)
        # The context tool_result and the current one merge into one user turn.
        blocks = msgs[-1]["content"]
        current = [b for b in blocks if b.get("tool_use_id") == "t1"][0]
        assert [c["type"] for c in current["content"]] == ["image"]

    def test_text_only_tool_result_keeps_string_content(self):
        from inferencesh.models.llm import ContextMessage, ContextMessageRole

        inp = LLMInput(
            context=[ContextMessage(role=ContextMessageRole.TOOL, tool_call_id="t0", text="42")],
            text="",
            role=ContextMessageRole.TOOL,
            tool_call_id="t1",
        )
        _, msgs = h.convert_messages_to_anthropic(inp)
        assert msgs[0]["content"][0] == {"type": "tool_result", "tool_use_id": "t0", "content": "42"}
        assert msgs[0]["content"][1] == {"type": "tool_result", "tool_use_id": "t1", "content": ""}

    def test_url_image_uses_url_source(self):
        block = h._build_tool_result_block("t", "ok", [NS(path=None, uri="https://example.com/a.png")])
        assert block["content"][1] == {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}


# ── images fitted to the model ────────────────────────────────────────

class TestImageFitting:
    def _shot(self, tmp_path, size, name="big.png"):
        from PIL import Image

        p = tmp_path / name
        Image.new("RGB", size, (0, 160, 0)).save(p, format="PNG")
        return str(p)

    def _size(self, block):
        import io
        from PIL import Image

        return Image.open(io.BytesIO(base64.b64decode(block["source"]["data"]))).size

    def _tool_image(self, msgs):
        return msgs[-1]["content"][0]["content"][1]

    def _input(self, path, n=1):
        from inferencesh import File
        from inferencesh.models.llm import ContextMessage, ContextMessageRole

        ctx = [ContextMessage(role=ContextMessageRole.USER, text="look", images=[File(path=path)] * (n - 1))] if n > 1 else []
        ctx.append(ContextMessage(
            role=ContextMessageRole.ASSISTANT, text="",
            tool_calls=[{"id": "t0", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}],
        ))
        return LLMInput(context=ctx, text="captured", role=ContextMessageRole.TOOL, tool_call_id="t0", images=[File(path=path)])

    def test_high_res_model_takes_2576_long_edge(self, tmp_path):
        _, msgs = h.convert_messages_to_anthropic(self._input(self._shot(tmp_path, (3840, 2160))), "claude-opus-5")
        assert self._size(self._tool_image(msgs)) == (2576, 1449)

    def test_earlier_model_takes_1568_and_its_token_budget(self, tmp_path):
        _, msgs = h.convert_messages_to_anthropic(self._input(self._shot(tmp_path, (3840, 2160))), "claude-sonnet-4-5-20250929")
        w, h_ = self._size(self._tool_image(msgs))
        assert max(w, h_) <= 1568 and w * h_ <= 1568 * 28 * 28

    def test_many_images_cap_the_long_edge_at_2000(self, tmp_path):
        _, msgs = h.convert_messages_to_anthropic(self._input(self._shot(tmp_path, (3840, 2160)), n=21), "claude-opus-5")
        assert max(self._size(self._tool_image(msgs))) <= 2000

    def test_small_image_is_sent_as_is(self, tmp_path):
        path = self._shot(tmp_path, (800, 600), "small.png")
        _, msgs = h.convert_messages_to_anthropic(self._input(path), "claude-opus-5")
        assert base64.b64decode(self._tool_image(msgs)["source"]["data"]) == open(path, "rb").read()
