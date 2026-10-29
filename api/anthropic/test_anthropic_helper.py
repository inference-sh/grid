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


class TestThinkingShapeByModel:
    """The thinking request shape per model generation; a wrong shape is a 400."""

    def _p(self, model, **kw):
        return h.build_params(LLMInput(text="hi", **kw), model)

    ADAPTIVE = {"type": "adaptive", "display": "summarized", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}

    @pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-mythos-5-1", "claude-opus-5-5", "claude-fable-5", "claude-opus-5"])
    def test_always_on_models_send_adaptive_and_run_low_effort_by_default(self, model):
        p = self._p(model)
        assert p["thinking"] == self.ADAPTIVE and "budget_tokens" not in p["thinking"]
        if model in h.EFFORT_MODELS:
            assert p["output_config"] == {"effort": "low"}
            p = self._p(model, reasoning_effort="high", reasoning_max_tokens=4096)
            assert p["thinking"] == self.ADAPTIVE and p["output_config"] == {"effort": "high"}

    def test_sonnet_5_5_off_is_between_tools(self):
        p = self._p("claude-sonnet-5-5")
        assert p["thinking"] == {"type": "between_tools"} and "output_config" not in p
        p = self._p("claude-sonnet-5-5", reasoning_effort="medium", reasoning_max_tokens=4096)
        assert p["thinking"] == self.ADAPTIVE and p["output_config"] == {"effort": "medium"}

    def test_haiku_5_5_is_adaptive_with_effort_and_disabled_when_off(self):
        p = self._p("claude-haiku-5-5")
        assert p["thinking"] == {"type": "disabled"} and "output_config" not in p
        p = self._p("claude-haiku-5-5", reasoning_effort="high", reasoning_max_tokens=4096)
        assert p["thinking"] == self.ADAPTIVE and p["output_config"] == {"effort": "high"}

    def test_budget_models_unchanged(self):
        p = self._p("claude-haiku-4-5", reasoning_effort="medium", reasoning_max_tokens=4096)
        assert p["thinking"] == {"type": "enabled", "budget_tokens": 4096}

    def test_no_budget_tokens_on_any_new_model(self):
        for model in ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"):
            p = self._p(model, reasoning_effort="high", reasoning_max_tokens=9000)
            assert "budget_tokens" not in (p.get("thinking") or {})

    @pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-mythos-5-1", "claude-opus-5-5", "claude-sonnet-5-5"])
    def test_forced_tool_choice_refused_before_the_request(self, model):
        for choice in (ToolChoice(mode=ToolChoiceMode.REQUIRED), ToolChoice(mode=ToolChoiceMode.FUNCTION, name="f")):
            with pytest.raises(ValueError, match="forced tool use"):
                h.build_params(LLMInput(text="hi", tools=TOOLS, tool_choice=choice), model)
        p = h.build_params(LLMInput(text="hi", tools=TOOLS, tool_choice=ToolChoice(mode=ToolChoiceMode.NONE)), model)
        assert p["tool_choice"] == {"type": "none"}

    def test_haiku_5_5_and_older_models_keep_forced_tool_choice(self):
        for model in ("claude-haiku-5-5", "claude-opus-5", "claude-sonnet-5"):
            p = h.build_params(LLMInput(text="hi", tools=TOOLS, tool_choice=ToolChoice(mode=ToolChoiceMode.REQUIRED)), model)
            assert p["tool_choice"] == {"type": "any"}


class TestMaxTokens:
    def test_callers_value_capped_at_the_model_limit(self):
        assert h.build_params(LLMInput(text="hi"), "claude-opus-5-5", 128000)["max_tokens"] == 64000  # LLMInput default
        assert h.build_params(LLMInput(text="hi", max_tokens=100000), "claude-opus-5-5", 128000)["max_tokens"] == 100000
        assert h.build_params(LLMInput(text="hi", max_tokens=500000), "claude-opus-5-5", 128000)["max_tokens"] == 128000
        assert h.build_params(LLMInput(text="hi", max_tokens=100000), "claude-haiku-4-5", 16000)["max_tokens"] == 16000


class TestStopReason:
    def test_stream_message_delta_sets_it(self):
        state = h.create_initial_state()
        for anthropic, platform in (("end_turn", "stop"), ("tool_use", "tool_calls"), ("max_tokens", "length"),
                                    ("stop_sequence", "stop"), ("refusal", "refusal"), ("pause_turn", "pause_turn")):
            h.apply_stream_event(NS(type="message_delta", delta=NS(stop_reason=anthropic), usage=NS(output_tokens=3)), state)
            state["input_tokens"] = 1
            assert h.build_output(state)["usage"].stop_reason == platform

    def test_unknown_stays_empty(self):
        state = h.create_initial_state(); state["input_tokens"] = 1
        assert h.build_output(state)["usage"].stop_reason == ""



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


# ── files as document blocks ──────────────────────────────────────────

class TestFiles:
    @pytest.fixture
    def pdf(self, tmp_path):
        p = tmp_path / "form.pdf"
        p.write_bytes(b"%PDF-1.4 fake")
        return str(p)

    def _ctx(self, text, **kw):
        from inferencesh.models.llm import ContextMessage, ContextMessageRole
        return ContextMessage(role=ContextMessageRole.USER, text=text, **kw)

    def test_pdf_path_is_a_base64_document(self, pdf):
        block = h._build_document_block(NS(path=pdf, uri="https://x/form.pdf", content_type="application/pdf", filename="form.pdf"))
        assert block["type"] == "document" and block["title"] == "form.pdf"
        assert block["source"]["type"] == "base64" and block["source"]["media_type"] == "application/pdf"
        assert base64.b64decode(block["source"]["data"]) == b"%PDF-1.4 fake"

    def test_pdf_url_only_is_a_url_document(self):
        block = h._build_document_block(NS(path=None, uri="https://x/a.pdf", content_type=None, filename=None))
        assert block == {"type": "document", "source": {"type": "url", "url": "https://x/a.pdf"}, "title": "a.pdf"}

    def test_type_guessed_from_name_when_octet_stream(self, pdf):
        block = h._build_document_block(NS(path=pdf, uri=None, content_type="application/octet-stream", filename="form.pdf"))
        assert block["type"] == "document" and block["source"]["media_type"] == "application/pdf"

    def test_text_file_is_a_text_document(self, tmp_path):
        p = tmp_path / "notes.md"
        p.write_text("# hello\n")
        block = h._build_document_block(NS(path=str(p), uri=None, content_type="text/markdown", filename="notes.md"))
        assert block == {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "# hello\n"}, "title": "notes.md"}

    def test_empty_text_file_is_a_note(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        block = h._build_document_block(NS(path=str(p), uri=None, content_type="text/plain", filename="empty.txt"))
        assert block["type"] == "text" and "empty" in block["text"]

    def test_unreadable_type_names_the_file(self):
        block = h._build_document_block(NS(path="/nope.docx", uri="https://x/cv.docx", content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", filename="cv.docx"))
        assert block["type"] == "text"
        assert "cv.docx" in block["text"] and "https://x/cv.docx" in block["text"] and "PDF and text" in block["text"]

    def test_oversized_pdf_is_a_note(self, pdf, monkeypatch):
        monkeypatch.setattr(h, "_MAX_DOCUMENT_BYTES", 4)
        block = h._build_document_block(NS(path=pdf, uri=None, content_type="application/pdf", filename="form.pdf"))
        assert block["type"] == "text" and "32MB" in block["text"]

    def test_current_input_file_follows_text(self, pdf):
        from inferencesh import File
        inp = LLMInput(context=[], text="do you see the contents", files=[File(path=pdf)])
        _, msgs = h.convert_messages_to_anthropic(inp)
        assert [b["type"] for b in msgs[-1]["content"]] == ["text", "document"]
        assert msgs[-1]["content"][1]["title"] == "form.pdf"

    def test_context_user_message_carries_file(self, pdf):
        from inferencesh import File
        inp = LLMInput(context=[self._ctx("here", files=[File(path=pdf)])], text="and?")
        _, msgs = h.convert_messages_to_anthropic(inp)
        # Two user turns in a row merge into one message, the document between them.
        assert len(msgs) == 1
        assert [b["type"] for b in msgs[0]["content"]] == ["text", "document", "text"]
        assert msgs[0]["content"][2] == {"type": "text", "text": "and?"}

    def test_tool_result_carries_document(self, pdf):
        block = h._build_tool_result_block("t", "fetched", None, files=[NS(path=pdf, uri=None, content_type="application/pdf", filename="form.pdf")])
        assert [c["type"] for c in block["content"]] == ["text", "document"]

    def test_no_files_keeps_message_shape(self):
        inp = LLMInput(context=[], text="hi")
        _, msgs = h.convert_messages_to_anthropic(inp)
        assert msgs == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


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


# ── reasoning_state: thinking blocks carried to the next request ──────

def ev_think_start(thinking="", signature=""):
    return NS(type="content_block_start", content_block=NS(type="thinking", thinking=thinking, signature=signature))


def ev_redacted(data):
    return NS(type="content_block_start", content_block=NS(type="redacted_thinking", data=data))


def ev_signature(sig):
    return NS(type="content_block_delta", delta=NS(type="signature_delta", signature=sig))


THINKING = {"type": "thinking", "thinking": "plan it", "signature": "sig-AB"}
REDACTED = {"type": "redacted_thinking", "data": "opaque=="}
TOOL_CALL = {"id": "toolu_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}


class TestReasoningStateCapture:
    def test_stream_keeps_every_block_whole_and_in_order(self):
        state, deltas = fold([
            ev_think_start(), ev_think("plan "), ev_think("it"), ev_signature("sig-"), ev_signature("AB"), ev_block_stop(),
            ev_redacted("opaque=="), ev_block_stop(),
            ev_tool_start("toolu_1", "get_weather"), ev_tool_json('{"city": "Paris"}'), ev_block_stop(),
        ])
        out = h.build_output(state)
        assert out["reasoning_state"] == {"anthropic": {"blocks": [THINKING, REDACTED]}}
        assert out["reasoning"] == "plan it"
        assert deltas[3] is None, "a signature is not reasoning the user reads"

    def test_no_thinking_no_state(self):
        state, _ = fold([ev_text("hi")])
        assert "reasoning_state" not in h.build_output(state)

    @pytest.mark.asyncio
    async def test_non_streaming_response(self):
        response = NS(usage=None, content=[
            NS(type="thinking", thinking="plan it", signature="sig-AB"),
            NS(type="redacted_thinking", data="opaque=="),
            NS(type="tool_use", id="toolu_1", name="get_weather", input={"city": "Paris"}),
        ])

        class Messages:
            async def create(self, **_):
                return response

        client = NS(with_options=lambda **_: NS(messages=Messages()))
        out = await h.complete(client, LLMInput(text="hi"), "claude-sonnet-5")
        assert out["reasoning_state"] == {"anthropic": {"blocks": [THINKING, REDACTED]}}


def assistant_turn(state, text=" "):
    return LLMInput(text="18C", role="tool", tool_call_id="toolu_1", context=[
        {"role": "user", "text": "weather in Paris?"},
        {"role": "assistant", "text": text, "tool_calls": [TOOL_CALL], "reasoning_state": state},
    ])


class TestReasoningStateReplay:
    def test_blocks_go_first_exactly_as_returned(self):
        state = {"anthropic": {"blocks": [THINKING, REDACTED]}, "gemini": {"signatures": ["x"]}}
        _, messages = h.convert_messages_to_anthropic(assistant_turn(state, text="Checking."))
        assert messages[1]["content"] == [
            THINKING, REDACTED,
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
        ]

    def test_platform_placeholder_text_is_not_added_beside_the_blocks(self):
        _, messages = h.convert_messages_to_anthropic(assistant_turn({"anthropic": {"blocks": [THINKING]}}))
        assert [b["type"] for b in messages[1]["content"]] == ["thinking", "tool_use"]

    def test_other_apps_entries_are_never_sent(self):
        for state in ({"gemini": {"signatures": ["x"]}, "openai": {"items": []}}, None):
            _, messages = h.convert_messages_to_anthropic(assistant_turn(state))
            assert [b["type"] for b in messages[1]["content"]] == ["text", "tool_use"], "unchanged from before"

    def test_every_turn_is_replayed(self):
        """A new user turn keeps the earlier turns' blocks: dropping them is a
        history edit that misses the prompt cache from that turn on."""
        old = {"anthropic": {"blocks": [dict(THINKING, signature="old")]}}
        new = {"anthropic": {"blocks": [THINKING]}}
        context = [
            {"role": "user", "text": "weather in Paris?"},
            {"role": "assistant", "text": " ", "tool_calls": [dict(TOOL_CALL, id="toolu_0")], "reasoning_state": old},
            {"role": "tool", "tool_call_id": "toolu_0", "text": "18C"},
            {"role": "assistant", "text": "18C.", "reasoning_state": old},
            {"role": "user", "text": "and Oslo?"},
            {"role": "assistant", "text": " ", "tool_calls": [TOOL_CALL], "reasoning_state": new},
        ]

        def thinking(messages):
            return [b["signature"] for m in messages if m["role"] == "assistant"
                    for b in m["content"] if b["type"] == "thinking"]

        _, in_loop = h.convert_messages_to_anthropic(
            LLMInput(text="9C", role="tool", tool_call_id="toolu_1", context=context))
        _, next_turn = h.convert_messages_to_anthropic(LLMInput(text="thanks", context=context))
        assert thinking(in_loop) == thinking(next_turn) == ["old", "old", "sig-AB"]
        assert in_loop[:-1] == next_turn[:-1], "the history is the same bytes on the next turn"

    def test_without_replay_no_blocks_are_sent(self):
        _, messages = h.convert_messages_to_anthropic(
            assistant_turn({"anthropic": {"blocks": [THINKING]}}), replay_thinking=False)
        assert [b["type"] for b in messages[1]["content"]] == ["text", "tool_use"], "as with no state"

    def test_consecutive_assistant_messages_stay_apart_with_their_blocks(self):
        first = {"anthropic": {"blocks": [THINKING]}}
        second = {"anthropic": {"blocks": [dict(THINKING, signature="second"), REDACTED]}}
        _, messages = h.convert_messages_to_anthropic(LLMInput(text="18C", role="tool", tool_call_id="toolu_1", context=[
            {"role": "user", "text": "weather in Paris?"},
            {"role": "assistant", "text": "Let me check.", "reasoning_state": first},
            {"role": "assistant", "text": " ", "tool_calls": [TOOL_CALL], "reasoning_state": second},
        ]))
        assert [m["role"] for m in messages] == ["user", "assistant", "assistant", "user"]
        assert [b["type"] for b in messages[1]["content"]] == ["thinking", "text"]
        assert [b["type"] for b in messages[2]["content"]] == ["thinking", "redacted_thinking", "tool_use"]

    def test_consecutive_user_messages_still_merge(self):
        _, messages = h.convert_messages_to_anthropic(LLMInput(text="b", context=[{"role": "user", "text": "a"}]))
        assert messages == [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]


class TestBlockBinding:
    def test_adaptive_thinking_drops_stale_blocks_under_the_beta(self):
        p = h.build_params(LLMInput(text="hi"), "claude-opus-5-5")
        assert p["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
        assert p["extra_headers"] == {"anthropic-beta": "thinking-binding-controls-2026-08-01"}
        assert "block_binding" not in h.ADAPTIVE_THINKING, "the shared default is not mutated"

    @pytest.mark.parametrize("model,kw", [
        ("claude-sonnet-5-5", {}),  # between_tools takes no other field
        ("claude-sonnet-4-6", {}),  # disabled
        ("claude-haiku-4-5", {"reasoning_effort": "medium", "reasoning_max_tokens": 4096}),  # budget_tokens
    ])
    def test_other_thinking_shapes_go_without(self, model, kw):
        p = h.build_params(LLMInput(text="hi", **kw), model)
        assert "block_binding" not in p["thinking"] and "extra_headers" not in p


class TestRefusedReplayRetry:
    class Refused(Exception):
        status_code = 400

    def _client(self, fail_first):
        calls = []
        refused = self.Refused("messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation.")

        class Messages:
            async def create(self, **params):
                calls.append(params)
                if fail_first and len(calls) == 1:
                    raise refused
                return NS(usage=None, stop_reason="end_turn", content=[NS(type="text", text="ok")])

        return NS(with_options=lambda **_: NS(messages=Messages())), calls

    @pytest.mark.asyncio
    async def test_a_refused_block_is_retried_once_without_replay(self):
        client, calls = self._client(fail_first=True)
        out = await h.complete(client, assistant_turn({"anthropic": {"blocks": [THINKING]}}), "claude-opus-5-5")
        assert out["response"] == "ok" and len(calls) == 2
        assert calls[0]["messages"][1]["content"][0] == THINKING
        assert [b["type"] for b in calls[1]["messages"][1]["content"]] == ["text", "tool_use"]
        assert calls[1]["stream"] is False

    @pytest.mark.asyncio
    async def test_other_errors_are_not_retried(self):
        class Messages:
            calls = 0

            async def create(self, **_):
                Messages.calls += 1
                raise TestRefusedReplayRetry.Refused("max_tokens: too large")

        client = NS(with_options=lambda **_: NS(messages=Messages()))
        with pytest.raises(RuntimeError):
            await h.complete(client, LLMInput(text="hi"), "claude-opus-5-5")
        assert Messages.calls == 1
