"""gemini_chat_helper: delta extraction and tool_choice/response_format mapping. No network."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("GCP_ACCESS_TOKEN", "x")
os.environ.setdefault("GCP_PROJECT_NUMBER", "1")

from google.genai import types  # noqa: E402
from inferencesh.llm_types_gen import (  # noqa: E402
    ResponseFormat,
    ResponseFormatType,
    ToolChoice,
    ToolChoiceMode,
)
from inferencesh.models.llm import LLMInput, ReasoningEffortEnum  # noqa: E402

import gemini_chat_helper as h  # noqa: E402


def text(t, thought=False):
    return SimpleNamespace(text=t, thought=thought, function_call=None)


def call(name, args):
    return SimpleNamespace(text=None, thought=False, function_call=SimpleNamespace(name=name, args=args))


class TestApplyParts:
    def test_text_and_thought_deltas(self):
        state = h.create_initial_state()
        assert h.apply_parts([text("plan", thought=True)], state) == {"reasoning": "plan"}
        assert h.apply_parts([text("Hel"), text("lo")], state) == {"response": "Hello"}
        assert state["thinking"] == "plan" and state["response"] == "Hello"

    def test_function_call_is_one_indexed_delta_with_full_arguments(self):
        state = h.create_initial_state()
        delta = h.apply_parts([call("get_weather", {"city": "Paris"})], state)
        tc = delta["tool_calls"][0]
        assert tc["index"] == 0 and tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}
        second = h.apply_parts([call("other", {})], state)
        assert second["tool_calls"][0]["index"] == 1
        assert [c["function"]["name"] for c in state["tool_calls"]] == ["get_weather", "other"]

    def test_empty_chunk_yields_no_delta(self):
        state = h.create_initial_state()
        assert h.apply_parts([], state) is None
        assert h.apply_parts([text("")], state) is None


class TestToolConfig:
    def test_none_when_unset(self):
        assert h.build_tool_config(None, has_tools=True) is None

    @pytest.mark.parametrize("mode,expected", [
        (ToolChoiceMode.NONE, types.FunctionCallingConfigMode.NONE),
        (ToolChoiceMode.AUTO, types.FunctionCallingConfigMode.AUTO),
        (ToolChoiceMode.REQUIRED, types.FunctionCallingConfigMode.ANY),
    ])
    def test_modes(self, mode, expected):
        cfg = h.build_tool_config(ToolChoice(mode=mode), has_tools=True)
        assert cfg.function_calling_config.mode == expected
        assert cfg.function_calling_config.allowed_function_names is None

    def test_named_function_restricts_allowed_names(self):
        cfg = h.build_tool_config(ToolChoice(mode=ToolChoiceMode.FUNCTION, name="f"), has_tools=True)
        assert cfg.function_calling_config.mode == types.FunctionCallingConfigMode.ANY
        assert cfg.function_calling_config.allowed_function_names == ["f"]

    def test_forcing_without_tools_is_rejected(self):
        with pytest.raises(ValueError, match="requires tools"):
            h.build_tool_config(ToolChoice(mode=ToolChoiceMode.REQUIRED), has_tools=False)


class TestResponseFormat:
    def test_text_or_unset_is_noop(self):
        assert h.build_response_format_config(None, has_tools=False) == {}
        assert h.build_response_format_config(ResponseFormat(type=ResponseFormatType.TEXT), has_tools=True) == {}

    def test_json_object(self):
        assert h.build_response_format_config(ResponseFormat(type=ResponseFormatType.JSON_OBJECT), has_tools=False) == {
            "response_mime_type": "application/json"}

    def test_json_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        cfg = h.build_response_format_config(
            ResponseFormat(type=ResponseFormatType.JSON_SCHEMA, json_schema=schema), has_tools=False)
        assert cfg == {"response_mime_type": "application/json", "response_json_schema": schema}
        types.GenerateContentConfig(**cfg)  # accepted by the SDK

    def test_json_with_tools_is_rejected(self):
        with pytest.raises(ValueError, match="not supported together with tools"):
            h.build_response_format_config(ResponseFormat(type=ResponseFormatType.JSON_OBJECT), has_tools=True)


class TestThinkingConfig:
    def test_none_effort_disables_when_allowed(self):
        i = LLMInput(text="x", reasoning_effort=ReasoningEffortEnum.NONE)
        assert h.build_thinking_config(i).thinking_budget == 0
        assert h.build_thinking_config(i, can_disable_thinking=False) is None


class TestToolResultImages:
    def test_tool_images_ride_in_function_response_parts(self, tmp_path):
        from inferencesh import File
        from inferencesh.models.llm import ContextMessage, ContextMessageRole

        png = tmp_path / "shot.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        inp = LLMInput(
            context=[ContextMessage(role=ContextMessageRole.TOOL, tool_call_id="t0", text="captured", images=[File(path=str(png))])],
            text='{"ok": true}',
            role=ContextMessageRole.TOOL,
            tool_call_id="t1",
            images=[File(path=str(png))],
        )
        _, contents = h.convert_messages_to_gemini(inp, "gemini-3.8-flash")
        for c in contents:
            assert c.role == "user" and len(c.parts) == 1
            fr = c.parts[0].function_response
            assert fr is not None and len(fr.parts) == 1
            assert fr.parts[0].inline_data.mime_type == "image/png"

    def test_text_only_tool_result_unchanged(self):
        from inferencesh.models.llm import ContextMessageRole

        _, contents = h.convert_messages_to_gemini(
            LLMInput(text='{"v": 42}', role=ContextMessageRole.TOOL, tool_call_id="t1")
        )
        assert len(contents) == 1 and len(contents[0].parts) == 1
        assert contents[0].parts[0].function_response.parts is None

    def test_large_tool_image_is_downsized(self, tmp_path):
        import io
        from PIL import Image
        from inferencesh import File
        from inferencesh.models.llm import ContextMessageRole

        png = tmp_path / "big.png"
        Image.new("RGB", (6000, 3000), (0, 160, 0)).save(png, format="PNG")
        _, contents = h.convert_messages_to_gemini(
            LLMInput(text="captured", role=ContextMessageRole.TOOL, tool_call_id="t1", images=[File(path=str(png))]),
            "gemini-3.8-flash",
        )
        blob = contents[0].parts[0].function_response.parts[0].inline_data
        assert Image.open(io.BytesIO(blob.data)).size == (3072, 1536)


class TestGemini25AndSignatures:
    def _tool_input(self, tmp_path):
        from inferencesh import File
        from inferencesh.models.llm import ContextMessage, ContextMessageRole

        png = tmp_path / "shot.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        calls = [{"id": f"c{i}", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}} for i in (1, 2)]
        return LLMInput(
            context=[
                ContextMessage(role=ContextMessageRole.USER, text="look"),
                ContextMessage(role=ContextMessageRole.ASSISTANT, text="", tool_calls=calls),
            ],
            text="captured",
            role=ContextMessageRole.TOOL,
            tool_call_id="c1",
            images=[File(path=str(png))],
        )

    def test_gemini_2_5_gets_images_after_the_function_response(self, tmp_path):
        _, contents = h.convert_messages_to_gemini(self._tool_input(tmp_path), "gemini-2.5-flash")
        parts = contents[-1].parts
        assert parts[0].function_response.parts is None
        assert parts[2].inline_data.mime_type == "image/png"

    def test_gemini_3_first_function_call_carries_the_skip_signature(self, tmp_path):
        _, contents = h.convert_messages_to_gemini(self._tool_input(tmp_path), "gemini-3.8-flash")
        calls = [p for p in contents[1].parts if p.function_call]
        dumped = [p.model_dump(mode="json", exclude_none=True).get("thought_signature") for p in calls]
        assert dumped == ["skip_thought_signature_validator", None]

    def test_gemini_2_5_sends_no_signature(self, tmp_path):
        _, contents = h.convert_messages_to_gemini(self._tool_input(tmp_path), "gemini-2.5-pro")
        assert all(p.thought_signature is None for p in contents[1].parts)
