"""Tests for melious.py usage reporting."""

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("melious", Path(__file__).with_name("melious.py"))
melious = importlib.util.module_from_spec(_spec)
sys.modules["melious"] = melious
_spec.loader.exec_module(melious)


class TestUsage:
    def test_cache_reads_and_writes_reach_usage_and_token_report(self):
        state = melious._create_initial_state()
        melious._handle_chunk({"id": "c1", "choices": [], "usage": {
            "prompt_tokens": 1000, "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 300},
            "completion_tokens_details": {"reasoning_tokens": 5}}}, state)
        out = melious._build_output(state, final=True)
        u = out["usage"]
        assert (u.prompt_tokens, u.cached_tokens, u.cache_write_tokens, u.completion_tokens) == (1000, 600, 300, 20)
        item = out["output_meta"].inputs[0]
        assert (item.tokens, item.cached_tokens, item.cache_write_tokens) == (1000, 600, 300)
        assert out["output_meta"].outputs[0].tokens == 20


class TestRequestBody:
    def test_reasoning_content_goes_back_only_for_models_that_replay_it(self):
        from inferencesh.models.llm import ContextMessage, ContextMessageRole, LLMInput

        call = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
        input_data = LLMInput(
            role=ContextMessageRole.TOOL, tool_call_id="call_1", text="18C", system_prompt="",
            context=[
                ContextMessage(role=ContextMessageRole.USER, text="weather?"),
                ContextMessage(role=ContextMessageRole.ASSISTANT, text="", tool_calls=[call], reasoning="call the tool"),
            ],
        )
        body = melious.build_request_body(input_data, "deepseek-v4.1-flash", vision=False, replays_reasoning=True)
        assistant = next(m for m in body["messages"] if m["role"] == "assistant")
        assert assistant["reasoning_content"] == "call the tool"

        body = melious.build_request_body(input_data, "glm-5.3", vision=False)
        assistant = next(m for m in body["messages"] if m["role"] == "assistant")
        assert "reasoning_content" not in assistant


class TestJSONSchemaDowngrade:
    @staticmethod
    def _input(system_prompt="Be terse."):
        from inferencesh.llm_types_gen import ResponseFormat, ResponseFormatType
        from inferencesh.models.llm import LLMInput

        return LLMInput(
            text="one fruit", system_prompt=system_prompt,
            response_format=ResponseFormat(
                type=ResponseFormatType.JSON_SCHEMA, name="fruit", strict=True,
                json_schema={"type": "object", "properties": {"fruit": {"type": "string"}}, "required": ["fruit"]},
            ),
        )

    def test_known_models_send_json_object_with_the_schema_in_the_system_prompt(self):
        body = melious.build_request_body(self._input(), "deepseek-v4.1-flash-1017", vision=True)
        assert body["response_format"] == {"type": "json_object"}
        system = body["messages"][0]
        assert system["role"] == "system"
        assert system["content"].startswith("Be terse.")
        assert '"required":["fruit"]' in system["content"]
        assert "(fruit)" in system["content"]

    def test_other_models_keep_json_schema(self):
        body = melious.build_request_body(self._input(), "deepseek-v4-flash-0731", vision=False)
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["schema"]["required"] == ["fruit"]
        assert body["messages"][0]["content"] == "Be terse."

    def test_downgrade_adds_a_system_message_when_there_is_none_and_runs_once(self):
        body = melious.build_request_body(self._input(system_prompt=""), "kimi-k3", vision=False)
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"].startswith("Respond with a single JSON object")
        assert body["messages"][1]["role"] == "user"
        # Already json_object: nothing left to downgrade, so a second rejection is final.
        assert melious._downgrade_json_schema(body) is False

    def test_rejection_is_recognised_only_for_schema_mode(self):
        err = {"code": "model_not_capable", "message": "The model 'x' does not support JSON schema mode. Use 'json_object' mode instead."}
        assert melious._is_schema_unsupported(err) is True
        assert melious._is_schema_unsupported({"code": "model_not_capable", "message": "no vision"}) is False
        assert melious._is_schema_unsupported("bad request") is False
        assert melious._is_schema_unsupported(None) is False

    def test_rejection_inside_the_stream_asks_for_the_downgrade(self):
        import pytest

        err = {"code": "model_not_capable", "message": "The model 'glm-5.3-flash' does not support JSON schema mode."}
        with pytest.raises(melious._SchemaUnsupported):
            melious._handle_chunk({"error": err}, melious._create_initial_state())
        with pytest.raises(RuntimeError, match="rate_limited"):
            melious._handle_chunk({"error": {"code": "rate_limited", "message": "slow down"}}, melious._create_initial_state())
