"""Tests for openrouter.py usage reporting."""

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("openrouter", Path(__file__).with_name("openrouter.py"))
openrouter = importlib.util.module_from_spec(_spec)
sys.modules["openrouter"] = openrouter
_spec.loader.exec_module(openrouter)


class TestUsage:
    def test_cache_reads_and_writes_reach_usage_and_token_report(self):
        state = openrouter._create_initial_state()
        openrouter._parse_sse_chunk({"id": "gen-1", "choices": [], "usage": {
            "prompt_tokens": 1000, "completion_tokens": 20, "cost": 0.0123,
            "prompt_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 300},
            "completion_tokens_details": {"reasoning_tokens": 5}}}, state)
        out = openrouter._build_output(state)
        u = out["usage"]
        assert (u.prompt_tokens, u.cached_tokens, u.cache_write_tokens, u.completion_tokens, u.reasoning_tokens) == (1000, 600, 300, 20, 5)
        text, raw = out["output_meta"].inputs
        # The text item comes first: formulas read inputs[0] as the prompt.
        assert (text.type, text.tokens, text.cached_tokens, text.cache_write_tokens) == ("text", 1000, 600, 300)
        assert raw.type == "raw" and raw.cost == 1.23
        assert out["output_meta"].outputs[0].tokens == 20

    def test_no_usage_reports_no_tokens(self):
        out = openrouter._build_output(openrouter._create_initial_state())
        assert "usage" not in out and "output_meta" not in out


CALL = {"id": "toolu_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}


def _chunk(details):
    return {"choices": [{"delta": {"reasoning_details": details}}]}


class TestReasoningState:
    # The fragments OpenRouter streamed for anthropic/claude-haiku-4.5 with a
    # tool call (the signature comes on a fragment of its own).
    STREAMED = [
        {"type": "reasoning.text", "text": "The user wants", "format": "anthropic-claude-v1", "index": 0},
        {"type": "reasoning.text", "text": " the weather.", "format": "anthropic-claude-v1", "index": 0},
        {"type": "reasoning.text", "signature": "c2ln", "format": "anthropic-claude-v1", "index": 0},
    ]
    BLOCK = [{"type": "reasoning.text", "text": "The user wants the weather.", "signature": "c2ln",
              "format": "anthropic-claude-v1", "index": 0}]

    def test_streamed_details_land_in_reasoning_state_as_whole_blocks(self):
        state = openrouter._create_initial_state()
        for d in self.STREAMED:
            openrouter._parse_sse_chunk(_chunk([d]), state)
        out = openrouter._build_output(state)
        assert out["reasoning_state"] == {"openrouter": self.BLOCK}
        assert "reasoning_details" not in out
        assert openrouter.OpenRouterOutput(**out).reasoning_state == {"openrouter": self.BLOCK}

    def test_no_details_no_state(self):
        assert "reasoning_state" not in openrouter._build_output(openrouter._create_initial_state())

    def test_own_entry_replayed_foreign_left_out(self):
        from inferencesh.models.llm import ContextMessage, ContextMessageRole, LLMInput
        inp = LLMInput(role=ContextMessageRole.TOOL, tool_call_id="toolu_1", text='{"temp_c": 17}', context=[
            ContextMessage(role=ContextMessageRole.USER, text="weather?"),
            ContextMessage(role=ContextMessageRole.ASSISTANT, text="", tool_calls=[CALL],
                           reasoning_state={"openrouter": self.BLOCK, "minimax": [{"type": "reasoning.text", "text": "x"}]}),
            ContextMessage(role=ContextMessageRole.USER, text="and now?"),
            ContextMessage(role=ContextMessageRole.ASSISTANT, text="ok", reasoning_state={"gemini": {"signatures": ["eA=="]}}),
        ])
        assistants = [m for m in openrouter._build_request_body(inp, "anthropic/claude-haiku-4.5")["messages"] if m["role"] == "assistant"]
        assert assistants[0]["reasoning_details"] == self.BLOCK
        assert "reasoning_details" not in assistants[1]
