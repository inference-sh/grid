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
