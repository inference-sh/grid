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
