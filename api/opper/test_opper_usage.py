import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import opper  # noqa: E402


def _out(usage):
    state = opper.create_initial_state()
    opper.read_usage(usage, state)
    return opper.build_output(state)


def _text_input(out):
    item = out["output_meta"].inputs[0]
    return item.tokens, item.cached_tokens, item.cache_write_tokens


def test_cache_read_is_added_back_into_the_input():
    # Opper's usage for a cache hit (probed): prompt_tokens leaves the cached part out.
    out = _out({"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0034656,
                "prompt_tokens_details": {"cached_tokens": 11202}})
    assert out["usage"].prompt_tokens == 11212
    assert _text_input(out) == (11212, 11202, 0)
    assert out["output_meta"].outputs[0].tokens == 5


def test_cache_write_is_added_back_into_the_input():
    out = _out({"prompt_tokens": 10, "completion_tokens": 5,
                "prompt_tokens_details": {"cache_creation_tokens": 11202, "cache_write_tokens": 11202}})
    assert _text_input(out) == (11212, 0, 11202)


def test_uncached_request():
    out = _out({"prompt_tokens": 11212, "completion_tokens": 5, "cost": 0.033711})
    assert _text_input(out) == (11212, 0, 0)
    raw = out["output_meta"].inputs[-1]
    assert raw.type == "raw" and abs(raw.cost - 3.3711) < 1e-9


def test_usage_object_from_the_openai_sdk():
    class Usage:
        def model_dump(self):
            return {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 90}}

    out = _out(Usage())
    assert _text_input(out) == (100, 90, 0)
