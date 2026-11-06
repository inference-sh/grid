"""No-network tests: OpenAI safety refusals raise ContentModerationError with error_code."""

import asyncio
import importlib.util
import sys
from pathlib import Path

import httpx
import openai
import pytest

_spec = importlib.util.spec_from_file_location("openai_helper", Path(__file__).with_name("openai_helper.py"))
openai_helper = importlib.util.module_from_spec(_spec)
sys.modules["openai_helper"] = openai_helper
_spec.loader.exec_module(openai_helper)


def bad_request(code):
    body = {"message": "Your request was rejected by the safety system.", "type": "image_generation_user_error",
            "param": None, "code": code}
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/images/generations"),
                              json={"error": body})
    return openai.BadRequestError(f"Error code: 400 - {{'error': {body}}}", response=response, body=body)


async def raising(exc):
    raise exc


@pytest.mark.parametrize("code", ["moderation_blocked", "content_policy_violation"])
def test_a_safety_refusal_is_content_moderation(code):
    exc = bad_request(code)
    with pytest.raises(openai_helper.ContentModerationError) as info:
        asyncio.run(openai_helper.moderated(raising(exc)))
    assert info.value.error_code == "content_moderation"
    assert str(info.value) == str(exc)
    assert info.value.__cause__ is exc


@pytest.mark.parametrize("code", ["invalid_value", "invalid_image_file", None])
def test_other_bad_requests_pass_through(code):
    exc = bad_request(code)
    with pytest.raises(openai.BadRequestError) as info:
        asyncio.run(openai_helper.moderated(raising(exc)))
    assert info.value is exc


def test_a_result_is_returned():
    async def ok():
        return 7
    assert asyncio.run(openai_helper.moderated(ok())) == 7
