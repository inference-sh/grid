"""No-network tests: Gemini safety refusals raise ContentModerationError with error_code."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import types

HERE = Path(__file__).parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HELPERS = [load("vertex_helper", HERE / "vertex_helper.py"),
           load("gemini_helper", HERE.parent.parent / "gemini-studio" / "gemini_helper.py")]


def blocked_response(reason):
    return SimpleNamespace(candidates=[], prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=reason))


def finished_response(reason):
    candidate = SimpleNamespace(finish_reason=reason, finish_message="Unable to show the generated image.",
                                content=SimpleNamespace(parts=[]))
    return SimpleNamespace(candidates=[candidate], usage_metadata=None)


@pytest.fixture(params=HELPERS, ids=["vertex", "studio"])
def helper(request):
    return request.param


@pytest.mark.parametrize("reason", ["SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "IMAGE_SAFETY"])
def test_a_blocked_prompt_is_content_moderation(helper, reason):
    with pytest.raises(helper.ContentModerationError) as info:
        helper.process_image_response(blocked_response(types.BlockedReason(reason)))
    assert info.value.error_code == "content_moderation"
    assert "Image generation blocked" in str(info.value)


def test_a_prompt_blocked_for_another_reason_stays_runtime_error(helper):
    with pytest.raises(RuntimeError) as info:
        helper.process_image_response(blocked_response(types.BlockedReason.OTHER))
    assert not isinstance(info.value, helper.ContentModerationError)


@pytest.mark.parametrize("reason", ["IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "PROHIBITED_CONTENT", "SAFETY", "SPII"])
def test_a_refused_image_is_content_moderation(helper, reason):
    result = helper.process_image_response(finished_response(types.FinishReason(reason)))
    with pytest.raises(helper.ContentModerationError) as info:
        helper.raise_no_images_error([result])
    assert f"finish_reason=FinishReason.{reason}" in str(info.value)


@pytest.mark.parametrize("reason", ["STOP", "IMAGE_RECITATION", "MAX_TOKENS"])
def test_other_empty_results_stay_runtime_error(helper, reason):
    result = helper.process_image_response(finished_response(types.FinishReason(reason)))
    with pytest.raises(RuntimeError) as info:
        helper.raise_no_images_error([result])
    assert not isinstance(info.value, helper.ContentModerationError)


def test_wrapping_keeps_the_class(helper):
    wrapped = helper.wrap_app_error(helper.ContentModerationError("No images were generated"), "Image generation failed")
    assert isinstance(wrapped, helper.ContentModerationError)
    assert str(wrapped) == "Image generation failed: No images were generated"
    plain = helper.wrap_app_error(ValueError("bad"), "Image generation failed")
    assert type(plain) is RuntimeError and str(plain) == "Image generation failed: bad"


VERTEX = HELPERS[0]

VEO_REFUSALS = [
    "The prompt could not be submitted. This prompt contains sensitive words that violate Google's Responsible AI practices.",
    "Veo could not generate videos because the input image violates Vertex AI's usage guidelines.",
    "The input image contains content that has been blocked by your current safety settings for person/face generation.",
]


@pytest.mark.parametrize("message", VEO_REFUSALS)
def test_a_veo_refusal_is_content_moderation(message):
    with pytest.raises(VERTEX.ContentModerationError) as info:
        VERTEX.raise_veo_no_videos({"error": {"code": 3, "message": message}})
    assert info.value.error_code == "content_moderation"
    assert str(info.value).startswith("Video generation failed: {'code': 3")


def test_a_veo_filtered_video_is_content_moderation():
    result = {"response": {"raiMediaFilteredCount": 1, "raiMediaFilteredReasons": [
        "1 videos were filtered out because they violated Vertex AI's usage guidelines."]}}
    with pytest.raises(VERTEX.ContentModerationError, match=r"blocked by content filtering \(1 filtered\)"):
        VERTEX.raise_veo_no_videos(result)


@pytest.mark.parametrize("result", [
    {"error": {"code": 8, "message": "The service is currently experiencing high load and cannot process your request."}},
    {"error": {"code": 13, "message": "Internal error. Please try again later."}},
    {"response": {}},
])
def test_other_veo_failures_stay_runtime_error(result):
    with pytest.raises(RuntimeError) as info:
        VERTEX.raise_veo_no_videos(result)
    assert not isinstance(info.value, VERTEX.ContentModerationError)
