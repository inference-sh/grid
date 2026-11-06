"""No-network tests for the typed ElevenLabs errors in elevenlabs_helper.

The error bodies are the ones ElevenLabs returned in production.
"""

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

import pytest
from elevenlabs.core.api_error import ApiError

os.environ.setdefault("ELEVENLABS_KEY", "test")
sys.path.insert(0, str(Path(__file__).parent))
helper = importlib.import_module("tts.elevenlabs_helper")

SAFETY = {
    "detail": {
        "type": "authorization_error",
        "code": "forbidden",
        "message": "Sorry, this prompt potentially doesn't follow our safety guidelines and therefore has been blocked.",
        "status": "blocked_generation",
        "request_id": "req-1",
    }
}
VOICE_LIMIT = {
    "detail": {
        "type": "invalid_request",
        "message": "You have reached your maximum amount of custom voices (660 / 660).",
        "status": "voice_limit_reached",
        "request_id": "req-2",
    }
}


def error(body, status=403):
    return ApiError(status_code=status, headers={"server": "uvicorn"}, body=body)


def test_blocked_generation_is_moderation_and_keeps_the_request_id():
    e = helper.classify_elevenlabs_error(error(SAFETY))
    assert type(e) is helper.ContentModerationError
    assert e.status == "blocked_generation"
    assert e.request_id == "req-1"
    assert "req-1" in str(e) and "safety guidelines" in str(e)
    assert "headers" not in str(e)


@pytest.mark.parametrize("message", [
    "Your prompt appears to have violated our Terms of Service. Please try again with a different prompt.",
    "Voice 'abc' may violate our Terms of Service and has been blocked.",
    "Voice 'abc' may violate our Terms of Service and requires verification.",
    "This voice requires verification. Please verify your voice to continue.",
])
def test_refusals_without_a_known_status_are_moderation(message):
    e = helper.classify_elevenlabs_error(error({"detail": {"message": message}}))
    assert type(e) is helper.ContentModerationError
    assert message in str(e)


def test_a_prompt_suggestion_stays_in_the_message():
    body = {"detail": {"message": "Your prompt appears to have violated our Terms of Service.", "data": {"prompt_suggestion": "A calm song"}}}
    assert "Suggested prompt: A calm song" in str(helper.classify_elevenlabs_error(error(body)))


def test_other_errors_are_plain_elevenlabs_errors():
    e = helper.classify_elevenlabs_error(error(VOICE_LIMIT, status=400))
    assert type(e) is helper.ElevenLabsError
    assert e.status == "voice_limit_reached"
    assert str(e).startswith("You have reached")


@pytest.mark.parametrize("body,expected", [
    ("upstream exploded", "upstream exploded"),
    ([{"loc": ["text"], "msg": "required"}], "required"),
    (None, "ElevenLabs answered 500"),
])
def test_bodies_without_a_detail_never_show_the_headers(body, expected):
    message = str(helper.classify_elevenlabs_error(error(body, status=500)))
    assert expected in message
    assert "uvicorn" not in message


def test_a_moderation_error_is_still_a_runtime_error():
    assert issubclass(helper.ContentModerationError, RuntimeError)


def test_a_moderation_error_carries_the_platform_error_code():
    # The engine's kernel reads error_code and the API stores it on the task.
    e = helper.classify_elevenlabs_error(error(SAFETY))
    assert e.error_code == "content_moderation"
    assert getattr(helper.error_from_response(502, "bad gateway"), "error_code", None) is None


def test_an_answer_read_outside_the_sdk_is_classified():
    e = helper.error_from_response(403, json.dumps(SAFETY))
    assert type(e) is helper.ContentModerationError
    assert type(helper.error_from_response(502, "<html>bad gateway</html>")) is helper.ElevenLabsError


def test_sync_helpers_raise_the_typed_error(monkeypatch):
    class Previews:
        def create_previews(self, **_):
            raise error(SAFETY)

    class Client:
        text_to_voice = Previews()

    monkeypatch.setattr(helper, "get_client", Client)
    with pytest.raises(helper.ContentModerationError):
        helper.design_voice("a voice")


def test_audio_errors_raised_while_streaming_are_typed():
    def chunks():
        raise error(SAFETY)
        yield b""

    with pytest.raises(helper.ContentModerationError):
        helper.save_audio_generator(chunks())


def test_async_helpers_raise_the_typed_error():
    @helper.typed_errors
    async def call():
        raise error(VOICE_LIMIT, status=400)

    with pytest.raises(helper.ElevenLabsError):
        asyncio.run(call())


def test_every_sdk_helper_is_typed():
    names = ["text_to_speech", "speech_to_text", "speech_to_speech", "isolate_voice", "generate_sound_effect",
             "compose_music", "create_dubbing", "poll_dubbing_status", "download_dubbed_audio", "clone_voice",
             "list_cloned_voices", "delete_voice", "design_voice", "remix_voice", "text_to_dialogue", "forced_alignment"]
    assert all(hasattr(getattr(helper, n), "__wrapped__") for n in names)
