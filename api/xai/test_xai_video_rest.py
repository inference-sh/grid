"""xai_video_rest: the REST requests the video apps send and how answers are read. No network."""

import asyncio
import importlib
import json
import logging
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("XAI_API_KEY", "test-key")

# The helper is a symlink inside each app package and imports its sibling relatively.
rest = importlib.import_module("grok-imagine-video.xai_video_rest")
helper = importlib.import_module("grok-imagine-video.xai_helper")


@pytest.fixture
def api(monkeypatch):
    """Route the helper's httpx.Client to a scripted transport; record requests."""
    seen, answers = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status, body = answers.pop(0)
        return httpx.Response(status, json=body)

    real = httpx.Client
    monkeypatch.setattr(rest.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return seen, answers


def sent(request):
    return request.method, str(request.url), json.loads(request.content) if request.content else None


def test_generation_request(api):
    seen, answers = api
    answers.append((200, {"request_id": "r1"}))
    rid = rest.start_generation(
        model="grok-imagine-video", prompt="p", duration=5, aspect_ratio="16:9", resolution="720p",
        image_url="data:image/png;base64,AAAA",
    )
    assert rid == "r1"
    assert sent(seen[0]) == ("POST", "https://api.x.ai/v1/videos/generations", {
        "model": "grok-imagine-video", "prompt": "p", "image": {"url": "data:image/png;base64,AAAA"},
        "duration": 5, "aspect_ratio": "16:9", "resolution": "720p",
    })
    assert seen[0].headers["Authorization"] == "Bearer test-key"


def test_reference_request(api):
    seen, answers = api
    answers.append((200, {"request_id": "r2"}))
    rest.start_generation(model="m", prompt="p", duration=5, aspect_ratio="1:1", resolution="480p",
                          reference_image_urls=["https://a/1.png", "https://a/2.png"])
    assert sent(seen[0])[2]["reference_images"] == [{"url": "https://a/1.png"}, {"url": "https://a/2.png"}]
    assert "image" not in sent(seen[0])[2]


def test_edit_and_extension_requests(api):
    seen, answers = api
    answers += [(200, {"request_id": "e"}), (200, {"request_id": "x"}), (200, {"request_id": "y"})]
    rest.start_edit(model="m", prompt="p", video_url="https://a/v.mp4")
    rest.start_extension(model="m", prompt="p", video_url="https://a/v.mp4", duration=6)
    rest.start_extension(model="m", prompt="p", video_url="https://a/v.mp4")
    assert sent(seen[0]) == ("POST", "https://api.x.ai/v1/videos/edits", {"model": "m", "prompt": "p", "video": {"url": "https://a/v.mp4"}})
    assert sent(seen[1]) == ("POST", "https://api.x.ai/v1/videos/extensions", {"model": "m", "prompt": "p", "video": {"url": "https://a/v.mp4"}, "duration": 6})
    assert "duration" not in sent(seen[2])[2]


def wait(rid="r1"):
    return asyncio.run(rest.wait_for_video(rid, logging.getLogger("t"), lambda: False, interval=0))


def test_wait_done(api):
    seen, answers = api
    answers += [
        (202, {"status": "pending", "progress": 10}),
        (200, {"status": "done", "model": "m", "video": {"url": "https://vidgen.x.ai/v.mp4", "duration": 6, "respect_moderation": True},
               "usage": {"cost_in_usd_ticks": 5_000_000_000}}),
    ]
    result = wait()
    assert sent(seen[0]) == ("GET", "https://api.x.ai/v1/videos/r1", None)
    assert (result.url, result.duration, result.respect_moderation) == ("https://vidgen.x.ai/v.mp4", 6, True)
    assert result.cost_usd == pytest.approx(0.5)
    assert helper.is_moderated(result) is False


def test_wait_moderated_and_no_usage(api):
    _, answers = api
    answers.append((200, {"status": "done", "video": {"url": None, "duration": 6, "respect_moderation": False}}))
    result = wait()
    assert helper.is_moderated(result) is True
    assert result.cost_usd is None
    with pytest.raises(ValueError):
        result.url


def test_wait_failed(api):
    _, answers = api
    answers.append((200, {"status": "failed", "error": {"code": "invalid_argument", "message": "bad prompt"}}))
    with pytest.raises(helper.XAIGenerationError, match=r"\(invalid_argument\): bad prompt"):
        wait()


def test_errors_read_like_the_sdk(api):
    _, answers = api
    answers += [(429, {"error": "slow down"}), (401, {"error": "bad key"}), (400, {"error": "bad"}), (503, {"error": "down"})]
    errors = []
    for _ in range(4):
        with pytest.raises(rest.XAIHTTPError) as e:
            rest.get_video("r")
        errors.append(e.value)
    assert helper.is_rate_limit_error(errors[0])
    assert not helper.is_rate_limit_error(errors[1])
    assert "authentication failed" in str(helper.classify_xai_error(errors[1]))
    assert isinstance(helper.classify_xai_error(errors[2]), helper.XAIGenerationError)
    assert isinstance(helper.classify_xai_error(errors[3]), helper.XAIUnavailableError)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY")
    with pytest.raises(RuntimeError, match="XAI_API_KEY"):
        rest.require_xai_api_key()
