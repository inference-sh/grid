"""claude-code apps against testdata/fake-belt (a stand-in `belt harness serve`).

Needs a Python venv with inferencesh and pytest (uv venv v && uv pip install --python v/bin/python inferencesh pytest):

  cd grid/api/claude-code && REAL_BELT=bin/belt-linux-amd64 CLAUDE_BIN=$(which claude) v/bin/python -m pytest test_claude_code_helper.py -q

REAL_BELT (built by build.sh) adds the test_real_* runs: the real `belt harness
serve` and the real claude (CLAUDE_BIN, else claude on PATH) against
testdata/fake-anthropic.py, which the tests start on a free port, with a fake
token and a throwaway HOME. They are skipped when REAL_BELT or claude is missing.
"""

import asyncio
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import claude_code_helper as h  # noqa: E402
from inferencesh.models.llm import LLMDelta  # noqa: E402

FAKE = HERE / "testdata" / "fake-belt"
APPS = {"opus-5-5": "claude-opus-5-5", "sonnet-5-5": "claude-sonnet-5-5",
        "haiku-5-5": "claude-haiku-5-5", "fable-5-1": "claude-fable-5-1"}


def load(app_dir):
    return importlib.import_module(f"{app_dir}.inference")


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("BELT_BIN", str(FAKE))
    monkeypatch.setenv("CLAUDE_BIN", "/fake/claude")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-good")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-leak")
    monkeypatch.setenv("FAKE_BELT_LOG", str(tmp_path / "requests.jsonl"))
    return tmp_path / "requests.jsonl"


async def drive(app, task_id, **inp):
    mod = sys.modules[type(app).__module__]
    app.context = type("C", (), {"metadata": type("M", (), {"task_id": task_id})()})()
    deltas, final = [], None
    async for item in app.run(mod.AppInput(**inp)):
        if isinstance(item, LLMDelta):
            deltas.append(item)
        else:
            final = item
    return deltas, final


async def started(app_dir):
    mod = load(app_dir)
    app = mod.App()
    await app.setup()
    return app


@pytest.mark.parametrize("app_dir", list(APPS))
def test_streams_text_with_model_and_task_id(env, app_dir):
    async def go():
        app = await started(app_dir)
        try:
            deltas, out = await drive(app, "task-1", text="hello")
        finally:
            await app.unload()
        return deltas, out
    deltas, out = asyncio.run(go())
    assert "".join(d.response for d in deltas) == f"[{APPS[app_dir]}] hello"
    assert out.response == f"[{APPS[app_dir]}] hello"
    assert out.output_meta is not None and out.usage.prompt_tokens == 12
    req = json.loads(env.read_text().splitlines()[0])
    assert req["id"] == "task-1" and req["model"] == APPS[app_dir] and req["harness"] == "claude"
    assert req["leaked_env"] == []  # no token or ANTHROPIC_* in serve's env


def test_tool_call(env):
    async def go():
        app = await started("opus-5-5")
        try:
            return await drive(app, "task-tool", text="TOOL",
                               tools=[{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}])
        finally:
            await app.unload()
    deltas, out = asyncio.run(go())
    print("deltas:", [d.model_dump(exclude_none=True) for d in deltas])
    print("output:", out.model_dump(exclude_none=True, mode="json"))
    call = out.model_dump(mode="json")["tool_calls"][0]
    assert call["function"]["name"] == "search"
    assert json.loads(call["function"]["arguments"]) == {"query": "weather in Paris"}
    assert out.usage.stop_reason == "tool_calls"


def test_auth_error_fails_run(env, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "bad")
    async def go():
        app = await started("sonnet-5-5")
        try:
            await drive(app, "task-auth", text="hi")
        finally:
            await app.unload()
    with pytest.raises(RuntimeError, match="reconnect Claude Code") as e:
        asyncio.run(go())
    assert getattr(e.value, "code", None) == "auth"


def test_missing_token_refused(env, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
    async def go():
        app = await started("haiku-5-5")
        try:
            await drive(app, "task-none", text="hi")
        finally:
            await app.unload()
    with pytest.raises(RuntimeError, match="No Claude Code credential"):
        asyncio.run(go())


def test_token_read_per_run_and_overlapping_runs(env, monkeypatch):
    async def go():
        app = await started("fable-5-1")
        try:
            a, b = await asyncio.gather(drive(app, "t-a", text="one"), drive(app, "t-b", text="two"))
            monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "bad")
            with pytest.raises(RuntimeError):
                await drive(app, "t-c", text="three")
            return a, b
        finally:
            await app.unload()
    a, b = asyncio.run(go())
    assert a[1].response.endswith("one") and b[1].response.endswith("two")


def test_openai_function(env):
    async def go():
        from inferencesh.openai import ChatCompletionInput
        app = await started("opus-5-5")
        app.context = None
        try:
            chunks = [c async for c in app.openai(ChatCompletionInput(messages=[{"role": "user", "content": "hey"}]))]
        finally:
            await app.unload()
        return chunks
    chunks = asyncio.run(go())
    assert chunks, "openai function yielded nothing"
    print("openai last:", chunks[-1].model_dump(exclude_none=True) if hasattr(chunks[-1], "model_dump") else chunks[-1])


@pytest.mark.parametrize("app_dir", list(APPS))
def test_usage_function(env, app_dir):
    async def go():
        app = await started(app_dir)
        app.context = type("C", (), {"metadata": type("M", (), {"task_id": "u1"})()})()
        try:
            return await app.usage(h.UsageInput())
        finally:
            await app.unload()
    out = asyncio.run(go())
    assert [(w.id, w.used_percent, w.label) for w in out.windows] == [("five_hour", 12.5, "5-hour")]
    assert out.windows[0].resets_at == "2026-10-25T15:00:00Z" and out.windows[0].duration_minutes == 300
    assert out.account is None and out.unavailable is None
    reqs = [json.loads(l) for l in env.read_text().splitlines()]
    assert reqs[-1]["usage"] is True and "input" not in reqs[-1] and reqs[-1]["leaked_env"] == []


def test_usage_unreadable(env, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "bad")
    async def go():
        app = await started("opus-5-5")
        app.context = None
        try:
            return await app.usage(h.UsageInput())
        finally:
            await app.unload()
    out = asyncio.run(go())
    assert out.windows == [] and "401" in out.unavailable


def test_usage_missing_token(env, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
    async def go():
        app = await started("opus-5-5")
        app.context = None
        try:
            await app.usage(h.UsageInput())
        finally:
            await app.unload()
    # The app's helper is its own module (opus-5-5.claude_code_helper).
    with pytest.raises(RuntimeError) as e:
        asyncio.run(go())
    assert type(e.value).__name__ == "ClaudeCodeError" and e.value.code == "auth"


def test_wire_input_files_as_strings(tmp_path):
    from inferencesh.models.llm import LLMInput
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG")
    d = h.wire_input(LLMInput(text="x", images=[str(img)], context=[{"role": "user", "text": "y", "images": [str(img)]}]))
    assert d["images"] == [str(img)] and d["context"][0]["images"] == [str(img)]


def test_find_belt_shipped(tmp_path, monkeypatch):
    monkeypatch.delenv("BELT_BIN", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    (tmp_path / "bin").mkdir()
    b = tmp_path / "bin" / f"belt-linux-{h._arch()}"
    b.write_text("#!/bin/sh\n")
    assert h.find_belt(tmp_path) == b and os.access(b, os.X_OK)


@pytest.fixture(scope="module")
def fake_anthropic():
    proc = subprocess.Popen([sys.executable, str(HERE / "testdata" / "fake-anthropic.py"), "0"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    line = proc.stdout.readline().strip()
    assert line.startswith("listening "), line
    yield line.split(" ", 1)[1]
    proc.kill()
    proc.wait()


@pytest.fixture
def real(monkeypatch, tmp_path, fake_anthropic):
    belt = os.environ.get("REAL_BELT")
    if not belt:
        pytest.skip("REAL_BELT not set")
    claude = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
    if not claude:
        pytest.skip("claude not found (CLAUDE_BIN or PATH)")
    monkeypatch.setenv("BELT_BIN", os.path.abspath(belt))
    monkeypatch.setenv("CLAUDE_BIN", claude)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-fake-test-token")
    monkeypatch.setenv("BELT_HARNESS_SERVE_TEST_BASE_URL", fake_anthropic)
    monkeypatch.setenv("HOME", str(tmp_path))
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(k, raising=False)


def test_real_serve(real):
    async def go():
        app = await started("opus-5-5")
        try:
            text = await drive(app, "real-text", text="Say hi")
            tool = await drive(app, "real-tool", text="What's the weather in Paris? Use the tool.",
                               tools=[{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                       "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])
        finally:
            await app.unload()
        return text, tool
    (td, to), (kd, ko) = asyncio.run(go())
    print("text deltas:", [d.model_dump(exclude_none=True) for d in td])
    print("text output:", to.model_dump(exclude_none=True, mode="json"))
    print("tool deltas:", [d.model_dump(exclude_none=True) for d in kd])
    print("tool output:", ko.model_dump(exclude_none=True, mode="json"))
    assert to.response
    call = ko.model_dump(mode="json")["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    assert to.usage.stop_reason == "stop" and ko.usage.stop_reason == "tool_calls"


def test_cancel_sends_cancel_line_and_ends_with_partial(env):
    async def go():
        app = await started("opus-5-5")
        try:
            run = asyncio.create_task(drive(app, "task-slow", text="SLOW"))
            await asyncio.sleep(0.3)
            assert await app.on_cancel() is True
            deltas, out = await asyncio.wait_for(run, 5)
            # serve is still usable afterwards
            _, after = await drive(app, "task-after", text="again")
            return deltas, out, after
        finally:
            await app.unload()
    deltas, out, after = asyncio.run(go())
    assert deltas and out.response == "".join(d.response for d in deltas)
    assert after.response.endswith("again")
    lines = [json.loads(l) for l in env.read_text().splitlines()]
    assert {"id": "task-slow", "cancel": True, "token": "[redacted]", "leaked_env": []} in lines


def test_abandoned_run_is_cancelled_in_serve(env):
    async def go():
        app = await started("opus-5-5")
        try:
            mod = sys.modules[type(app).__module__]
            app.context = type("C", (), {"metadata": type("M", (), {"task_id": "task-drop"})()})()
            gen = app.run(mod.AppInput(text="SLOW"))
            await gen.__anext__()
            await gen.aclose()  # the consumer went away
            await asyncio.sleep(0.2)
        finally:
            await app.unload()
    asyncio.run(go())
    lines = [json.loads(l) for l in env.read_text().splitlines()]
    assert any(l.get("id") == "task-drop" and l.get("cancel") for l in lines)


def test_real_serve_usage(real):
    async def go():
        app = await started("opus-5-5")
        app.context = None
        try:
            return await app.usage(h.UsageInput())
        finally:
            await app.unload()
    out = asyncio.run(go())
    print("usage:", out.model_dump(exclude_none=True))
    got = {w.id: w.used_percent for w in out.windows}
    assert got == {"five_hour": 12.5, "seven_day": 30}, got
    assert out.unavailable is None


def test_real_serve_cancel(real):
    async def go():
        app = await started("opus-5-5")
        try:
            run = asyncio.create_task(drive(app, "real-cancel", text="Answer slow please"))
            await asyncio.sleep(3)
            await app.on_cancel()
            deltas, out = await asyncio.wait_for(run, 10)
            after = await drive(app, "real-after", text="Say hi")
            return deltas, out, after
        finally:
            await app.unload()
    deltas, out, (_, after) = asyncio.run(go())
    print("cancel deltas:", len(deltas), "partial output:", out.model_dump(exclude_none=True, mode="json"))
    print("next run:", after.response)
    assert deltas and out.response and "part39" not in out.response
    assert after.response
