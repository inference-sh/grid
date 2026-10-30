"""Claude Code on the user's own Claude subscription, as an LLM app.

The app runs one long-lived `belt harness serve` and writes one JSON request
per run on its stdin:

    {"id": <task id>, "harness": "claude", "model": <model id>,
     "token": <CLAUDE_CODE_OAUTH_TOKEN>, "input": <LLMInput>}

serve answers on stdout with zero or more {"id", "delta"} lines, then exactly
one {"id", "output"} or {"id", "error", "code"} line. A request with
"usage": true instead of "input" is answered with one {"id", "usage"} line:
the token's plan windows, what Claude Code's /usage shows. Requests may overlap;
lines are routed to their run by id. serve finds the `claude` binary through
CLAUDE_BIN, which setup points at a pinned @anthropic-ai/claude-code installed
with the worker image's npm.

serve is the latest belt release: setup copies the belt shipped in the app
dir (bin/belt-linux-<arch>) to the worker's cache and runs `belt update` on
the copy, since `harness` commands never update themselves. When the update
can't be had, the copy runs as it is.

The token is read from the environment at every run and only ever written to
serve's stdin: it is never put in serve's environment, logged, or echoed in an
error.
"""

import asyncio
import fcntl
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from inferencesh import BaseAppInput, BaseAppOutput, OutputMeta, function
from inferencesh.models.llm import LLMUsage

logger = logging.getLogger(__name__)

TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"

# Claude Code release installed when CLAUDE_BIN is unset. The npm package
# carries the native binary for the platform as an optional dependency.
CLAUDE_CODE_VERSION = "2.1.302"

MISSING_TOKEN = (
    "No Claude Code credential: connect Claude Code (credentials → Claude Code) "
    "with a token from `claude setup-token`."
)
AUTH_FALLBACK = "Claude rejected the credential; reconnect Claude Code (credentials → Claude Code)"

# serve's own environment: never the user's Anthropic credentials. serve
# builds claude's environment itself; this keeps them out of serve as well.
_STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", TOKEN_ENV)

# The app dir: the helper is symlinked into it (dereferenced in the package).
APP_DIR = Path(os.path.abspath(__file__)).parent


class ClaudeCodeError(RuntimeError):
    def __init__(self, message: str, code: str = "failed"):
        super().__init__(message)
        self.code = code


def _arch() -> str:
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "amd64"
    if m in ("aarch64", "arm64"):
        return "arm64"
    return m


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "inferencesh" / "claude-code"


# How long setup waits for `belt update` before it runs the copy it has.
BELT_UPDATE_TIMEOUT = 120


def find_belt(app_dir: Path = APP_DIR) -> Path:
    """BELT_BIN, else the latest belt: the one shipped with the app for this
    platform, copied to the cache and updated there."""
    explicit = os.environ.get("BELT_BIN")
    if explicit:
        return Path(explicit)
    system = platform.system().lower()
    shipped = app_dir / "bin" / f"belt-{system}-{_arch()}"
    if not shipped.exists():
        raise ClaudeCodeError(f"belt binary for {system}/{_arch()} not found at {shipped}; run build.sh before deploying")
    return _updated_belt(shipped)


def _updated_belt(shipped: Path) -> Path:
    """The cache's belt, updated to the latest release. The cache starts from
    the shipped belt (the app dir may be read-only, and the package may not
    keep the mode bit); an update that fails leaves the copy as it was."""
    dst = _cache_root() / "bin" / shipped.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst.parent / f".{shipped.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not dst.exists():
            tmp = dst.with_suffix(f".{uuid.uuid4().hex}")
            shutil.copyfile(shipped, tmp)
            tmp.chmod(0o755)
            os.replace(tmp, dst)
        env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
        try:
            r = subprocess.run([str(dst), "update"], env=env, stdin=subprocess.DEVNULL, capture_output=True,
                               text=True, timeout=BELT_UPDATE_TIMEOUT)
            logger.info("belt update: %s", (r.stdout + r.stderr).strip()[-300:])
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning("belt update failed, running the belt the cache has: %s", e)
    return dst


def ensure_claude() -> str:
    """CLAUDE_BIN, else the pinned Claude Code, installed once per worker
    with npm (the worker image ships Node.js 22)."""
    explicit = os.environ.get("CLAUDE_BIN")
    if explicit:
        return explicit
    root = _cache_root() / CLAUDE_CODE_VERSION
    binary = root / "node_modules" / ".bin" / "claude"
    if binary.exists():
        return str(binary)
    npm = shutil.which("npm")
    if not npm:
        raise ClaudeCodeError("npm not found: the worker needs Node.js to install Claude Code")
    root.mkdir(parents=True, exist_ok=True)
    with open(root.parent / f".{CLAUDE_CODE_VERSION}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.exists():
            print(f"installing @anthropic-ai/claude-code@{CLAUDE_CODE_VERSION}", flush=True)
            subprocess.run(
                [npm, "install", "--prefix", str(root), "--no-audit", "--no-fund", "--loglevel=error",
                 f"@anthropic-ai/claude-code@{CLAUDE_CODE_VERSION}"],
                check=True,
            )
    if not binary.exists():
        raise ClaudeCodeError(f"Claude Code install did not produce {binary}")
    return str(binary)


def _file_ref(v: Any) -> Any:
    """A serialized File as the string the wire LLMInput carries: the local
    path when the file was downloaded (serve runs on the same worker), else
    its uri."""
    if isinstance(v, dict):
        return v.get("path") or v.get("uri") or ""
    return v


def wire_input(input_data) -> Dict[str, Any]:
    """The run's LLMInput as JSON in the platform's wire shape (images and
    files as strings, context always a list)."""
    d = input_data.model_dump(mode="json", exclude_none=True)
    for key in ("images", "files"):
        if d.get(key):
            d[key] = [_file_ref(f) for f in d[key]]
    for msg in d.get("context") or []:
        for key in ("images", "files"):
            if msg.get(key):
                msg[key] = [_file_ref(f) for f in msg[key]]
    d.setdefault("context", [])
    return d


def with_output_meta(output: Dict[str, Any]) -> Dict[str, Any]:
    """serve's LLMOutput plus the output_meta the platform bills from, as
    anthropic_helper.build_output sets it."""
    out = {k: v for k, v in output.items() if v is not None}
    usage = out.get("usage")
    if usage:
        u = LLMUsage(**usage) if isinstance(usage, dict) else usage
        out["usage"] = u
        out["output_meta"] = OutputMeta.from_llm_usage(u)
    return out


class Serve:
    """One `belt harness serve` process shared by every run of this app process."""

    def __init__(self, belt: Path, claude_bin: str):
        self.belt = belt
        self.claude_bin = claude_bin
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.queues: Dict[str, asyncio.Queue] = {}
        self._readers: Tuple[asyncio.Task, ...] = ()
        self._start_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    def _env(self) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
        env["CLAUDE_BIN"] = self.claude_bin
        return env

    async def start(self):
        async with self._start_lock:
            if self.proc and self.proc.returncode is None:
                return
            self.proc = await asyncio.create_subprocess_exec(
                str(self.belt), "harness", "serve",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env(),
                limit=64 * 1024 * 1024,
            )
            logger.info("belt harness serve started (pid %s)", self.proc.pid)
            self._readers = (
                asyncio.create_task(self._read_stdout(self.proc)),
                asyncio.create_task(self._read_stderr(self.proc)),
            )

    async def _read_stdout(self, proc):
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("belt harness serve: unreadable line (%d bytes)", len(line))
                continue
            q = self.queues.get(msg.get("id"))
            if q is not None:
                q.put_nowait(msg)
        await proc.wait()
        # Fail every run still waiting on this process.
        for rid, q in list(self.queues.items()):
            q.put_nowait({"id": rid, "error": f"belt harness serve exited ({proc.returncode})", "code": "failed"})

    async def _read_stderr(self, proc):
        # serve redacts token values from its diagnostics.
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            print(f"[belt harness serve] {line.decode(errors='replace').rstrip()}", file=sys.stderr, flush=True)

    async def request(self, rid: str, model: str, token: str, input_json: Optional[Dict[str, Any]], usage: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        """Yields serve's lines for one request, ending with its output, usage
        or error line, or {"cancelled": True} after cancel(rid). A request
        abandoned before its end (the generator closed) is cancelled in serve.
        usage asks for the token's plan usage instead of a model call."""
        await self.start()
        if rid in self.queues:
            rid = f"{rid}-{uuid.uuid4().hex[:8]}"
        q: asyncio.Queue = asyncio.Queue()
        self.queues[rid] = q
        done = False
        try:
            req: Dict[str, Any] = {"id": rid, "harness": "claude", "model": model, "token": token}
            if usage:
                req["usage"] = True
            else:
                req["input"] = input_json
            line = json.dumps(req)
            async with self._write_lock:
                self.proc.stdin.write(line.encode() + b"\n")
                await self.proc.stdin.drain()
            while True:
                msg = await q.get()
                if "output" in msg or "usage" in msg or "error" in msg or msg.get("cancelled"):
                    done = True
                yield msg
                if done:
                    return
        finally:
            if self.queues.pop(rid, None) is not None and not done:
                self._send_cancel(rid)

    def _send_cancel(self, rid: str):
        # A plain write: this also runs from a closing generator, where it
        # cannot await. The line is small; the pipe takes it.
        proc = self.proc
        if proc and proc.returncode is None and not proc.stdin.is_closing():
            try:
                proc.stdin.write(json.dumps({"id": rid, "cancel": True}).encode() + b"\n")
            except (BrokenPipeError, ConnectionResetError, RuntimeError):
                pass

    def cancel_all(self):
        """Cancels every request in flight: serve stops it and sends nothing
        more for it; its run ends with what it streamed so far."""
        for rid, q in list(self.queues.items()):
            self.queues.pop(rid, None)
            self._send_cancel(rid)
            q.put_nowait({"id": rid, "cancelled": True})

    async def close(self):
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.stdin.close()
                await asyncio.wait_for(self.proc.wait(), timeout=10)
            except (asyncio.TimeoutError, Exception):
                self.proc.kill()
        for t in self._readers:
            t.cancel()
        self.proc = None


async def stream_run(serve: Serve, rid: str, model: str, input_data) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
    """One run: yields ("delta", LLMDelta dict) ... then ("output", LLMOutput dict).
    Raises ClaudeCodeError on a missing token or serve's error line."""
    token = (os.environ.get(TOKEN_ENV) or "").strip()
    if not token:
        raise ClaudeCodeError(MISSING_TOKEN, "auth")
    partial: Dict[str, str] = {"response": "", "reasoning": ""}
    async for msg in serve.request(rid, model, token, wire_input(input_data)):
        if msg.get("cancelled"):
            # Cancelled: end with the text streamed so far.
            yield "output", {k: v for k, v in partial.items() if v or k == "response"}
            return
        if "delta" in msg:
            d = msg["delta"] or {}
            for k in partial:
                partial[k] += d.get(k) or ""
            yield "delta", d
        elif "output" in msg:
            yield "output", with_output_meta(msg["output"] or {})
        else:
            code = msg.get("code") or "failed"
            text = msg.get("error") or (AUTH_FALLBACK if code == "auth" else "Claude Code failed")
            raise ClaudeCodeError(text.replace(token, "[redacted]"), code)


class UsageInput(BaseAppInput):
    pass


class UsageWindow(BaseModel):
    id: str = Field(description="Stable window id: five_hour, seven_day, or seven_day_<models> for a weekly limit scoped to models.")
    kind: str = Field(description="session (the 5-hour window) or weekly.")
    label: str = Field(description="Display name, such as 5-hour, Weekly or Weekly · Opus.")
    used_percent: float = Field(description="How much of the window is used, 0 to 100.")
    resets_at: Optional[str] = Field(default=None, description="When the window resets (RFC 3339), when Claude says.")
    duration_minutes: Optional[int] = Field(default=None, description="The window's length in minutes.")


class UsageAccount(BaseModel):
    plan: Optional[str] = Field(default=None, description="The Claude plan id, such as max or pro.")
    plan_label: Optional[str] = Field(default=None, description="The plan's display name, such as Max 20x.")
    email: Optional[str] = Field(default=None, description="The Claude account's email.")
    workspace: Optional[str] = Field(default=None, description="The organization the account belongs to.")


class UsageOutput(BaseAppOutput):
    windows: List[UsageWindow] = Field(default_factory=list, description="The plan's rate-limit windows and how much of each is used.")
    checked_at: Optional[str] = Field(default=None, description="When the windows were read (RFC 3339).")
    account: Optional[UsageAccount] = Field(default=None, description="The account, when the credential can read it. A `claude setup-token` token cannot, so this is usually empty.")
    unavailable: Optional[str] = Field(default=None, description="Why the windows could not be read, when they could not.")


def usage_output(snapshot: Dict[str, Any]) -> UsageOutput:
    """UsageOutput from serve's usage snapshot (agentprotocol UsageSnapshot)."""
    limits = snapshot.get("limits") or {}
    acct = snapshot.get("account") or {}
    account = None
    if any(acct.get(k) for k in ("plan", "plan_label", "email", "workspace")):
        account = UsageAccount(**{k: acct.get(k) or None for k in ("plan", "plan_label", "email", "workspace")})
    unavailable = None
    if limits.get("unavailable"):
        u = limits["unavailable"]
        unavailable = u.get("message") or u.get("reason") or "unavailable"
    return UsageOutput(
        windows=[
            UsageWindow(
                id=w.get("id", ""),
                kind=w.get("kind", ""),
                label=w.get("label", ""),
                used_percent=float(w.get("used_percent") or 0),
                resets_at=w.get("resets_at") or None,
                duration_minutes=w.get("duration_minutes") or None,
            )
            for w in limits.get("windows") or []
        ],
        checked_at=limits.get("checked_at") or None,
        account=account,
        unavailable=unavailable,
    )


async def read_usage(serve: Serve, rid: str) -> UsageOutput:
    """The connected token's plan usage. Raises ClaudeCodeError on a missing
    token or serve's error line."""
    token = (os.environ.get(TOKEN_ENV) or "").strip()
    if not token:
        raise ClaudeCodeError(MISSING_TOKEN, "auth")
    async for msg in serve.request(rid, "", token, None, usage=True):
        if "usage" in msg:
            out = usage_output(msg["usage"] or {})
            if out.unavailable:
                out.unavailable = out.unavailable.replace(token, "[redacted]")
            return out
        if "error" in msg:
            code = msg.get("code") or "failed"
            text = msg.get("error") or (AUTH_FALLBACK if code == "auth" else "Claude Code failed")
            raise ClaudeCodeError(text.replace(token, "[redacted]"), code)
    raise ClaudeCodeError("belt harness serve ended the usage request without an answer")


class UsageMixin:
    """The app's usage function: the connected Claude plan's rate-limit
    windows. Needs self.serve, a started Serve."""

    @function(annotations={"title": "Claude plan usage", "readOnlyHint": True, "idempotentHint": True})
    async def usage(self, input_data: UsageInput) -> UsageOutput:
        """How much of your Claude plan is used: the 5-hour and weekly windows
        (and any weekly limit scoped to a model), each with when it resets,
        as Claude Code's /usage shows them. No inference.sh charge; with a
        setup-token it sends Claude Code's own one-token quota check, which
        counts that token against your plan."""
        if not getattr(self, "serve", None):
            raise RuntimeError("belt harness serve not started. Call setup() first.")
        context = getattr(self, "context", None)
        metadata = getattr(context, "metadata", None)
        rid = getattr(metadata, "task_id", None) or "local"
        return await read_usage(self.serve, f"{rid}-usage")
