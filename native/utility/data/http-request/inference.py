"""
HTTP Request - make one HTTP request to a public URL and return the response.

Every connection, including each redirect hop and each input-file download,
is checked at dial time against loopback, private, link-local, metadata and
other non-public ranges (see safehttp.py). A non-2xx response is a result
(ok=false); a refused destination, network error, timeout or oversize body
fails the task.
"""

import asyncio
import http.cookiejar
import json
import logging
import mimetypes
import os
import re
import sys
import tempfile
import time
import urllib.parse
from email.message import Message
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import BaseModel, Field, field_validator, model_validator

from .safehttp import (
    PLAIN_HEADERS,
    BlockedAddressError,
    GuardedAsyncTransport,
    GuardedSyncTransport,
    blocked_cause,
    clean_request_headers,
)

MB = 1024 * 1024
DEFAULT_MAX_BYTES = 10 * MB
HARD_MAX_BYTES = 100 * MB
INLINE_TEXT_BYTES = 1 * MB  # auto mode: text above this goes to body_file
INLINE_JSON_BYTES = 5 * MB  # auto mode: JSON above this goes to body_file
SPILL_BYTES = 1 * MB  # responses above this stream to disk instead of memory
INPUT_FILE_MAX_BYTES = 100 * MB
INPUT_FILE_TIMEOUT = 60.0
USER_AGENT = "inference.sh-http/1.0"
REDIRECT_STATUSES = (301, 302, 303, 307, 308)
HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")

Scalar = Union[str, int, float, bool]


# Task logs are the process's stdout; a logger with no handler logs nothing.
_logger = logging.getLogger("infsh.http")
if not _logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


class RequestError(Exception):
    """A failure the caller should see verbatim."""


def _download_input_file(uri: str) -> dict:
    """Fetch a remote input file through the guarded client.

    The SDK's File would download it with urllib, which has no address check:
    a file URI is as much a request target as `url` is.
    """
    with httpx.Client(
        transport=GuardedSyncTransport(),
        follow_redirects=True,
        max_redirects=5,
        timeout=INPUT_FILE_TIMEOUT,
        trust_env=False,
    ) as client:
        try:
            with client.stream("GET", uri) as resp:
                if resp.status_code >= 400:
                    raise ValueError(f"could not fetch input file: HTTP {resp.status_code}")
                name = os.path.basename(urllib.parse.urlparse(str(resp.url)).path) or "upload"
                fd, path = tempfile.mkstemp(prefix="infsh-http-in-", suffix="-" + _safe_name(name))
                total = 0
                with os.fdopen(fd, "wb") as f:
                    for chunk in resp.iter_bytes():
                        total += len(chunk)
                        if total > INPUT_FILE_MAX_BYTES:
                            raise ValueError(f"input file is larger than {INPUT_FILE_MAX_BYTES} bytes")
                        f.write(chunk)
                ctype = (resp.headers.get("content-type") or "").split(";")[0].strip() or None
                return {"path": path, "filename": name, "content_type": ctype}
        except BlockedAddressError as e:
            raise ValueError(f"input file: {e}") from None
        except httpx.HTTPError as e:
            cause = blocked_cause(e)
            if cause:
                raise ValueError(f"input file: {cause}") from None
            raise ValueError(f"could not fetch input file: {type(e).__name__}: {e}") from None


class UploadPart(BaseModel):
    field: str = Field(default="file", description="Multipart form field name for this file.")
    file: File = Field(description="The file to upload (a platform file or an http(s) URL).")
    filename: Optional[str] = Field(default=None, description="Filename sent in the part. Defaults to the file's own name.")
    content_type: Optional[str] = Field(default=None, description="Content-Type of the part. Defaults to the file's detected type.")

    @field_validator("file", mode="before")
    @classmethod
    def _fetch_guarded(cls, v: Any) -> Any:
        if isinstance(v, File) or v is None:
            return v
        uri = v if isinstance(v, str) else (v.get("uri") if isinstance(v, dict) else None)
        if uri and uri.lower().startswith(("http://", "https://")):
            return _download_input_file(uri)
        if uri and uri.lower().startswith("data:"):
            return v
        if os.environ.get("INFSH_HTTP_ALLOW_LOCAL_FILES") == "1":
            return v
        raise ValueError("file must be a platform file, an http(s) URL or a data URI")


class AppInput(BaseAppInput):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"] = Field(
        default="GET", description="HTTP method."
    )
    url: str = Field(description="Request URL. http or https only; must resolve to a public internet address.")
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Request headers. Connection-level headers (Host, Content-Length, Connection, Transfer-Encoding, ...) are dropped. A value may hold {{secret.HTTP_SECRET_NAME}}, filled from a secret given to the task.",
    )
    query: Optional[Dict[str, Union[Scalar, List[Scalar]]]] = Field(
        default=None, description="Query parameters, added to any already in the URL. A list repeats the key."
    )
    body_json: Optional[Any] = Field(
        default=None, description="JSON request body (any JSON value). Sets Content-Type: application/json."
    )
    body_text: Optional[str] = Field(
        default=None, description="Raw text request body. Set Content-Type in headers (default text/plain; charset=utf-8)."
    )
    form: Optional[Dict[str, Scalar]] = Field(
        default=None,
        description="Form fields. Sent application/x-www-form-urlencoded, or as multipart fields when files are given.",
    )
    files: Optional[List[UploadPart]] = Field(
        default=None, description="Files to upload as multipart/form-data, optionally with form fields."
    )
    timeout_seconds: float = Field(
        default=30, gt=0, le=120, description="Deadline for the whole request: connect, redirects and reading the body."
    )
    follow_redirects: bool = Field(default=True, description="Follow 3xx redirects. Each hop is checked like the first.")
    max_redirects: int = Field(default=10, ge=0, le=10, description="Most redirects to follow.")
    max_response_bytes: int = Field(
        default=DEFAULT_MAX_BYTES, ge=1, le=HARD_MAX_BYTES, description="Largest response body to read (decoded bytes)."
    )
    truncate: bool = Field(
        default=False,
        description="When the body exceeds max_response_bytes: true returns the first max_response_bytes with truncated=true, false fails the task.",
    )
    response_as: Literal["auto", "json", "text", "file"] = Field(
        default="auto",
        description="auto: JSON to body_json, small text to body_text, anything else to body_file. json/text/file force one.",
    )

    @model_validator(mode="after")
    def _one_body(self):
        given = [n for n in ("body_json", "body_text") if getattr(self, n) is not None]
        if self.form is not None or self.files:
            given.append("form/files")
        if len(given) > 1:
            raise ValueError(f"give at most one request body; got {', '.join(given)}")
        return self


class AppOutput(BaseAppOutput):
    status: int = Field(description="HTTP status code of the final response.")
    ok: bool = Field(description="True when status is 2xx.")
    headers: Dict[str, str] = Field(description="Response headers, lower-case names; repeated headers joined with ', '.")
    final_url: str = Field(description="URL of the final response, after redirects.")
    redirects: int = Field(default=0, description="Redirects followed.")
    elapsed_ms: int = Field(description="Time inside the app from sending the request to the end of the body.")
    size_bytes: int = Field(default=0, description="Body bytes read (after content decoding).")
    truncated: bool = Field(default=False, description="True when the body was cut at max_response_bytes.")
    body_json: Optional[Any] = Field(default=None, description="Parsed body, when it is JSON.")
    body_text: Optional[str] = Field(default=None, description="Body as text, when it is text.")
    body_file: Optional[File] = Field(default=None, description="Body as a file, when binary, large, or response_as=file.")


def _safe_name(name: str) -> str:
    name = os.path.basename(name.replace("\\", "/"))
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
    return name[:120] or "response"


def _build_url(url: str, query: Optional[dict]) -> httpx.URL:
    try:
        u = httpx.URL(url)
    except Exception as e:
        raise RequestError(f"invalid url: {e}")
    if u.scheme not in ("http", "https"):
        raise RequestError(f"only http and https URLs are allowed, got {u.scheme or 'none'!r}")
    if not u.host:
        raise RequestError("url has no host")
    if query:
        params = list(u.params.multi_items())
        for k, v in query.items():
            for item in v if isinstance(v, list) else [v]:
                params.append((k, _scalar(item)))
        u = u.copy_with(params=params)
    return u


def _scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


# {{secret.NAME}} in a header value is filled from the task's environment:
# the platform hands an agent's HTTP tool its auth this way (a task secret,
# never in the input). Only HTTP_SECRET_* names are read, so a caller cannot
# send anything else the process environment holds.
SECRET_REF_RE = re.compile(r"\{\{secret\.([A-Za-z0-9_]+)\}\}")
SECRET_PREFIX = "HTTP_SECRET_"


def _fill_secrets(headers: dict) -> dict:
    def value_of(m: "re.Match") -> str:
        name = m.group(1)
        if not name.startswith(SECRET_PREFIX):
            raise RequestError(f"{{{{secret.{name}}}}}: only {SECRET_PREFIX}* secrets can be sent")
        v = os.environ.get(name)
        if v is None:
            raise RequestError(f"{{{{secret.{name}}}}}: no such secret was given to this task")
        return v

    return {k: SECRET_REF_RE.sub(value_of, v) for k, v in headers.items()}


def _validate_headers(headers: Optional[dict]) -> dict:
    out = {}
    for k, v in (headers or {}).items():
        if not HEADER_NAME_RE.match(k):
            raise RequestError(f"invalid header name {k!r}")
        if any(c in v for c in "\r\n\0"):
            raise RequestError(f"header {k} contains a line break or NUL")
        out[k] = v
    return clean_request_headers(out.items())


def _origin(u: httpx.URL):
    return (u.scheme, u.host.lower(), u.port or (443 if u.scheme == "https" else 80))


def _is_texty(ctype: str) -> bool:
    return (
        ctype.startswith("text/")
        or ctype in ("application/xml", "application/javascript", "application/x-www-form-urlencoded",
                     "application/ecmascript", "image/svg+xml", "application/x-ndjson")
        or ctype.endswith("+xml")
    )


def _is_json(ctype: str) -> bool:
    return ctype == "application/json" or ctype.endswith("+json")


class _Body:
    """Response body read with a byte cap, in memory until SPILL_BYTES, then on disk."""

    def __init__(self, workdir: str):
        self.workdir = workdir
        self.buf = bytearray()
        self.path: Optional[str] = None
        self._f = None
        self.size = 0

    def write(self, chunk: bytes):
        self.size += len(chunk)
        if self._f is None and len(self.buf) + len(chunk) <= SPILL_BYTES:
            self.buf += chunk
            return
        if self._f is None:
            fd, self.path = tempfile.mkstemp(dir=self.workdir, prefix="body-")
            self._f = os.fdopen(fd, "wb")
            self._f.write(self.buf)
            self.buf = bytearray()
        self._f.write(chunk)

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def read(self) -> bytes:
        if self.path:
            with open(self.path, "rb") as f:
                return f.read()
        return bytes(self.buf)

    def to_file(self, name: str) -> str:
        dest = os.path.join(self.workdir, name)
        if self.path:
            os.replace(self.path, dest)
            self.path = dest
        else:
            with open(dest, "wb") as f:
                f.write(self.buf)
        return dest


class App(BaseApp):
    async def setup(self, config=None):
        # One pooled client for the worker's life: warm runs reuse TLS
        # connections. The pool only ever holds connections that passed the
        # dial check.
        self.client = httpx.AsyncClient(
            transport=GuardedAsyncTransport(),
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": USER_AGENT},
            # The client outlives the task: a cookie one caller's response set
            # must never ride along on the next caller's request.
            cookies=_no_cookies(),
        )

    def _log(self):
        return getattr(self, "logger", None) or _logger

    def _request_factory(self, inp: AppInput, base_headers: dict):
        """Builds a fresh request per hop, so a 307/308 can re-send the body."""

        def build(method: str, url: httpx.URL, headers: dict, with_body: bool, timeout: httpx.Timeout) -> httpx.Request:
            kwargs: Dict[str, Any] = {}
            if with_body:
                if inp.body_json is not None:
                    kwargs["content"] = json.dumps(inp.body_json).encode()
                    headers = {"Content-Type": "application/json", **headers}
                elif inp.body_text is not None:
                    kwargs["content"] = inp.body_text.encode()
                    headers = {"Content-Type": "text/plain; charset=utf-8", **headers}
                elif inp.files:
                    kwargs["data"] = {k: _scalar(v) for k, v in (inp.form or {}).items()}
                    parts = []
                    for part in inp.files:
                        path = part.file.path
                        if not path or not os.path.exists(path):
                            raise RequestError(f"file for field {part.field!r} is not available")
                        fname = part.filename or part.file.filename or os.path.basename(path)
                        ctype = (part.content_type or part.file.content_type
                                 or mimetypes.guess_type(fname)[0] or "application/octet-stream")
                        parts.append((part.field, (fname, open(path, "rb"), ctype)))
                    kwargs["files"] = parts
                    # httpx must set the multipart boundary itself.
                    headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
                elif inp.form is not None:
                    kwargs["data"] = {k: _scalar(v) for k, v in inp.form.items()}
            else:
                headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
            return self.client.build_request(method, url, headers=headers, timeout=timeout, **kwargs)

        return build

    async def run(self, input_data: AppInput) -> AppOutput:
        log = self._log()
        if not hasattr(self, "client"):
            await self.setup()
        inp = input_data
        url = _build_url(inp.url, inp.query)
        # Filled first, so the header rules (_validate_headers) also hold for
        # the secrets' values.
        headers = _validate_headers(_fill_secrets(inp.headers or {}))
        has_body = inp.body_json is not None or inp.body_text is not None or inp.form is not None or bool(inp.files)
        # Header names only: values (Authorization, cookies, keys) and bodies are never logged.
        log.info(
            f"{inp.method} {url.scheme}://{url.host}{url.path} header_names={sorted(h.lower() for h in headers)} "
            f"body={'yes' if has_body else 'no'} files={len(inp.files or [])} response_as={inp.response_as}"
        )

        t = inp.timeout_seconds
        timeout = httpx.Timeout(t, connect=min(10.0, t))
        build = self._request_factory(inp, headers)
        workdir = tempfile.mkdtemp(prefix="infsh-http-")
        started = time.monotonic()
        try:
            async with asyncio.timeout(t):
                resp, final_url, hops = await self._send_following(inp, build, url, headers, has_body, timeout)
                try:
                    body, truncated = await self._read_body(resp, inp, workdir)
                finally:
                    await resp.aclose()
        except TimeoutError:
            raise RequestError(f"request timed out after {t:g}s") from None
        except httpx.TimeoutException as e:
            raise RequestError(f"request timed out ({type(e).__name__}) after {t:g}s limit") from None
        except BlockedAddressError as e:
            raise RequestError(str(e)) from None
        except httpx.HTTPError as e:
            cause = blocked_cause(e)
            if cause:
                raise RequestError(str(cause)) from None
            raise RequestError(f"request failed: {type(e).__name__}: {e}") from None
        elapsed_ms = int((time.monotonic() - started) * 1000)

        out = AppOutput(
            status=resp.status_code,
            ok=200 <= resp.status_code < 300,
            headers=_flatten_headers(resp.headers),
            final_url=str(final_url),
            redirects=hops,
            elapsed_ms=elapsed_ms,
            size_bytes=body.size,
            truncated=truncated,
        )
        self._shape_body(out, resp, body, inp, final_url)
        log.info(
            f"status={out.status} bytes={body.size} truncated={truncated} redirects={hops} elapsed_ms={elapsed_ms} "
            f"as={'json' if out.body_json is not None else 'text' if out.body_text is not None else 'file' if out.body_file else 'none'}"
        )
        return out

    async def _send_following(self, inp, build, url, headers, has_body, timeout):
        method = inp.method
        first = url
        with_body = has_body
        hops = 0
        cross_origin = False
        while True:
            req = build(method, url, dict(headers), with_body, timeout)
            resp = await self.client.send(req, stream=True)
            location = resp.headers.get("location")
            if not (inp.follow_redirects and resp.status_code in REDIRECT_STATUSES and location):
                return resp, url, hops
            if hops >= inp.max_redirects:
                await resp.aclose()
                raise RequestError(f"stopped after {hops} redirects (max_redirects={inp.max_redirects})")
            await resp.aclose()
            try:
                nxt = url.join(location)
            except Exception:
                raise RequestError(f"redirect to an invalid location {location!r}")
            if nxt.scheme not in ("http", "https"):
                raise RequestError(f"blocked redirect to scheme {nxt.scheme!r}")
            nxt = nxt.copy_with(fragment=None)
            code = resp.status_code
            if code == 303 and method != "HEAD" or code in (301, 302) and method == "POST":
                method, with_body = "GET", False
            if not cross_origin and _origin(nxt) != _origin(first):
                cross_origin = True
                # What was meant for the first host stays there: credentials,
                # API keys, custom headers. A body is never re-sent elsewhere.
                headers = {k: v for k, v in headers.items() if k.lower() in PLAIN_HEADERS}
            if cross_origin and with_body:
                raise RequestError(
                    f"refusing to re-send the request body to another origin ({nxt.scheme}://{nxt.host}) on HTTP {code}"
                )
            if nxt.userinfo:
                nxt = nxt.copy_with(username=None, password=None)
            url = nxt
            hops += 1

    async def _read_body(self, resp: httpx.Response, inp: AppInput, workdir: str):
        body = _Body(workdir)
        cap = inp.max_response_bytes
        truncated = False
        if inp.method == "HEAD" or resp.status_code in (204, 304):
            return body, False
        declared = resp.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > cap and not inp.truncate and not resp.headers.get("content-encoding"):
            raise RequestError(
                f"response is {declared} bytes, over max_response_bytes={cap} (set truncate=true to keep the first {cap})"
            )
        try:
            async for chunk in resp.aiter_bytes():
                room = cap - body.size
                if len(chunk) > room:
                    if not inp.truncate:
                        raise RequestError(
                            f"response is larger than max_response_bytes={cap} (set truncate=true to keep the first {cap})"
                        )
                    body.write(chunk[:room])
                    truncated = True
                    break
                body.write(chunk)
        finally:
            body.close()
        return body, truncated

    def _shape_body(self, out: AppOutput, resp: httpx.Response, body: _Body, inp: AppInput, final_url: httpx.URL):
        ctype_full = resp.headers.get("content-type") or ""
        ctype = ctype_full.split(";")[0].strip().lower()
        mode = inp.response_as
        if mode == "file":
            out.body_file = File(path=body.to_file(_filename(resp, final_url, ctype)))
            return
        if body.size == 0:
            if mode == "text":
                out.body_text = ""
            return
        if mode == "json":
            try:
                if out.truncated:
                    raise ValueError("body was truncated")
                out.body_json = json.loads(body.read())
            except ValueError as e:
                if out.ok:
                    raise RequestError(f"response_as=json but the body is not JSON: {e}")
                out.body_text = _decode(body.read(), ctype_full)
            return
        if mode == "text":
            out.body_text = _decode(body.read(), ctype_full)
            return
        # auto
        if _is_json(ctype) and not out.truncated and body.size <= INLINE_JSON_BYTES:
            try:
                out.body_json = json.loads(body.read())
                return
            except ValueError:
                pass
        if body.size <= INLINE_TEXT_BYTES:
            raw = body.read()
            if _is_texty(ctype) or _is_json(ctype) or (not ctype and _looks_text(raw)):
                out.body_text = _decode(raw, ctype_full)
                return
        out.body_file = File(path=body.to_file(_filename(resp, final_url, ctype)))

    async def unload(self):
        client = getattr(self, "client", None)
        if client is not None:
            await client.aclose()

    async def on_cancel(self):
        return True


def _no_cookies() -> http.cookiejar.CookieJar:
    return http.cookiejar.CookieJar(policy=http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))


def _flatten_headers(h: httpx.Headers) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in h.multi_items():
        k = k.lower()
        out[k] = f"{out[k]}, {v}" if k in out else v
    return out


def _decode(raw: bytes, ctype_full: str) -> str:
    m = Message()
    m["content-type"] = ctype_full or "application/octet-stream"
    charset = m.get_content_charset() or "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _looks_text(raw: bytes) -> bool:
    sample = raw[:4096]
    if b"\0" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError as e:
        return e.start > len(sample) - 4  # a multibyte char cut at the sample edge


def _filename(resp: httpx.Response, url: httpx.URL, ctype: str) -> str:
    name = None
    cd = resp.headers.get("content-disposition")
    if cd:
        m = Message()
        m["content-disposition"] = cd
        name = m.get_filename()
    if not name:
        name = os.path.basename(url.path) or ""
    name = _safe_name(name) if name else "response"
    if "." not in name:
        ext = mimetypes.guess_extension(ctype) if ctype else None
        name += ext or ".bin"
    return name
