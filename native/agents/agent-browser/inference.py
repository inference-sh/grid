"""
Agentic Browser

Browser automation for agents to surf the web using Playwright.
Uses sessions to maintain browser state across function calls.

Functions:
- open: Navigate to URL, configure browser (entry point)
- snapshot: Get page state with @e refs for interactive elements
- interact: Click, type, scroll, hover, or send keys using @e refs
- screenshot: Take page screenshot
- execute: Run JavaScript code
- close: Close browser session

Element refs use @e1, @e2, etc. format (like Vercel's agent-browser).
Refs are invalidated after navigation - always re-snapshot after clicks that navigate.
"""

import asyncio
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Literal
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from playwright.async_api import async_playwright, Page, Browser, BrowserContext


# --- Browser launch ---

CHROMIUM = '/usr/bin/chromium'
CHROMIUM_ARGS = ['--disable-dev-shm-usage']


async def launch_chromium(playwright, **opts) -> Browser:
    """Launch Chromium with its sandbox.

    The app declares sandbox.user_namespaces (inf.yml), which lets Chromium
    build its namespace sandbox in the container. On an engine or host that
    does not allow user namespaces the sandboxed launch fails; then it runs
    with --no-sandbox, loudly, so the app keeps working.
    """
    try:
        return await playwright.chromium.launch(
            args=CHROMIUM_ARGS, headless=True, executable_path=CHROMIUM, chromium_sandbox=True, **opts
        )
    except Exception as e:
        print(
            "[agent-browser] WARNING: Chromium could not start with its sandbox "
            f"({str(e).splitlines()[0] if str(e) else type(e).__name__}); "
            "running it with --no-sandbox. The engine or host does not allow user namespaces.",
            file=sys.stderr, flush=True,
        )
        return await playwright.chromium.launch(
            args=CHROMIUM_ARGS + ['--no-sandbox'], headless=True, executable_path=CHROMIUM, **opts
        )


# --- URLs ---

# Schemes the agent may open. file: is refused: it reads the container's own
# files (the app's code, its environment, mounted caches). data: is allowed:
# its content is the caller's own, it reads nothing local, and its page is an
# opaque origin that Chromium does not let load file: either; the caller can
# already run any script in a page (execute).
ALLOWED_URL_SCHEMES = ('http', 'https', 'data')

# A scheme: RFC 3986, then ':'. "example.com:8080" is a host and port, not a
# scheme: a digit follows the colon.
_SCHEME = re.compile(r'^([a-zA-Z][a-zA-Z0-9+.-]*):(?!\d)')


def normalize_url(url: str) -> str:
    """The URL to open: https:// prepended when it has no scheme. A scheme
    other than http, https or data (file:, javascript:, chrome:,
    view-source:, ...) is refused."""
    url = (url or '').strip()
    if not url:
        raise ValueError("url is empty")
    m = _SCHEME.match(url)
    if m is None:
        return f"https://{url}"
    scheme = m.group(1).lower()
    if scheme not in ALLOWED_URL_SCHEMES:
        raise ValueError(f"{scheme}: URLs are not allowed; use http, https or data")
    return url


# --- Uploads ---

# The most an upload reads into memory, per file.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def _is_remote(uri: str) -> bool:
    scheme = (uri or '').split(':', 1)[0].lower().strip()
    return ':' in (uri or '') and scheme in ('http', 'https', 'data')


def local_upload_path(path: str, upload_dir: str) -> str:
    """A local file the page may be given: a regular file whose real path
    (symlinks resolved) is inside upload_dir. Anything else in the container
    (its environment, the app's code, other callers' files) is refused."""
    if not path or not upload_dir:
        raise ValueError("upload: no such file")
    root = os.path.realpath(upload_dir)
    real = os.path.realpath(path)
    if os.path.commonpath([root, real]) != root or real == root or not os.path.isfile(real):
        raise ValueError(f"upload: {path} is not an uploadable file; pass the file as a URL or data URI")
    return real


def _payload(name: str, data: bytes) -> dict:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"upload: {name} is larger than {MAX_UPLOAD_BYTES} bytes")
    mime = mimetypes.guess_type(name)[0] or 'application/octet-stream'
    return {'name': name, 'mimeType': mime, 'buffer': data}


def _fetch_uri(uri: str) -> str:
    """Download a URL or data URI into a fresh local file (the SDK's own
    download); never a path the caller named."""
    return File(uri=uri).path


def upload_payloads(file_paths, files, upload_dir: str, fetch=_fetch_uri) -> list[dict]:
    """What an upload gives the page's file input, as in-memory payloads:
    each URL or data URI downloaded afresh, and local paths only from
    upload_dir. A task input File counts by its URI alone: its local path
    may be one the caller named."""
    uris, locals_ = [], []
    for p in file_paths or []:
        (uris if _is_remote(p) else locals_).append(p)
    for f in files or []:
        uri = getattr(f, 'uri', None) or ''
        if not _is_remote(uri):
            raise ValueError("upload: a file input must be a URL or data URI")
        uris.append(uri)
    out = []
    for p in locals_:
        real = local_upload_path(p, upload_dir)
        with open(real, 'rb') as fh:
            out.append(_payload(os.path.basename(real), fh.read(MAX_UPLOAD_BYTES + 1)))
    for uri in uris:
        local = fetch(uri)
        name = os.path.basename(uri.split('?', 1)[0]) if not uri.lower().startswith('data:') else os.path.basename(local)
        with open(local, 'rb') as fh:
            out.append(_payload(name or 'upload', fh.read(MAX_UPLOAD_BYTES + 1)))
    if not out:
        raise ValueError("'file_paths' or 'files' required for upload action")
    return out


def is_local_page(url: str) -> bool:
    """A page the browser should never show: a local file."""
    return (url or '').strip().lower().startswith('file:')


# --- Output Types ---

class ElementInfo(BaseAppOutput):
    """Interactive element with @e ref."""
    ref: str = Field(description="Element reference (e.g., @e1, @e2)")
    desc: str = Field(description="Human-readable description like '@e1 [button] \"Submit\"'")
    tag: str = Field(description="HTML tag name")
    text: str = Field(description="Element text content (truncated)")
    role: Optional[str] = Field(default=None, description="ARIA role")
    name: Optional[str] = Field(default=None, description="Accessible name")
    href: Optional[str] = Field(default=None, description="Link URL if applicable")
    input_type: Optional[str] = Field(default=None, description="Input type if applicable")


class PageSnapshot(BaseAppOutput):
    """Page snapshot with elements."""
    url: str = Field(description="Current page URL")
    title: str = Field(description="Page title")
    screenshot: Optional[File] = Field(default=None, description="Page screenshot for vision agents")
    elements: list[ElementInfo] = Field(description="Interactive elements with @e refs")
    elements_text: str = Field(default="", description="Text summary of elements for non-vision agents")


# --- Input Schemas ---

class OpenInput(BaseAppInput):
    """Open URL and configure browser."""
    url: str = Field(description="URL to navigate to")
    width: int = Field(default=1280, description="Viewport width in pixels")
    height: int = Field(default=720, description="Viewport height in pixels")
    user_agent: Optional[str] = Field(default=None, description="Custom user agent string")
    # Video recording
    record_video: bool = Field(default=False, description="Record video of browser session (returned on close)")
    show_cursor: bool = Field(default=False, description="Show cursor indicator in screenshots/video")
    # Proxy support
    proxy_url: Optional[str] = Field(default=None, description="Proxy server URL (e.g., http://proxy:8080)")
    proxy_username: Optional[str] = Field(default=None, description="Proxy authentication username")
    proxy_password: Optional[str] = Field(default=None, description="Proxy authentication password")


class SnapshotInput(BaseAppInput):
    """Get current page snapshot."""
    pass


class InteractInput(BaseAppInput):
    """Interact with page elements using @e refs."""
    action: Literal["click", "dblclick", "type", "fill", "scroll", "press", "select", "hover", "drag", "upload", "check", "uncheck", "back", "wait", "goto"] = Field(
        description="Action to perform"
    )
    ref: Optional[str] = Field(default=None, description="Element ref from snapshot (e.g., @e1)")
    text: Optional[str] = Field(default=None, description="Text for type/fill/press/select actions")
    direction: Optional[Literal["up", "down", "left", "right"]] = Field(default=None, description="Scroll direction")
    scroll_amount: Optional[int] = Field(default=None, description="Scroll amount in pixels (default 400)")
    wait_ms: Optional[int] = Field(default=None, description="Milliseconds to wait (for wait action)")
    url: Optional[str] = Field(default=None, description="URL to navigate to (for goto action)")
    # Drag and drop
    target_ref: Optional[str] = Field(default=None, description="Target element ref for drag action")
    # File upload
    file_paths: Optional[list[str]] = Field(default=None, description="File(s) for upload action: http(s) URLs or data URIs. Local paths are refused unless under the app's upload directory")
    files: Optional[list[File]] = Field(default=None, description="File(s) for upload action, as task input files (URLs or data URIs)")


class ScreenshotInput(BaseAppInput):
    """Take page screenshot."""
    full_page: bool = Field(default=False, description="Capture full scrollable page")


class ExecuteInput(BaseAppInput):
    """Execute JavaScript code."""
    code: str = Field(description="JavaScript code to execute")


class CloseInput(BaseAppInput):
    """Close browser session."""
    pass


# --- Output Schemas ---

class OpenOutput(PageSnapshot):
    """Open result with initial snapshot."""
    pass


class SnapshotOutput(PageSnapshot):
    """Current page snapshot."""
    pass


class InteractOutput(BaseAppOutput):
    """Interaction result."""
    success: bool = Field(description="Whether action succeeded")
    action: str = Field(description="Action that was performed")
    message: Optional[str] = Field(default=None, description="Additional info or error")
    screenshot: Optional[File] = Field(default=None, description="Page screenshot after action")
    snapshot: Optional[PageSnapshot] = Field(default=None, description="Page snapshot after action (also includes screenshot)")


class ScreenshotOutput(BaseAppOutput):
    """Screenshot result."""
    screenshot: File = Field(description="Screenshot image file")
    width: int = Field(description="Image width")
    height: int = Field(description="Image height")


class ExecuteOutput(BaseAppOutput):
    """JavaScript execution result."""
    result: Optional[str] = Field(default=None, description="Return value as string")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    screenshot: Optional[File] = Field(default=None, description="Page screenshot")


class CloseOutput(BaseAppOutput):
    """Close result."""
    success: bool = Field(description="Whether browser closed successfully")
    video: Optional[File] = Field(default=None, description="Video recording if record_video was enabled")


class App(BaseApp):
    """
    Browser automation using Playwright.
    State persists across function calls within a session.
    """

    async def setup(self):
        """Initialize Playwright browser."""
        # Install ffmpeg for video recording (fast if already installed)
        import sys
        start = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "ffmpeg"],
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start
        print(f"[agent-browser] playwright install ffmpeg: {elapsed:.2f}s")
        if result.returncode != 0:
            print(f"[agent-browser] ffmpeg install warning: {result.stderr[:200]}")

        # The only local files an upload may hand a page (local_upload_path).
        self.upload_dir = tempfile.mkdtemp(prefix='agent-browser-uploads-')
        self.playwright = await async_playwright().start()
        self.browser: Optional[Browser] = await launch_chromium(self.playwright)
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.elements: dict[str, dict] = {}  # Map @e refs to selectors
        self.width = 1280
        self.height = 720
        # Proxy state (set at browser level)
        self._proxy_url: Optional[str] = None
        self._proxy_username: Optional[str] = None
        self._proxy_password: Optional[str] = None
        # Video recording state
        self.video_dir: Optional[str] = None
        self.recording_video: bool = False
        self.show_cursor: bool = False
        print("[agent-browser] Setup complete")

    def _parse_ref(self, ref: str) -> str:
        """Parse @e ref and return selector."""
        if not ref or not ref.startswith('@e'):
            raise ValueError(f"Invalid ref format: {ref}. Expected @e1, @e2, etc.")

        if ref not in self.elements:
            raise ValueError(f"Unknown ref: {ref}. Run 'snapshot' to get current elements.")

        return self.elements[ref]['selector']

    async def _ensure_context(
        self,
        width: int,
        height: int,
        user_agent: Optional[str] = None,
        record_video: bool = False,
        show_cursor: bool = False,
        proxy_url: Optional[str] = None,
        proxy_username: Optional[str] = None,
        proxy_password: Optional[str] = None
    ):
        """Ensure browser context exists with given settings."""
        # Relaunch browser if proxy settings changed
        proxy_changed = (
            proxy_url != self._proxy_url or
            proxy_username != self._proxy_username or
            proxy_password != self._proxy_password
        )
        if proxy_changed and self.browser:
            if self.context:
                await self.context.close()
                self.context = None
                self.page = None
            await self.browser.close()

            launch_opts = {}
            if proxy_url:
                proxy_opts = {'server': proxy_url}
                if proxy_username:
                    proxy_opts['username'] = proxy_username
                if proxy_password:
                    proxy_opts['password'] = proxy_password
                launch_opts['proxy'] = proxy_opts

            self.browser = await launch_chromium(self.playwright, **launch_opts)
            self._proxy_url = proxy_url
            self._proxy_username = proxy_username
            self._proxy_password = proxy_password

        if (self.context is None or self.width != width or self.height != height):
            if self.context:
                await self.context.close()

            self.width = width
            self.height = height
            self.show_cursor = show_cursor

            context_opts = {'viewport': {'width': width, 'height': height}}
            if user_agent:
                context_opts['user_agent'] = user_agent

            # Video recording
            if record_video:
                self.video_dir = tempfile.mkdtemp(prefix='agent-browser-video-')
                context_opts['record_video_dir'] = self.video_dir
                context_opts['record_video_size'] = {'width': width, 'height': height}
                self.recording_video = True
            else:
                self.video_dir = None
                self.recording_video = False

            self.context = await self.browser.new_context(**context_opts)
            self.page = await self.context.new_page()
            self.elements = {}

    async def _inject_cursor(self):
        """Inject visible cursor indicator that follows mouse movements."""
        if not self.page or not self.show_cursor:
            return

        cursor_script = """
        (() => {
            if (document.getElementById('__agentic_cursor__')) return;

            const cursor = document.createElement('div');
            cursor.id = '__agentic_cursor__';
            cursor.style.cssText = `
                position: fixed;
                width: 20px;
                height: 20px;
                background: radial-gradient(circle, rgba(255,0,0,0.8) 0%, rgba(255,0,0,0.4) 40%, transparent 70%);
                border-radius: 50%;
                pointer-events: none;
                z-index: 999999;
                transform: translate(-50%, -50%);
                transition: transform 0.05s ease-out;
            `;
            document.body.appendChild(cursor);

            document.addEventListener('mousemove', (e) => {
                cursor.style.left = e.clientX + 'px';
                cursor.style.top = e.clientY + 'px';
            });

            document.addEventListener('mousedown', () => {
                cursor.style.transform = 'translate(-50%, -50%) scale(0.8)';
                cursor.style.background = 'radial-gradient(circle, rgba(255,100,100,1) 0%, rgba(255,0,0,0.6) 40%, transparent 70%)';
            });

            document.addEventListener('mouseup', () => {
                cursor.style.transform = 'translate(-50%, -50%) scale(1)';
                cursor.style.background = 'radial-gradient(circle, rgba(255,0,0,0.8) 0%, rgba(255,0,0,0.4) 40%, transparent 70%)';
            });
        })();
        """
        try:
            await self.page.evaluate(cursor_script)
        except Exception as e:
            print(f"[agent-browser] Cursor injection warning: {e}")

    async def _get_elements(self) -> list[ElementInfo]:
        """Extract interactive elements and assign @e refs."""
        if not self.page:
            return []

        js_code = """
        () => {
            const elements = [];
            const selectors = [
                'a[href]', 'button', 'input', 'textarea', 'select',
                '[role="button"]', '[role="link"]', '[role="checkbox"]',
                '[role="radio"]', '[role="tab"]', '[role="menuitem"]',
                '[onclick]', '[tabindex]:not([tabindex="-1"])'
            ];

            const seen = new Set();

            for (const selector of selectors) {
                for (const el of document.querySelectorAll(selector)) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) continue;
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') continue;

                    const key = el.outerHTML.slice(0, 200);
                    if (seen.has(key)) continue;
                    seen.add(key);

                    let text = (el.innerText || el.value || el.placeholder || el.alt || '').trim();
                    text = text.replace(/\\s+/g, ' ').slice(0, 80);

                    elements.push({
                        tag: el.tagName.toLowerCase(),
                        text: text,
                        role: el.getAttribute('role'),
                        name: el.getAttribute('aria-label') || el.getAttribute('name') || el.getAttribute('title'),
                        href: el.href || null,
                        inputType: el.type || null,
                        selector: generateSelector(el)
                    });
                }
            }

            function generateSelector(el) {
                if (el.id) return '#' + CSS.escape(el.id);

                let path = [];
                while (el && el.nodeType === Node.ELEMENT_NODE) {
                    let selector = el.tagName.toLowerCase();
                    if (el.id) {
                        selector = '#' + CSS.escape(el.id);
                        path.unshift(selector);
                        break;
                    }
                    let sibling = el, nth = 1;
                    while (sibling = sibling.previousElementSibling) {
                        if (sibling.tagName === el.tagName) nth++;
                    }
                    if (nth > 1) selector += ':nth-of-type(' + nth + ')';
                    path.unshift(selector);
                    el = el.parentElement;
                }
                return path.join(' > ');
            }

            return elements.slice(0, 50);
        }
        """

        try:
            raw = await self.page.evaluate(js_code)
            self.elements = {}
            result = []

            for i, el in enumerate(raw):
                ref = f"@e{i + 1}"
                self.elements[ref] = {'selector': el['selector'], **el}

                # Build human-readable description
                desc_parts = [ref, f"[{el['tag']}"]
                if el.get('inputType'):
                    desc_parts[-1] += f" type={el['inputType']}"
                desc_parts[-1] += "]"
                if el.get('text'):
                    desc_parts.append(f'"{el["text"]}"')
                elif el.get('name'):
                    desc_parts.append(f'name="{el["name"]}"')
                if el.get('href'):
                    # Truncate long hrefs
                    href = el['href'][:50] + "..." if len(el['href']) > 50 else el['href']
                    desc_parts.append(f'href="{href}"')
                desc = " ".join(desc_parts)

                result.append(ElementInfo(
                    ref=ref,
                    desc=desc,
                    tag=el['tag'],
                    text=el['text'],
                    role=el.get('role'),
                    name=el.get('name'),
                    href=el.get('href'),
                    input_type=el.get('inputType')
                ))
            return result
        except Exception as e:
            print(f"[agent-browser] Error getting elements: {e}")
            return []

    async def _screenshot(self, full_page: bool = False) -> Optional[File]:
        """Take screenshot."""
        if not self.page:
            return None

        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        path = tmp.name
        tmp.close()

        try:
            await self.page.screenshot(path=path, full_page=full_page)
            return File(path=path)
        except Exception as e:
            print(f"[agent-browser] Screenshot error: {e}")
            Path(path).unlink(missing_ok=True)
            return None

    async def _snapshot(self) -> PageSnapshot:
        """Get full page snapshot."""
        await self._refuse_local_page()
        elements = await self._get_elements()
        screenshot = await self._screenshot()

        # Build text summary for non-vision agents
        elements_text = "\n".join(el.desc for el in elements) if elements else "(no interactive elements found)"

        return PageSnapshot(
            url=self.page.url if self.page else "",
            title=await self.page.title() if self.page else "",
            elements=elements,
            elements_text=elements_text,
            screenshot=screenshot
        )

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by prepending https:// if no protocol specified.
        Refuses file: and every scheme but http, https and data."""
        return normalize_url(url)

    async def _refuse_local_page(self):
        """Leave a page that shows a local file, however it got there."""
        if self.page and is_local_page(self.page.url):
            await self.page.goto('about:blank')
            raise ValueError("file: URLs are not allowed")

    async def open(self, input_data: OpenInput) -> OpenOutput:
        """Navigate to URL and configure browser."""
        url = self._normalize_url(input_data.url)
        await self._ensure_context(
            width=input_data.width,
            height=input_data.height,
            user_agent=input_data.user_agent,
            record_video=input_data.record_video,
            show_cursor=input_data.show_cursor,
            proxy_url=input_data.proxy_url,
            proxy_username=input_data.proxy_username,
            proxy_password=input_data.proxy_password
        )

        try:
            await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            await asyncio.sleep(0.5)
            await self._inject_cursor()
        except Exception as e:
            raise RuntimeError(f"Failed to open URL: {e}")

        snap = await self._snapshot()
        return OpenOutput(url=snap.url, title=snap.title, elements=snap.elements, elements_text=snap.elements_text, screenshot=snap.screenshot)

    async def snapshot(self, input_data: SnapshotInput) -> SnapshotOutput:
        """Get current page snapshot with @e refs."""
        if not self.page:
            raise RuntimeError("Browser not started. Call 'open' first.")

        snap = await self._snapshot()
        return SnapshotOutput(url=snap.url, title=snap.title, elements=snap.elements, elements_text=snap.elements_text, screenshot=snap.screenshot)

    async def interact(self, input_data: InteractInput) -> InteractOutput:
        """Interact with page using @e refs."""
        if not self.page:
            raise RuntimeError("Browser not started. Call 'open' first.")

        action = input_data.action
        success = True
        message = None

        try:
            if action == "click":
                selector = self._parse_ref(input_data.ref)
                await self.page.click(selector, timeout=5000)
                await asyncio.sleep(0.3)

            elif action == "dblclick":
                selector = self._parse_ref(input_data.ref)
                await self.page.dblclick(selector, timeout=5000)
                await asyncio.sleep(0.3)

            elif action == "type":
                if input_data.text is None:
                    raise ValueError("'text' required for type action")
                await self.page.keyboard.type(input_data.text)

            elif action == "fill":
                selector = self._parse_ref(input_data.ref)
                if input_data.text is None:
                    raise ValueError("'text' required for fill action")
                await self.page.fill(selector, input_data.text, timeout=5000)

            elif action == "scroll":
                direction = input_data.direction or "down"
                amount = input_data.scroll_amount or 400
                if input_data.ref:
                    # Scroll a specific element into view
                    selector = self._parse_ref(input_data.ref)
                    await self.page.evaluate(f"document.querySelector('{selector}')?.scrollIntoView({{behavior: 'smooth', block: 'center'}})")
                else:
                    # Find the actual scroll container (many sites use overflow-y-auto on a child, not body)
                    dx = amount if direction == "right" else (-amount if direction == "left" else 0)
                    dy = amount if direction == "down" else (-amount if direction == "up" else 0)
                    scrolled = await self.page.evaluate(f"""(() => {{
                        function findScrollContainer(el) {{
                            while (el && el !== document.documentElement) {{
                                const style = getComputedStyle(el);
                                const overflowY = style.overflowY;
                                if ((overflowY === 'auto' || overflowY === 'scroll') && el.scrollHeight > el.clientHeight) {{
                                    return el;
                                }}
                                el = el.parentElement;
                            }}
                            return null;
                        }}
                        // Check body children for scroll containers (common in SPA layouts)
                        let container = null;
                        for (const child of document.body.querySelectorAll('*')) {{
                            const style = getComputedStyle(child);
                            if ((style.overflowY === 'auto' || style.overflowY === 'scroll') && child.scrollHeight > child.clientHeight) {{
                                container = child;
                                break;
                            }}
                        }}
                        if (container) {{
                            const before = container.scrollTop;
                            container.scrollBy({{ left: {dx}, top: {dy}, behavior: 'instant' }});
                            return container.scrollTop !== before;
                        }}
                        // Fallback to window scroll
                        const before = window.scrollY;
                        window.scrollBy({{ left: {dx}, top: {dy}, behavior: 'instant' }});
                        return window.scrollY !== before;
                    }})()""")
                    if not scrolled:
                        # Last resort: mouse.wheel (works when JS scroll can't find container)
                        if direction == "down":
                            await self.page.mouse.wheel(0, amount)
                        elif direction == "up":
                            await self.page.mouse.wheel(0, -amount)
                        elif direction == "right":
                            await self.page.mouse.wheel(amount, 0)
                        elif direction == "left":
                            await self.page.mouse.wheel(-amount, 0)
                await asyncio.sleep(0.2)

            elif action == "press":
                if input_data.text is None:
                    raise ValueError("'text' required for press action (e.g., 'Enter')")
                await self.page.keyboard.press(input_data.text)

            elif action == "select":
                selector = self._parse_ref(input_data.ref)
                if input_data.text is None:
                    raise ValueError("'text' required for select action")
                await self.page.select_option(selector, label=input_data.text)

            elif action == "hover":
                selector = self._parse_ref(input_data.ref)
                await self.page.hover(selector, timeout=5000)

            elif action == "drag":
                if input_data.target_ref is None:
                    raise ValueError("'target_ref' required for drag action")
                source_selector = self._parse_ref(input_data.ref)
                target_selector = self._parse_ref(input_data.target_ref)
                await self.page.drag_and_drop(source_selector, target_selector)
                await asyncio.sleep(0.3)

            elif action == "upload":
                selector = self._parse_ref(input_data.ref)
                payloads = await asyncio.to_thread(upload_payloads, input_data.file_paths, input_data.files, self.upload_dir)
                await self.page.set_input_files(selector, payloads if len(payloads) > 1 else payloads[0])
                await asyncio.sleep(0.3)

            elif action == "check":
                selector = self._parse_ref(input_data.ref)
                await self.page.check(selector, timeout=5000)

            elif action == "uncheck":
                selector = self._parse_ref(input_data.ref)
                await self.page.uncheck(selector, timeout=5000)

            elif action == "back":
                await self.page.go_back(wait_until='domcontentloaded')
                await asyncio.sleep(0.3)
                await self._inject_cursor()

            elif action == "wait":
                wait_ms = input_data.wait_ms or 1000
                await asyncio.sleep(wait_ms / 1000)

            elif action == "goto":
                if input_data.url is None:
                    raise ValueError("'url' required for goto action")
                url = self._normalize_url(input_data.url)
                await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
                await asyncio.sleep(0.5)
                await self._inject_cursor()

            else:
                raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            success = False
            message = str(e)

        snap = await self._snapshot()
        # Extract screenshot for top-level field, clear from snapshot to avoid duplicate File references
        # TODO: Remove this workaround once engine deduplicates file processing
        screenshot = snap.screenshot
        snap.screenshot = None
        return InteractOutput(success=success, action=action, message=message, screenshot=screenshot, snapshot=snap)

    async def screenshot(self, input_data: ScreenshotInput) -> ScreenshotOutput:
        """Take a screenshot."""
        if not self.page:
            raise RuntimeError("Browser not started. Call 'open' first.")

        await self._refuse_local_page()
        shot = await self._screenshot(full_page=input_data.full_page)
        if shot is None:
            raise RuntimeError("Failed to take screenshot")

        return ScreenshotOutput(screenshot=shot, width=self.width, height=self.height)

    async def execute(self, input_data: ExecuteInput) -> ExecuteOutput:
        """Execute JavaScript on the page."""
        if not self.page:
            raise RuntimeError("Browser not started. Call 'open' first.")

        await self._refuse_local_page()
        result, error = None, None
        try:
            raw = await self.page.evaluate(input_data.code)
            result = str(raw) if raw is not None else None
        except Exception as e:
            error = str(e)

        shot = await self._screenshot()
        return ExecuteOutput(result=result, error=error, screenshot=shot)

    async def close(self, input_data: CloseInput) -> CloseOutput:
        """Close browser session."""
        success = True
        video_file = None

        try:
            # Get video path before closing (if recording)
            video_path = None
            if self.recording_video and self.page:
                try:
                    video = self.page.video
                    if video:
                        video_path = await video.path()
                except Exception as e:
                    print(f"[agent-browser] Error getting video path: {e}")

            if self.context:
                await self.context.close()
                self.context = None
                self.page = None

            # Return video file if we were recording
            if video_path and Path(video_path).exists():
                video_file = File(path=video_path)

            self.elements = {}
            self.recording_video = False
            self.video_dir = None

        except Exception as e:
            print(f"[agent-browser] Close error: {e}")
            success = False

        return CloseOutput(success=success, video=video_file)

    async def run(self, input_data: OpenInput) -> OpenOutput:
        """Default function - same as open."""
        return await self.open(input_data)

    async def unload(self):
        """Clean up Playwright."""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            print(f"[agent-browser] Unload error: {e}")
        print("[agent-browser] Unload complete")
