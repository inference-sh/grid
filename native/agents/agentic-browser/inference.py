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
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Literal
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from playwright.async_api import async_playwright, Page, Browser, BrowserContext


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
    file_paths: Optional[list[str]] = Field(default=None, description="File path(s) for upload action")


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
        print(f"[agentic-browser] playwright install ffmpeg: {elapsed:.2f}s")
        if result.returncode != 0:
            print(f"[agentic-browser] ffmpeg install warning: {result.stderr[:200]}")

        self.playwright = await async_playwright().start()
        self.browser: Optional[Browser] = await self.playwright.chromium.launch(
            args=['--no-sandbox', '--disable-dev-shm-usage'],
            headless=True,
            executable_path='/usr/bin/chromium'
        )
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.elements: dict[str, dict] = {}  # Map @e refs to selectors
        self.width = 1280
        self.height = 720
        # Video recording state
        self.video_dir: Optional[str] = None
        self.recording_video: bool = False
        self.show_cursor: bool = False
        print("[agentic-browser] Setup complete")

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
        if (self.context is None or self.width != width or self.height != height):
            if self.context:
                await self.context.close()

            self.width = width
            self.height = height
            self.show_cursor = show_cursor

            context_opts = {'viewport': {'width': width, 'height': height}}
            if user_agent:
                context_opts['user_agent'] = user_agent

            # Proxy configuration
            if proxy_url:
                proxy_opts = {'server': proxy_url}
                if proxy_username:
                    proxy_opts['username'] = proxy_username
                if proxy_password:
                    proxy_opts['password'] = proxy_password
                context_opts['proxy'] = proxy_opts

            # Video recording
            if record_video:
                self.video_dir = tempfile.mkdtemp(prefix='agentic-browser-video-')
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
            print(f"[agentic-browser] Cursor injection warning: {e}")

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
            print(f"[agentic-browser] Error getting elements: {e}")
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
            print(f"[agentic-browser] Screenshot error: {e}")
            Path(path).unlink(missing_ok=True)
            return None

    async def _snapshot(self) -> PageSnapshot:
        """Get full page snapshot."""
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

    async def open(self, input_data: OpenInput) -> OpenOutput:
        """Navigate to URL and configure browser."""
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
            await self.page.goto(input_data.url, wait_until='domcontentloaded', timeout=30000)
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
                if input_data.file_paths is None or len(input_data.file_paths) == 0:
                    raise ValueError("'file_paths' required for upload action")
                files = input_data.file_paths if len(input_data.file_paths) > 1 else input_data.file_paths[0]
                await self.page.set_input_files(selector, files)
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
                await self.page.goto(input_data.url, wait_until='domcontentloaded', timeout=30000)
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

        shot = await self._screenshot(full_page=input_data.full_page)
        if shot is None:
            raise RuntimeError("Failed to take screenshot")

        return ScreenshotOutput(screenshot=shot, width=self.width, height=self.height)

    async def execute(self, input_data: ExecuteInput) -> ExecuteOutput:
        """Execute JavaScript on the page."""
        if not self.page:
            raise RuntimeError("Browser not started. Call 'open' first.")

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
                    print(f"[agentic-browser] Error getting video path: {e}")

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
            print(f"[agentic-browser] Close error: {e}")
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
            print(f"[agentic-browser] Unload error: {e}")
        print("[agentic-browser] Unload complete")
