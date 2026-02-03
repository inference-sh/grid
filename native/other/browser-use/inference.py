"""
Browser Use App

Browser automation for agents to surf the web using Playwright.
Uses sessions to maintain browser state across function calls.

Functions:
- open: Navigate to URL, configure browser settings (entry point)
- state: Get current page state with clickable elements
- interact: Click, type, scroll, or send keys
- screenshot: Take page screenshot
- execute: Run JavaScript code
- close: Close browser session
"""

import asyncio
import tempfile
import re
from pathlib import Path
from typing import Optional, Literal
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from playwright.async_api import async_playwright, Page, Browser, BrowserContext


# --- Shared Output Types ---

class ElementInfo(BaseAppOutput):
    """Information about an interactive element."""
    index: int = Field(description="Element index for interaction")
    tag: str = Field(description="HTML tag name")
    text: str = Field(description="Element text content")
    role: Optional[str] = Field(default=None, description="ARIA role")
    name: Optional[str] = Field(default=None, description="Accessible name")
    href: Optional[str] = Field(default=None, description="Link URL if applicable")
    type: Optional[str] = Field(default=None, description="Input type if applicable")


class PageState(BaseAppOutput):
    """Current page state with elements."""
    url: str = Field(description="Current page URL")
    title: str = Field(description="Page title")
    elements: list[ElementInfo] = Field(description="Interactive elements with indices")
    screenshot: Optional[File] = Field(default=None, description="Page screenshot")


# --- Input Schemas ---

class OpenInput(BaseAppInput):
    """Open URL and configure browser."""
    url: str = Field(description="URL to navigate to")
    width: int = Field(default=1280, description="Viewport width in pixels")
    height: int = Field(default=720, description="Viewport height in pixels")
    user_agent: Optional[str] = Field(default=None, description="Custom user agent string")


class StateInput(BaseAppInput):
    """Get current page state."""
    pass


class InteractInput(BaseAppInput):
    """Interact with page elements."""
    action: Literal["click", "type", "input", "scroll", "keys", "select", "hover", "back", "wait"] = Field(
        description="Action to perform"
    )
    index: Optional[int] = Field(default=None, description="Element index (from state)")
    text: Optional[str] = Field(default=None, description="Text for type/input/keys/select actions")
    direction: Optional[Literal["up", "down"]] = Field(default=None, description="Scroll direction")
    wait_ms: Optional[int] = Field(default=None, description="Milliseconds to wait (for wait action)")


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

class OpenOutput(PageState):
    """Open result with initial page state."""
    pass


class StateOutput(PageState):
    """Current page state."""
    pass


class InteractOutput(BaseAppOutput):
    """Interaction result."""
    success: bool = Field(description="Whether action succeeded")
    action: str = Field(description="Action that was performed")
    message: Optional[str] = Field(default=None, description="Additional info or error")
    screenshot: Optional[File] = Field(default=None, description="Page screenshot after action")
    state: Optional[PageState] = Field(default=None, description="Page state after action")


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


class App(BaseApp):
    """
    Browser automation app using Playwright directly.
    State persists across function calls within a session.
    """

    async def setup(self):
        """Initialize Playwright browser."""
        self.playwright = await async_playwright().start()
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.elements: list = []  # Store elements for index-based interaction
        self.width = 1280
        self.height = 720
        print("[browser-use] Setup complete")

    async def _ensure_browser(self, width: int, height: int, user_agent: Optional[str] = None):
        """Ensure browser is started with given settings."""
        if self.browser is None:
            self.browser = await self.playwright.chromium.launch(
                args=['--no-sandbox', '--disable-dev-shm-usage'],
                headless=True
            )

        # Create new context if settings changed
        if (self.context is None or
            self.width != width or
            self.height != height):

            if self.context:
                await self.context.close()

            self.width = width
            self.height = height

            context_opts = {
                'viewport': {'width': width, 'height': height}
            }
            if user_agent:
                context_opts['user_agent'] = user_agent

            self.context = await self.browser.new_context(**context_opts)
            self.page = await self.context.new_page()

    async def _get_interactive_elements(self) -> list[ElementInfo]:
        """Extract interactive elements from page with indices."""
        if not self.page:
            return []

        # JavaScript to find interactive elements
        js_code = """
        () => {
            const elements = [];
            const selectors = [
                'a[href]',
                'button',
                'input',
                'textarea',
                'select',
                '[role="button"]',
                '[role="link"]',
                '[role="checkbox"]',
                '[role="radio"]',
                '[role="tab"]',
                '[role="menuitem"]',
                '[onclick]',
                '[tabindex]:not([tabindex="-1"])'
            ];

            const seen = new Set();

            for (const selector of selectors) {
                for (const el of document.querySelectorAll(selector)) {
                    // Skip hidden elements
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) continue;
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') continue;

                    // Skip duplicates
                    const key = el.outerHTML.slice(0, 200);
                    if (seen.has(key)) continue;
                    seen.add(key);

                    // Get text content (truncated)
                    let text = (el.innerText || el.value || el.placeholder || '').trim();
                    text = text.slice(0, 100);

                    elements.push({
                        tag: el.tagName.toLowerCase(),
                        text: text,
                        role: el.getAttribute('role'),
                        name: el.getAttribute('aria-label') || el.getAttribute('name') || el.getAttribute('title'),
                        href: el.href || null,
                        type: el.type || null,
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

                    let sibling = el;
                    let nth = 1;
                    while (sibling = sibling.previousElementSibling) {
                        if (sibling.tagName === el.tagName) nth++;
                    }
                    if (nth > 1) selector += ':nth-of-type(' + nth + ')';

                    path.unshift(selector);
                    el = el.parentElement;
                }
                return path.join(' > ');
            }

            return elements.slice(0, 100);  // Limit to 100 elements
        }
        """

        try:
            raw_elements = await self.page.evaluate(js_code)
            self.elements = raw_elements  # Store for later interaction

            return [
                ElementInfo(
                    index=i,
                    tag=el.get('tag', ''),
                    text=el.get('text', ''),
                    role=el.get('role'),
                    name=el.get('name'),
                    href=el.get('href'),
                    type=el.get('type')
                )
                for i, el in enumerate(raw_elements)
            ]
        except Exception as e:
            print(f"[browser-use] Error getting elements: {e}")
            return []

    async def _take_screenshot(self, full_page: bool = False) -> Optional[File]:
        """Take screenshot and return as File."""
        if not self.page:
            return None

        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        path = tmp.name
        tmp.close()

        try:
            await self.page.screenshot(path=path, full_page=full_page)
            return File(path=path)
        except Exception as e:
            print(f"[browser-use] Screenshot error: {e}")
            Path(path).unlink(missing_ok=True)
            return None

    async def _get_state_with_screenshot(self) -> PageState:
        """Get current page state with elements and screenshot."""
        elements = await self._get_interactive_elements()
        screenshot = await self._take_screenshot()

        return PageState(
            url=self.page.url if self.page else "",
            title=await self.page.title() if self.page else "",
            elements=elements,
            screenshot=screenshot
        )

    async def open(self, input_data: OpenInput) -> OpenOutput:
        """Navigate to URL and configure browser settings."""
        await self._ensure_browser(
            input_data.width,
            input_data.height,
            input_data.user_agent
        )

        try:
            await self.page.goto(input_data.url, wait_until='domcontentloaded', timeout=30000)
            # Wait a bit for dynamic content
            await asyncio.sleep(0.5)
        except Exception as e:
            raise RuntimeError(f"Failed to open URL: {e}")

        state = await self._get_state_with_screenshot()

        return OpenOutput(
            url=state.url,
            title=state.title,
            elements=state.elements,
            screenshot=state.screenshot
        )

    async def state(self, input_data: StateInput) -> StateOutput:
        """Get current page state with elements and screenshot."""
        if not self.page:
            raise RuntimeError("Browser not started. Call 'open' first.")

        state = await self._get_state_with_screenshot()

        return StateOutput(
            url=state.url,
            title=state.title,
            elements=state.elements,
            screenshot=state.screenshot
        )

    async def interact(self, input_data: InteractInput) -> InteractOutput:
        """Interact with the page: click, type, scroll, etc."""
        if not self.page:
            raise RuntimeError("Browser not started. Call 'open' first.")

        action = input_data.action
        success = True
        message = None

        try:
            if action == "click":
                if input_data.index is None:
                    raise ValueError("'index' required for click action")
                if input_data.index >= len(self.elements):
                    raise ValueError(f"Invalid index {input_data.index}. Run 'state' to get current elements.")

                selector = self.elements[input_data.index].get('selector')
                await self.page.click(selector, timeout=5000)
                await asyncio.sleep(0.3)

            elif action == "type":
                if input_data.text is None:
                    raise ValueError("'text' required for type action")
                await self.page.keyboard.type(input_data.text)

            elif action == "input":
                if input_data.index is None or input_data.text is None:
                    raise ValueError("'index' and 'text' required for input action")
                if input_data.index >= len(self.elements):
                    raise ValueError(f"Invalid index {input_data.index}")

                selector = self.elements[input_data.index].get('selector')
                await self.page.click(selector, timeout=5000)
                await self.page.keyboard.type(input_data.text)

            elif action == "scroll":
                direction = input_data.direction or "down"
                delta = 300 if direction == "down" else -300
                await self.page.mouse.wheel(0, delta)
                await asyncio.sleep(0.2)

            elif action == "keys":
                if input_data.text is None:
                    raise ValueError("'text' required for keys action")
                await self.page.keyboard.press(input_data.text)

            elif action == "select":
                if input_data.index is None or input_data.text is None:
                    raise ValueError("'index' and 'text' required for select action")
                if input_data.index >= len(self.elements):
                    raise ValueError(f"Invalid index {input_data.index}")

                selector = self.elements[input_data.index].get('selector')
                await self.page.select_option(selector, label=input_data.text)

            elif action == "hover":
                if input_data.index is None:
                    raise ValueError("'index' required for hover action")
                if input_data.index >= len(self.elements):
                    raise ValueError(f"Invalid index {input_data.index}")

                selector = self.elements[input_data.index].get('selector')
                await self.page.hover(selector, timeout=5000)

            elif action == "back":
                await self.page.go_back(wait_until='domcontentloaded')
                await asyncio.sleep(0.3)

            elif action == "wait":
                wait_ms = input_data.wait_ms or 1000
                await asyncio.sleep(wait_ms / 1000)

            else:
                raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            success = False
            message = str(e)

        # Get updated state and screenshot
        state = await self._get_state_with_screenshot()

        return InteractOutput(
            success=success,
            action=action,
            message=message,
            screenshot=state.screenshot,
            state=state
        )

    async def screenshot(self, input_data: ScreenshotInput) -> ScreenshotOutput:
        """Take a screenshot of the current page."""
        if not self.page:
            raise RuntimeError("Browser not started. Call 'open' first.")

        screenshot = await self._take_screenshot(full_page=input_data.full_page)

        if screenshot is None:
            raise RuntimeError("Failed to take screenshot")

        return ScreenshotOutput(
            screenshot=screenshot,
            width=self.width,
            height=self.height
        )

    async def execute(self, input_data: ExecuteInput) -> ExecuteOutput:
        """Execute JavaScript code on the page."""
        if not self.page:
            raise RuntimeError("Browser not started. Call 'open' first.")

        result = None
        error = None

        try:
            raw_result = await self.page.evaluate(input_data.code)
            result = str(raw_result) if raw_result is not None else None
        except Exception as e:
            error = str(e)

        screenshot = await self._take_screenshot()

        return ExecuteOutput(
            result=result,
            error=error,
            screenshot=screenshot
        )

    async def close(self, input_data: CloseInput) -> CloseOutput:
        """Close the browser session."""
        success = True

        try:
            if self.context:
                await self.context.close()
                self.context = None
                self.page = None
            if self.browser:
                await self.browser.close()
                self.browser = None
        except Exception as e:
            print(f"[browser-use] Close error: {e}")
            success = False

        self.elements = []

        return CloseOutput(success=success)

    async def run(self, input_data: OpenInput) -> OpenOutput:
        """Default function - same as open."""
        return await self.open(input_data)

    async def unload(self):
        """Clean up Playwright resources."""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            print(f"[browser-use] Unload error: {e}")
        print("[browser-use] Unload complete")
