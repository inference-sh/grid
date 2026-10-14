from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Literal, Optional
import tempfile
from pathlib import Path
from playwright.async_api import async_playwright

class AppInput(BaseAppInput):
    html_content: str = Field(description="The HTML string to render")
    width: int = Field(default=800, description="Viewport width in pixels")
    height: int = Field(default=600, description="Viewport height in pixels")
    output_format: Literal["jpeg", "png"] = Field(
        default="png", 
        description="Output image format (jpeg or png)"
    )
    full_page: bool = Field(
        default=True,
        description="Capture the whole scrollable page; false captures only the width x height viewport"
    )
    quality: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="JPEG quality (1-100); ignored for png"
    )
    wait_until: Literal["load", "networkidle"] = Field(
        default="load",
        description="When to capture: after the load event and web fonts, or after the network has also been idle for 500 ms (for pages that fetch content after load)"
    )

class AppOutput(BaseAppOutput):
    image: File = Field(description="The rendered image file")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize Playwright resources."""

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            args=['--no-sandbox'],
            executable_path='/usr/bin/chromium'
        )

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Convert HTML to image using Playwright."""
        with tempfile.NamedTemporaryFile(
            suffix=f'.{input_data.output_format}',
            delete=False
        ) as tmp:
            output_path = tmp.name

        # Isolated context per run
        context = await self.browser.new_context(viewport={
            "width": input_data.width,
            "height": input_data.height
        })
        try:
            page = await context.new_page()

            await page.set_content(input_data.html_content, wait_until=input_data.wait_until)
            await page.evaluate("document.fonts.ready.then(() => undefined)")
            # Ensure body fills viewport
            await page.add_style_tag(content="html,body{margin:0;padding:0;height:100%;min-height:100%;}")

            shot = {"path": output_path, "type": input_data.output_format, "full_page": input_data.full_page}
            if input_data.output_format == "jpeg" and input_data.quality is not None:
                shot["quality"] = input_data.quality
            await page.screenshot(**shot)

            return AppOutput(image=File(path=output_path))

        except Exception as e:
            Path(output_path).unlink(missing_ok=True)
            raise ValueError(f"Failed to convert HTML to image: {str(e)}")
        finally:
            await context.close()

    async def unload(self):
        """Clean up Playwright resources."""
        if hasattr(self, 'browser'):
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
