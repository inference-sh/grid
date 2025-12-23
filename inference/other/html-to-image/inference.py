from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Literal
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
        self.context = await self.browser.new_context(viewport={'width': 1920, 'height': 1080})
        
    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Convert HTML to image using Playwright."""
        if input_data.output_format not in ("jpeg", "png"):
            raise ValueError("output_format must be either 'jpeg' or 'png'")

        with tempfile.NamedTemporaryFile(
            suffix=f'.{input_data.output_format}',
            delete=False
        ) as tmp:
            output_path = tmp.name

        try:
            # Create isolated context for this run
            context = await self.browser.new_context(viewport={
                "width": input_data.width,
                "height": input_data.height
            })
            page = await context.new_page()

            # Ensure body fills viewport
            await page.set_content(input_data.html_content, wait_until="networkidle")
            await page.add_style_tag(content="html,body{margin:0;padding:0;height:100%;min-height:100%;}")

            # Capture full page to avoid background cutoff
            await page.screenshot(
                path=output_path,
                type=input_data.output_format,
                full_page=True,
            )

            await context.close()
            return AppOutput(image=File(path=output_path))

        except Exception as e:
            Path(output_path).unlink(missing_ok=True)
            raise ValueError(f"Failed to convert HTML to image: {str(e)}")
        async def unload(self):
            """Clean up Playwright resources."""
            if hasattr(self, 'context'):
                await self.context.close()
            if hasattr(self, 'browser'):
                await self.browser.close()
            if hasattr(self, 'playwright'):
                await self.playwright.stop()