import os
import logging
import tempfile
import pypandoc
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, TextMeta
from pydantic import Field
from typing import Optional


class AppInput(BaseAppInput):
    markdown: Optional[str] = Field(None, description="Markdown text to convert")
    file: Optional[File] = Field(None, description="Markdown file to convert (.md)")
    reference_doc: Optional[File] = Field(None, description="Word template (.docx) for styling")


class AppOutput(BaseAppOutput):
    document: File = Field(description="Generated Word document (.docx)")


class App(BaseApp):
    async def setup(self, config):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def run(self, input_data: AppInput) -> AppOutput:
        if input_data.file:
            self.logger.info(f"Converting markdown file: {input_data.file.path}")
            with open(input_data.file.path, "r") as f:
                md_content = f.read()
        elif input_data.markdown:
            md_content = input_data.markdown
            self.logger.info(f"Converting inline markdown ({len(md_content)} chars)")
        else:
            raise ValueError("Provide either 'markdown' text or a 'file'")

        output_path = os.path.join(tempfile.gettempdir(), "output.docx")

        extra_args = []
        if input_data.reference_doc and input_data.reference_doc.path:
            self.logger.info(f"Using reference doc: {input_data.reference_doc.path}")
            extra_args.extend(["--reference-doc", input_data.reference_doc.path])

        pypandoc.convert_text(
            md_content,
            "docx",
            format="markdown",
            outputfile=output_path,
            extra_args=extra_args,
        )

        file_size = os.path.getsize(output_path)
        self.logger.info(f"Generated docx: {file_size} bytes")

        return AppOutput(
            document=File(path=output_path),
            output_meta=OutputMeta(
                outputs=[TextMeta(characters=len(md_content))],
            ),
        )
