from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
import requests
import os
import tempfile
import logging

logging.basicConfig(level=logging.INFO)


class AppInput(BaseAppInput):
    identifier: str = Field(
        description="the archive.org item identifier"
    )
    filename: str = Field(
        description="the specific file to download from the item"
    )


class AppOutput(BaseAppOutput):
    file: File = Field(description="the downloaded file")
    size: int = Field(description="file size in bytes")


class App(BaseApp):
    async def setup(self, metadata):
        logging.info("archive download app initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        url = f"https://archive.org/download/{input_data.identifier}/{input_data.filename}"

        logging.info(f"downloading {input_data.filename} from {input_data.identifier}")

        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()

            # get file extension from filename
            _, ext = os.path.splitext(input_data.filename)
            if not ext:
                ext = ".bin"

            # stream to temp file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            size = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
                    size += len(chunk)
            tmp.close()

            logging.info(f"downloaded {size} bytes to {tmp.name}")

            return AppOutput(
                file=File(path=tmp.name),
                size=size,
            )

        except requests.exceptions.RequestException as e:
            logging.error(f"archive.org download failed: {e}")
            raise ValueError(f"archive.org download failed: {str(e)}")
