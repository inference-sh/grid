from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
import requests
import logging

logging.basicConfig(level=logging.INFO)


class AppInput(BaseAppInput):
    identifier: str = Field(
        description="the archive.org item identifier (e.g. 'greatgatsby00fitzgerald')"
    )


class AppOutput(BaseAppOutput):
    metadata: dict = Field(description="the item's metadata")
    files: list = Field(description="list of files in the item")
    files_count: int = Field(description="number of files")
    item_size: int = Field(description="total size in bytes")
    raw: dict = Field(description="full api response")


class App(BaseApp):
    async def setup(self, metadata):
        logging.info("archive metadata app initialized")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        url = f"https://archive.org/metadata/{input_data.identifier}"

        logging.info(f"fetching metadata for: {input_data.identifier}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            item_metadata = data.get("metadata", {})
            files = data.get("files", [])
            files_count = data.get("files_count", len(files))
            item_size = data.get("item_size", 0)

            logging.info(f"got metadata for {input_data.identifier}: {files_count} files, {item_size} bytes")

            return AppOutput(
                metadata=item_metadata,
                files=files,
                files_count=files_count,
                item_size=item_size,
                raw=data,
            )

        except requests.exceptions.RequestException as e:
            logging.error(f"archive.org api request failed: {e}")
            raise ValueError(f"archive.org metadata fetch failed: {str(e)}")
