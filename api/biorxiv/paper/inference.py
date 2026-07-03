import logging
import tempfile
import requests
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional


class AppInput(BaseAppInput):
    doi: str = Field(description="paper DOI like 10.1101/2023.01.01.123456")
    server: Optional[str] = Field(default="biorxiv", description="server: biorxiv or medrxiv")
    fetch_pdf: bool = Field(default=False, description="download the pdf file")


class AppOutput(BaseAppOutput):
    paper: dict = Field(description="paper metadata")
    raw: dict = Field(description="raw api response")
    pdf: Optional[File] = Field(default=None, description="downloaded pdf file (when fetch_pdf is true)")


class App(BaseApp):

    async def setup(self):
        self.base_url = "https://api.biorxiv.org/details"
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("biorxiv paper app initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        server = input_data.server or "biorxiv"
        url = f"{self.base_url}/{server}/{input_data.doi}/na/json"

        self.logger.info(f"fetching {server} paper: {input_data.doi}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        collection = data.get("collection", [])

        if not collection:
            raise ValueError(f"paper not found: {input_data.doi}")

        paper = collection[0]
        self.logger.info(f"fetched paper: {paper.get('title', '')[:80]}")

        pdf_file = None
        if input_data.fetch_pdf:
            jatsxml = paper.get("jatsxml", "")
            # Build PDF URL from the DOI
            doi = paper.get("doi", input_data.doi)
            pdf_url = f"https://www.{server}.org/content/{doi}v{paper.get('version', '1')}.full.pdf"

            self.logger.info(f"downloading pdf from {pdf_url}")
            pdf_resp = requests.get(pdf_url, timeout=120, stream=True)
            pdf_resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                for chunk in pdf_resp.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                pdf_file = File(path=tmp.name)
            self.logger.info("pdf downloaded")

        return AppOutput(paper=paper, raw=data, pdf=pdf_file)
