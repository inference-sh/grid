import os
import shutil
import tempfile
import torch
import logging

# DeepSeek-OCR model code uses total_mem but PyTorch only exposes total_memory
if not hasattr(torch._C._CudaDeviceProperties, 'total_mem'):
    torch._C._CudaDeviceProperties.total_mem = property(lambda self: self.total_memory)
from typing import Literal, List
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator
import fitz  # pymupdf for PDF-to-image conversion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppInput(BaseAppInput):
    image: File = Field(description="Input image or PDF file to perform OCR on")
    prompt: str = Field(
        default="<image>\nFree OCR. ",
        description="Prompt for the OCR model. Use '<image>\nFree OCR.' for basic OCR or '<image>\n<|grounding|>Convert the document to markdown.' for markdown conversion"
    )
    mode: Literal["tiny", "small", "base", "large", "gundam"] = Field(
        default="gundam",
        description="Model mode affecting image processing parameters. Gundam is optimized for general use."
    )

class AppOutput(BaseAppOutput):
    text: str = Field(description="The extracted text from the image")

def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 200) -> List[str]:
    """Convert PDF pages to PNG images using pymupdf."""
    doc = fitz.open(pdf_path)
    image_paths = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix)
        img_path = os.path.join(output_dir, f"page_{i:04d}.png")
        pix.save(img_path)
        image_paths.append(img_path)
    doc.close()
    return image_paths


class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_name = "deepseek-ai/DeepSeek-OCR"

    async def setup(self, metadata):
        """Initialize the OCR model and resources."""
        logger.info("Initializing DeepSeek OCR model...")

        accelerator = Accelerator()
        self.device = accelerator.device
        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            self.model = AutoModel.from_pretrained(
                self.model_name,
                attn_implementation='eager',
                trust_remote_code=True,
                use_safetensors=True
            )

            # Move model to device and set to evaluation mode
            self.model = self.model.eval().to(self.device).to(torch.bfloat16)
            logger.info("Model initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _ocr_single_image(self, image_path: str, prompt: str, params: dict) -> str:
        """Run OCR on a single image file."""
        temp_dir = tempfile.mkdtemp()
        try:
            with torch.inference_mode():
                self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=image_path,
                    output_path=temp_dir,
                    base_size=params["base_size"],
                    image_size=params["image_size"],
                    crop_mode=params["crop_mode"],
                    save_results=True,
                    test_compress=False
                )
            with open(os.path.join(temp_dir, "result.mmd"), "r") as f:
                return f.read()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run OCR inference on the input image or PDF."""
        logger.info(f"Processing file with mode={input_data.mode}")
        try:
            mode_params = {
                "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
                "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
                "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
                "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
                "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True}
            }
            params = mode_params[input_data.mode]
            file_path = input_data.image.path

            is_pdf = file_path.lower().endswith('.pdf')
            if not is_pdf:
                # sniff the first bytes for PDF magic number
                try:
                    with open(file_path, 'rb') as f:
                        is_pdf = f.read(5) == b'%PDF-'
                except Exception:
                    pass

            if is_pdf:
                logger.info("Detected PDF input, converting pages to images")
                pdf_dir = tempfile.mkdtemp()
                try:
                    page_images = pdf_to_images(file_path, pdf_dir)
                    logger.info(f"Converted PDF to {len(page_images)} page images")
                    page_results = []
                    for i, img_path in enumerate(page_images):
                        logger.info(f"Processing page {i + 1}/{len(page_images)}")
                        text = self._ocr_single_image(img_path, input_data.prompt, params)
                        page_results.append(text)
                    result = "\n\n---\n\n".join(page_results)
                finally:
                    shutil.rmtree(pdf_dir, ignore_errors=True)
            else:
                result = self._ocr_single_image(file_path, input_data.prompt, params)

            logger.info(f"OCR complete, extracted {len(result)} chars")
            return AppOutput(text=result)

        except Exception as e:
            error_msg = f"Inference failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
