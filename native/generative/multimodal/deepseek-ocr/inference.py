import os
import shutil
import tempfile
import torch
import logging
from typing import Literal
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppInput(BaseAppInput):
    image: File = Field(description="Input image file to perform OCR on")
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

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run OCR inference on the input image."""
        logger.info(f"Processing image with mode={input_data.mode}")
        try:
            mode_params = {
                "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
                "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
                "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
                "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
                "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True}
            }

            params = mode_params[input_data.mode]
            temp_dir = tempfile.mkdtemp()

            with torch.inference_mode():
                result = self.model.infer(
                    self.tokenizer,
                    prompt=input_data.prompt,
                    image_file=input_data.image.path,
                    output_path=temp_dir,
                    base_size=params["base_size"],
                    image_size=params["image_size"],
                    crop_mode=params["crop_mode"],
                    save_results=True,
                    test_compress=False
                )

            result_file = "result.mmd"
            with open(os.path.join(temp_dir, result_file), "r") as f:
                result = f.read()
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"OCR complete, extracted {len(result)} chars")
            return AppOutput(text=result)

        except Exception as e:
            error_msg = f"Inference failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
