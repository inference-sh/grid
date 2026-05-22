import os
import shutil
import tempfile
import torch
import logging

# DeepSeek-OCR model code uses total_mem but PyTorch only exposes total_memory
if not hasattr(torch._C._CudaDeviceProperties, 'total_mem'):
    torch._C._CudaDeviceProperties.total_mem = property(lambda self: self.total_memory)
import re
from typing import Literal, List
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from transformers import AutoModel, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from accelerate import Accelerator
import fitz  # pymupdf for PDF-to-image conversion


class WindowedNGramProcessor(LogitsProcessor):
    """Windowed n-gram repetition blocker with token whitelist.
    Matches vLLM's NGramPerReqLogitsProcessor from DeepSeek-OCR official config."""
    def __init__(self, ngram_size: int = 30, window_size: int = 90, whitelist_token_ids: set = None):
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for batch_idx in range(scores.shape[0]):
            ids = input_ids[batch_idx].tolist()
            if len(ids) < self.ngram_size:
                continue
            prefix = tuple(ids[-(self.ngram_size - 1):])
            start = max(0, len(ids) - self.window_size)
            end = len(ids) - self.ngram_size + 1
            banned = set()
            for i in range(start, end):
                ngram = tuple(ids[i : i + self.ngram_size])
                if ngram[:-1] == prefix:
                    banned.add(ngram[-1])
            banned -= self.whitelist_token_ids
            if banned:
                scores[batch_idx, list(banned)] = -float("inf")
        return scores

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_PROMPTS = {
    "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
    "ocr": "<image>\n<|grounding|>OCR this image.",
    "text": "<image>\nFree OCR. ",
    "figure": "<image>\nParse the figure.",
    "describe": "<image>\nDescribe this image in detail.",
}

class AppInput(BaseAppInput):
    image: File = Field(description="Input image or PDF file to perform OCR on")
    task: Literal["markdown", "ocr", "text", "figure", "describe"] = Field(
        default="markdown",
        description="markdown: structured document conversion (best for papers, tables, math). ocr: general OCR with layout detection. text: plain text extraction. figure: parse charts and diagrams. describe: image description."
    )
    mode: Literal["tiny", "small", "base", "large", "gundam"] = Field(
        default="gundam",
        description="Model mode affecting image processing parameters. Gundam is optimized for general use."
    )

class AppOutput(BaseAppOutput):
    text: str = Field(description="The extracted text from the image")

def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 144) -> List[str]:
    """Convert PDF pages to PNG images using pymupdf. 144 DPI per official recommendation."""
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


LAYOUT_LABELS = {'text', 'sub_title', 'title', 'header', 'footer', 'image',
                  'figure_title', 'figure_caption', 'image_caption',
                  'table', 'table_caption', 'equation', 'page_number'}

def postprocess_ocr(text: str) -> str:
    """Clean up model output: strip grounding tags, layout labels, fix LaTeX symbols."""
    text = re.sub(r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[.*?\]\]<\|/det\|>', r'\1', text)
    text = re.sub(r'<\|ref\|>(.*?)<\|/ref\|>', r'\1', text)
    text = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text)
    # strip standalone layout type labels (v2 format)
    lines = text.split('\n')
    lines = [l for l in lines if l.strip() not in LAYOUT_LABELS]
    text = '\n'.join(lines)
    text = text.replace(r'\coloneqq', ':=')
    text = text.replace(r'\eqqcolon', '=:')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_name = "deepseek-ai/DeepSeek-OCR-2"

    async def setup(self, metadata):
        """Initialize the OCR model and resources."""
        logger.info("Initializing DeepSeek OCR-2 model...")

        accelerator = Accelerator()
        self.device = accelerator.device
        logger.info(f"Using device: {self.device}")

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

            self.model = self.model.eval().to(self.device).to(torch.bfloat16)

            # Patch generate() to inject windowed n-gram processor
            _original_generate = self.model.generate
            td_ids = set()
            for tag in ["<td>", "</td>"]:
                td_ids.update(self.tokenizer.encode(tag, add_special_tokens=False))
            logger.info(f"Whitelist token IDs for table tags: {td_ids}")
            ngram_proc = WindowedNGramProcessor(ngram_size=30, window_size=90, whitelist_token_ids=td_ids)

            def _patched_generate(*args, **kwargs):
                existing = kwargs.get("logits_processor", LogitsProcessorList())
                if not isinstance(existing, LogitsProcessorList):
                    existing = LogitsProcessorList(existing)
                existing.append(ngram_proc)
                kwargs["logits_processor"] = existing
                kwargs.pop("no_repeat_ngram_size", None)
                return _original_generate(*args, **kwargs)

            self.model.generate = _patched_generate
            logger.info("Model initialized with windowed n-gram processor")

        except Exception as e:
            error_msg = f"Failed to initialize model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _ocr_single_image(self, image_path: str, prompt: str, params: dict) -> str:
        """Run OCR on a single image file."""
        temp_dir = tempfile.mkdtemp()
        try:
            with torch.inference_mode():
                return self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=image_path,
                    output_path=temp_dir,
                    base_size=params["base_size"],
                    image_size=params["image_size"],
                    crop_mode=params["crop_mode"],
                    eval_mode=True,
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run OCR inference on the input image or PDF."""
        prompt = TASK_PROMPTS[input_data.task]
        logger.info(f"Processing file with task={input_data.task}, mode={input_data.mode}")
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
                        text = self._ocr_single_image(img_path, prompt, params)
                        page_results.append(postprocess_ocr(text))
                    result = "\n\n---\n\n".join(page_results)
                finally:
                    shutil.rmtree(pdf_dir, ignore_errors=True)
            else:
                result = postprocess_ocr(
                    self._ocr_single_image(file_path, prompt, params)
                )

            logger.info(f"OCR complete, extracted {len(result)} chars")
            return AppOutput(text=result)

        except Exception as e:
            error_msg = f"Inference failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
