import os
import shutil
import tempfile
import torch
import logging
import re
from typing import Literal, List
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForImageTextToText,
    AutoProcessor,
    LogitsProcessor,
    LogitsProcessorList,
)
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
    auto_rotate: bool = Field(
        default=True,
        description="Detect pages rotated by 90, 180 or 270 degrees and turn them upright before OCR.",
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


# Image processing per mode. DeepSeek-OCR-2's encoder has query embeddings for
# 768 px (144 tokens) and 1024 px (256 tokens) views only, so every mode maps to
# one of those: gundam is the model's default (a 1024 global view plus 2-6
# local 768 tiles), the others a single global view. Without crops the
# processor sizes the global view by tile_size but counts its tokens by size,
# so both are set.
def _single_view(px):
    return {"crop_to_patches": False, "size": {"height": px, "width": px}, "tile_size": px}

MODE_IMAGE_KWARGS = {
    "tiny": _single_view(768),
    "small": _single_view(768),
    "base": _single_view(1024),
    "large": _single_view(1024),
    "gundam": {"crop_to_patches": True, "size": {"height": 1024, "width": 1024}, "tile_size": 768},
}

MAX_NEW_TOKENS = 8192

# Page orientation classifier (PP-LCNet, 4 classes: 0/90/180/270, clockwise).
# Pages are only turned when it is this sure; tiny crops such as a single
# handwritten word score lower and stay as they are.
ORIENTATION_MODEL = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors"
ORIENTATION_REVISION = "b2716954d2e321dd4436400b37584fab641027ab"
ORIENTATION_MIN_CONFIDENCE = 0.85


class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.device = None
        self.model_name = "deepseek-community/DeepSeek-OCR-2"

    async def setup(self, metadata):
        """Initialize the OCR model and resources."""
        logger.info("Initializing DeepSeek OCR-2 model (native transformers)...")

        accelerator = Accelerator()
        self.device = accelerator.device
        logger.info(f"Using device: {self.device}")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16,
            ).eval().to(self.device)

            self.ori_processor = AutoImageProcessor.from_pretrained(ORIENTATION_MODEL, revision=ORIENTATION_REVISION)
            self.ori_model = AutoModelForImageClassification.from_pretrained(
                ORIENTATION_MODEL, revision=ORIENTATION_REVISION
            ).eval().to(self.device)

            tokenizer = self.processor.tokenizer
            td_ids = set()
            for tag in ["<td>", "</td>"]:
                td_ids.update(tokenizer.encode(tag, add_special_tokens=False))
            logger.info(f"Whitelist token IDs for table tags: {td_ids}")
            self.ngram_proc = WindowedNGramProcessor(ngram_size=30, window_size=90, whitelist_token_ids=td_ids)
            # Decoded with special tokens kept (grounding tags); these are stripped after.
            self.strip_tokens = [t for t in (tokenizer.eos_token, tokenizer.pad_token, tokenizer.bos_token) if t]
            logger.info("Model initialized with windowed n-gram processor")

        except Exception as e:
            error_msg = f"Failed to initialize model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _upright(self, image: Image.Image) -> Image.Image:
        """Turn a page rotated by 90/180/270 degrees upright."""
        inputs = self.ori_processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            # PPLCNetForImageClassification returns its logits as last_hidden_state
            probs = self.ori_model(**inputs).last_hidden_state.softmax(-1)[0]
        idx = int(probs.argmax())
        angle = int(self.ori_model.config.id2label[idx])
        confidence = float(probs[idx])
        if angle == 0 or confidence < ORIENTATION_MIN_CONFIDENCE:
            return image
        logger.info(f"Page rotated {angle} degrees clockwise (p={confidence:.2f}); turning it upright")
        return image.rotate(angle, expand=True)

    def _ocr_single_image(self, image_path: str, prompt: str, image_kwargs: dict, auto_rotate: bool = True) -> str:
        """Run OCR on a single image file."""
        image = Image.open(image_path).convert("RGB")
        if auto_rotate:
            image = self._upright(image)
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt", **image_kwargs
        ).to(self.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                logits_processor=LogitsProcessorList([self.ngram_proc]),
            )
        text = self.processor.decode(
            output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False
        )
        for tok in self.strip_tokens:
            text = text.replace(tok, "")
        return text.strip()

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run OCR inference on the input image or PDF."""
        prompt = TASK_PROMPTS[input_data.task]
        logger.info(f"Processing file with task={input_data.task}, mode={input_data.mode}")
        try:
            params = MODE_IMAGE_KWARGS[input_data.mode]
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
                        text = self._ocr_single_image(img_path, prompt, params, input_data.auto_rotate)
                        page_results.append(postprocess_ocr(text))
                    result = "\n\n---\n\n".join(page_results)
                finally:
                    shutil.rmtree(pdf_dir, ignore_errors=True)
            else:
                result = postprocess_ocr(
                    self._ocr_single_image(file_path, prompt, params, input_data.auto_rotate)
                )

            logger.info(f"OCR complete, extracted {len(result)} chars")
            return AppOutput(text=result)

        except Exception as e:
            error_msg = f"Inference failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
