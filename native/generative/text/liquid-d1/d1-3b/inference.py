"""
d1-3B — Liquid AI's open 3B decision model for text and images, run on our own GPUs.

One request evaluates one `state` (text or JSON, plus up to 8 images) against any number of
typed questions and returns a probability for every option. The state and images are read
once and shared by all questions; no text is generated. Schemas are shared with the other
liquid/d1-* apps: see d1_helper.py.

Metering: inputs=[TextMeta(tokens=<input tokens, image tokens included>)], outputs=[TextMeta(tokens=0)].

Model: https://huggingface.co/LiquidAI/d1-3B (LFM Open License v1.0)
"""

import asyncio
import importlib
import logging
import time
from typing import Any, Dict, List

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, TextMeta
from pydantic import Field, model_validator

from .d1_helper import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    Structured,
    build_answers,
    build_questions,
    check_questions,
    load_model,
    open_images,
    vram_gb,
)

MODEL_ID = "LiquidAI/d1-3B"
# The repository runs its own code (trust_remote_code). Review the diff before moving this.
REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
# bfloat16 is the dtype the checkpoint is stored in.
DTYPE = "bfloat16"
# The backbone's position limit. The model code does not check it, so run() does.
MAX_INPUT_TOKENS = 32768
MAX_IMAGES = 8
# Upper bound on one image's tokens: 10 tiles and a thumbnail at 256 tokens each, plus markers.
IMAGE_TOKENS = 3000


class AppInput(BaseAppInput):
    state: Structured = Field(
        default="",
        description="The text to evaluate: a string, or a JSON object / array of related context (messages, records, a policy). Every question sees the same state and images. May be empty when `images` carries the content. Input over the model's 32,768-token limit is rejected, never truncated.",
        examples=["Seller title: wireless earbuds, barely used"],
    )
    images: List[File] = Field(
        default_factory=list,
        max_length=MAX_IMAGES,
        description="Up to 8 images the questions are about. Every question sees them, placed before the state. Images over 1 megapixel are downscaled.",
    )
    choices: List[ChoiceQuestion] = Field(default_factory=list, description="Choice questions: pick one option from a set.")
    scores: List[ScoreQuestion] = Field(default_factory=list, description="Score questions: place the state on ordered levels.")
    nouls: List[NoulQuestion] = Field(default_factory=list, description="Noul questions: probability that the answer is yes.")

    @model_validator(mode="after")
    def _check(self):
        check_questions(self.choices, self.scores, self.nouls)
        if self.state == "" and not self.images:
            raise ValueError("send a `state`, `images`, or both")
        return self


class AppOutput(BaseAppOutput):
    choices: Dict[str, ChoiceAnswer] = Field(default_factory=dict, description="Choice answers by question id.")
    scores: Dict[str, ScoreAnswer] = Field(default_factory=dict, description="Score answers by question id.")
    nouls: Dict[str, NoulAnswer] = Field(default_factory=dict, description="Noul answers by question id.")
    model: str = Field(description="The model that answered: `LiquidAI/d1-3B`.")
    input_tokens: int = Field(default=0, description="Input tokens read: the state and images once, plus each question. Image tokens are included.")


def check_length(model: Any, state: Any, questions: Dict[str, Dict[str, Any]], image_count: int) -> None:
    """Raise if the longest question over this state, with room for the images, is over the position limit."""
    # The repository's own question classes, from the package its model class was loaded from.
    prompt = importlib.import_module(".prompt", type(model).__module__.rpartition(".")[0])
    text_tokens = model.engine.tokens(state, [prompt.as_question(q) for q in questions.values()])
    total = text_tokens + image_count * IMAGE_TOKENS
    if total > MAX_INPUT_TOKENS:
        raise RuntimeError(
            f"input is too long: {text_tokens} text tokens for the longest question"
            + (f", plus up to {IMAGE_TOKENS} for each of {image_count} images" if image_count else "")
            + f". This model's limit is {MAX_INPUT_TOKENS} tokens."
        )


def decide(model: Any, input_data: AppInput, logger) -> AppOutput:
    """Evaluate the state and images against every question. Blocking: call it in a thread."""
    import torch

    images = open_images(input_data.images)
    questions = build_questions(input_data.choices, input_data.scores, input_data.nouls)
    state = None if input_data.state == "" else input_data.state
    check_length(model, state, questions, len(images))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = model.system_one(state, questions, images=images)
    elapsed_ms = (time.monotonic() - started) * 1000

    input_tokens = int(result["usage"]["input_tokens"])
    logger.info(
        f"answered by {MODEL_ID}: {input_tokens} input tokens, {len(questions)} questions, "
        f"{len(images)} images, {elapsed_ms:.1f} ms, peak {vram_gb(peak=True):.1f} GB VRAM"
    )
    return AppOutput(
        **build_answers(result["answers"], questions),
        model=MODEL_ID,
        input_tokens=input_tokens,
        output_meta=OutputMeta(inputs=[TextMeta(tokens=input_tokens)], outputs=[TextMeta(tokens=0)]),
    )


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.model = await asyncio.to_thread(load_model, MODEL_ID, REVISION, DTYPE, self.logger)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Evaluate the state and images against every question."""
        return await asyncio.to_thread(decide, self.model, input_data, self.logger)

    async def unload(self):
        import torch

        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()
