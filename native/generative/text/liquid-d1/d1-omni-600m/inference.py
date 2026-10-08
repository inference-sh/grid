"""
d1-omni-600M — Liquid AI's open 587M decision model for text, images and speech, on our own GPUs.

One request evaluates one `state` (text or JSON, plus up to 4 images or one audio clip)
against any number of typed questions and returns a probability for every option. No text is
generated. Schemas are shared with the other liquid/d1-* apps: see d1_helper.py.

The model truncates as it was trained: the state is cut on the right to the room left after
the question (16,384 tokens for text, 15,360 with audio, 896 with images). run() reports that
in `state_truncated`. Liquid marks this model an early research release.

Metering: inputs=[TextMeta(tokens=<positions read, image and audio included>)], outputs=[TextMeta(tokens=0)].

Model: https://huggingface.co/LiquidAI/d1-omni-600M (LFM Open License v1.0)
"""

import asyncio
import importlib
import logging
import time
from typing import Any, Dict, List, Optional

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

MODEL_ID = "LiquidAI/d1-omni-600M"
# The repository runs its own code (trust_remote_code). Review the diff before moving this.
REVISION = "02b55d7076f15129e59ab3f94783f32c4b088674"
# The model card: float16 keeps the float32 answers on GPU; bfloat16 changes some.
DTYPE = "float16"
MAX_IMAGES = 4
SAMPLE_RATE = 16000
# The model cuts audio at 30 s. A longer clip is rejected here so that nothing is dropped silently.
MAX_AUDIO_SECONDS = 30


class AppInput(BaseAppInput):
    state: Structured = Field(
        default="",
        description="The text to evaluate: a string, or a JSON object / array of related context. Every question sees the same state. May be empty when `images` or `audio` carries the content. A state over the model's limit is cut at the end and `state_truncated` is set: the state and one question share 16,384 tokens for text, 15,360 with audio, 896 with images.",
        examples=["Seller title: wireless earbuds, barely used"],
    )
    images: List[File] = Field(
        default_factory=list,
        max_length=MAX_IMAGES,
        description="Up to 4 images the questions are about. Cannot be combined with `audio`.",
    )
    audio: Optional[File] = Field(
        default=None,
        description="One speech clip the questions are about, up to 30 seconds (WAV, FLAC, OGG or MP3). Trained on English requests to an assistant. Cannot be combined with `images`.",
    )
    choices: List[ChoiceQuestion] = Field(default_factory=list, description="Choice questions: pick one option from a set.")
    scores: List[ScoreQuestion] = Field(default_factory=list, description="Score questions: place the state on ordered levels.")
    nouls: List[NoulQuestion] = Field(default_factory=list, description="Noul questions: probability that the answer is yes.")

    @model_validator(mode="after")
    def _check(self):
        check_questions(self.choices, self.scores, self.nouls)
        if self.images and self.audio is not None:
            raise ValueError("send `images` or `audio`, not both")
        if self.state == "" and not self.images and self.audio is None:
            raise ValueError("send a `state`, `images` or `audio`")
        return self


class AppOutput(BaseAppOutput):
    choices: Dict[str, ChoiceAnswer] = Field(default_factory=dict, description="Choice answers by question id.")
    scores: Dict[str, ScoreAnswer] = Field(default_factory=dict, description="Score answers by question id.")
    nouls: Dict[str, NoulAnswer] = Field(default_factory=dict, description="Noul answers by question id.")
    model: str = Field(description="The model that answered: `LiquidAI/d1-omni-600M`.")
    input_tokens: int = Field(default=0, description="Positions read, summed over the questions. Image and audio positions are included.")
    state_truncated: bool = Field(default=False, description="True when the state did not fit and its end was cut for at least one question.")


def load_audio(file: File) -> Any:
    """One clip as mono 16 kHz float32 samples."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    if not file.exists():
        raise RuntimeError(f"audio does not exist at path: {file.path}")
    samples, rate = sf.read(file.path, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    seconds = len(samples) / rate
    if seconds > MAX_AUDIO_SECONDS:
        raise RuntimeError(f"audio is {seconds:.1f} s long; this model reads at most {MAX_AUDIO_SECONDS} s")
    if rate != SAMPLE_RATE:
        divisor = np.gcd(SAMPLE_RATE, rate)
        samples = resample_poly(samples, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
    return samples


def state_is_truncated(model: Any, state: Any, questions: Dict[str, Dict[str, Any]], text_length: int) -> bool:
    """True if any question over this state is longer than `text_length` tokens before the model cuts it."""
    # The repository's own prompt code, from the package its model class was loaded from.
    prompt = importlib.import_module(".prompt", type(model).__module__.rpartition(".")[0])
    unbounded = 10 ** 9
    return any(
        len(prompt.encode(model.tokenizer, state, prompt.as_question(q), unbounded)[0]) > text_length
        for q in questions.values()
    )


def decide(model: Any, input_data: AppInput, logger) -> AppOutput:
    """Evaluate the state and its images or audio against every question. Blocking: call it in a thread."""
    import torch

    images = open_images(input_data.images)
    audio = load_audio(input_data.audio) if input_data.audio is not None else None
    questions = build_questions(input_data.choices, input_data.scores, input_data.nouls)

    config = model.config
    text_length = config.image_text_length if images else config.audio_text_length if audio is not None else config.max_length
    truncated = state_is_truncated(model, input_data.state, questions, text_length)
    if truncated:
        logger.warning(f"state cut to fit {text_length} tokens per question")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = model.system_one(input_data.state or None, questions, images=images or None, audio=audio)
    elapsed_ms = (time.monotonic() - started) * 1000

    input_tokens = int(result["usage"]["input_tokens"])
    seconds = 0.0 if audio is None else len(audio) / SAMPLE_RATE
    logger.info(
        f"answered by {MODEL_ID}: {input_tokens} input tokens, {len(questions)} questions, "
        f"{len(images)} images, {seconds:.1f} s audio, {elapsed_ms:.1f} ms, peak {vram_gb(peak=True):.1f} GB VRAM"
    )
    return AppOutput(
        **build_answers(result["answers"], questions),
        model=MODEL_ID,
        input_tokens=input_tokens,
        state_truncated=truncated,
        output_meta=OutputMeta(inputs=[TextMeta(tokens=input_tokens)], outputs=[TextMeta(tokens=0)]),
    )


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.model = await asyncio.to_thread(load_model, MODEL_ID, REVISION, DTYPE, self.logger)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Evaluate the state and its images or audio against every question."""
        return await asyncio.to_thread(decide, self.model, input_data, self.logger)

    async def unload(self):
        import torch

        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()
