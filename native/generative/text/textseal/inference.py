import logging
from enum import Enum
from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, BaseAppSetup, OutputMeta, TextMeta
from pydantic import Field

# Upstream keeps chunks near 2000 chars; a rephrased chunk stays well under this many tokens.
MAX_GEN_TOKENS = 1024
# prf_hires multiplies the key by 40503 into an int64, and JS clients lose precision above 2^53.
MAX_SECRET_KEY = 2**46


class ModelEnum(str, Enum):
    qwen_0_8b = "Qwen/Qwen3.5-0.8B"
    qwen_2b = "Qwen/Qwen3.5-2B"
    qwen_4b = "Qwen/Qwen3.5-4B"
    qwen_9b = "Qwen/Qwen3.5-9B"


class AppSetup(BaseAppSetup):
    model: ModelEnum = Field(
        default=ModelEnum.qwen_2b,
        description="Model that writes the text. Detect with the model that watermarked it.",
    )


class KeyInput(BaseAppInput):
    secret_key: int = Field(
        ge=0,
        le=MAX_SECRET_KEY,
        description="Private key. Detection needs the same key. Use a random number and keep it secret.",
    )


class SamplingInput(KeyInput):
    temperature: float = Field(default=1.0, ge=0.1, le=1.5, description="Sampling temperature")
    top_p: float = Field(default=0.95, gt=0.0, le=1.0, description="Top-p sampling")


class WatermarkInput(SamplingInput):
    text: str = Field(min_length=1, max_length=20000, description="Text to rephrase with a watermark")


class GenerateInput(SamplingInput):
    prompt: str = Field(min_length=1, max_length=20000, description="Prompt to answer with watermarked text")
    max_tokens: int = Field(default=512, ge=1, le=MAX_GEN_TOKENS, description="Maximum tokens to generate")


class DetectInput(KeyInput):
    text: str = Field(min_length=1, max_length=100000, description="Text to check for a watermark")


class Detection(BaseAppOutput):
    detected: bool = Field(description="True when the p-value is under 0.001")
    p_value: float = Field(description="Chance of this score on unwatermarked text. Lower is stronger evidence.")
    tokens_scored: int = Field(description="Tokens tested. Short texts cannot reach detection.")


class TextOutput(Detection):
    text: str = Field(description="Watermarked text")


class App(BaseApp):
    async def setup(self, config: AppSetup):
        self.logger = logging.getLogger(__name__)
        from textseal import EvaluationConfig, ModelConfig, PostHocWatermarker

        model_name = config.model.value
        self.logger.info("Loading %s", model_name)
        self.wm = PostHocWatermarker(
            watermark_config=self._config(0),
            model_config=ModelConfig(model_name=model_name),
            evaluation_config=EvaluationConfig(
                enable_semantic_similarity=False,
                enable_rouge=False,
                enable_bleu=False,
            ),
        )

    @staticmethod
    def _config(secret_key: int):
        from textseal import WatermarkConfig

        return WatermarkConfig(watermark_type="textseal2", secret_key=secret_key, ngram=3, rho=0.9, prf="hires")

    def _use_key(self, secret_key: int) -> None:
        from textseal.watermarking.detector import build_detector
        from textseal.watermarking.generator import build_generator

        config = self._config(secret_key)
        self.wm.watermark_config = config
        self.wm.generator = build_generator(self.wm.model, self.wm.tokenizer, config)
        self.wm.detector = build_detector(self.wm.tokenizer, config, self.wm.model)

    @staticmethod
    def _detection(result: dict) -> dict:
        return {
            "detected": bool(result["det"]),
            "p_value": float(result["p_value"]),
            "tokens_scored": int(result["toks_scored"]),
        }

    def _meta(self, source: str, output: Optional[str] = None) -> OutputMeta:
        outputs = [TextMeta(tokens=self.wm.count_tokens(output))] if output is not None else []
        return OutputMeta(inputs=[TextMeta(tokens=self.wm.count_tokens(source))], outputs=outputs)

    async def watermark(self, input_data: WatermarkInput) -> TextOutput:
        self._use_key(input_data.secret_key)
        self.wm.processing_config.generation_mode = False
        process = (
            self.wm.process_large_text
            if len(input_data.text) > self.wm.processing_config.target_chunk_size
            else self.wm.process_text
        )
        result = process(
            input_data.text,
            max_gen_len=MAX_GEN_TOKENS,
            temperature=input_data.temperature,
            top_p=input_data.top_p,
        )
        text = result["wm_text"]
        detection = self._detection(result["wm_eval"])
        self.logger.info("Watermarked %d chars into %d: %s", len(input_data.text), len(text), detection)
        return TextOutput(text=text, output_meta=self._meta(input_data.text, text), **detection)

    async def generate(self, input_data: GenerateInput) -> TextOutput:
        self._use_key(input_data.secret_key)
        self.wm.processing_config.generation_mode = True
        text = self.wm.rephrase_with_watermark(
            input_data.prompt,
            max_gen_len=input_data.max_tokens,
            temperature=input_data.temperature,
            top_p=input_data.top_p,
        )
        detection = self._detection(self.wm.evaluate_watermark(text))
        self.logger.info("Generated %d chars: %s", len(text), detection)
        return TextOutput(text=text, output_meta=self._meta(input_data.prompt, text), **detection)

    async def detect(self, input_data: DetectInput) -> Detection:
        self._use_key(input_data.secret_key)
        detection = self._detection(self.wm.evaluate_watermark(input_data.text))
        self.logger.info("Checked %d chars: %s", len(input_data.text), detection)
        return Detection(output_meta=self._meta(input_data.text), **detection)
