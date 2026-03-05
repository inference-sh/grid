from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from enum import Enum
from typing import Optional
import torch
import torchaudio as ta

# Patch perth watermarker if it fails to initialize (common GPU env issue)
import perth
if perth.PerthImplicitWatermarker is None:
    class _NoOpWatermarker:
        def apply_watermark(self, wav, sample_rate):
            return wav
    perth.PerthImplicitWatermarker = _NoOpWatermarker


class Model(str, Enum):
    TURBO = "turbo"
    MULTILINGUAL = "multilingual"
    ORIGINAL = "original"


class Mode(str, Enum):
    TTS = "tts"
    VC = "vc"


# Supported languages for multilingual model
SUPPORTED_LANGUAGES = [
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh"
]

MAX_CHUNK_SIZE = 1024


def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """Split text into chunks, preferring sentence boundaries."""
    if len(text) <= max_size:
        return [text]

    chunks = []
    current = ""

    # Split on sentence endings
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_size:
            current = f"{current} {sentence}".strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            # If single sentence is too long, split by words
            if len(sentence) > max_size:
                words = sentence.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 <= max_size:
                        current = f"{current} {word}".strip() if current else word
                    else:
                        if current:
                            chunks.append(current)
                        current = word
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


class AppInput(BaseAppInput):
    model: Model = Field(
        Model.ORIGINAL,
        description="Model to use: 'original' (works without voice ref), 'turbo' (fastest, requires voice ref, supports [laugh] tags), or 'multilingual' (23+ languages)"
    )
    mode: Mode = Field(
        Mode.TTS,
        description="Mode of operation: 'tts' (Text-to-Speech) or 'vc' (Voice Conversion, original model only)"
    )
    text: str = Field(
        description="Text to convert to speech. Turbo model supports tags like [laugh], [cough], [chuckle]"
    )
    voice_reference: Optional[File] = Field(
        None,
        description="Reference audio file for voice cloning. Required for turbo model."
    )
    language: str = Field(
        "en",
        description="Language code for multilingual model (ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh)"
    )
    source_audio: Optional[File] = Field(
        None,
        description="Source audio for voice conversion (VC mode with original model only)"
    )
    target_voice_style: Optional[File] = Field(
        None,
        description="Target voice style for voice conversion (VC mode with original model only)"
    )
    exaggeration: float = Field(
        0.5,
        description="Expressiveness control (original/multilingual). Higher (0.7+) = dramatic, lower (0.3-) = neutral"
    )
    cfg_weight: float = Field(
        0.5,
        description="Voice matching strength (original/multilingual). Lower (0.3) = natural variation, higher (0.7+) = closer match"
    )


class AppOutput(BaseAppOutput):
    audio_output: File = Field(description="Generated audio file")


class App(BaseApp):
    async def setup(self):
        """Initialize device - models loaded on demand."""
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self._turbo_model = None
        self._multilingual_model = None
        self._original_model = None
        self._vc_model = None

    def _get_turbo_model(self):
        if self._turbo_model is None:
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                self._turbo_model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            except Exception as e:
                import traceback
                raise RuntimeError(f"Failed to load ChatterboxTurboTTS: {e}\n{traceback.format_exc()}")
        return self._turbo_model

    def _get_multilingual_model(self):
        if self._multilingual_model is None:
            try:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self._multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            except Exception as e:
                import traceback
                raise RuntimeError(f"Failed to load ChatterboxMultilingualTTS: {e}\n{traceback.format_exc()}")
        return self._multilingual_model

    def _get_original_model(self):
        if self._original_model is None:
            try:
                from chatterbox.tts import ChatterboxTTS
                self._original_model = ChatterboxTTS.from_pretrained(device=self.device)
            except Exception as e:
                import traceback
                raise RuntimeError(f"Failed to load ChatterboxTTS: {e}\n{traceback.format_exc()}")
        return self._original_model

    def _get_vc_model(self):
        if self._vc_model is None:
            try:
                from chatterbox.vc import ChatterboxVC
                self._vc_model = ChatterboxVC.from_pretrained(device=self.device)
            except Exception as e:
                import traceback
                raise RuntimeError(f"Failed to load ChatterboxVC: {e}\n{traceback.format_exc()}")
        return self._vc_model

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run TTS or VC inference."""
        output_path = "/tmp/output.wav"

        if input_data.mode == Mode.VC:
            # Voice Conversion - only works with original model
            if not input_data.source_audio or not input_data.target_voice_style:
                raise ValueError("Voice conversion requires both source_audio and target_voice_style")

            vc_model = self._get_vc_model()
            wav = vc_model.generate(
                audio=input_data.source_audio.path,
                target_voice_path=input_data.target_voice_style.path
            )
            ta.save(output_path, wav, vc_model.sr)

        elif input_data.model == Model.TURBO:
            if not input_data.voice_reference:
                raise ValueError("Turbo model requires a voice_reference audio file")

            model = self._get_turbo_model()
            chunks = chunk_text(input_data.text)
            wavs = []
            for chunk in chunks:
                wav = model.generate(
                    chunk,
                    audio_prompt_path=input_data.voice_reference.path
                )
                wavs.append(wav)
            wav = torch.cat(wavs, dim=1) if len(wavs) > 1 else wavs[0]
            ta.save(output_path, wav, model.sr)

        elif input_data.model == Model.MULTILINGUAL:
            if input_data.language not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language: {input_data.language}. Supported: {SUPPORTED_LANGUAGES}")

            model = self._get_multilingual_model()
            kwargs = {
                "language_id": input_data.language,
                "exaggeration": input_data.exaggeration,
                "cfg_weight": input_data.cfg_weight,
            }
            if input_data.voice_reference:
                kwargs["audio_prompt_path"] = input_data.voice_reference.path

            chunks = chunk_text(input_data.text)
            wavs = []
            for chunk in chunks:
                wav = model.generate(chunk, **kwargs)
                wavs.append(wav)
            wav = torch.cat(wavs, dim=1) if len(wavs) > 1 else wavs[0]
            ta.save(output_path, wav, model.sr)

        else:  # Original model
            model = self._get_original_model()
            kwargs = {
                "exaggeration": input_data.exaggeration,
                "cfg_weight": input_data.cfg_weight,
            }
            if input_data.voice_reference:
                kwargs["audio_prompt_path"] = input_data.voice_reference.path

            chunks = chunk_text(input_data.text)
            wavs = []
            for chunk in chunks:
                wav = model.generate(chunk, **kwargs)
                wavs.append(wav)
            wav = torch.cat(wavs, dim=1) if len(wavs) > 1 else wavs[0]
            ta.save(output_path, wav, model.sr)

        return AppOutput(audio_output=File(path=output_path))
