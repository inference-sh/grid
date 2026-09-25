"""DramaBox — expressive TTS with voice cloning (Resemble AI, built on LTX-2.3).

Vendored upstream code lives in ./DramaBox (see DramaBox/UPSTREAM.txt). The
warm TTSServer keeps Gemma, the audio DiT, VAE and vocoder resident so each
request is a single denoising pass.
"""
import logging
import os
import sys
from typing import Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "DramaBox", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Above this estimated length the prompt is chunked and stitched (upstream default).
MAX_CHUNK_SECONDS = 45.0
TARGET_CHUNK_SECONDS = 37.0
CROSSFADE_MS = 50.0
REF_SECONDS = 10.0


class AppInput(BaseAppInput):
    prompt: str = Field(
        description=(
            'Scene-style prompt: <speaker description>, "<dialogue>" <stage direction> "<more dialogue>". '
            'Text inside quotes is spoken; text outside quotes directs delivery (e.g. She sighs deeply.). '
            'Laughs and sounds go inside quotes as one word ("Hahaha", "Mmmm", "Ugh"). '
            'End the prompt at the last closing quote. Long prompts are chunked automatically.'
        ),
        examples=['A woman speaks warmly, "Hello, how are you today?" She laughs, "Hahaha, it is so good to see you!"'],
    )
    voice_reference: Optional[File] = Field(
        None,
        description="Optional voice reference to clone (10+ seconds recommended; first 10 s are used). Match the speaker description to the reference's gender and age.",
    )
    cfg_scale: float = Field(2.5, ge=1.0, le=10.0, description="Classifier-free guidance. Lower = more natural, higher = more text-faithful.")
    stg_scale: float = Field(1.5, ge=0.0, le=5.0, description="Skip-token guidance scale.")
    duration: float = Field(
        0.0, ge=0.0, le=300.0,
        description="Target output length in seconds. 0 = estimate automatically from the prompt.",
    )
    duration_multiplier: float = Field(
        1.1, ge=0.5, le=2.0,
        description="Headroom applied to the auto-estimated duration (ignored when duration is set).",
    )
    seed: int = Field(42, ge=0, description="Random seed for reproducible output.")


class AppOutput(BaseAppOutput):
    audio: File = Field(description="Generated speech (WAV, 48 kHz, watermarked with Resemble Perth)")
    duration_seconds: float = Field(description="Length of the generated audio in seconds")


class App(BaseApp):
    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        from accelerate import Accelerator
        from model_downloader import get_all_paths
        from inference_server import TTSServer

        device = Accelerator().device
        self.logger.info(f"Fetching DramaBox checkpoints (device={device})")
        paths = get_all_paths()

        compile_model = os.environ.get("DRAMABOX_COMPILE", "0") == "1"
        self.server = TTSServer(
            checkpoint=paths["transformer"],
            full_checkpoint=paths["audio_components"],
            gemma_root=paths["gemma_root"],
            device=str(device),
            dtype=os.environ.get("LTX_DTYPE", "bf16"),
            compile_model=compile_model,
            bnb_4bit=True,  # unsloth Gemma checkpoint is pre-quantized
        )
        import perth
        self._perth = perth.PerthImplicitWatermarker()
        self.logger.info("DramaBox ready")

    def _watermark(self, wav, sr: int):
        """Resemble Perth watermark on mono, re-broadcast to the original channel count."""
        import numpy as np
        import torch

        mono = wav.mean(dim=0).numpy() if wav.shape[0] > 1 else wav[0].numpy()
        mono_wm = self._perth.apply_watermark(mono, sample_rate=sr)
        mono_wm = torch.from_numpy(np.asarray(mono_wm, dtype=np.float32)).unsqueeze(0)
        return mono_wm if wav.shape[0] == 1 else mono_wm.repeat(wav.shape[0], 1)

    async def run(self, input_data: AppInput) -> AppOutput:
        import soundfile as sf
        from inference_server import estimate_duration

        voice_ref = None
        if input_data.voice_reference is not None:
            if not input_data.voice_reference.exists():
                raise RuntimeError(f"Voice reference not found at {input_data.voice_reference.path}")
            voice_ref = input_data.voice_reference.path

        gen_kwargs = dict(
            voice_ref=voice_ref,
            cfg_scale=input_data.cfg_scale,
            stg_scale=input_data.stg_scale,
            seed=input_data.seed,
            ref_duration=REF_SECONDS,
            gen_duration=input_data.duration,
            duration_multiplier=input_data.duration_multiplier,
            denoise_ref=False,  # RE-USE denoiser is non-commercial and its mamba deps are not installed
        )

        est = input_data.duration if input_data.duration > 0 else estimate_duration(
            input_data.prompt, input_data.duration_multiplier)
        self.logger.info(
            f"prompt={input_data.prompt[:80]!r} voice_ref={'yes' if voice_ref else 'no'} "
            f"est={est:.1f}s cfg={input_data.cfg_scale} stg={input_data.stg_scale} seed={input_data.seed}"
        )

        if est > MAX_CHUNK_SECONDS:
            waveform, sr = self.server.generate_long(
                input_data.prompt,
                max_chunk_duration=MAX_CHUNK_SECONDS,
                target_chunk_duration=TARGET_CHUNK_SECONDS,
                crossfade_ms=CROSSFADE_MS,
                **gen_kwargs,
            )
        else:
            waveform, sr = self.server.generate(input_data.prompt, **gen_kwargs)

        wav = waveform.detach().cpu().float()
        wav = self._watermark(wav, sr)

        output_path = "/tmp/output.wav"
        sf.write(output_path, wav.T.numpy(), sr)

        duration_seconds = round(wav.shape[-1] / sr, 2)
        self.logger.info(f"Generated {duration_seconds}s audio at {sr} Hz")

        return AppOutput(
            audio=File(path=output_path),
            duration_seconds=duration_seconds,
            output_meta=OutputMeta(inputs=[], outputs=[AudioMeta(seconds=duration_seconds)]),
        )

    async def unload(self):
        import gc
        import torch
        self.server = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
