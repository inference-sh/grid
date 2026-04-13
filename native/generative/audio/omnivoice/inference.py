from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, AudioMeta
from pydantic import Field
from typing import Optional
from enum import Enum


SAMPLE_RATE = 24000


class Mode(str, Enum):
    VOICE_CLONING = "voice_cloning"
    VOICE_DESIGN = "voice_design"
    AUTO = "auto"


class AppInput(BaseAppInput):
    text: str = Field(description="Text to convert to speech. Supports non-verbal tags like [laughter], [sigh], and pronunciation control.")
    mode: Mode = Field(
        Mode.AUTO,
        description="Generation mode: 'voice_cloning' (clone from reference audio), 'voice_design' (describe voice attributes), or 'auto' (model picks a voice)"
    )
    ref_audio: Optional[File] = Field(
        None,
        description="Reference audio for voice cloning (3-10 seconds recommended). Required for voice_cloning mode."
    )
    ref_text: Optional[str] = Field(
        None,
        description="Transcription of the reference audio. If omitted, Whisper ASR auto-transcribes it."
    )
    instruct: Optional[str] = Field(
        None,
        description="Voice design attributes for voice_design mode. Comma-separated: e.g. 'female, low pitch, british accent'. Supports gender, age, pitch, accent, whisper."
    )
    num_step: int = Field(
        32,
        description="Number of diffusion steps. Lower (16) = faster, higher (32) = better quality.",
        ge=4,
        le=64
    )
    speed: float = Field(
        1.0,
        description="Speaking speed factor. >1.0 = faster, <1.0 = slower.",
        ge=0.25,
        le=4.0
    )
    duration: Optional[float] = Field(
        None,
        description="Fixed output duration in seconds. Overrides speed if set.",
        ge=0.5,
        le=120.0
    )


class AppOutput(BaseAppOutput):
    audio: File = Field(description="Generated speech audio file (WAV, 24kHz)")


class App(BaseApp):
    async def setup(self):
        import torch
        from omnivoice import OmniVoice

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Loading OmniVoice on {device} with {dtype}")

        self.model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map=device,
            dtype=dtype,
        )
        print("OmniVoice model loaded")

    async def run(self, input_data: AppInput) -> AppOutput:
        import soundfile as sf

        output_path = "/tmp/output.wav"

        kwargs = {
            "text": input_data.text,
            "num_step": input_data.num_step,
            "speed": input_data.speed,
        }

        if input_data.duration is not None:
            kwargs["duration"] = input_data.duration

        if input_data.mode == Mode.VOICE_CLONING:
            if not input_data.ref_audio:
                raise ValueError("Voice cloning mode requires ref_audio")
            kwargs["ref_audio"] = input_data.ref_audio.path
            if input_data.ref_text:
                kwargs["ref_text"] = input_data.ref_text
            print(f"Voice cloning: text='{input_data.text[:80]}', ref_text={'provided' if input_data.ref_text else 'auto-transcribe'}")

        elif input_data.mode == Mode.VOICE_DESIGN:
            if not input_data.instruct:
                raise ValueError("Voice design mode requires instruct")
            kwargs["instruct"] = input_data.instruct
            print(f"Voice design: text='{input_data.text[:80]}', instruct='{input_data.instruct}'")

        else:
            print(f"Auto voice: text='{input_data.text[:80]}'")

        audio = self.model.generate(**kwargs)
        print("Generation complete")

        # audio is a list of np.ndarray at 24kHz
        wav = audio[0]
        sf.write(output_path, wav, SAMPLE_RATE)

        duration_secs = round(len(wav) / SAMPLE_RATE, 2)
        print(f"Output: {duration_secs}s audio")

        return AppOutput(
            audio=File(path=output_path),
            output_meta=OutputMeta(
                outputs=[AudioMeta(seconds=duration_secs)]
            ),
        )
