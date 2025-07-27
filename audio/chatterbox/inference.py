from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from enum import Enum
from typing import Optional
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC

class Mode(str, Enum):
    TTS = "tts"
    VC = "vc"

class AppInput(BaseAppInput):
    mode: Mode = Field(
        Mode.TTS,
        description="Mode of operation: 'tts' (Text-to-Speech with optional voice cloning) or 'vc' (Voice Conversion between two speakers)"
    )
    text: str = Field(
        description="The text content to be converted to speech (used in TTS mode)"
    )
    voice_reference: Optional[File] = Field(
        None, 
        description="Reference audio file containing the desired voice style for TTS. The generated speech will mimic this voice's characteristics"
    )
    source_audio: Optional[File] = Field(
        None, 
        description="Source audio file containing the speech to be converted in voice conversion (VC) mode. This is the 'what is being said'"
    )
    target_voice_style: Optional[File] = Field(
        None, 
        description="Reference audio file containing the target voice style for voice conversion. The source_audio will be converted to match this voice's characteristics"
    )
    exaggeration: float = Field(
        0.5, 
        description="Controls the expressiveness of the generated speech. Higher values (0.7+) create more dramatic speech, lower values (0.3-) create more neutral speech"
    )
    cfg_weight: float = Field(
        0.5, 
        description="Controls how closely the output follows the reference voice style. Lower values (0.3-) allow more natural variation, higher values (0.7+) enforce closer matching"
    )

class AppOutput(BaseAppOutput):
    audio_output: File = Field(description="The generated audio file containing the synthesized or converted speech")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize Chatterbox models."""
        # Determine the device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        # Initialize both models
        self.tts_model = ChatterboxTTS.from_pretrained(device=self.device)
        self.vc_model = ChatterboxVC.from_pretrained(device=self.device)

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run TTS or VC inference based on the input mode."""
        output_path = "/tmp/output.wav"
        
        if input_data.mode == "tts":
            # Text-to-Speech generation
            wav = self.tts_model.generate(
                text=input_data.text,
                audio_prompt_path=input_data.voice_reference.path if input_data.voice_reference else None,
                exaggeration=input_data.exaggeration,
                cfg_weight=input_data.cfg_weight
            )
            ta.save(output_path, wav, self.tts_model.sr)
            
        elif input_data.mode == "vc":
            # Voice Conversion
            if not input_data.source_audio or not input_data.target_voice_style:
                raise ValueError("Voice conversion requires both source_audio (what is being said) and target_voice_style (how it should sound)")
                
            wav = self.vc_model.generate(
                audio=input_data.source_audio.path,
                target_voice_path=input_data.target_voice_style.path
            )
            ta.save(output_path, wav, self.vc_model.sr)
            
        else:
            raise ValueError(f"Invalid mode: {input_data.mode}. Must be either 'tts' or 'vc'")

        return AppOutput(
            audio_output=File(path=output_path)
        )

    async def unload(self):
        """Clean up resources."""
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()