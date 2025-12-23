import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import whisper_timestamped as whisper
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional, Union, List, Dict, Any
import soundfile as sf

def get_duration_in_seconds(audio_path):
    f = sf.SoundFile(audio_path)
    return f.frames / f.samplerate


class AppInput(BaseAppInput):
    audio: File = Field(
        description="The audio file to transcribe"
    )
    
    language: str = Field(
        default="english",
        description="Optional language of the audio (e.g. 'english', 'french'). If not provided, will be auto-detected.",
        enum=["english", "french", "german", "spanish", "italian", "japanese", "chinese", "portuguese", "russian", "korean"]
    )
    
    task: str = Field(
        default="transcribe",
        description="Whether to transcribe the audio in its original language or translate to English",
        enum=["transcribe", "translate"]
    )
    
    return_timestamps: str = Field(
        default="word",
        description="Whether to return timestamps. Use 'none' for no timestamps, 'word' for word-level timestamps, or 'sentence' for sentence-level timestamps.",
        enum=["none", "word", "sentence"]
    )

    vad: bool = Field(
        default=False,
        description="Whether to use Voice Activity Detection (VAD) to improve accuracy on files with silence"
    )

    detect_disfluencies: bool = Field(
        default=False,
        description="Whether to detect speech disfluencies (marked as [*] in transcript)"
    )

class AppOutput(BaseAppOutput):
    text: str = Field(description="The transcribed/translated text")
    segments: Optional[List[Dict[str, Any]]] = Field(description="Detailed segments with word-level timestamps and confidence scores", default=None)
    language: Optional[str] = Field(description="Detected language", default=None)

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize Whisper model with whisper-timestamped"""
        # Load model with CUDA device
        self.model = whisper.load_model("large-v3", device="cuda")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run transcription/translation on the input audio"""
        
        # Prepare transcription options
        transcribe_options = {
            "task": input_data.task,
            "vad": input_data.vad,
            "detect_disfluencies": input_data.detect_disfluencies,
        }
        
        # Set language if specified (whisper-timestamped uses None for auto-detection)
        if input_data.language and input_data.language != "auto":
            transcribe_options["language"] = input_data.language
        
        # Run transcription with whisper-timestamped
        result = whisper.transcribe(
            self.model,
            input_data.audio.path,
            **transcribe_options
        )

        # Extract text from all segments
        full_text = result.get("text", "")
        
        # Process segments based on timestamp requirements
        output_segments = None
        if input_data.return_timestamps == "word":
            # Return detailed word-level information with full timestamps
            output_segments = result.get("segments", [])
        elif input_data.return_timestamps == "sentence":
            # Return simplified sentence-level segments without word details
            output_segments = []
            for segment in result.get("segments", []):
                output_segments.append({
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": segment.get("text")
                })
        # For "none", output_segments remains None

        return AppOutput(
            text=full_text,
            segments=output_segments,
            language=result.get("language")
        )

    async def unload(self):
        """Clean up resources"""
        del self.model
        torch.cuda.empty_cache()