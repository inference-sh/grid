from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from enum import Enum
import os
import soundfile as sf
import tempfile
from pydub import AudioSegment
import numpy as np
from dia.model import Dia
import torch
import librosa
from typing import Optional
class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"

class AppInput(BaseAppInput):
    text: str = Field(..., description="The text to convert to speech. Use [S1] and [S2] tags for different speakers.")
    format: AudioFormat = Field(default=AudioFormat.WAV, description="The output audio format")
    speed: float = Field(default=1.0, description="Speech speed (0.1 to 0.5)")
    clone_from_text: Optional[str] = Field(default=None, description="The transcript text used for voice cloning. Must use [S1] and [S2] tags.")
    clone_from_audio: Optional[File] = Field(default=None, description="The audio file to clone voices from")

class AppOutput(BaseAppOutput):
    audio: File = Field(..., description="The generated audio file")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the DIA TTS model."""
        print("Initializing DIA model...")
        self.model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")
        print("DIA model initialized successfully")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate speech from text using DIA TTS."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{input_data.format.value}", delete=False) as temp_file:
            output_path = temp_file.name
        
        # Generate speech using DIA
        if input_data.clone_from_text and input_data.clone_from_audio:
            # For voice cloning, concatenate the clone text with generation text
            full_text = input_data.clone_from_text + input_data.text
            output = self.model.generate(
                full_text,
                audio_prompt=input_data.clone_from_audio.path,
                use_torch_compile=True,
                verbose=True
            )
        else:
            # Regular generation without cloning
            output = self.model.generate(
                input_data.text,
                use_torch_compile=True,
                verbose=True
            )

        # Get sample rate (assuming 44100 as in the example)
        output_sr = 44100

        # Adjust speed if needed
        if output is not None:
            original_len = len(output)
            # Ensure speed_factor is positive and not excessively small/large
            speed_factor = max(0.1, min(input_data.speed, 5))
            target_len = int(original_len / speed_factor)  # Target length based on speed_factor

            if target_len != original_len and target_len > 0:  # Only interpolate if length changes and is valid
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                output = np.interp(x_resampled, x_original, output)
                output = output.astype(np.float32)  # Ensure float32 type
                print(f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed.")
                try:
                    # Calculate pitch shift in semitones
                    # When we speed up (speed_factor > 1), we need to pitch down
                    # When we slow down (speed_factor < 1), we need to pitch up
                    n_steps = -12 * np.log2(speed_factor)  # Convert speed ratio to semitones
                    
                    # Apply pitch shift to compensate
                    output = librosa.effects.pitch_shift(
                        y=output,
                        sr=output_sr,
                        n_steps=n_steps,
                        bins_per_octave=12
                    )
                    
                    print(f"Adjusted audio speed to {speed_factor:.2f}x with pitch compensation")
                except Exception as e:
                    print(f"Warning: Pitch adjustment failed, using speed-only version: {e}")
                    # Fall back to speed-only version if pitch adjustment fails
                    output = output.astype(np.float32)
            else:
                print(f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")

            # Ensure the output is in the correct format and range
            output = np.clip(output, -1.0, 1.0)  # Clip to valid range
                
            print(f"Audio processing successful. Final shape: {output.shape}, Sample Rate: {output_sr}")
        else:
            print("Generation produced no valid output.")
            output = np.zeros(1, dtype=np.float32)
        
        # Save the audio in the requested format
        if input_data.format == AudioFormat.WAV:
            self.model.save_audio(output_path, output)
        else:
            # For other formats, first save as WAV and then convert
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            self.model.save_audio(temp_wav, output)
            
            # Convert to the requested format using pydub
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(output_path, format=input_data.format.value)
            os.remove(temp_wav)
        
        return AppOutput(audio=File(path=output_path))

    async def unload(self):
        """Clean up resources."""
        # Clear the model from memory
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()