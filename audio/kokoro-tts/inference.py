from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from enum import Enum
import os
import soundfile as sf
import tempfile
from kokoro import KPipeline
from pydub import AudioSegment
import numpy as np
import torch

# Define the supported voice types with descriptive human-readable values
class VoiceType(str, Enum):
    # American English voices
    AMERICAN_FEMALE_HEART = "American Female (Heart)"     # Highest quality American female voice
    AMERICAN_FEMALE_BELLA = "American Female (Bella)"     # High quality American female voice
    AMERICAN_FEMALE_NICOLE = "American Female (Nicole)"   # American female voice with headphones
    AMERICAN_FEMALE_KORE = "American Female (Kore)"       # American female voice
    AMERICAN_FEMALE_SARAH = "American Female (Sarah)"     # American female voice
    AMERICAN_MALE_FENRIR = "American Male (Fenrir)"       # American male voice
    AMERICAN_MALE_MICHAEL = "American Male (Michael)"     # American male voice
    AMERICAN_MALE_PUCK = "American Male (Puck)"           # American male voice
    
    # British English voices
    BRITISH_FEMALE_EMMA = "British Female (Emma)"         # Best British female voice
    BRITISH_MALE_FABLE = "British Male (Fable)"           # British male voice
    BRITISH_MALE_GEORGE = "British Male (George)"         # British male voice
    
    # Japanese voices
    JAPANESE_FEMALE_ALPHA = "Japanese Female (Alpha)"     # Japanese female voice
    JAPANESE_MALE_KUMO = "Japanese Male (Kumo)"           # Japanese male voice
    
    # Mandarin Chinese voices
    CHINESE_FEMALE_XIAOBEI = "Chinese Female (Xiaobei)"   # Mandarin Chinese female voice
    CHINESE_MALE_YUNJIAN = "Chinese Male (Yunjian)"       # Mandarin Chinese male voice
    
    # Spanish voices
    SPANISH_FEMALE_DORA = "Spanish Female (Dora)"         # Spanish female voice
    SPANISH_MALE_ALEX = "Spanish Male (Alex)"             # Spanish male voice
    
    # French voice
    FRENCH_FEMALE_SIWIS = "French Female (Siwis)"         # French female voice
    
    # Hindi voices
    HINDI_FEMALE_ALPHA = "Hindi Female (Alpha)"           # Hindi female voice
    HINDI_MALE_OMEGA = "Hindi Male (Omega)"               # Hindi male voice
    
    # Italian voices
    ITALIAN_FEMALE_SARA = "Italian Female (Sara)"         # Italian female voice
    ITALIAN_MALE_NICOLA = "Italian Male (Nicola)"         # Italian male voice
    
    # Brazilian Portuguese voices
    PORTUGUESE_FEMALE_DORA = "Portuguese Female (Dora)"   # Brazilian Portuguese female voice
    PORTUGUESE_MALE_ALEX = "Portuguese Male (Alex)"       # Brazilian Portuguese male voice

# Define the supported audio formats
class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"

class AppInput(BaseAppInput):
    text: str = Field(..., description="The text to convert to speech")
    voice: VoiceType = Field(default=VoiceType.AMERICAN_FEMALE_HEART, description="The voice type to use")
    format: AudioFormat = Field(default=AudioFormat.WAV, description="The output audio format")
    speed: float = Field(default=1.0, description="Speech speed (0.5 to 2.0)")

class AppOutput(BaseAppOutput):
    audio: File = Field(..., description="The generated audio file")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Kokoro TTS model."""
        
        # Map voice types to their corresponding Kokoro voice IDs
        self.voice_id_map = {
            VoiceType.AMERICAN_FEMALE_HEART: "af_heart",
            VoiceType.AMERICAN_FEMALE_BELLA: "af_bella",
            VoiceType.AMERICAN_FEMALE_NICOLE: "af_nicole",
            VoiceType.AMERICAN_FEMALE_KORE: "af_kore",
            VoiceType.AMERICAN_FEMALE_SARAH: "af_sarah",
            VoiceType.AMERICAN_MALE_FENRIR: "am_fenrir",
            VoiceType.AMERICAN_MALE_MICHAEL: "am_michael",
            VoiceType.AMERICAN_MALE_PUCK: "am_puck",
            VoiceType.BRITISH_FEMALE_EMMA: "bf_emma",
            VoiceType.BRITISH_MALE_FABLE: "bm_fable",
            VoiceType.BRITISH_MALE_GEORGE: "bm_george",
            VoiceType.JAPANESE_FEMALE_ALPHA: "jf_alpha",
            VoiceType.JAPANESE_MALE_KUMO: "jm_kumo",
            VoiceType.CHINESE_FEMALE_XIAOBEI: "zf_xiaobei",
            VoiceType.CHINESE_MALE_YUNJIAN: "zm_yunjian",
            VoiceType.SPANISH_FEMALE_DORA: "ef_dora",
            VoiceType.SPANISH_MALE_ALEX: "em_alex",
            VoiceType.FRENCH_FEMALE_SIWIS: "ff_siwis",
            VoiceType.HINDI_FEMALE_ALPHA: "hf_alpha",
            VoiceType.HINDI_MALE_OMEGA: "hm_omega",
            VoiceType.ITALIAN_FEMALE_SARA: "if_sara",
            VoiceType.ITALIAN_MALE_NICOLA: "im_nicola",
            VoiceType.PORTUGUESE_FEMALE_DORA: "pf_dora",
            VoiceType.PORTUGUESE_MALE_ALEX: "pm_alex"
        }

        # Map voice prefixes to language codes for KPipeline
        self.lang_code_map = {
            "af": "a",  # American English
            "am": "a",
            "bf": "b",  # British English
            "bm": "b",
            "jf": "j",  # Japanese
            "jm": "j",
            "zf": "z",  # Mandarin Chinese
            "zm": "z",
            "ef": "e",  # Spanish
            "em": "e",
            "ff": "f",  # French
            "fm": "f",
            "hf": "h",  # Hindi
            "hm": "h",
            "if": "i",  # Italian
            "im": "i",
            "pf": "p",  # Brazilian Portuguese
            "pm": "p"
        }
        
        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        
        # Pre-load pipelines for all supported languages
        self.pipelines = {}
        for lang_code in ["a", "b"]: 
            self.pipelines[lang_code] = KPipeline(lang_code=lang_code)
        
        # Add custom pronunciation for "kokoro"
        if "a" in self.pipelines:
            self.pipelines["a"].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
        if "b" in self.pipelines:
            self.pipelines["b"].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
        
        print("Kokoro TTS model initialized successfully")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate speech from text using Kokoro TTS."""
        # Get the voice ID for the selected voice
        voice_id = self.voice_id_map[input_data.voice]
        
        # Get the language code by extracting the first two characters of the voice ID
        voice_prefix = voice_id[:2]
        lang_code = self.lang_code_map[voice_prefix]
        
        # Get or initialize the pipeline for this language
        if lang_code not in self.pipelines:
            self.pipelines[lang_code] = KPipeline(lang_code=lang_code)
        
        pipeline = self.pipelines[lang_code]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{input_data.format.value}", delete=False) as temp_file:
            output_path = temp_file.name
        
        # Generate speech using the voice ID (not the enum value)
        generator = pipeline(
            input_data.text,
            voice=voice_id,  # Use the mapped voice ID here
            speed=input_data.speed,
            split_pattern=r'\n+'
        )
        
        # Process the generated audio chunks
        full_audio = []
        for _, _, audio in generator:
            full_audio.append(audio)
        
        # Combine all audio chunks
        combined_audio = np.concatenate(full_audio)

        # Save the audio in the requested format
        if input_data.format == AudioFormat.WAV:
            sf.write(output_path, combined_audio, 24000)
        else:
            # For other formats, first save as WAV and then convert
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(temp_wav, combined_audio, 24000)
            
            # Convert to the requested format using pydub
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(output_path, format=input_data.format.value)
            os.remove(temp_wav)
        
        return AppOutput(audio=File(path=output_path))

    async def unload(self):
        """Clean up resources."""
        # Clear any loaded pipelines
        self.pipelines = {}