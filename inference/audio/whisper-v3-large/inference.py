import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional, Union

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
        default="none",
        description="Whether to return timestamps. Use 'none' for no timestamps, 'word' for word-level timestamps, or 'sentence' for sentence-level timestamps.",
        enum=["none", "word", "sentence"]
    )

    batch_size: int = Field(
        default=16,
        description="Batch size for processing long audio files",
        ge=1,
        le=32
    )

class AppOutput(BaseAppOutput):
    text: str = Field(description="The transcribed/translated text")
    chunks: Optional[list] = Field(description="Timestamp chunks if requested", default=None)

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize Whisper model with optimizations"""
        self.device = "cuda"
        torch.set_float32_matmul_precision("high")
        
        model_id = "openai/whisper-large-v3"
        torch_dtype = torch.float16

        # Initialize model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa"
        ).to(self.device)

        # Enable static cache and compile
        #self.model.generation_config.cache_implementation = "static"
        #self.model.forward = torch.compile(
        #    self.model.forward,
        #    mode="reduce-overhead",
        #    fullgraph=True
        #)

        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            torch_dtype=torch_dtype,
            device=self.device,
        )

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run transcription/translation on the input audio"""
        # Prepare generation kwargs
        generate_kwargs = {
            "task": input_data.task,
            "language": input_data.language if input_data.language != "auto" else None,
        }

        timestamps = True if input_data.return_timestamps == "sentence" else ("word" if input_data.return_timestamps == "word" else False)
        
        # Run inference
        result = self.pipe(
            input_data.audio.path,
            batch_size=input_data.batch_size,
            return_timestamps=timestamps,
            generate_kwargs=generate_kwargs
        )

        # Return results
        if timestamps:
            return AppOutput(
                text=result["text"],
                chunks=result["chunks"]
            )
        else:
            return AppOutput(text=result["text"])

    async def unload(self):
        """Clean up resources"""
        del self.model
        del self.processor
        del self.pipe
        torch.cuda.empty_cache()