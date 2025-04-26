import os
import sys
sys.path.append(os.path.dirname(__file__))  # Add just the current directory

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field

from magi.pipeline import MagiPipeline
from typing import Optional
from enum import Enum

# Set environment variables for MAGI pipeline
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "6009" 
os.environ["GPUS_PER_NODE"] = "1"
os.environ["NNODES"] = "1"
os.environ["WORLD_SIZE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ["PAD_HQ"] = "1"
os.environ["PAD_DURATION"] = "1"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OFFLOAD_T5_CACHE"] = "true"
os.environ["OFFLOAD_VAE_CACHE"] = "true"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9;9.0"


class Mode(str, Enum):
    T2V = "t2v"
    I2V = "i2v"
    V2V = "v2v"

class AppInput(BaseAppInput):
    mode: Mode = Field(description="The mode of the pipeline")
    prompt: str = Field(description="The prompt for the pipeline. Used for all modes.")
    image_path: Optional[str] = Field(None, description="For i2v mode, the source image path")
    prefix_video_path: Optional[str] = Field(None, description="For v2v mode, the source video path")

class AppOutput(BaseAppOutput):
    video: File

class App(BaseApp):
    async def setup(self):
        """Initialize the MAGI-1 pipeline."""
        config_file = os.path.join(os.path.dirname(__file__), "example/4.5B/4.5B_config.json")
        self.pipeline = MagiPipeline(config_file)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run the appropriate MAGI pipeline based on input mode."""
        output_path = "/tmp/output.mp4"
        
        if input_data.mode == Mode.T2V:
            self.pipeline.run_text_to_video(
                prompt=input_data.prompt,
                output_path=output_path
            )
        elif input_data.mode == Mode.I2V:
            if not input_data.image_path:
                raise ValueError("image_path is required for i2v mode")
            self.pipeline.run_image_to_video(
                prompt=input_data.prompt,
                image_path=input_data.image_path,
                output_path=output_path
            )
        elif input_data.mode == Mode.V2V:
            if not input_data.prefix_video_path:
                raise ValueError("prefix_video_path is required for v2v mode")
            self.pipeline.run_video_to_video(
                prompt=input_data.prompt,
                prefix_video_path=input_data.prefix_video_path,
                output_path=output_path
            )
        
        return AppOutput(video=File(path=output_path))

    async def unload(self):
        """Clean up resources."""
        # Add any cleanup code if needed
        pass