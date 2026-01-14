from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, BaseAppSetup, File, OutputMeta, VideoMeta
import os
import torch
import numpy as np
from typing import Optional, List
from pydantic import BaseModel, Field
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class LoraConfig(BaseModel):
    """Configuration for a single LoRA adapter."""
    adapter_name: str = Field(description="Name for the LoRA adapter")
    lora_url: Optional[str] = Field(
        default=None,
        description="URL to LoRA file (.safetensors), or HuggingFace repo path (e.g. 'Lightricks/LTX-2-19b-IC-LoRA-Canny-Control')"
    )
    lora_file: Optional[File] = Field(
        default=None,
        description="Uploaded LoRA file (.safetensors)"
    )
    lora_multiplier: float = Field(
        default=1.0,
        description="Multiplier for the LoRA effect (0.0-2.0 typical)"
    )
    fuse: bool = Field(
        default=False,
        description="Whether to fuse the LoRA into the model for faster inference"
    )


class AppSetup(BaseAppSetup):
    """Setup configuration for LTX-2 Video Generation.
    
    LTX 2.0 uses a lot of memory; CPU offloading is enabled by default.
    """
    # TODO: Uncomment when diffusers PR #12934 is merged
    # use_distilled: bool = Field(
    #     default=False,
    #     description="Use distilled model for faster inference (requires fewer steps)"
    # )
    pass


class RunInput(BaseAppInput):
    """Input schema for LTX-2 video generation.
    
    Supports:
    - Text-to-Video (T2V): Just provide prompt
    - Image-to-Video (I2V): Provide start_frame + prompt
    - Long Video Generation: Enable long_video mode with multi-prompt scheduling
    """
    prompt: str = Field(
        description="Text prompt to guide video generation. For long videos, use '|' to separate prompts for different segments (e.g. 'scene 1|scene 2|scene 3')"
    )
    negative_prompt: Optional[str] = Field(
        default="shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static.",
        description="Negative prompt to specify undesired features"
    )
    start_frame: Optional[File] = Field(
        default=None,
        description="Optional start frame image for Image-to-Video (I2V) generation. When provided, the I2V pipeline is used instead of T2V."
    )
    width: int = Field(
        default=768,
        description="Width of the output video frames"
    )
    height: int = Field(
        default=512,
        description="Height of the output video frames"
    )
    num_frames: int = Field(
        default=121,
        description="Number of frames to generate. Max ~20 seconds duration (e.g. 481 frames at 24fps, 1001 at 50fps). Default 121 (~5 seconds at 24fps)."
    )
    frame_rate: float = Field(
        default=24.0,
        description="Frame rate for the output video"
    )
    num_inference_steps: int = Field(
        default=40,
        description="Number of denoising steps. Use 8 for distilled models."
    )
    guidance_scale: float = Field(
        default=4.0,
        description="Scale for classifier-free guidance. Use 1.0 for distilled models."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. If not provided, a random seed is used."
    )
    # Long video options - NOT YET AVAILABLE FOR LTX-2
    # The LTXI2VLongMultiPromptPipeline is for LTX-1 (0.9.8), not LTX-2
    # TODO: Uncomment when LTX-2 long video support is added
    # long_video: bool = Field(
    #     default=False,
    #     description="Enable long video generation with sliding windows and multi-prompt scheduling"
    # )
    # temporal_tile_size: int = Field(
    #     default=120,
    #     description="Size of temporal tiles for long video generation"
    # )
    # temporal_overlap: int = Field(
    #     default=32,
    #     description="Overlap between temporal tiles for smooth transitions"
    # )
    # adain_factor: float = Field(
    #     default=0.25,
    #     description="AdaIN normalization factor for color/contrast consistency across windows"
    # )
    # LoRA options
    loras: Optional[List[LoraConfig]] = Field(
        default=None,
        description="List of LoRA adapters to apply"
    )


class RunOutput(BaseAppOutput):
    """Output schema for LTX-2 video generation."""
    video: File = Field(description="Generated video file with synced audio")
    output_meta: OutputMeta = Field(description="Usage metadata for pricing")


def load_lora_adapter(pipeline, lora_config: LoraConfig) -> bool:
    """Load a LoRA adapter from config."""
    try:
        if lora_config.lora_file:
            # Direct file upload
            pipeline.load_lora_weights(lora_config.lora_file.path, adapter_name=lora_config.adapter_name)
            logger.info(f"Loaded LoRA adapter '{lora_config.adapter_name}' from file")
        elif lora_config.lora_url:
            # URL or HuggingFace repo
            if lora_config.lora_url.endswith('.safetensors'):
                # Direct safetensors file
                pipeline.load_lora_weights(lora_config.lora_url, adapter_name=lora_config.adapter_name)
            elif '/' in lora_config.lora_url and not lora_config.lora_url.startswith('http'):
                # HuggingFace repo format
                pipeline.load_lora_weights(lora_config.lora_url, adapter_name=lora_config.adapter_name)
            else:
                logger.warning(f"Unsupported LoRA URL format: {lora_config.lora_url}")
                return False
            logger.info(f"Loaded LoRA adapter '{lora_config.adapter_name}' from URL")
        
        if lora_config.fuse:
            pipeline.fuse_lora(components=["transformer"], lora_scale=lora_config.lora_multiplier)
            logger.info(f"Fused LoRA adapter '{lora_config.adapter_name}'")
        
        return True
    except Exception as e:
        logger.error(f"Failed to load LoRA adapter '{lora_config.adapter_name}': {e}")
        return False


class App(BaseApp):
    """LTX 2.0 Video Generation App.
    
    LTX 2.0 is an audio-video foundation model that generates videos with synced audio.
    Supports:
    - Text-to-Video (T2V): Generate video from text prompt
    - Image-to-Video (I2V): Generate video from start frame image and text prompt
    - Long Video Generation: Multi-prompt sliding window generation for longer videos
    - LoRA Support: Load custom LoRA adapters including IC-LoRAs for control
    """
    
    async def setup(self, config: AppSetup):
        """Initialize the LTX-2 model with CPU offloading enabled."""
        from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2ImageToVideoPipeline
        
        # Determine model to use
        # TODO: Uncomment when diffusers PR #12934 is merged
        # if config.use_distilled:
        #     model_id = "Lightricks/LTX-Video-0.9.8-13B-distilled"
        # else:
        #     model_id = "Lightricks/LTX-2"
        # self.use_distilled = config.use_distilled
        model_id = "Lightricks/LTX-2"
        
        # Load the T2V pipeline
        self.t2v_pipe = LTX2Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        )
        self.t2v_pipe.enable_model_cpu_offload()
        
        # Load the I2V pipeline
        self.i2v_pipe = LTX2ImageToVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        )
        self.i2v_pipe.enable_model_cpu_offload()
        
        # Long video pipeline - NOT YET AVAILABLE FOR LTX-2
        # The LTXI2VLongMultiPromptPipeline is for LTX-1 (0.9.8), not LTX-2
        # TODO: Uncomment when LTX-2 long video support is added
        # try:
        #     from diffusers.pipelines.ltx import LTXI2VLongMultiPromptPipeline
        #     self.long_pipe = LTXI2VLongMultiPromptPipeline.from_pretrained(
        #         model_id,
        #         torch_dtype=torch.bfloat16
        #     )
        #     self.long_pipe.enable_model_cpu_offload()
        #     self.has_long_pipeline = True
        # except ImportError:
        #     logger.warning("LTXI2VLongMultiPromptPipeline not available in this diffusers version")
        #     self.has_long_pipeline = False
        self.has_long_pipeline = False
        
        # Track loaded LoRAs
        self.loaded_loras = {}
        
    async def run(self, input_data: RunInput) -> RunOutput:
        """Run video generation with LTX-2.
        
        Uses I2V pipeline when start_frame is provided, otherwise uses T2V pipeline.
        Uses long video pipeline when long_video is enabled.
        """
        from diffusers.pipelines.ltx2.export_utils import encode_video
        import random
        
        # Set seed for reproducibility
        seed = input_data.seed if input_data.seed is not None else random.randint(0, 2**32 - 1)
        generator = torch.Generator().manual_seed(seed)
        
        # Validate num_frames (max 20 seconds)
        max_duration_seconds = 20
        max_frames = int(max_duration_seconds * input_data.frame_rate) + 1
        if input_data.num_frames > max_frames:
            raise ValueError(
                f"num_frames ({input_data.num_frames}) exceeds maximum of {max_frames} "
                f"frames ({max_duration_seconds} seconds at {input_data.frame_rate} fps). "
                f"LTX-2 supports up to 20 seconds per generation."
            )
        
        # Determine which pipeline to use
        # Long video mode is NOT YET AVAILABLE for LTX-2
        # if input_data.long_video and input_data.start_frame and self.has_long_pipeline:
        #     pipe = self.long_pipe
        #     mode = "long"
        if input_data.start_frame is not None:
            pipe = self.i2v_pipe
            mode = "i2v"
        else:
            pipe = self.t2v_pipe
            mode = "t2v"
        
        # Load LoRAs if specified
        if input_data.loras:
            adapter_names = []
            adapter_weights = []
            for lora in input_data.loras:
                if lora.adapter_name not in self.loaded_loras:
                    success = load_lora_adapter(pipe, lora)
                    if success:
                        self.loaded_loras[lora.adapter_name] = True
                
                if lora.adapter_name in self.loaded_loras and not lora.fuse:
                    adapter_names.append(lora.adapter_name)
                    adapter_weights.append(lora.lora_multiplier)
            
            if adapter_names:
                pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        
        # Prepare common kwargs
        common_kwargs = {
            "prompt": input_data.prompt,
            "negative_prompt": input_data.negative_prompt,
            "width": input_data.width,
            "height": input_data.height,
            "num_frames": input_data.num_frames,
            "guidance_scale": input_data.guidance_scale,
            "generator": generator,
            "output_type": "np",
            "return_dict": False,
        }
        
        # TODO: Uncomment when diffusers PR #12934 is merged
        # Distilled models require specific sigmas instead of num_inference_steps
        # if self.use_distilled:
        #     common_kwargs["sigmas"] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
        # else:
        #     common_kwargs["num_inference_steps"] = input_data.num_inference_steps
        common_kwargs["num_inference_steps"] = input_data.num_inference_steps
        
        # Long video mode is NOT YET AVAILABLE for LTX-2
        # The code below was for LTX-1's LTXI2VLongMultiPromptPipeline
        # if mode == "long":
        #     # Long video with multi-prompt sliding windows
        #     image = Image.open(input_data.start_frame.path).convert("RGB")
        #     sigmas = [1.0, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250, 0.4219, 0.0]
        #     latents = pipe(
        #         prompt=input_data.prompt,
        #         negative_prompt=input_data.negative_prompt,
        #         width=input_data.width,
        #         height=input_data.height,
        #         num_frames=input_data.num_frames,
        #         temporal_tile_size=input_data.temporal_tile_size,
        #         temporal_overlap=input_data.temporal_overlap,
        #         sigmas=sigmas,
        #         guidance_scale=input_data.guidance_scale,
        #         cond_image=image,
        #         adain_factor=input_data.adain_factor,
        #         output_type="latent",
        #     ).frames
        #     video_pil = pipe.vae_decode_tiled(
        #         latents, 
        #         decode_timestep=0.05, 
        #         decode_noise_scale=0.025, 
        #         output_type="np"
        #     )[0]
        #     video = video_pil
        #     audio = None
        
        if mode == "i2v":
            # Image-to-Video (I2V) mode
            image = Image.open(input_data.start_frame.path).convert("RGB")
            common_kwargs["image"] = image
            common_kwargs["frame_rate"] = input_data.frame_rate
            video, audio = pipe(**common_kwargs)
            
        else:
            # Text-to-Video (T2V) mode
            common_kwargs["frame_rate"] = input_data.frame_rate
            video, audio = pipe(**common_kwargs)
        
        # Convert video to tensor format for encoding
        video = (video * 255).round().astype("uint8")
        video = torch.from_numpy(video)
        
        # Output path
        output_path = f"/tmp/ltx2_output_{seed}.mp4"
        
        # Encode video with audio if available
        if audio is not None:
            encode_video(
                video[0],
                fps=input_data.frame_rate,
                audio=audio[0].float().cpu(),
                audio_sample_rate=self.i2v_pipe.vocoder.config.output_sampling_rate,
                output_path=output_path,
            )
        else:
            # Fallback: encode without audio
            encode_video(
                video[0],
                fps=input_data.frame_rate,
                output_path=output_path,
            )
        
        # Calculate duration for pricing
        duration_seconds = input_data.num_frames / input_data.frame_rate
        
        return RunOutput(
            video=File(path=output_path),
            output_meta=OutputMeta(
                video=VideoMeta(duration_seconds=duration_seconds)
            )
        )
