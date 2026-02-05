# Wan 2.1 Image-to-Video Generation App

This app implements the Wan 2.1 fast image-to-video generation model, based on the [multimodalart/wan2-1-fast](https://huggingface.co/spaces/multimodalart/wan2-1-fast) HuggingFace space.

## Features

- **Fast 4-step generation** using CausVid LoRA weights
- **14B parameter model** for high-quality video generation
- **Image-to-Video animation** from static images
- **Customizable parameters** for motion control and quality
- **GPU optimized** for efficient inference

## Requirements

- NVIDIA GPU with at least 13GB VRAM (for 14B model)
- CUDA support
- Python 3.8+

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have CUDA properly configured for PyTorch.

## Model Details

- **Model**: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
- **LoRA**: Kijai/WanVideo_comfy (CausVid 4-step generation)
- **Resolution**: Up to 896x896 pixels
- **Frame Rate**: 24 FPS
- **Duration**: 0.3 to 3.4 seconds

## Input Parameters

- `input_image`: Image file to animate (PNG, JPG, etc.)
- `prompt`: Text description of desired motion/animation
- `height`: Target video height (128-896, adjusted to multiple of 32)
- `width`: Target video width (128-896, adjusted to multiple of 32)  
- `negative_prompt`: What to avoid in the generation
- `duration_seconds`: Video length in seconds (0.3-3.4)
- `guidance_scale`: How closely to follow the prompt (0.0-20.0)
- `steps`: Number of inference steps (1-30, default 4 for fast generation)
- `seed`: Random seed for reproducible results
- `randomize_seed`: Whether to use random seed instead

## Output

- `video_output`: Generated MP4 video file
- `seed_used`: The actual seed used for generation

## Usage Example

The app takes an input image and animates it based on your text prompt. For example:

- **Input**: Portrait photo
- **Prompt**: "gentle head movement, natural breathing, cinematic lighting"
- **Output**: Animated video of the portrait with subtle motion

## Performance Notes

- First run will download the model (~13GB)
- Generation typically takes 10-30 seconds depending on settings
- Higher steps and longer duration will take more time
- Memory optimization is built-in for consumer GPUs

## Model License

Please check the respective model licenses:
- [Wan 2.1 Model License](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers)
- [CausVid LoRA License](https://huggingface.co/Kijai/WanVideo_comfy) 