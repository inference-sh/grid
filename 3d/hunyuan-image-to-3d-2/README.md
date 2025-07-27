# Hunyuan3D-2: Image/Text to 3D Generation

High-resolution 3D asset generation using Tencent's Hunyuan3D-2 diffusion models. This app supports both image-to-3D and text-to-3D generation with multiple model variants optimized for different performance and quality requirements.

## Features

- **Image-to-3D**: Convert any image to a high-quality 3D model
- **Text-to-3D**: Generate 3D models directly from text descriptions  
- **Multiple Model Variants**: Choose from 6 different model configurations
- **Texture Painting**: Optional high-quality texture generation
- **Post-processing**: Automatic mesh cleanup and optimization

## Model Variants

### Mini (600M parameters)
- **mini**: Standard mini model for fast generation
- **mini_turbo**: Accelerated version with FlashVDM for even faster generation

### Multi-View (1.1B parameters)  
- **mv**: Multi-view model for better 3D consistency
- **mv_turbo**: Accelerated multi-view model with FlashVDM

### Standard (1.1B parameters)
- **standard**: Full-featured model for highest quality
- **standard_turbo**: Accelerated standard model with FlashVDM

## Usage

### Input Parameters

- **prompt** (optional): Text description to guide 3D generation
- **input_image** (optional): Input image to convert to 3D model
- **num_inference_steps**: Number of denoising steps (1-100, default: 30)
- **seed**: Random seed for reproducible results (default: 2025)
- **paint_texture**: Whether to generate textures (default: true)

### Output

- **result**: Generated 3D model in GLB format

## Requirements

- CUDA-compatible GPU with 8GB+ VRAM (mini) or 12GB+ VRAM (mv/standard)
- 16GB+ system RAM
- Python 3.11

## Model Information

Based on [Tencent Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) - scaling diffusion models for high-resolution textured 3D asset generation.

## Citation

```bibtex
@misc{hunyuan3d22025tencent,
    title={Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation},
    author={Tencent Hunyuan3D Team},
    year={2025},
    eprint={2501.12202},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
