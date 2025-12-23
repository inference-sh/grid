# Qwen Image Edit Lightning Plus

Fast multi-image editing using Qwen-Image-Edit-2509 with Lightning LoRA for 8-step inference.

## Features

- **Multi-image editing**: Support for 1-3 input images
- **Fast inference**: 8 steps with Lightning LoRA (vs 40 steps default)
- **Flexible editing**: Text-guided image modifications
- **High quality**: Based on Qwen-Image-Edit-2509 model

## Model Information

- **Base Model**: [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- **LoRA**: [lightx2v/Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning)
- **License**: Apache 2.0

## Usage

### Input Parameters

- `images` (required): List of 1-3 input images for editing
- `prompt` (required): Text description of desired edit
- `negative_prompt` (optional): What to avoid in generation (default: " ")
- `num_inference_steps` (optional): Number of denoising steps, 1-50 (default: 8)
- `guidance_scale` (optional): Guidance scale, 0-10 (default: 1.0)
- `true_cfg_scale` (optional): True CFG scale, 0-10 (default: 4.0)
- `seed` (optional): Random seed for reproducibility (default: 42)

### Example Input

```json
{
  "images": [
    {"path": "path/to/image1.jpg"}
  ],
  "prompt": "a photo of a cat wearing sunglasses",
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "true_cfg_scale": 4.0,
  "seed": 42
}
```

### Multiple Images Example

```json
{
  "images": [
    {"path": "path/to/person.jpg"},
    {"path": "path/to/background.jpg"}
  ],
  "prompt": "combine the person with the background scene",
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "true_cfg_scale": 4.0,
  "seed": 42
}
```

## Output

Returns a single edited image in PNG format.

## Resource Requirements

- **VRAM**: 24GB GPU
- **RAM**: 32GB system memory
- **Python**: 3.10

## Local Testing

```bash
# Generate example input
infsh run

# Test with custom input
infsh run input.json
```

## Deployment

```bash
infsh deploy
```

## Use Cases

- Product image editing
- Person/object editing with consistency
- Multi-image composition
- Style transfer
- Background replacement
- Object addition/removal
- Text editing with custom fonts/colors

## Technical Details

- Uses `FlowMatchEulerDiscreteScheduler` for efficient sampling
- BFloat16 precision for optimal memory usage
- Lightning LoRA fused for fast 8-step generation
- Supports 1-3 input images for flexible editing scenarios
