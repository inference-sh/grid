# Model: fal-ai/patina/material

## Endpoint
`fal-ai/patina/material`

## Category


## Description


## Input Schema

### Required Fields
- `prompt` (string): The text prompt describing the material/texture to generate. (example: mossy stone wall)

### Optional Fields
- `image_size` (unknown, default: square_hd): Output texture dimensions in pixels.
- `num_inference_steps` (integer, default: 8): Number of denoising steps for texture generation.
- `seed` (unknown): Random seed for reproducible generation.
- `num_images` (integer, default: 1): Number of texture images to generate.
- `enable_prompt_expansion` (boolean, default: True): Expand prompt with an LLM for richer detail. Adds ~0.0025 credits.
- `enable_safety_checker` (boolean, default: True): Enable the safety checker for generated images.
- `tiling_mode` (string, default: both): Tiling direction: 'both' (omnidirectional), 'horizontal', or 'vertical'.
- `tile_size` (integer, default: 128): Tile size in latent space (64 = 512px, 128 = 1024px).
- `tile_stride` (integer, default: 64): Tile stride in latent space.
- `image_url` (unknown): URL of an input image for image-to-image or inpainting. When provided without mask_url, performs image-to-image; with mask_url, performs inpainting.
- `mask_url` (unknown): URL of a mask image for inpainting. White regions are regenerated, black regions are preserved. Requires image_url.
- `strength` (number, default: 0.75): How much to transform the input image. Only used when image_url is provided.
- `maps` (array, default: ['basecolor', 'normal', 'roughness', 'metalness', 'height']): Which PBR maps to predict. Deselect all to skip PBR estimation entirely. Defaults to all five.
- `upscale_factor` (integer, default: 0): Upscale factor for predicted PBR maps via SeedVR seamless upscaling. 0 = no upscaling, 2 = 2× resolution, 4 = 4× resolution. The base texture image is not upscaled.
- `output_format` (string, default: png): Output image format for textures and PBR maps.

## Output Schema
- `images` (array): Generated tileable texture image(s) from z-image and predicted PBR material maps from PATINA.
- `seed` (integer): Seed used for texture generation.
- `prompt` (string): The prompt used for texture generation (possibly expanded).
- `timings` (object): End-to-end timing breakdown (seconds).

## Notes
- [Add implementation notes here]
