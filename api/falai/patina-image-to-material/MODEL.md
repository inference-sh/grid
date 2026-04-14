# Model: fal-ai/patina

## Endpoint
`fal-ai/patina`

## Category
image-to-image

## Description
PATINA creates seamless high-resolution normal, roughness, basecolor (albedo), height (displacement) and metalness maps from images

## Input Schema

### Required Fields
- `image_url` (string): URL of the input image (photograph or render). (example: https://storage.googleapis.com/falserverless/gallery/patina-blog-hero-render.png)

### Optional Fields
- `maps` (array, default: ['basecolor', 'normal', 'roughness', 'metalness', 'height']): Which PBR maps to predict. Defaults to all five.
- `seed` (unknown): Random seed for reproducible denoising. If not set, a random seed is used.
- `sync_mode` (boolean, default: False): If True, return images as data URIs instead of CDN URLs.
- `enable_safety_checker` (boolean, default: True): Enable the safety checker for images.
- `output_format` (string, default: png): Output image format.

## Output Schema
- `images` (array): Predicted PBR material maps.
- `timings` (object): Timing breakdown (seconds).
- `seed` (integer): The seed used for denoising.

## Notes
- [Add implementation notes here]
