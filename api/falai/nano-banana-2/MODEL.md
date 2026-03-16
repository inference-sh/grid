# Model: Nano Banana 2

## Endpoints
- Text-to-image: `fal-ai/nano-banana-2`
- Image editing: `fal-ai/nano-banana-2/edit`

## Category
text-to-image / image-to-image (consolidated)

## Description
Nano Banana 2 is Google's new state-of-the-art fast image generation and editing model.
This app consolidates both endpoints - when input images are provided, it uses edit mode.

## Input Schema

### Required Fields
- `prompt` (string, 3-50000 chars): The text prompt to generate or edit an image

### Optional Fields
- `images` (list of files): Input images for editing. If provided, enables edit mode.
- `num_images` (int, 1-4, default: 1): Number of images to generate
- `aspect_ratio` (enum, default: "auto"): auto, 21:9, 16:9, 3:2, 4:3, 5:4, 1:1, 4:5, 3:4, 2:3, 9:16, 4:1, 1:4, 8:1, 1:8
- `resolution` (enum, default: "1K"): 0.5K, 1K, 2K, 4K
- `output_format` (enum, default: "png"): jpeg, png, webp
- `safety_tolerance` (enum, default: "4"): 1 (most strict) to 6 (least strict)
- `seed` (int): Random seed for reproducibility
- `enable_web_search` (bool, default: false): Enable web search for latest info
- `thinking_level` (enum): "minimal" or "high" to enable model thinking

## Output Schema
- `images` (list of files): Generated images
- `description` (string): Text description from the model

## Notes
- Consolidation: text-to-image when no images, edit when images provided
- Supports extreme aspect ratios (4:1, 1:4, 8:1, 1:8)
- Web search can use latest information for generation
- Thinking mode includes model reasoning in generation
