# Model: Reve (Consolidated)

Consolidates three fal.ai endpoints:
- `fal-ai/reve/text-to-image` - Generate images from text
- `fal-ai/reve/edit` - Edit existing images with prompts
- `fal-ai/reve/remix` - Style remix of existing images

## Endpoints

| Mode | Endpoint | Trigger |
|------|----------|---------|
| text-to-image | `fal-ai/reve/text-to-image` | No image provided (or mode=text-to-image) |
| edit | `fal-ai/reve/edit` | Image provided (default) |
| remix | `fal-ai/reve/remix` | Image provided + mode=remix |

## Category
text-to-image / image-to-image

## Description
Reve generates detailed visual output with strong aesthetic quality and accurate text rendering. Supports text-to-image generation, image editing, and style remix.

## Input Schema

### Required Fields
- `prompt` (string): Text description for generation or editing

### Optional Fields
- `image` (File): Input image for edit/remix modes
- `mode` (enum): Operation mode - auto, edit, remix, text-to-image (default: auto)
- `output_format` (enum): png, jpeg, webp (default: png)

## Output Schema
- `images` (array of File): Generated/edited images

## Modes

### Text-to-Image
No image provided, generates from prompt only.
```json
{"prompt": "A serene mountain landscape at sunset"}
```

### Edit
Image provided, applies prompt-based edits.
```json
{
  "prompt": "Add a rainbow in the sky",
  "image": {"uri": "https://..."}
}
```

### Remix
Image provided with mode=remix, style transfer.
```json
{
  "prompt": "In the style of Van Gogh",
  "image": {"uri": "https://..."},
  "mode": "remix"
}
```

## Notes
- Auto mode detects: no image → text-to-image, with image → edit
- All modes use same pricing (per compute second)
