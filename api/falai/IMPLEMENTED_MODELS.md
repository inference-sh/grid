# Implemented fal.ai Models

This file tracks which fal.ai models have been implemented as inference.sh apps.

## Format

Each entry: `our-app-name` -> `fal-endpoint-id` (category)

## Implemented Models

| Our App | fal.ai Endpoint | Category |
|---------|-----------------|----------|
| dia-tts | fal-ai/dia-tts | text-to-speech |
| fabric-1-0 | fal-ai/fabric | image-to-image |
| flux-2-dev | fal-ai/flux/dev | text-to-image |
| flux-2-dev-turbo | fal-ai/flux/dev/turbo | text-to-image |
| flux-2-klein-lora | fal-ai/flux-2-klein-lora | text-to-image |
| flux-dev-lora | fal-ai/flux-lora | text-to-image |
| imagine-art-1-5-pro-preview | fal-ai/imagineart/v1.5/pro/preview | text-to-image |
| nano-banana | fal-ai/nano-banana | text-to-image |
| nano-banana-pro | fal-ai/nano-banana-pro | text-to-image |
| nano-banana-pro-edit | fal-ai/nano-banana-pro/edit | image-to-image |
| omni-human-1-5 | fal-ai/omni-human-1.5 | image-to-video |
| pixverse-lipsync | fal-ai/pixverse/lipsync | video-lipsync |
| reve | fal-ai/reve | text-to-image |
| seedance-1-5-pro-i2v | fal-ai/bytedance/seedance/v1.5/pro/image-to-video | image-to-video |
| seedance-1-5-pro-t2v | fal-ai/bytedance/seedance/v1.5/pro/text-to-video | text-to-video |
| seedream-v4-edit | fal-ai/seedream/v4/edit | image-to-image |
| topaz-image-upscaler | fal-ai/topaz/image-upscaler | image-upscale |
| topaz-video-upscaler | fal-ai/topaz/video-upscaler | video-upscale |
| wan-2-5 | fal-ai/wan/v2.5/text-to-video | text-to-video |
| wan-2-5-i2v | fal-ai/wan/v2.5/image-to-video | image-to-video |

## Last Updated

2026-02-05

## Notes

- Use `fal-model-search` skill to discover new models
- Use `fal-add-model` skill to implement new models
- Some fal.ai models have multiple functions (e.g., text-to-image + image-to-image) - we consolidate these into single apps where practical
