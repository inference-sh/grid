# Wan 2.5 Image-to-Video App

This inference.sh app provides an interface to the fal.ai Wan 2.5 image-to-video model, allowing you to generate high-quality videos from static images using state-of-the-art AI.

## Overview

The app acts as a bridge between the inference.sh platform and fal.ai's Wan 2.5 model, handling file uploads, API communication, and video download seamlessly.

## Features

- **High-quality video generation** from static images
- **Multiple resolution options**: 480p, 720p, 1080p
- **Audio support**: Optional background music integration
- **Prompt enhancement**: Automatic prompt expansion using LLM
- **Negative prompts**: Specify content to avoid
- **Reproducible results**: Seed-based generation
- **Progress logging**: Real-time generation updates

## Setup

### 1. API Key Configuration

You need a fal.ai API key to use this app. You can provide it in two ways:

#### Option 1: Environment Variable (Recommended)
Set your fal.ai API key in the `inf.yml` environment configuration:

```yaml
env:
    FAL_KEY: "your_fal_ai_api_key_here"
```

#### Option 2: Request Parameter
Provide the API key directly in your request:

```json
{
  "fal_api_key": "your_fal_ai_api_key_here",
  // ... other parameters
}
```

### 2. Getting a fal.ai API Key

1. Visit [fal.ai](https://fal.ai)
2. Create an account or log in
3. Navigate to your API keys section
4. Generate a new API key
5. Copy the key and configure it as described above

## Input Parameters

### Required Parameters

- **`prompt`** (string): Text describing the desired video motion (max 800 characters)
  - Example: "The white dragon warrior stands still, eyes full of determination and strength. The camera slowly moves closer or circles around the warrior, highlighting the powerful presence and heroic spirit of the character."

- **`image`** (File): Input image file to use as the first frame
  - Supported formats: JPEG, PNG, WebP
  - Can be a local file path or URL

### Optional Parameters

- **`audio`** (File, optional): Background audio file
  - Supported formats: WAV, MP3
  - Duration: 3-30 seconds
  - Max file size: 15MB
  - Audio will be truncated if longer than video duration

- **`resolution`** (string, default: "1080p"): Video resolution
  - Options: "480p", "720p", "1080p"
  - Higher resolutions provide better quality but take longer to generate

- **`negative_prompt`** (string, optional): Content to avoid (max 500 characters)
  - Example: "low resolution, error, worst quality, low quality, defects"

- **`enable_prompt_expansion`** (boolean, default: true): Enable LLM-based prompt rewriting
  - Improves prompt quality and generation results

- **`seed`** (integer, optional): Random seed for reproducibility
  - If not provided, a random seed will be chosen

- **`fal_api_key`** (string, optional): fal.ai API key
  - Only needed if not set as environment variable

## Output

The app returns:

- **`video`** (File): Generated MP4 video file
- **`seed`** (integer): Seed used for generation (for reproducibility)
- **`actual_prompt`** (string, optional): The enhanced prompt used (if prompt expansion was enabled)

## Example Usage

### Basic Example

```json
{
  "prompt": "A serene lake with gentle ripples, surrounded by mountains. The camera slowly pans across the water surface, capturing the peaceful movement and reflections.",
  "image": {
    "path": "https://example.com/lake-image.jpg"
  },
  "resolution": "1080p"
}
```

### Advanced Example with Audio

```json
{
  "prompt": "The white dragon warrior stands still, eyes full of determination and strength. The camera slowly moves closer or circles around the warrior, highlighting the powerful presence and heroic spirit of the character.",
  "image": {
    "path": "/path/to/dragon-warrior.jpg"
  },
  "audio": {
    "path": "/path/to/epic-music.mp3"
  },
  "resolution": "1080p",
  "negative_prompt": "low resolution, error, worst quality, low quality, defects",
  "enable_prompt_expansion": true,
  "seed": 12345
}
```

## Deployment

### Local Testing

1. Set up your API key in `inf.yml` or prepare to include it in requests
2. Install dependencies: `pip install -r requirements.txt`
3. Test with the provided example:

```bash
infsh run input.json
```

### Production Deployment

```bash
infsh deploy
```

## Tips for Best Results

### Prompt Writing
- Be specific about camera movements and visual effects
- Describe the mood and atmosphere you want
- Mention lighting conditions and environmental details
- Keep prompts under 800 characters for best performance

### Image Selection
- Use high-quality images with clear subjects
- Images with good contrast and lighting work best
- Avoid overly complex compositions for first-time use

### Resolution Choice
- **480p**: Fastest generation, suitable for previews
- **720p**: Good balance between quality and speed
- **1080p**: Best quality, longer generation time

### Audio Integration
- Keep audio files under 15MB
- WAV format recommended for best quality
- Ensure audio duration matches desired video length

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your fal.ai API key is valid and active
   - Check that the key is properly set in environment or request

2. **File Upload Errors**
   - Verify image and audio files exist and are accessible
   - Check file formats are supported
   - Ensure audio files are under 15MB

3. **Generation Failures**
   - Try simpler prompts if generation fails
   - Reduce resolution if experiencing timeouts
   - Check fal.ai service status

4. **Network Issues**
   - Ensure stable internet connection for API calls
   - Check firewall settings allow outbound connections

### Debug Logging

The app provides detailed logging during generation:
- File upload progress
- API communication status
- Generation progress from fal.ai
- Download completion status

## Resource Requirements

- **RAM**: 4GB minimum (for file handling and downloads)
- **Storage**: Sufficient space for temporary video files
- **Network**: Stable internet connection for API calls
- **GPU**: Not required (processing handled by fal.ai)

## Cost Considerations

Video generation costs are handled by fal.ai based on:
- Resolution selected
- Video duration (typically 5-10 seconds)
- API usage

Check fal.ai pricing for current rates.

## Support

For issues related to:
- **This app**: Check the troubleshooting section above
- **fal.ai API**: Visit [fal.ai documentation](https://docs.fal.ai)
- **inference.sh platform**: Check the platform documentation

## License

This app is designed for use with the inference.sh platform and requires a valid fal.ai API key for operation.