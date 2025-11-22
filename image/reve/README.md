# Reve Image Editor App

This inference.sh app provides an interface to Reve's advanced image editing model, allowing you to transform existing images using natural language prompts.

## Overview

Reve's edit model lets you upload an existing image and then transform it via a text prompt. The app acts as a bridge between the inference.sh platform and the Reve model, handling file uploads, API communication, and image downloads seamlessly.

## Features

- **Image-to-image editing** using natural language prompts
- **Single image transformation** with precise control
- **Multiple output formats**: PNG, JPEG, WebP
- **Flexible output modes**: URLs or data URIs
- **Progress logging**: Real-time generation updates

## Setup

### API Key Configuration

You need a FAL API key to use this app. Set your API key in the `inf.yml` environment configuration:

```yaml
env:
    FAL_KEY: "your_api_key_here"
```

### Getting an API Key

1. Visit [fal.ai](https://fal.ai)
2. Create an account or log in
3. Navigate to your API keys section
4. Generate a new API key
5. Copy the key and configure it as described above

## Input Parameters

### Required Parameters

- **`prompt`** (string): The text description of how to edit the provided image
  - Example: "Give him a friend"
  - Example: "Turn the sky into a beautiful sunset"
  - Example: "Add flowers to the garden"

- **`image`** (File): The reference image to edit
  - Supported formats: PNG, JPEG, WebP, AVIF, and HEIF
  - Can be a local file path or URL
  - Must be publicly accessible if using URL

### Optional Parameters

- **`output_format`** (string, default: "png"): Output format for generated image
  - Options: "png", "jpeg", "webp"
  - PNG supports transparency and lossless quality
  - JPEG is smaller for photographs
  - WebP offers good compression with quality

- **`sync_mode`** (boolean, default: false): Output format preference
  - false: Returns URL to download image (recommended)
  - true: Returns image as base64 data URI (not available in request history)

## Output

The app returns:

- **`images`** (List[File]): The edited image

## Example Usage

### Basic Example

```json
{
  "prompt": "Give him a friend",
  "image": {
    "path": "https://v3b.fal.media/files/b/koala/sZE6zNTKjOKc4kcUdVlu__26bac54c-3e94-43e9-aeff-f2efc2631ef0.webp"
  }
}
```

### Local File Example

```json
{
  "prompt": "Turn the building into a futuristic skyscraper with glass walls",
  "image": {
    "path": "/path/to/your/building.jpg"
  },
  "output_format": "jpeg"
}
```

### Advanced Example with WebP Output

```json
{
  "prompt": "Add vibrant flowers throughout the garden",
  "image": {
    "path": "https://example.com/garden.jpg"
  },
  "output_format": "webp",
  "sync_mode": false
}
```

### Style Transfer Example

```json
{
  "prompt": "Make this look like a watercolor painting",
  "image": {
    "path": "/path/to/photo.png"
  },
  "output_format": "png"
}
```

## Deployment

### Local Testing

1. Set up your API key in `inf.yml`
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
- Be specific about the changes you want to make
- Describe what to add, modify, or transform
- Reference specific objects or elements in the image
- Use clear, descriptive language
- Examples:
  - "Add a sunset sky in the background"
  - "Give the person sunglasses"
  - "Turn the room into a modern minimalist design"
  - "Add snow to the landscape"

### Image Selection
- Use high-quality input images for best results
- Ensure images are well-lit and clear
- Consider the composition and what you want to change
- Images with clear subjects work better

### Output Configuration
- Use **PNG** for:
  - Images that need transparency
  - Graphics and illustrations
  - When quality is paramount
- Use **JPEG** for:
  - Photographs
  - Smaller file sizes
  - When slight compression is acceptable
- Use **WebP** for:
  - Balance between quality and file size
  - Modern web applications
  - Best compression with good quality
- Use **sync_mode=false** (default) to keep results in request history
- Use **sync_mode=true** for immediate one-time processing

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your FAL_KEY is valid and active
   - Check that the key is properly set in inf.yml
   - Verify you have credits/quota available

2. **File Upload Errors**
   - Verify image file exists and is accessible
   - Check file format is supported (PNG, JPEG, WebP, AVIF, HEIF)
   - Ensure file size is reasonable (under 10MB recommended)
   - If using URLs, ensure they are publicly accessible

3. **Generation Failures**
   - Try simpler prompts if generation fails
   - Check input image quality and clarity
   - Ensure the requested edit is feasible
   - Review error messages for specific issues

4. **Memory Issues**
   - Reduce input image size if processing large files
   - Use JPEG format for smaller outputs
   - Compress images before uploading

### Debug Logging

The app provides detailed logging during generation:
- File upload progress
- API communication status
- Generation progress updates
- Download completion status
- Error messages with context

## Resource Requirements

- **RAM**: 4GB minimum (for file handling and downloads)
- **Storage**: Sufficient space for temporary image files (typically 10-100MB)
- **Network**: Stable internet connection for API calls
- **GPU**: Not required (processing handled by external service)

## Cost Considerations

Image generation costs are handled by fal.ai based on:
- Number of images generated
- Output resolution and complexity
- API usage and credits

Check [fal.ai pricing](https://fal.ai/pricing) for current rates.

## Supported Use Cases

- **Photo editing**: Modify existing photographs
- **Object addition**: Add new elements to images
- **Style transfer**: Apply artistic styles to images
- **Object replacement**: Replace or modify objects in images
- **Scene modification**: Change backgrounds, lighting, or atmosphere
- **Creative enhancement**: Add effects or transformations
- **Content creation**: Generate variations of existing images

## API Reference

For more details on the underlying API:
- [Reve Model Playground](https://fal.ai/models/fal-ai/reve/edit)
- [Reve API Documentation](https://fal.ai/models/fal-ai/reve/edit/api)

## Support

For issues related to:
- **This app**: Check the troubleshooting section above
- **Reve model API**: Visit [fal.ai documentation](https://docs.fal.ai)
- **inference.sh platform**: Check the [platform documentation](https://docs.inference.sh)

## License

This app is designed for use with the inference.sh platform and requires a valid FAL API key for operation.
