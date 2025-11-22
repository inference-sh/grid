# Nano Banana Image Editor App

This inference.sh app provides an interface to Google's Nano Banana image generation and editing model, allowing you to edit and transform images using natural language prompts.

## Overview

The app acts as a bridge between the inference.sh platform and the Nano Banana model, handling file uploads, API communication, and image download seamlessly.

## Features

- **Advanced image editing** using natural language prompts
- **Multi-image input** support for complex editing tasks
- **Multiple output formats**: JPEG, PNG
- **Batch generation**: Generate 1-4 images per request
- **Flexible output modes**: URLs or data URIs
- **Progress logging**: Real-time generation updates

## Setup

### API Key Configuration

You need an API key to use this app. Set your API key in the `inf.yml` environment configuration:

```yaml
env:
    FAL_KEY: "your_api_key_here"
```

### Getting an API Key

1. Visit the model provider's platform
2. Create an account or log in
3. Navigate to your API keys section
4. Generate a new API key
5. Copy the key and configure it as described above

## Input Parameters

### Required Parameters

- **`prompt`** (string): The prompt for image editing
  - Example: "make a photo of the man driving the car down the california coastline"

- **`images`** (List[File]): List of input images for editing
  - Supported formats: JPEG, PNG, WebP
  - Can be local file paths or URLs
  - Minimum 1 image required

### Optional Parameters

- **`num_images`** (integer, default: 1): Number of images to generate
  - Range: 1-4 images
  - Higher numbers increase generation time

- **`output_format`** (string, default: "jpeg"): Output format for generated images
  - Options: "jpeg", "png"
  - JPEG is smaller, PNG supports transparency

- **`sync_mode`** (boolean, default: false): Output format preference
  - false: Returns URLs to download images
  - true: Returns images as base64 data URIs

## Output

The app returns:

- **`images`** (List[File]): The edited images
- **`description`** (string): Text description or response from the model

## Example Usage

### Basic Example

```json
{
  "prompt": "turn the building into a futuristic skyscraper with glass walls",
  "images": [
    {
      "path": "https://example.com/building.jpg"
    }
  ]
}
```

### Multi-Image Editing

```json
{
  "prompt": "make a photo of the man driving the car down the california coastline",
  "images": [
    {
      "path": "/path/to/person.jpg"
    },
    {
      "path": "/path/to/car.jpg"
    }
  ],
  "num_images": 2,
  "output_format": "png"
}
```

### Advanced Example

```json
{
  "prompt": "combine these images into a surreal landscape with vibrant colors",
  "images": [
    {
      "path": "https://example.com/landscape1.jpg"
    },
    {
      "path": "https://example.com/landscape2.jpg"
    }
  ],
  "num_images": 4,
  "output_format": "jpeg",
  "sync_mode": false
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
- Describe the style, mood, and visual elements
- Reference specific objects or people in the images
- Use clear, descriptive language

### Image Selection
- Use high-quality input images
- Ensure images are relevant to your editing goal
- Multiple related images often produce better results
- Consider image composition and lighting

### Output Configuration
- Use JPEG for smaller file sizes
- Use PNG when transparency is needed
- Generate multiple images to get variety
- Use sync_mode=true for immediate processing of small images

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your API key is valid and active
   - Check that the key is properly set in the environment

2. **File Upload Errors**
   - Verify image files exist and are accessible
   - Check file formats are supported (JPEG, PNG, WebP)
   - Ensure file sizes are reasonable

3. **Generation Failures**
   - Try simpler prompts if generation fails
   - Reduce the number of output images
   - Check input image quality and relevance

4. **Memory Issues**
   - Reduce image sizes if processing large files
   - Lower num_images parameter
   - Use JPEG format for smaller outputs

### Debug Logging

The app provides detailed logging during generation:
- File upload progress
- API communication status
- Generation progress updates
- Download completion status

## Resource Requirements

- **RAM**: 4GB minimum (for file handling and downloads)
- **Storage**: Sufficient space for temporary image files
- **Network**: Stable internet connection for API calls
- **GPU**: Not required (processing handled by external service)

## Cost Considerations

Image generation costs are handled by the service provider based on:
- Number of images generated
- Output resolution and format
- API usage

Check the provider's pricing for current rates.

## Supported Use Cases

- **Photo editing**: Modify existing photographs
- **Style transfer**: Apply artistic styles to images
- **Object replacement**: Replace or modify objects in images
- **Scene composition**: Combine multiple images into new scenes
- **Creative enhancement**: Add effects, lighting, or atmosphere
- **Content creation**: Generate variations of existing images

## Support

For issues related to:
- **This app**: Check the troubleshooting section above
- **Model API**: Visit the provider's documentation
- **inference.sh platform**: Check the platform documentation

## License

This app is designed for use with the inference.sh platform and requires a valid API key for operation.