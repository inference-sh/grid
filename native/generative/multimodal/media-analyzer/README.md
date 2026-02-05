# Media Analyzer

AI-powered media analyzer that can analyze images and audio files using OpenAI's multimodal AI models.

## Features

- **Image Analysis**: Analyze images using vision models (JPEG, PNG, GIF, BMP, WebP, TIFF)
- **Audio Analysis**: Analyze audio files using audio-capable models (WAV, MP3, M4A, OGG, FLAC, AAC, WMA)
- **Multi-file Support**: Analyze multiple media files in a single request
- **Metadata Extraction**: Automatically extract detailed metadata including:
  - File size and format
  - Image dimensions (width, height)
  - Audio duration (for audio files)
- **Flexible Questions**: Ask any question about the media
- **Model Selection**: Choose from various OpenAI models

## Input Schema

```json
{
  "question": "Describe this media (describe all media if there are multiple as input):",
  "media": [
    {"path": "path/to/image.jpg"},
    {"path": "path/to/audio.wav"}
  ],
  "model": "gpt-4o-mini",
  "temperature": 0.7,
  "max_tokens": null
}
```

### Parameters

- **question** (string): Question or prompt about the media. Default: "Describe this media (describe all media if there are multiple as input):"
- **media** (array): List of media files to analyze (images or audio)
- **model** (string): OpenAI model to use. Default: "gpt-4o-mini"
  - For images: `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`, etc.
  - For audio: `gpt-4o-audio-preview` or other audio-capable models
- **temperature** (float): Response randomness (0.0-2.0). Default: 0.7
- **max_tokens** (int, optional): Maximum tokens in response

## Output Schema

```json
{
  "analysis": "Detailed analysis of the media",
  "question": "Original question asked",
  "media_count": 1,
  "media_types": ["image"],
  "media_metadata": [
    {
      "filename": "image.jpg",
      "format": "jpg",
      "size_bytes": 1024000,
      "size_mb": 0.98,
      "media_type": "image",
      "width": 1920,
      "height": 1080,
      "dimensions": "1920x1080",
      "duration_seconds": null,
      "duration_formatted": null
    }
  ],
  "model_used": "gpt-4o-mini"
}
```

### Metadata Fields

The app extracts detailed metadata for all input media files:

**Common fields (all media):**
- `filename`: Name of the file
- `format`: File format/extension
- `size_bytes`: File size in bytes
- `size_mb`: File size in megabytes
- `media_type`: Type of media (image, audio, unknown)

**Image-specific fields:**
- `width`: Width in pixels
- `height`: Height in pixels
- `dimensions`: Dimensions as string (e.g., "1920x1080")

**Audio-specific fields:**
- `duration_seconds`: Duration in seconds
- `duration_formatted`: Duration as MM:SS or HH:MM:SS

## Supported Media Types

### Images
- JPEG/JPG
- PNG
- GIF
- BMP
- WebP
- TIFF/TIF

### Audio
- WAV
- MP3
- M4A
- OGG
- FLAC
- AAC
- WMA

## Example Usage

### Analyze a Single Image

```json
{
  "question": "What's in this image?",
  "media": [
    {"path": "https://example.com/image.jpg"}
  ],
  "model": "gpt-4o-mini"
}
```

### Analyze Audio

```json
{
  "question": "What is in this recording?",
  "media": [
    {"path": "/path/to/audio.wav"}
  ],
  "model": "gpt-4o-audio-preview"
}
```

### Analyze Multiple Media Files

```json
{
  "question": "Compare and describe all the media files",
  "media": [
    {"path": "image1.jpg"},
    {"path": "image2.png"},
    {"path": "audio.wav"}
  ],
  "model": "gpt-4o"
}
```

## Environment Variables

- **OPENAI_API_KEY**: Your OpenAI API key (required)

## Deployment

### Local Testing

```bash
# Generate example input
infsh run

# Test with input.json
infsh run input.json
```

### Deploy to inference.sh

```bash
infsh deploy
```

## Model Recommendations

- **Images only**: `gpt-4o-mini` (fast and cost-effective) or `gpt-4o` (more detailed)
- **Audio only**: `gpt-4o-audio-preview`
- **Mixed media**: `gpt-4o` (supports both images and text)

## Notes

- The app automatically detects media type based on file extension
- Images are converted to base64 for API transmission
- Audio files are also base64-encoded for the API
- Multiple media files can be analyzed in a single request
- The default question works well for general media description

## License

MIT
