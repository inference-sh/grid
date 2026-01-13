# YouTube Downloader

Download audio and video from YouTube.

## Features

- Audio-only (MP3, AAC, etc.)
- Video with audio (MP4, WebM)
- Quality selection
- Metadata embedding

## Usage

```json
{
  "url": "https://youtube.com/watch?v=...",
  "export_format": "audio_only",
  "audio_format": "mp3"
}
```

## Development

```bash
infsh app run    # Test locally
infsh app deploy # Deploy
```

## Documentation

Full docs: **[app.inference.sh/docs/extend](https://app.inference.sh/docs/extend)**
