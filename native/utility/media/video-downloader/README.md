# Video Downloader

Download audio and video from YouTube, Instagram, Twitter/X, TikTok, and 1000+ sites.

## Features

- Audio-only (MP3, AAC, etc.)
- Video with audio (MP4, WebM)
- Quality selection
- Metadata embedding
- Transcript extraction (when available)
- Proxy support

## Usage

```json
{
  "url": "https://youtube.com/watch?v=...",
  "export_format": "audio_only",
  "audio_codec": "mp3"
}
```

With proxy and transcript:
```json
{
  "url": "https://instagram.com/reel/...",
  "export_format": "video_with_audio",
  "include_transcript": true,
  "proxy": "http://user:pass@host:port"
}
```

## Development

```bash
infsh app run    # Test locally
infsh app deploy # Deploy
```

## Documentation

Full docs: **[app.inference.sh/docs/extend](https://app.inference.sh/docs/extend)**
