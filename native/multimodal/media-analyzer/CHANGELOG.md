# Media Analyzer - Changelog

## Latest Updates - Metadata Extraction Feature

### New Features Added

#### 1. Automatic Metadata Extraction

The app now automatically extracts and returns detailed metadata for ALL input media files:

**For Images:**
- Filename
- Format (jpg, png, etc.)
- File size (in bytes and MB)
- Dimensions (width x height in pixels)
- Media type

**For Audio:**
- Filename
- Format (wav, mp3, etc.)
- File size (in bytes and MB)
- Duration (in seconds and formatted as MM:SS or HH:MM:SS)
- Media type

#### 2. Enhanced Output Schema

The `AppOutput` now includes a `media_metadata` field containing an array of `MediaMetadata` objects - one for each input media file.

```python
class MediaMetadata(BaseModel):
    """Metadata about a media file."""
    filename: str
    format: str
    size_bytes: int
    size_mb: float
    media_type: str  # "image", "audio", or "unknown"
    # Image-specific
    width: Optional[int] = None
    height: Optional[int] = None
    dimensions: Optional[str] = None
    # Audio-specific
    duration_seconds: Optional[float] = None
    duration_formatted: Optional[str] = None
```

#### 3. Enhanced Logging

The app now logs detailed metadata for each file during processing:

```
📊 Extracting media metadata...
  📄 nature-boardwalk.jpg
     Type: image, Format: jpg
     Size: 1.39 MB (1,458,432 bytes)
     Dimensions: 2560x1440 (2560x2560)
```

### Technical Implementation

**Dependencies Added:**
- `mutagen` - For audio metadata extraction (duration, bitrate, etc.)
- `PIL/Pillow` - For image dimension extraction (already included)

**New Methods:**
- `_extract_metadata(media_file)` - Extracts metadata from a single media file
- `_log_metadata(metadata)` - Logs metadata in a readable format

**Error Handling:**
- Gracefully handles missing metadata libraries
- Continues processing even if metadata extraction fails
- Provides warning logs for extraction errors

### Example Output

```json
{
  "analysis": "This image shows a beautiful nature boardwalk...",
  "question": "Describe this media",
  "media_count": 1,
  "media_types": ["image"],
  "media_metadata": [
    {
      "filename": "nature-boardwalk.jpg",
      "format": "jpg",
      "size_bytes": 1458432,
      "size_mb": 1.39,
      "media_type": "image",
      "width": 2560,
      "height": 1440,
      "dimensions": "2560x1440",
      "duration_seconds": null,
      "duration_formatted": null
    }
  ],
  "model_used": "gpt-4o-mini"
}
```

### Multi-File Support

When analyzing multiple media files, metadata is extracted for ALL files:

```json
{
  "media_count": 3,
  "media_metadata": [
    {
      "filename": "image1.jpg",
      "format": "jpg",
      "size_mb": 2.5,
      "width": 1920,
      "height": 1080,
      ...
    },
    {
      "filename": "audio.mp3",
      "format": "mp3",
      "size_mb": 5.2,
      "duration_seconds": 180.5,
      "duration_formatted": "03:00",
      ...
    },
    {
      "filename": "image2.png",
      "format": "png",
      "size_mb": 3.1,
      "width": 2560,
      "height": 1440,
      ...
    }
  ]
}
```

### Benefits

1. **Comprehensive Analysis**: Users get both AI analysis AND technical metadata
2. **Quality Validation**: Quickly verify file dimensions and sizes
3. **Debugging**: Helps identify issues with media files
4. **Documentation**: Automatic documentation of processed media
5. **Multi-file Tracking**: Easy to see details of all processed files

### Backwards Compatibility

This is a **non-breaking change**. The existing functionality remains unchanged, with metadata being an additional feature in the output.
