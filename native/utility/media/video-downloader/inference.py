from inferencesh import BaseApp, BaseAppSetup, File, OutputMeta, AudioMeta, VideoMeta
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import tempfile
import os
import logging
import json
import yt_dlp

logging.basicConfig(level=logging.INFO)


class ExportFormat(str, Enum):
    """Export format options"""
    AUDIO_ONLY = "audio_only"
    VIDEO_WITH_AUDIO = "video_with_audio"
    VIDEO_ONLY = "video_only"


class AudioQuality(str, Enum):
    """Audio quality options"""
    BEST = "best"
    HIGH = "320"  # 320kbps
    MEDIUM = "192"  # 192kbps
    LOW = "128"  # 128kbps
    LOWEST = "64"  # 64kbps


class VideoQuality(str, Enum):
    """Video quality/resolution options"""
    BEST = "best"
    UHD_4K = "2160"
    QHD_1440P = "1440"
    FHD_1080P = "1080"
    HD_720P = "720"
    SD_480P = "480"
    LOW_360P = "360"


class AudioCodec(str, Enum):
    """Audio codec options"""
    AAC = "aac"  # Fast - native format on many sites
    OPUS = "opus"  # Fast - native format
    MP3 = "mp3"  # Slow - requires re-encoding
    VORBIS = "vorbis"
    FLAC = "flac"  # Slow - requires conversion
    WAV = "wav"  # Slow - requires conversion


class VideoCodec(str, Enum):
    """Video codec options"""
    MP4 = "mp4"
    WEBM = "webm"
    MKV = "mkv"


class AppSetup(BaseAppSetup):
    """Setup configuration for Video Downloader."""
    pass


class RunInput(BaseModel):
    """Input for downloading video/audio content.

    Provide a URL from YouTube, Instagram, Twitter/X, TikTok, or 1000+ supported sites.
    Just provide the URL for a quick download with good defaults!
    """
    url: str = Field(
        description="Video URL (YouTube, Instagram, Twitter/X, TikTok, and 1000+ sites supported)"
    )
    export_format: ExportFormat = Field(
        default=ExportFormat.AUDIO_ONLY,
        description="Export format: audio_only (default), video_with_audio, or video_only"
    )
    audio_quality: AudioQuality = Field(
        default=AudioQuality.HIGH,
        description="Audio quality/bitrate: best, 320 (default), 192, 128, 64 kbps"
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.FHD_1080P,
        description="Video resolution: best, 2160, 1440, 1080 (default), 720, 480, 360"
    )
    audio_codec: AudioCodec = Field(
        default=AudioCodec.AAC,
        description="Audio codec/format: aac (default, fastest), mp3, opus, vorbis, flac, wav"
    )
    video_codec: VideoCodec = Field(
        default=VideoCodec.MP4,
        description="Video container format: mp4 (default), webm, mkv"
    )
    include_thumbnail: bool = Field(
        default=True,
        description="Embed thumbnail in the output file (default: true)"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include metadata like title, artist, etc. in the output file (default: true)"
    )
    include_transcript: bool = Field(
        default=False,
        description="Include transcript/subtitles if available (default: false). Returns auto-generated or manual captions."
    )
    transcript_langs: Optional[List[str]] = Field(
        default=None,
        description="Preferred transcript languages in order of priority (e.g. ['en', 'es']). Defaults to ['en'] if not specified."
    )
    proxy: Optional[str] = Field(
        default=None,
        description="Proxy URL for downloading (e.g. http://user:pass@host:port or socks5://host:port)"
    )


class TranscriptSegment(BaseModel):
    """A single transcript/subtitle segment"""
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcript text")


class VideoInfo(BaseModel):
    """Metadata about the downloaded video/audio"""
    title: str = Field(description="Video title")
    channel: str = Field(description="Channel/uploader name")
    duration_seconds: float = Field(description="Duration in seconds")
    view_count: Optional[int] = Field(default=None, description="Number of views")
    upload_date: Optional[str] = Field(default=None, description="Upload date")
    description: Optional[str] = Field(default=None, description="Video description (truncated)")
    source: Optional[str] = Field(default=None, description="Source platform (e.g. youtube, instagram, twitter)")


class RunOutput(BaseModel):
    """Output from the video downloader."""
    file: File = Field(description="Downloaded audio/video file")
    info: VideoInfo = Field(description="Video/audio metadata")
    format_downloaded: str = Field(description="Format that was downloaded (e.g., 'audio mp3 320kbps')")
    transcript: Optional[List[TranscriptSegment]] = Field(default=None, description="Transcript segments with timestamps, if requested and available")
    transcript_text: Optional[str] = Field(default=None, description="Full transcript as plain text, if requested and available")
    output_meta: Optional[OutputMeta] = Field(default=None, description="Usage metadata for pricing")



class App(BaseApp):

    async def setup(self, config: AppSetup):
        """Initialize the video downloader."""
        logging.info("Video Downloader initialized")

    def _get_video_info(self, url: str, proxy: Optional[str] = None, write_subs: bool = False, sub_langs: List[str] = None) -> dict:
        """Fetch video metadata using yt-dlp Python API."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        if proxy:
            ydl_opts['proxy'] = proxy
        if write_subs:
            ydl_opts['writesubtitles'] = True
            ydl_opts['writeautomaticsub'] = True
            ydl_opts['subtitleslangs'] = sub_langs or ['en']
            ydl_opts['subtitlesformat'] = 'json3'

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info

    def _extract_transcript(self, info: dict, sub_langs: List[str]) -> Optional[List[TranscriptSegment]]:
        """Extract transcript segments from yt-dlp subtitle data."""
        subtitles = info.get('subtitles', {})
        automatic_captions = info.get('automatic_captions', {})

        # Try requested languages in order, manual subs first then auto
        for lang in sub_langs:
            for source in [subtitles, automatic_captions]:
                if lang not in source:
                    continue
                formats = source[lang]
                # Prefer json3 format for structured segments
                json3 = next((f for f in formats if f.get('ext') == 'json3'), None)
                if json3 and 'url' in json3:
                    return self._fetch_json3_transcript(json3['url'])
                # Fall back to any format with direct data
                for fmt in formats:
                    if 'data' in fmt:
                        return self._parse_subtitle_data(fmt)

        return None

    def _fetch_json3_transcript(self, url: str) -> Optional[List[TranscriptSegment]]:
        """Fetch and parse json3 subtitle format from URL."""
        import urllib.request
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            segments = []
            for event in data.get('events', []):
                if 'segs' not in event:
                    continue
                text = ''.join(seg.get('utf8', '') for seg in event['segs']).strip()
                if not text or text == '\n':
                    continue
                start_ms = event.get('tStartMs', 0)
                dur_ms = event.get('dDurationMs', 0)
                segments.append(TranscriptSegment(
                    start=start_ms / 1000.0,
                    end=(start_ms + dur_ms) / 1000.0,
                    text=text
                ))
            return segments if segments else None
        except Exception as e:
            logging.warning(f"Failed to fetch json3 transcript: {e}")
            return None

    def _parse_subtitle_data(self, fmt: dict) -> Optional[List[TranscriptSegment]]:
        """Parse inline subtitle data."""
        try:
            data = fmt.get('data', '')
            if isinstance(data, str):
                data = json.loads(data)
            segments = []
            for event in data.get('events', []):
                if 'segs' not in event:
                    continue
                text = ''.join(seg.get('utf8', '') for seg in event['segs']).strip()
                if not text or text == '\n':
                    continue
                start_ms = event.get('tStartMs', 0)
                dur_ms = event.get('dDurationMs', 0)
                segments.append(TranscriptSegment(
                    start=start_ms / 1000.0,
                    end=(start_ms + dur_ms) / 1000.0,
                    text=text
                ))
            return segments if segments else None
        except Exception:
            return None

    def _build_ydl_opts(
        self,
        output_path: str,
        export_format: ExportFormat,
        audio_quality: AudioQuality,
        video_quality: VideoQuality,
        audio_codec: AudioCodec,
        video_codec: VideoCodec,
        include_thumbnail: bool,
        include_metadata: bool,
        proxy: Optional[str] = None
    ) -> dict:
        """Build yt-dlp options dict based on user preferences."""

        # Base options
        opts = {
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': False,
        }

        if proxy:
            opts['proxy'] = proxy

        # Postprocessors list
        postprocessors = []

        if export_format == ExportFormat.AUDIO_ONLY:
            # Audio-only download
            # For native formats (AAC, Opus), request directly to avoid re-encoding
            if audio_codec == AudioCodec.AAC:
                # Request m4a directly (contains AAC) - no conversion needed
                opts['format'] = 'bestaudio[ext=m4a]/bestaudio/best'
                opts['postprocessor_args'] = {'ffmpeg': ['-c:a', 'copy']}  # Copy, don't re-encode
            elif audio_codec == AudioCodec.OPUS:
                # Request opus/webm directly - no conversion needed
                opts['format'] = 'bestaudio[ext=webm]/bestaudio/best'
                opts['postprocessor_args'] = {'ffmpeg': ['-c:a', 'copy']}
            else:
                # For other codecs (MP3, FLAC, WAV, etc.), we need to convert
                opts['format'] = 'bestaudio/best'

                # Audio extraction postprocessor (will re-encode)
                audio_pp = {
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': audio_codec.value,
                }

                if audio_quality != AudioQuality.BEST:
                    audio_pp['preferredquality'] = audio_quality.value
                else:
                    audio_pp['preferredquality'] = '0'  # Best quality

                postprocessors.append(audio_pp)

        elif export_format == ExportFormat.VIDEO_WITH_AUDIO:
            # Video + audio
            if video_quality == VideoQuality.BEST:
                opts['format'] = 'bestvideo+bestaudio/best'
            else:
                opts['format'] = f'bestvideo[height<={video_quality.value}]+bestaudio/best[height<={video_quality.value}]/best'

            opts['merge_output_format'] = video_codec.value

        elif export_format == ExportFormat.VIDEO_ONLY:
            # Video without audio
            if video_quality == VideoQuality.BEST:
                opts['format'] = 'bestvideo'
            else:
                opts['format'] = f'bestvideo[height<={video_quality.value}]'

            opts['merge_output_format'] = video_codec.value

        # Thumbnail embedding
        if include_thumbnail:
            opts['writethumbnail'] = True
            postprocessors.append({
                'key': 'EmbedThumbnail',
            })

        # Metadata embedding
        if include_metadata:
            postprocessors.append({
                'key': 'FFmpegMetadata',
                'add_metadata': True,
            })

        if postprocessors:
            opts['postprocessors'] = postprocessors

        return opts

    async def run(self, input_data: RunInput) -> RunOutput:
        """Download video/audio content with specified options.

        Supports YouTube, Instagram, Twitter/X, TikTok, and 1000+ sites.
        Just provide a URL for a quick download, or customize all options!
        """

        logging.info(f"Downloading: {input_data.url}")
        logging.info(f"Format: {input_data.export_format.value}, Audio: {input_data.audio_quality.value}, Video: {input_data.video_quality.value}")

        sub_langs = input_data.transcript_langs or ['en']

        # Fetch video info first (with subtitle info if transcript requested)
        info = self._get_video_info(
            input_data.url,
            proxy=input_data.proxy,
            write_subs=input_data.include_transcript,
            sub_langs=sub_langs
        )

        # Extract transcript if requested
        transcript = None
        transcript_text = None
        if input_data.include_transcript:
            transcript = self._extract_transcript(info, sub_langs)
            if transcript:
                transcript_text = '\n'.join(seg.text for seg in transcript)
                logging.info(f"Extracted transcript: {len(transcript)} segments")
            else:
                logging.info("No transcript available for this video")

        # Determine output extension
        if input_data.export_format == ExportFormat.AUDIO_ONLY:
            ext = input_data.audio_codec.value
        else:
            ext = input_data.video_codec.value

        # Create temp directory and output path
        temp_dir = tempfile.mkdtemp()
        # Use a template without extension - yt-dlp will add it
        output_template = os.path.join(temp_dir, "download.%(ext)s")

        # Build yt-dlp options
        ydl_opts = self._build_ydl_opts(
            output_path=output_template,
            export_format=input_data.export_format,
            audio_quality=input_data.audio_quality,
            video_quality=input_data.video_quality,
            audio_codec=input_data.audio_codec,
            video_codec=input_data.video_codec,
            include_thumbnail=input_data.include_thumbnail,
            include_metadata=input_data.include_metadata,
            proxy=input_data.proxy
        )

        logging.info(f"yt-dlp options: {ydl_opts}")

        # Download the content
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([input_data.url])

        # Find the output file (yt-dlp adds extension automatically)
        output_files = [f for f in os.listdir(temp_dir) if f.startswith("download") and not f.endswith(('.jpg', '.webp', '.png'))]

        if not output_files:
            raise RuntimeError(f"No output file found in {temp_dir}. Files present: {os.listdir(temp_dir)}")

        output_path = os.path.join(temp_dir, output_files[0])
        logging.info(f"Output file: {output_path}")

        # Build format description
        if input_data.export_format == ExportFormat.AUDIO_ONLY:
            quality_str = f"{input_data.audio_quality.value}kbps" if input_data.audio_quality != AudioQuality.BEST else "best"
            format_str = f"audio {input_data.audio_codec.value} {quality_str}"
        elif input_data.export_format == ExportFormat.VIDEO_WITH_AUDIO:
            quality_str = f"{input_data.video_quality.value}p" if input_data.video_quality != VideoQuality.BEST else "best"
            format_str = f"video+audio {input_data.video_codec.value} {quality_str}"
        else:
            quality_str = f"{input_data.video_quality.value}p" if input_data.video_quality != VideoQuality.BEST else "best"
            format_str = f"video-only {input_data.video_codec.value} {quality_str}"

        # Detect source platform from extractor
        source = info.get('extractor_key', info.get('extractor', '')).lower()
        if 'youtube' in source:
            source = 'youtube'
        elif 'instagram' in source:
            source = 'instagram'
        elif 'twitter' in source or 'x' in source:
            source = 'twitter'
        elif 'tiktok' in source:
            source = 'tiktok'

        # Build video info
        video_info = VideoInfo(
            title=info.get("title", "Unknown"),
            channel=info.get("channel", info.get("uploader", "Unknown")),
            duration_seconds=info.get("duration", 0),
            view_count=info.get("view_count"),
            upload_date=info.get("upload_date"),
            description=info.get("description", "")[:500] if info.get("description") else None,
            source=source or None
        )

        # Build output metadata for pricing
        duration = info.get("duration", 0)

        if input_data.export_format == ExportFormat.AUDIO_ONLY:
            output_meta = OutputMeta(
                outputs=[AudioMeta(seconds=float(duration))]
            )
        else:
            width = info.get("width", 1920)
            height = info.get("height", 1080)
            resolution = f"{height}p" if height else "unknown"
            output_meta = OutputMeta(
                outputs=[VideoMeta(
                    width=width,
                    height=height,
                    resolution=resolution,
                    seconds=float(duration)
                )]
            )

        return RunOutput(
            file=File(path=output_path),
            info=video_info,
            format_downloaded=format_str,
            transcript=transcript,
            transcript_text=transcript_text,
            output_meta=output_meta
        )
