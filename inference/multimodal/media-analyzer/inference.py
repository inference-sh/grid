import os
import logging
import base64
from typing import List, Optional, Dict, Any
from pathlib import Path
from openai import OpenAI
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel
from PIL import Image

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import audio metadata library
try:
    from mutagen import File as MutagenFile
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    logger.warning("mutagen not available - audio duration will not be extracted")



# Supported media types
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac', '.wma'}


class MediaMetadata(BaseModel):
    """Metadata about a media file."""
    filename: str = Field(description="Name of the file")
    format: str = Field(description="File format/extension")
    size_bytes: int = Field(description="File size in bytes")
    size_mb: float = Field(description="File size in megabytes")
    media_type: str = Field(description="Type of media (image, audio, unknown)")
    # Image-specific fields
    width: Optional[int] = Field(default=None, description="Width in pixels (images only)")
    height: Optional[int] = Field(default=None, description="Height in pixels (images only)")
    dimensions: Optional[str] = Field(default=None, description="Dimensions as string (images only)")
    # Audio-specific fields
    duration_seconds: Optional[float] = Field(default=None, description="Duration in seconds (audio only)")
    duration_formatted: Optional[str] = Field(default=None, description="Duration as HH:MM:SS (audio only)")

class AppInput(BaseAppInput):
    """Input schema for the media analyzer."""
    question: str = Field(
        default="Describe this media (describe all media if there are multiple as input):",
        description="Question or prompt about the media"
    )
    media: List[File] = Field(
        description="List of media files to analyze (images or audio)"
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for analysis (gpt-4o-mini, gpt-4o, gpt-4o-audio-preview, etc.)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for response randomness (0.0-2.0)"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens in response (optional)"
    )

class AppOutput(BaseAppOutput):
    """Output schema for the media analyzer."""
    analysis: str = Field(description="Analysis of the media")


class App(BaseApp):
    """Media Analyzer App for inference.sh platform."""

    def __init__(self):
        super().__init__()
        self.client = None

    async def setup(self, metadata=None):
        """Initialize the media analyzer."""
        logger.info("🔧 Setting up Media Analyzer...")
        logger.info("✅ Media Analyzer setup complete")

    async def run(self, input_data: AppInput, metadata=None) -> AppOutput:
        """Analyze media files (images and/or audio) with AI."""
        try:
            logger.info(f"🔍 Analyzing {len(input_data.media)} media file(s)")
            logger.info(f"📝 Question: {input_data.question}")

            # Get API key from environment variable
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")

            # Initialize client with API key from environment
            self.client = OpenAI(api_key=api_key)

            # Validate media files exist
            if not input_data.media or len(input_data.media) == 0:
                raise ValueError("No media files provided")

            # Extract metadata for all media files
            logger.info("📊 Extracting media metadata...")
            media_metadata_list = []
            for media_file in input_data.media:
                file_metadata = self._extract_metadata(media_file)
                media_metadata_list.append(file_metadata)
                self._log_metadata(file_metadata)

            # Categorize media by type
            media_info = self._categorize_media(input_data.media)

            # Perform analysis
            analysis_result = await self._analyze_media(
                question=input_data.question,
                media_info=media_info,
                model=input_data.model,
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens
            )

            logger.info(f"✅ Analysis completed successfully")
            
            analysis = analysis_result["analysis"]
            
            # Add metadata section in markdown format
            analysis += "\n\n## Media Metadata\n"
            for metadata in media_metadata_list:
                analysis += f"\n### {metadata.filename}\n"
                analysis += f"- **Type**: {metadata.media_type}\n"
                analysis += f"- **Format**: {metadata.format}\n"
                analysis += f"- **Size**: {metadata.size_mb} MB ({metadata.size_bytes:,} bytes)\n"
                
                if metadata.dimensions:
                    analysis += f"- **Dimensions**: {metadata.dimensions}\n"
                
                if metadata.duration_formatted:
                    analysis += f"- **Duration**: {metadata.duration_formatted} ({metadata.duration_seconds}s)\n"
            
            return AppOutput(analysis=analysis)

        except Exception as e:
            logger.error(f"❌ Analysis error: {str(e)}")
            raise ValueError(f"Media analysis failed: {str(e)}")

    def _categorize_media(self, media_files: List[File]) -> Dict[str, Any]:
        """Categorize media files into images and audio."""
        images = []
        audio = []
        types = set()

        for media_file in media_files:
            file_path = Path(media_file.path)
            extension = file_path.suffix.lower()

            if extension in IMAGE_EXTENSIONS:
                images.append(media_file)
                types.add("image")
                logger.info(f"  📷 Image detected: {file_path.name}")
            elif extension in AUDIO_EXTENSIONS:
                audio.append(media_file)
                types.add("audio")
                logger.info(f"  🎵 Audio detected: {file_path.name}")
            else:
                logger.warning(f"  ⚠️ Unknown media type: {file_path.name} (extension: {extension})")
                # Try to process as image by default
                images.append(media_file)
                types.add("unknown")

        return {
            "images": images,
            "audio": audio,
            "types": list(types)
        }

    def _extract_metadata(self, media_file: File) -> MediaMetadata:
        """Extract metadata from a media file."""
        file_path = Path(media_file.path)
        file_size = os.path.getsize(media_file.path)
        file_size_mb = file_size / (1024 * 1024)
        extension = file_path.suffix.lower()

        metadata = {
            "filename": file_path.name,
            "format": extension.strip('.'),
            "size_bytes": file_size,
            "size_mb": round(file_size_mb, 2),
            "media_type": "unknown"
        }

        # Extract image metadata
        if extension in IMAGE_EXTENSIONS:
            metadata["media_type"] = "image"
            try:
                with Image.open(media_file.path) as img:
                    metadata["width"] = img.width
                    metadata["height"] = img.height
                    metadata["dimensions"] = f"{img.width}x{img.height}"
            except Exception as e:
                logger.warning(f"Could not extract image dimensions for {file_path.name}: {e}")

        # Extract audio metadata
        elif extension in AUDIO_EXTENSIONS:
            metadata["media_type"] = "audio"
            if MUTAGEN_AVAILABLE:
                try:
                    audio = MutagenFile(media_file.path)
                    if audio and hasattr(audio.info, 'length'):
                        duration = audio.info.length
                        metadata["duration_seconds"] = round(duration, 2)
                        # Format as HH:MM:SS
                        hours = int(duration // 3600)
                        minutes = int((duration % 3600) // 60)
                        seconds = int(duration % 60)
                        if hours > 0:
                            metadata["duration_formatted"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        else:
                            metadata["duration_formatted"] = f"{minutes:02d}:{seconds:02d}"
                except Exception as e:
                    logger.warning(f"Could not extract audio duration for {file_path.name}: {e}")
            else:
                logger.debug(f"mutagen not available - skipping audio duration for {file_path.name}")

        return MediaMetadata(**metadata)

    def _log_metadata(self, metadata: MediaMetadata):
        """Log metadata in a readable format."""
        logger.info(f"  📄 {metadata.filename}")
        logger.info(f"     Type: {metadata.media_type}, Format: {metadata.format}")
        logger.info(f"     Size: {metadata.size_mb} MB ({metadata.size_bytes:,} bytes)")

        if metadata.dimensions:
            logger.info(f"     Dimensions: {metadata.dimensions} ({metadata.width}x{metadata.height})")

        if metadata.duration_formatted:
            logger.info(f"     Duration: {metadata.duration_formatted} ({metadata.duration_seconds}s)")

    async def _analyze_media(
        self,
        question: str,
        media_info: Dict[str, Any],
        model: str,
        temperature: float,
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """Analyze media using OpenAI API."""
        try:
            images = media_info["images"]
            audio_files = media_info["audio"]

            # Build content array for the API request
            content = []

            # Add the question as text
            content.append({
                "type": "text",
                "text": question
            })

            # Process images
            for img_file in images:
                # For images, we can use the file path directly or convert to base64
                # OpenAI API supports both URLs and base64
                with open(img_file.path, "rb") as f:
                    image_data = f.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')

                # Determine image format from extension
                file_ext = Path(img_file.path).suffix.lower().strip('.')
                if file_ext == 'jpg':
                    file_ext = 'jpeg'

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{file_ext};base64,{base64_image}"
                    }
                })
                logger.info(f"    📷 Added image to analysis")

            # Process audio files
            for audio_file in audio_files:
                with open(audio_file.path, "rb") as f:
                    audio_data = f.read()
                    base64_audio = base64.b64encode(audio_data).decode('utf-8')

                # Determine audio format from extension
                audio_format = Path(audio_file.path).suffix.lower().strip('.')

                content.append({
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64_audio,
                        "format": audio_format
                    }
                })
                logger.info(f"    🎵 Added audio to analysis")

            # Make API call
            logger.info(f"    🤖 Using model: {model}")
            logger.info(f"    🌡️ Temperature: {temperature}")

            # Prepare API parameters
            api_params = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "temperature": temperature
            }

            # Add optional parameters
            if max_tokens:
                api_params["max_tokens"] = max_tokens

            # For audio analysis, we may need to specify modalities
            if audio_files:
                # If using audio-capable model, add modalities
                if "audio" in model.lower():
                    api_params["modalities"] = ["text", "audio"]
                    # For audio output, we can optionally specify the voice
                    # api_params["audio"] = {"voice": "alloy", "format": "wav"}

            response = self.client.chat.completions.create(**api_params)

            # Extract response content
            if response.choices and len(response.choices) > 0:
                analysis = response.choices[0].message.content

                return {
                    "analysis": analysis,
                    "tokens_used": response.usage.total_tokens if response.usage else None
                }
            else:
                raise ValueError("No response received from API")

        except Exception as e:
            logger.error(f"    ❌ Analysis API error: {str(e)}")
            raise ValueError(f"Media analysis API failed: {str(e)}")
