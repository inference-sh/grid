import os
import io
import asyncio
import base64
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional, List, Tuple
from PIL import Image

# X API media size limits
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_GIF_SIZE = 15 * 1024 * 1024   # 15MB
MAX_VIDEO_SIZE = 512 * 1024 * 1024  # 512MB


class AppInput(BaseAppInput):
    """Input schema for posting a tweet."""
    text: str = Field(description="Tweet text to post (max 280 characters)", min_length=1, max_length=280)
    reply_to_tweet_id: Optional[str] = Field(None, description="Tweet ID to reply to (optional)")
    quote_tweet_id: Optional[str] = Field(None, description="Tweet ID to quote (optional)")
    media: Optional[List[File]] = Field(
        None,
        description="Media files to attach (up to 4 images, or 1 video/GIF). Supports JPG, PNG, GIF, WEBP, MP4."
    )


class AppOutput(BaseAppOutput):
    """Output schema for posted tweet."""
    tweet_id: str = Field(description="ID of the posted tweet")
    tweet_url: str = Field(description="URL of the posted tweet")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        """Initialize the X.com client with OAuth 2.0 access token."""
        access_token = os.environ.get("X_ACCESS_TOKEN")

        if not access_token:
            raise ValueError(
                "X_ACCESS_TOKEN not found. "
                "Please ensure the X.com integration is connected in Settings."
            )

        # Create client with OAuth 2.0 access token
        self.client = Client(access_token=access_token)

        print("X.com client initialized")

    def _get_media_category(self, content_type: str) -> str:
        """Determine the media category based on content type."""
        if content_type.startswith("video/"):
            return "tweet_video"
        elif content_type == "image/gif":
            return "tweet_gif"
        else:
            return "tweet_image"

    def _resize_image_to_fit(self, file_data: bytes, content_type: str, max_size: int = MAX_IMAGE_SIZE) -> Tuple[bytes, str]:
        """Resize an image to fit under the size limit while maintaining aspect ratio.

        Returns tuple of (resized_data, content_type).
        """
        # Open the image
        img = Image.open(io.BytesIO(file_data))
        original_format = img.format or "PNG"

        # Convert to RGB if necessary (for JPEG output)
        if img.mode in ("RGBA", "P"):
            # Keep PNG for images with transparency
            output_format = "PNG"
        else:
            # Use JPEG for better compression
            output_format = "JPEG"
            if img.mode != "RGB":
                img = img.convert("RGB")

        current_size = len(file_data)
        original_width, original_height = img.size
        scale = 1.0

        print(f"Original image: {original_width}x{original_height}, {current_size} bytes, format: {original_format}")

        # Binary search for the right scale factor
        while current_size > max_size and scale > 0.1:
            # Reduce scale by 10%
            scale *= 0.9
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            # Ensure minimum dimensions
            if new_width < 100 or new_height < 100:
                break

            # Resize the image
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save to buffer
            buffer = io.BytesIO()
            if output_format == "JPEG":
                # Try different quality levels for JPEG
                quality = 85
                while quality >= 50:
                    buffer.seek(0)
                    buffer.truncate()
                    resized.save(buffer, format=output_format, quality=quality, optimize=True)
                    if buffer.tell() <= max_size:
                        break
                    quality -= 10
            else:
                resized.save(buffer, format=output_format, optimize=True)

            current_size = buffer.tell()

            if current_size <= max_size:
                print(f"Resized image: {new_width}x{new_height}, {current_size} bytes, format: {output_format}")
                buffer.seek(0)
                new_content_type = "image/jpeg" if output_format == "JPEG" else "image/png"
                return buffer.read(), new_content_type

        # If we couldn't resize enough, raise an error
        if current_size > max_size:
            raise ValueError(f"Could not resize image to fit under {max_size // (1024*1024)}MB limit")

        return file_data, content_type

    async def _upload_media_chunked(self, file_data: bytes, content_type: str, media_category: str) -> str:
        """Upload media using chunked upload."""
        file_size = len(file_data)
        chunk_size = 1 * 1024 * 1024  # 1MB chunks

        # Step 1: Initialize upload
        print(f"Initializing upload: {file_size} bytes, {content_type}, {media_category}")
        init_response = self.client.media.initialize_upload(body={
            "total_bytes": file_size,
            "media_type": content_type,
            "media_category": media_category
        })
        # XDK returns InitializeUploadResponse with data dict containing 'id'
        media_id = init_response.data["id"]
        print(f"Upload initialized with media_id: {media_id}")

        # Step 2: Append chunks (media must be base64 encoded)
        segment_index = 0
        offset = 0
        while offset < file_size:
            chunk = file_data[offset:offset + chunk_size]
            chunk_b64 = base64.b64encode(chunk).decode('ascii')
            print(f"Uploading chunk {segment_index}: {len(chunk)} bytes")
            self.client.media.append_upload(
                id=media_id,
                body={
                    "media": chunk_b64,
                    "segment_index": segment_index
                }
            )
            offset += chunk_size
            segment_index += 1

        # Step 3: Finalize upload
        print("Finalizing upload...")
        finalize_response = self.client.media.finalize_upload(id=media_id)
        print(f"Finalize response: {finalize_response}")

        # Step 4: Wait for processing if needed (videos)
        # XDK response has .data dict
        data = finalize_response.data if hasattr(finalize_response, 'data') else {}
        processing_info = data.get("processing_info") if isinstance(data, dict) else None

        while processing_info:
            state = processing_info.get("state")
            print(f"Processing state: {state}")

            if state == "succeeded":
                break
            elif state == "failed":
                error = processing_info.get("error", {})
                raise ValueError(f"Media processing failed: {error.get('message', 'Unknown error')}")

            # Wait before checking again
            check_after = processing_info.get("check_after_secs", 5)
            print(f"Waiting {check_after}s for processing...")
            await asyncio.sleep(check_after)

            # Check status
            status_response = self.client.media.get_upload_status(media_id=media_id)
            status_data = status_response.data if hasattr(status_response, 'data') else {}
            processing_info = status_data.get("processing_info") if isinstance(status_data, dict) else None

        print(f"Media upload complete: {media_id}")
        return media_id

    def _get_content_type(self, file_path: str) -> str:
        """Get content type from file extension."""
        ext = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else ""
        content_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
            "mp4": "video/mp4",
            "mov": "video/quicktime",
            "avi": "video/x-msvideo",
        }
        return content_types.get(ext, "application/octet-stream")

    async def _upload_media(self, media_file: File) -> str:
        """Upload a single media file and return its media_id."""
        # Read the file data from the local path
        file_path = media_file.path
        with open(file_path, "rb") as f:
            file_data = f.read()

        content_type = media_file.content_type or self._get_content_type(file_path)
        file_size = len(file_data)

        print(f"Uploading media: {file_path}, {content_type}, {file_size} bytes")

        media_category = self._get_media_category(content_type)

        # Check size limits and resize if needed
        if media_category == "tweet_image":
            if file_size > MAX_IMAGE_SIZE:
                print(f"Image exceeds {MAX_IMAGE_SIZE // (1024*1024)}MB limit, resizing...")
                file_data, content_type = self._resize_image_to_fit(file_data, content_type)
                file_size = len(file_data)
                print(f"Resized to {file_size} bytes")
        elif media_category == "tweet_gif":
            if file_size > MAX_GIF_SIZE:
                raise ValueError(f"GIF exceeds {MAX_GIF_SIZE // (1024*1024)}MB limit ({file_size // (1024*1024)}MB). Please use a smaller GIF.")
        elif media_category == "tweet_video":
            if file_size > MAX_VIDEO_SIZE:
                raise ValueError(f"Video exceeds {MAX_VIDEO_SIZE // (1024*1024)}MB limit ({file_size // (1024*1024)}MB). Please use a smaller video.")

        return await self._upload_media_chunked(file_data, content_type, media_category)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Post a tweet to X.com."""
        # Validate tweet length
        if len(input_data.text) > 280:
            raise ValueError(f"Tweet text exceeds 280 characters ({len(input_data.text)} chars)")

        print(f"Posting tweet: {input_data.text[:50]}...")

        try:
            # Upload media if provided
            media_ids = []
            if input_data.media:
                # Validate media count
                if len(input_data.media) > 4:
                    raise ValueError("Maximum 4 media files allowed per tweet")

                # Check for mixed media types
                def get_type(m: File) -> str:
                    return m.content_type or self._get_content_type(m.path)

                has_video = any(get_type(m).startswith("video/") for m in input_data.media)
                has_gif = any(get_type(m) == "image/gif" for m in input_data.media)

                if has_video and len(input_data.media) > 1:
                    raise ValueError("Only 1 video allowed per tweet")
                if has_gif and len(input_data.media) > 1:
                    raise ValueError("Only 1 GIF allowed per tweet")

                # Upload each media file
                for media_file in input_data.media:
                    media_id = await self._upload_media(media_file)
                    media_ids.append(media_id)

                print(f"Uploaded {len(media_ids)} media files: {media_ids}")

            # Build tweet payload
            payload = {"text": input_data.text}

            if media_ids:
                payload["media"] = {"media_ids": media_ids}

            if input_data.reply_to_tweet_id:
                payload["reply"] = {"in_reply_to_tweet_id": input_data.reply_to_tweet_id}
                print(f"Replying to tweet: {input_data.reply_to_tweet_id}")

            if input_data.quote_tweet_id:
                payload["quote_tweet_id"] = input_data.quote_tweet_id
                print(f"Quoting tweet: {input_data.quote_tweet_id}")

            # Post the tweet
            response = self.client.posts.create(body=payload)

            tweet_id = response.data["id"]
            tweet_url = f"https://x.com/i/web/status/{tweet_id}"

            print(f"Tweet posted successfully: {tweet_url}")

            return AppOutput(
                tweet_id=tweet_id,
                tweet_url=tweet_url
            )

        except ValueError:
            raise
        except Exception as e:
            print(f"X API raw error: {type(e).__name__}: {e}")
            error_msg = str(e).lower()
            if "duplicate" in error_msg:
                raise ValueError("This tweet was already posted (duplicate content)")
            elif "rate limit" in error_msg:
                raise ValueError("Rate limit exceeded. Please try again later.")
            elif "unauthorized" in error_msg or "forbidden" in error_msg or "401" in error_msg or "403" in error_msg:
                raise ValueError(f"Authorization failed: {e}")
            else:
                raise ValueError(f"X.com API error: {e}")

    async def unload(self):
        """Cleanup resources."""
        self.client = None
