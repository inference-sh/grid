import os
import io
import asyncio
import base64
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional, List, Tuple
from PIL import Image

MAX_IMAGE_SIZE = 5 * 1024 * 1024
MAX_GIF_SIZE = 15 * 1024 * 1024
MAX_VIDEO_SIZE = 512 * 1024 * 1024


class AppInput(BaseAppInput):
    text: str = Field(description="Post text (max 280 characters)", min_length=1, max_length=280)
    reply_to_tweet_id: Optional[str] = Field(None, description="Tweet ID to reply to")
    quote_tweet_id: Optional[str] = Field(None, description="Tweet ID to quote")
    media: Optional[List[File]] = Field(None, description="Media files (up to 4 images, or 1 video/GIF)")


class AppOutput(BaseAppOutput):
    tweet_id: str = Field(description="ID of the created tweet")
    post_url: str = Field(description="URL of the created post")


class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    def _get_media_category(self, content_type: str) -> str:
        if content_type.startswith("video/"):
            return "tweet_video"
        elif content_type == "image/gif":
            return "tweet_gif"
        return "tweet_image"

    def _get_content_type(self, file_path: str) -> str:
        ext = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else ""
        return {
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp", "mp4": "video/mp4",
        }.get(ext, "application/octet-stream")

    def _resize_image(self, data: bytes, content_type: str) -> Tuple[bytes, str]:
        img = Image.open(io.BytesIO(data))
        if img.mode in ("RGBA", "P"):
            fmt = "PNG"
        else:
            fmt = "JPEG"
            if img.mode != "RGB":
                img = img.convert("RGB")

        scale = 1.0
        while len(data) > MAX_IMAGE_SIZE and scale > 0.1:
            scale *= 0.9
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            if new_size[0] < 100 or new_size[1] < 100:
                break
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format=fmt, quality=85 if fmt == "JPEG" else None, optimize=True)
            data = buf.getvalue()

        return data, f"image/{fmt.lower()}"

    async def _upload_media(self, file: File) -> str:
        with open(file.path, "rb") as f:
            data = f.read()

        content_type = file.content_type or self._get_content_type(file.path)
        category = self._get_media_category(content_type)

        if category == "tweet_image" and len(data) > MAX_IMAGE_SIZE:
            data, content_type = self._resize_image(data, content_type)
        elif category == "tweet_gif" and len(data) > MAX_GIF_SIZE:
            raise ValueError(f"GIF exceeds 15MB limit")
        elif category == "tweet_video" and len(data) > MAX_VIDEO_SIZE:
            raise ValueError(f"Video exceeds 512MB limit")

        init = self.client.media.initialize_upload(body={
            "total_bytes": len(data), "media_type": content_type, "media_category": category
        })
        media_id = init.data["id"]

        chunk_size = 1024 * 1024
        for i, offset in enumerate(range(0, len(data), chunk_size)):
            chunk = base64.b64encode(data[offset:offset + chunk_size]).decode()
            self.client.media.append_upload(id=media_id, body={"media": chunk, "segment_index": i})

        finalize = self.client.media.finalize_upload(id=media_id)
        processing = finalize.data.get("processing_info") if hasattr(finalize, 'data') else None

        while processing and processing.get("state") not in ("succeeded", None):
            if processing.get("state") == "failed":
                raise ValueError(f"Media processing failed: {processing.get('error', {}).get('message')}")
            await asyncio.sleep(processing.get("check_after_secs", 5))
            status = self.client.media.get_upload_status(media_id=media_id)
            processing = status.data.get("processing_info") if hasattr(status, 'data') else None

        return media_id

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            media_ids = []
            if input_data.media:
                if len(input_data.media) > 4:
                    raise ValueError("Maximum 4 media files allowed")
                for m in input_data.media:
                    media_ids.append(await self._upload_media(m))

            payload = {"text": input_data.text}
            if media_ids:
                payload["media"] = {"media_ids": media_ids}
            if input_data.reply_to_tweet_id:
                payload["reply"] = {"in_reply_to_tweet_id": input_data.reply_to_tweet_id}
            if input_data.quote_tweet_id:
                payload["quote_tweet_id"] = input_data.quote_tweet_id

            response = self.client.posts.create(body=payload)
            tweet_id = response.data["id"]

            return AppOutput(tweet_id=tweet_id, post_url=f"https://x.com/i/web/status/{tweet_id}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"X API error: {e}")

    async def unload(self):
        self.client = None
