"""Shared X API helpers for media upload via xdk.

All xdk response objects are Pydantic models — use attribute access (.id, .processing_info),
not dict subscript (["id"]) or .get().
"""

import io
import asyncio
import base64
from typing import Tuple
from PIL import Image

MAX_IMAGE_SIZE = 5 * 1024 * 1024       # 5MB
MAX_GIF_SIZE = 15 * 1024 * 1024        # 15MB
MAX_VIDEO_SIZE = 512 * 1024 * 1024     # 512MB


def get_media_category(content_type: str) -> str:
    """Determine the X API media category from content type."""
    if content_type.startswith("video/"):
        return "tweet_video"
    elif content_type == "image/gif":
        return "tweet_gif"
    return "tweet_image"


def get_content_type(file_path: str) -> str:
    """Guess content type from file extension."""
    ext = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else ""
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
        "gif": "image/gif", "webp": "image/webp", "mp4": "video/mp4",
        "mov": "video/quicktime", "avi": "video/x-msvideo",
    }.get(ext, "application/octet-stream")


def resize_image(file_data: bytes, content_type: str, max_size: int = MAX_IMAGE_SIZE) -> Tuple[bytes, str]:
    """Resize an image to fit under the size limit while maintaining aspect ratio."""
    img = Image.open(io.BytesIO(file_data))

    if img.mode in ("RGBA", "P"):
        output_format = "PNG"
    else:
        output_format = "JPEG"
        if img.mode != "RGB":
            img = img.convert("RGB")

    current_size = len(file_data)
    original_width, original_height = img.size
    scale = 1.0

    print(f"Original image: {original_width}x{original_height}, {current_size} bytes")

    while current_size > max_size and scale > 0.1:
        scale *= 0.9
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        if new_width < 100 or new_height < 100:
            break

        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        if output_format == "JPEG":
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

    if current_size > max_size:
        raise ValueError(f"Could not resize image to fit under {max_size // (1024*1024)}MB limit")

    return file_data, content_type


async def upload_media(client, file_data: bytes, content_type: str, media_category: str) -> str:
    """Upload media via xdk chunked upload. Returns media_id."""
    file_size = len(file_data)
    chunk_size = 1024 * 1024  # 1MB

    # Step 1: Initialize
    print(f"Initializing upload: {file_size} bytes, {content_type}, {media_category}")
    init = client.media.initialize_upload(body={
        "total_bytes": file_size,
        "media_type": content_type,
        "media_category": media_category,
    })
    media_id = init.data.id
    print(f"Upload initialized: media_id={media_id}")

    # Step 2: Append chunks
    for i, offset in enumerate(range(0, file_size, chunk_size)):
        chunk = base64.b64encode(file_data[offset:offset + chunk_size]).decode()
        print(f"Uploading chunk {i}: {min(chunk_size, file_size - offset)} bytes")
        client.media.append_upload(id=media_id, body={"media": chunk, "segment_index": i})

    # Step 3: Finalize
    print("Finalizing upload...")
    finalize = client.media.finalize_upload(id=media_id)

    # Step 4: Wait for processing (videos)
    data = finalize.data if hasattr(finalize, "data") else None
    processing = getattr(data, "processing_info", None) if data else None

    while processing:
        state = getattr(processing, "state", None)
        print(f"Processing state: {state}")

        if state == "succeeded":
            break
        elif state == "failed":
            error = getattr(processing, "error", None)
            msg = getattr(error, "message", "Unknown error") if error else "Unknown error"
            raise ValueError(f"Media processing failed: {msg}")

        check_after = getattr(processing, "check_after_secs", 5)
        print(f"Waiting {check_after}s for processing...")
        await asyncio.sleep(check_after)

        status = client.media.get_upload_status(media_id=media_id)
        status_data = status.data if hasattr(status, "data") else None
        processing = getattr(status_data, "processing_info", None) if status_data else None

    print(f"Media upload complete: {media_id}")
    return media_id


async def upload_file(client, file_path: str, file_content_type: str = None) -> str:
    """Read a file, validate size, resize if needed, and upload. Returns media_id."""
    with open(file_path, "rb") as f:
        data = f.read()

    content_type = file_content_type or get_content_type(file_path)
    category = get_media_category(content_type)

    if category == "tweet_image" and len(data) > MAX_IMAGE_SIZE:
        print(f"Image exceeds {MAX_IMAGE_SIZE // (1024*1024)}MB limit, resizing...")
        data, content_type = resize_image(data, content_type)
    elif category == "tweet_gif" and len(data) > MAX_GIF_SIZE:
        raise ValueError(f"GIF exceeds {MAX_GIF_SIZE // (1024*1024)}MB limit")
    elif category == "tweet_video" and len(data) > MAX_VIDEO_SIZE:
        raise ValueError(f"Video exceeds {MAX_VIDEO_SIZE // (1024*1024)}MB limit")

    return await upload_media(client, data, content_type, category)
