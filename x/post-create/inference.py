import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional, List
from .x_helper import upload_file, get_content_type


class AppInput(BaseAppInput):
    """Input schema for creating a post."""
    text: str = Field(description="Post text (max 280 characters)", min_length=1, max_length=280)
    reply_to_tweet_id: Optional[str] = Field(None, description="Tweet ID to reply to (optional)")
    quote_tweet_id: Optional[str] = Field(None, description="Tweet ID to quote (optional)")
    media: Optional[List[File]] = Field(
        None,
        description="Media files to attach (up to 4 images, or 1 video/GIF). Supports JPG, PNG, GIF, WEBP, MP4."
    )


class AppOutput(BaseAppOutput):
    """Output schema for created post."""
    tweet_id: str = Field(description="ID of the created post")
    tweet_url: str = Field(description="URL of the created post")


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
        self.client = Client(access_token=access_token)
        print("X.com client initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Create a post on X.com."""
        if len(input_data.text) > 280:
            raise ValueError(f"Post text exceeds 280 characters ({len(input_data.text)} chars)")

        print(f"Creating post: {input_data.text[:50]}...")

        try:
            media_ids = []
            if input_data.media:
                if len(input_data.media) > 4:
                    raise ValueError("Maximum 4 media files allowed per post")

                # Check for mixed media types
                def get_type(m: File) -> str:
                    return m.content_type or get_content_type(m.path)

                has_video = any(get_type(m).startswith("video/") for m in input_data.media)
                has_gif = any(get_type(m) == "image/gif" for m in input_data.media)

                if has_video and len(input_data.media) > 1:
                    raise ValueError("Only 1 video allowed per post")
                if has_gif and len(input_data.media) > 1:
                    raise ValueError("Only 1 GIF allowed per post")

                for media_file in input_data.media:
                    media_id = await upload_file(self.client, media_file.path, media_file.content_type)
                    media_ids.append(media_id)

                print(f"Uploaded {len(media_ids)} media files: {media_ids}")

            payload = {"text": input_data.text}

            if media_ids:
                payload["media"] = {"media_ids": media_ids}

            if input_data.reply_to_tweet_id:
                payload["reply"] = {"in_reply_to_tweet_id": input_data.reply_to_tweet_id}
                print(f"Replying to: {input_data.reply_to_tweet_id}")

            if input_data.quote_tweet_id:
                payload["quote_tweet_id"] = input_data.quote_tweet_id
                print(f"Quoting: {input_data.quote_tweet_id}")

            response = self.client.posts.create(body=payload)

            tweet_id = response.data.id
            tweet_url = f"https://x.com/i/web/status/{tweet_id}"

            print(f"Post created successfully: {tweet_url}")

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
                raise ValueError("This post was already created (duplicate content)")
            elif "rate limit" in error_msg:
                raise ValueError("Rate limit exceeded. Please try again later.")
            elif "unauthorized" in error_msg or "forbidden" in error_msg or "401" in error_msg or "403" in error_msg:
                raise ValueError(f"Authorization failed: {e}")
            else:
                raise ValueError(f"X.com API error: {e}")

    async def unload(self):
        """Cleanup resources."""
        self.client = None
