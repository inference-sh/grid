import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional, List
from .x_helper import upload_file


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

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            media_ids = []
            if input_data.media:
                if len(input_data.media) > 4:
                    raise ValueError("Maximum 4 media files allowed")
                for m in input_data.media:
                    media_id = await upload_file(self.client, m.path, m.content_type)
                    media_ids.append(media_id)

            payload = {"text": input_data.text}
            if media_ids:
                payload["media"] = {"media_ids": media_ids}
            if input_data.reply_to_tweet_id:
                payload["reply"] = {"in_reply_to_tweet_id": input_data.reply_to_tweet_id}
            if input_data.quote_tweet_id:
                payload["quote_tweet_id"] = input_data.quote_tweet_id

            response = self.client.posts.create(body=payload)
            tweet_id = response.data.id

            return AppOutput(tweet_id=tweet_id, post_url=f"https://x.com/i/web/status/{tweet_id}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"X API error: {e}")

    async def unload(self):
        self.client = None
