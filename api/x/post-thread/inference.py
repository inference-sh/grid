import os
import logging
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, TextMeta, RawMeta
from pydantic import BaseModel, Field
from typing import Optional, List
from .x_helper import upload_file, get_content_type

logger = logging.getLogger(__name__)


class ThreadTweet(BaseModel):
    """A single tweet within a thread."""
    text: str = Field(description="Tweet text (max 280 characters)", min_length=1, max_length=280)
    media: Optional[List[File]] = Field(
        None,
        description="Media files to attach (up to 4 images, or 1 video/GIF). Supports JPG, PNG, GIF, WEBP, MP4."
    )


class RunInput(BaseAppInput):
    """Input schema for creating a thread."""
    tweets: List[ThreadTweet] = Field(
        description="Ordered list of tweets to post as a thread (2-25 tweets)",
        min_length=2,
        max_length=25
    )


class PostedTweet(BaseModel):
    """A single posted tweet result."""
    tweet_id: str = Field(description="ID of the posted tweet")
    tweet_url: str = Field(description="URL of the posted tweet")
    index: int = Field(description="Position in the thread (0-based)")


class RunOutput(BaseAppOutput):
    """Output schema for created thread."""
    thread_id: str = Field(description="ID of the first tweet (thread root)")
    thread_url: str = Field(description="URL of the first tweet")
    tweets: List[PostedTweet] = Field(description="All posted tweets in order")
    count: int = Field(description="Number of tweets posted")


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
        logger.info("X.com client initialized")

    async def run(self, input_data: RunInput) -> RunOutput:
        """Create a thread on X.com by posting tweets sequentially as replies."""
        logger.info(f"Creating thread with {len(input_data.tweets)} tweets")

        posted: List[PostedTweet] = []
        previous_tweet_id: Optional[str] = None
        total_chars = 0

        try:
            for i, tweet in enumerate(input_data.tweets):
                if len(tweet.text) > 280:
                    raise ValueError(f"Tweet {i} exceeds 280 characters ({len(tweet.text)} chars)")

                logger.info(f"Posting tweet {i + 1}/{len(input_data.tweets)}: {tweet.text[:50]}...")

                media_ids = []
                if tweet.media:
                    if len(tweet.media) > 4:
                        raise ValueError(f"Tweet {i}: maximum 4 media files allowed per tweet")

                    def get_type(m: File) -> str:
                        return m.content_type or get_content_type(m.path)

                    has_video = any(get_type(m).startswith("video/") for m in tweet.media)
                    has_gif = any(get_type(m) == "image/gif" for m in tweet.media)

                    if has_video and len(tweet.media) > 1:
                        raise ValueError(f"Tweet {i}: only 1 video allowed per tweet")
                    if has_gif and len(tweet.media) > 1:
                        raise ValueError(f"Tweet {i}: only 1 GIF allowed per tweet")

                    for media_file in tweet.media:
                        media_id = await upload_file(self.client, media_file.path, media_file.content_type)
                        media_ids.append(media_id)

                    logger.info(f"Tweet {i}: uploaded {len(media_ids)} media files")

                payload = {"text": tweet.text}

                if media_ids:
                    payload["media"] = {"media_ids": media_ids}

                if previous_tweet_id:
                    payload["reply"] = {"in_reply_to_tweet_id": previous_tweet_id}

                response = self.client.posts.create(body=payload)
                tweet_id = response.data.id
                tweet_url = f"https://x.com/i/web/status/{tweet_id}"

                logger.info(f"Tweet {i + 1} posted: {tweet_url}")

                posted.append(PostedTweet(
                    tweet_id=tweet_id,
                    tweet_url=tweet_url,
                    index=i
                ))

                previous_tweet_id = tweet_id
                total_chars += len(tweet.text)

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"X API error on tweet {len(posted)}: {type(e).__name__}: {e}")
            error_msg = str(e).lower()
            already_posted = f" ({len(posted)} tweets were already posted)" if posted else ""
            if "duplicate" in error_msg:
                raise ValueError(f"Duplicate content detected on tweet {len(posted)}{already_posted}")
            elif "rate limit" in error_msg:
                raise ValueError(f"Rate limit exceeded on tweet {len(posted)}{already_posted}")
            elif "unauthorized" in error_msg or "forbidden" in error_msg or "401" in error_msg or "403" in error_msg:
                raise ValueError(f"Authorization failed: {e}")
            else:
                raise ValueError(f"X.com API error on tweet {len(posted)}: {e}{already_posted}")

        thread_root = posted[0]
        logger.info(f"Thread created: {len(posted)} tweets, {total_chars} total chars, root={thread_root.tweet_url}")

        return RunOutput(
            thread_id=thread_root.tweet_id,
            thread_url=thread_root.tweet_url,
            tweets=posted,
            count=len(posted),
            output_meta=OutputMeta(
                outputs=[RawMeta(extra={"api_calls": len(posted)})]
            )
        )

    async def unload(self):
        """Cleanup resources."""
        self.client = None
