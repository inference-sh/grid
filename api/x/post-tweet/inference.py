import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional


class AppInput(BaseAppInput):
    """Input schema for posting a tweet."""
    text: str = Field(description="Tweet text to post (max 280 characters)", min_length=1, max_length=280)
    reply_to_tweet_id: Optional[str] = Field(None, description="Tweet ID to reply to (optional)")
    quote_tweet_id: Optional[str] = Field(None, description="Tweet ID to quote (optional)")


class AppOutput(BaseAppOutput):
    """Output schema for posted tweet."""
    tweet_id: str = Field(description="ID of the posted tweet")
    tweet_url: str = Field(description="URL of the posted tweet")


class App(BaseApp):
    client: Client = None

    async def setup(self, metadata):
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

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Post a tweet to X.com."""
        # Validate tweet length
        if len(input_data.text) > 280:
            raise ValueError(f"Tweet text exceeds 280 characters ({len(input_data.text)} chars)")
        
        print(f"Posting tweet: {input_data.text[:50]}...")
        
        try:
            # Build tweet payload
            payload = {"text": input_data.text}
            
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
            
        except Exception as e:
            error_msg = str(e).lower()
            if "duplicate" in error_msg:
                raise ValueError("This tweet was already posted (duplicate content)")
            elif "rate limit" in error_msg:
                raise ValueError("Rate limit exceeded. Please try again later.")
            elif "unauthorized" in error_msg or "forbidden" in error_msg or "401" in error_msg or "403" in error_msg:
                raise ValueError(
                    "Authorization failed. Please reconnect the X.com integration in Settings."
                )
            else:
                raise ValueError(f"X.com API error: {e}")

    async def unload(self):
        """Cleanup resources."""
        self.client = None
