import os
import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional, List
from .discord_helper import send_message


class EmbedField(BaseAppInput):
    """A field within a Discord embed."""
    name: str = Field(description="Field name")
    value: str = Field(description="Field value")
    inline: bool = Field(default=False, description="Display inline")


class Embed(BaseAppInput):
    """Discord rich embed object."""
    title: Optional[str] = Field(None, description="Embed title (max 256 characters)")
    description: Optional[str] = Field(None, description="Embed description (max 4096 characters)")
    url: Optional[str] = Field(None, description="URL the title links to")
    color: Optional[int] = Field(None, description="Color code (decimal, e.g. 5814783 for blue)")
    image_url: Optional[str] = Field(None, description="Image URL to display in the embed")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL (small image, top-right)")
    fields: Optional[List[EmbedField]] = Field(None, description="Embed fields (max 25)")


class AppInput(BaseAppInput):
    """Input schema for sending a Discord message."""
    channel_id: str = Field(description="Discord channel ID to send the message to")
    content: Optional[str] = Field(None, description="Message text (max 2000 characters)")
    embeds: Optional[List[Embed]] = Field(None, description="Rich embeds to include (max 10)")


class AppOutput(BaseAppOutput):
    """Output schema for the sent message."""
    message_id: str = Field(description="ID of the sent message")
    channel_id: str = Field(description="Channel the message was sent to")
    message_url: str = Field(description="URL to the message")


class App(BaseApp):
    client: httpx.AsyncClient = None
    bot_token: str = None

    async def setup(self):
        """Initialize the Discord bot client."""
        self.bot_token = os.environ.get("DISCORD_BOT_TOKEN")
        if not self.bot_token:
            raise ValueError(
                "DISCORD_BOT_TOKEN not found. "
                "Please connect your Discord integration in Settings and provide a bot token."
            )
        self.client = httpx.AsyncClient(timeout=30)
        self.logger.info("Discord bot client initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Send a message to a Discord channel."""
        if input_data.content and len(input_data.content) > 2000:
            raise ValueError(f"Message content exceeds 2000 characters ({len(input_data.content)} chars)")

        self.logger.info(f"Sending message to channel {input_data.channel_id}")

        # Build embeds payload
        embeds = None
        if input_data.embeds:
            if len(input_data.embeds) > 10:
                raise ValueError("Maximum 10 embeds allowed per message")
            embeds = []
            for e in input_data.embeds:
                embed = {}
                if e.title:
                    embed["title"] = e.title[:256]
                if e.description:
                    embed["description"] = e.description[:4096]
                if e.url:
                    embed["url"] = e.url
                if e.color is not None:
                    embed["color"] = e.color
                if e.image_url:
                    embed["image"] = {"url": e.image_url}
                if e.thumbnail_url:
                    embed["thumbnail"] = {"url": e.thumbnail_url}
                if e.fields:
                    embed["fields"] = [
                        {"name": f.name, "value": f.value, "inline": f.inline}
                        for f in e.fields[:25]
                    ]
                embeds.append(embed)

        result = await send_message(
            self.client,
            self.bot_token,
            input_data.channel_id,
            content=input_data.content,
            embeds=embeds,
        )

        message_id = result["id"]
        channel_id = result["channel_id"]
        guild_id = result.get("guild_id", "@me")
        message_url = f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"

        self.logger.info(f"Message sent: {message_url}")

        return AppOutput(
            message_id=message_id,
            channel_id=channel_id,
            message_url=message_url,
        )

    async def unload(self):
        """Cleanup resources."""
        if self.client:
            await self.client.aclose()
