"""Shared Discord API helpers for all discord/* apps.

Uses Discord REST API v10 with bot token authentication.
All responses are JSON — use .json() on httpx responses.
"""

import httpx

BASE_URL = "https://discord.com/api/v10"


def bot_headers(bot_token: str) -> dict:
    """Return headers for bot-authenticated requests."""
    return {
        "Authorization": f"Bot {bot_token}",
        "Content-Type": "application/json",
    }


async def send_message(
    client: httpx.AsyncClient,
    bot_token: str,
    channel_id: str,
    content: str | None = None,
    embeds: list[dict] | None = None,
) -> dict:
    """Send a message to a channel. Returns the created message object.

    At least one of content or embeds must be provided.
    See: https://discord.com/developers/docs/resources/message#create-message
    """
    payload = {}
    if content:
        payload["content"] = content
    if embeds:
        payload["embeds"] = embeds

    if not payload:
        raise ValueError("At least one of content or embeds must be provided")

    resp = await client.post(
        f"{BASE_URL}/channels/{channel_id}/messages",
        headers=bot_headers(bot_token),
        json=payload,
    )

    if resp.status_code == 403:
        raise ValueError(
            f"Bot lacks permission to send messages in channel {channel_id}. "
            "Ensure the bot has 'Send Messages' permission in this channel."
        )
    if resp.status_code == 404:
        raise ValueError(
            f"Channel {channel_id} not found. Check the channel ID is correct "
            "and the bot has access to this channel."
        )

    resp.raise_for_status()
    return resp.json()


async def get_message(
    client: httpx.AsyncClient,
    bot_token: str,
    channel_id: str,
    message_id: str,
) -> dict:
    """Get a single message by ID."""
    resp = await client.get(
        f"{BASE_URL}/channels/{channel_id}/messages/{message_id}",
        headers=bot_headers(bot_token),
    )
    resp.raise_for_status()
    return resp.json()


async def delete_message(
    client: httpx.AsyncClient,
    bot_token: str,
    channel_id: str,
    message_id: str,
) -> None:
    """Delete a message by ID."""
    resp = await client.delete(
        f"{BASE_URL}/channels/{channel_id}/messages/{message_id}",
        headers=bot_headers(bot_token),
    )
    resp.raise_for_status()


async def add_reaction(
    client: httpx.AsyncClient,
    bot_token: str,
    channel_id: str,
    message_id: str,
    emoji: str,
) -> None:
    """Add a reaction to a message. emoji should be URL-encoded (e.g. '%F0%9F%91%8D' or 'custom:id')."""
    resp = await client.put(
        f"{BASE_URL}/channels/{channel_id}/messages/{message_id}/reactions/{emoji}/@me",
        headers=bot_headers(bot_token),
    )
    resp.raise_for_status()


async def list_guilds(
    client: httpx.AsyncClient,
    bot_token: str,
) -> list[dict]:
    """List guilds the bot is a member of."""
    resp = await client.get(
        f"{BASE_URL}/users/@me/guilds",
        headers=bot_headers(bot_token),
    )
    resp.raise_for_status()
    return resp.json()


async def list_channels(
    client: httpx.AsyncClient,
    bot_token: str,
    guild_id: str,
) -> list[dict]:
    """List channels in a guild."""
    resp = await client.get(
        f"{BASE_URL}/guilds/{guild_id}/channels",
        headers=bot_headers(bot_token),
    )
    resp.raise_for_status()
    return resp.json()


async def create_dm(
    client: httpx.AsyncClient,
    bot_token: str,
    recipient_id: str,
) -> dict:
    """Open a DM channel with a user. Returns channel object."""
    resp = await client.post(
        f"{BASE_URL}/users/@me/channels",
        headers=bot_headers(bot_token),
        json={"recipient_id": recipient_id},
    )
    resp.raise_for_status()
    return resp.json()
