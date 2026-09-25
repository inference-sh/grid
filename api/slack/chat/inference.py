"""Slack chat — post a message, edit it, delete it, find the channel.

Outbound and unprompted: nobody has to message the bot first. That is what
separates this from a conversation transport, and it is what a monitor or a
triage agent needs when it decides at 03:00 that something is worth writing
down.

The bot token comes from the team's Slack integration (credential), never a request field.
Free to run — every function reports empty usage metas so pricing zeroes it.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, OutputMeta
from pydantic import BaseModel, Field, model_validator

from .slack_http import SlackClient

FREE = OutputMeta(inputs=[], outputs=[])

CHANNEL_HELP = (
    "Channel id (C0123456789) or name with or without the hash. Ids survive renames, so "
    "prefer them for anything long-lived; use list_channels to find one. The bot must be "
    "in the channel — invite it with /invite @yourbot, or posting fails with not_in_channel."
)
TEXT_HELP = (
    "Message text, in Slack's mrkdwn: *bold*, _italic_, `code`, ```block```, <url|label>, "
    "<@U123> to mention. When blocks are given this becomes the notification fallback, so "
    "keep it meaningful rather than empty."
)
THREAD_HELP = (
    "Reply inside a thread by passing the parent message's ts. Threading a findings feed "
    "keeps a channel readable: one message per incident, updates as replies."
)


class SlackInput(BaseAppInput):
    """Every function shares the workspace connection."""


class PostMessageInput(SlackInput):
    channel: str = Field(description=CHANNEL_HELP, examples=["#alerts"])
    text: str = Field(description=TEXT_HELP, examples=["Deploy finished: api-v1048"])
    blocks: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Block Kit blocks for rich layout. Each item is one block object, e.g. "
        '{"type": "section", "text": {"type": "mrkdwn", "text": "*hello*"}}. Leave unset '
        "for a plain message.",
    )
    thread_ts: Optional[str] = Field(default=None, description=THREAD_HELP)
    reply_broadcast: bool = Field(
        default=False,
        description="For a threaded reply, also show it in the channel. Use sparingly: it "
        "notifies everyone, which is the thing threading was avoiding.",
    )
    unfurl_links: bool = Field(
        default=True, description="Expand link previews. Turn off for noisy log links."
    )
    username: Optional[str] = Field(
        default=None,
        description="Override the bot's display name for this message. Needs chat:write.customize.",
    )
    icon_emoji: Optional[str] = Field(
        default=None,
        description="Override the bot's avatar with an emoji, e.g. :rotating_light:. "
        "Needs chat:write.customize.",
    )

    @model_validator(mode="after")
    def _check_broadcast(self):
        if self.reply_broadcast and not self.thread_ts:
            raise ValueError("reply_broadcast only applies to a threaded reply; set thread_ts")
        return self


class PostMessageOutput(BaseAppOutput):
    ts: str = Field(
        description="Message timestamp — its id. Keep it to edit, delete, or thread replies "
        "onto this message."
    )
    channel: str = Field(description="Channel id the message landed in.")
    permalink: str = Field(description="Link to the message, empty if Slack did not return one.")


class UpdateMessageInput(SlackInput):
    channel: str = Field(description="Channel id the message is in.")
    ts: str = Field(description="Timestamp of the message to edit, from post_message.")
    text: str = Field(description=TEXT_HELP)
    blocks: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Replacement Block Kit blocks. Omitted, existing blocks are cleared "
        "and the message becomes plain text.",
    )


class UpdateMessageOutput(BaseAppOutput):
    ts: str = Field(description="Timestamp of the edited message.")
    channel: str = Field(description="Channel id the message is in.")


class DeleteMessageInput(SlackInput):
    channel: str = Field(description="Channel id the message is in.")
    ts: str = Field(description="Timestamp of the message to delete.")


class DeleteMessageOutput(BaseAppOutput):
    ts: str = Field(description="Timestamp of the deleted message.")
    channel: str = Field(description="Channel id the message was in.")
    deleted: bool = Field(description="True when Slack accepted the deletion.")


class ListChannelsInput(SlackInput):
    types: List[Literal["public_channel", "private_channel", "mpim", "im"]] = Field(
        default_factory=lambda: ["public_channel"],
        description="Which conversation kinds to list. Private channels only appear when "
        "the bot is a member.",
    )
    limit: int = Field(
        default=100, ge=1, le=1000, description="Channels per page (1 to 1000)."
    )
    cursor: Optional[str] = Field(
        default=None, description="next_cursor from a previous call, to fetch the next page."
    )
    exclude_archived: bool = Field(default=True, description="Skip archived channels.")


class Channel(BaseModel):
    id: str = Field(description="Channel id — pass this as `channel` when posting.")
    name: str = Field(description="Channel name without the hash.")
    is_private: bool = Field(description="Whether it is a private channel.")
    is_member: bool = Field(
        description="Whether the bot is in it. Posting to a channel it is not in fails."
    )
    topic: str = Field(description="Channel topic, empty when unset.")


class ListChannelsOutput(BaseAppOutput):
    channels: List[Channel] = Field(description="Matching channels.")
    count: int = Field(description="Number of channels returned.")
    next_cursor: str = Field(
        description="Pass back as `cursor` for the next page. Empty when this is the last page."
    )


class App(BaseApp):
    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.client = SlackClient(cancelled=self._cancelled)
        self.logger.info("slack chat app initialized")

    def _cancelled(self) -> bool:
        context = getattr(self, "context", None)
        return bool(getattr(context, "cancel_requested", False))

    async def post_message(self, input_data: PostMessageInput) -> PostMessageOutput:
        """Post a message to a channel, or as a threaded reply."""
        body: Dict[str, Any] = {
            "channel": _channel(input_data.channel),
            "text": input_data.text,
            "unfurl_links": input_data.unfurl_links,
        }
        for field, value in (
            ("blocks", input_data.blocks),
            ("thread_ts", input_data.thread_ts),
            ("username", input_data.username),
            ("icon_emoji", input_data.icon_emoji),
        ):
            if value:
                body[field] = value
        if input_data.reply_broadcast:
            body["reply_broadcast"] = True

        self.logger.info(
            f"posting to {input_data.channel}"
            f"{' in thread ' + input_data.thread_ts if input_data.thread_ts else ''}: "
            f"{input_data.text[:100]}"
        )
        result = await self.client.call("chat.postMessage", body)

        ts = str(result.get("ts") or "")
        channel_id = str(result.get("channel") or "")
        permalink = await self.client.permalink(channel_id, ts)
        self.logger.info(f"posted ts={ts} channel={channel_id}")

        return PostMessageOutput(
            ts=ts, channel=channel_id, permalink=permalink, output_meta=FREE
        )

    async def update_message(self, input_data: UpdateMessageInput) -> UpdateMessageOutput:
        """Edit a message already posted — the findings-feed update path."""
        body: Dict[str, Any] = {
            "channel": _channel(input_data.channel),
            "ts": input_data.ts,
            "text": input_data.text,
        }
        # Slack keeps old blocks unless they are explicitly replaced or cleared.
        body["blocks"] = input_data.blocks if input_data.blocks else []

        self.logger.info(f"updating {input_data.ts} in {input_data.channel}")
        result = await self.client.call("chat.update", body)
        return UpdateMessageOutput(
            ts=str(result.get("ts") or input_data.ts),
            channel=str(result.get("channel") or ""),
            output_meta=FREE,
        )

    async def delete_message(self, input_data: DeleteMessageInput) -> DeleteMessageOutput:
        """Delete a message the bot posted."""
        self.logger.info(f"deleting {input_data.ts} in {input_data.channel}")
        result = await self.client.call(
            "chat.delete",
            {"channel": _channel(input_data.channel), "ts": input_data.ts},
        )
        return DeleteMessageOutput(
            ts=str(result.get("ts") or input_data.ts),
            channel=str(result.get("channel") or ""),
            deleted=True,
            output_meta=FREE,
        )

    async def list_channels(self, input_data: ListChannelsInput) -> ListChannelsOutput:
        """List conversations, to find a channel id or check the bot is a member."""
        params: Dict[str, Any] = {
            "types": ",".join(input_data.types),
            "limit": input_data.limit,
            "exclude_archived": input_data.exclude_archived,
        }
        if input_data.cursor:
            params["cursor"] = input_data.cursor

        result = await self.client.call(
            "conversations.list", params, method="GET"
        )
        channels = [
            Channel(
                id=str(item.get("id") or ""),
                name=str(item.get("name") or ""),
                is_private=bool(item.get("is_private")),
                is_member=bool(item.get("is_member")),
                topic=str(((item.get("topic") or {}).get("value")) or ""),
            )
            for item in (result.get("channels") or [])
        ]
        cursor = str(((result.get("response_metadata") or {}).get("next_cursor")) or "")
        self.logger.info(f"list_channels returned {len(channels)} channels")

        return ListChannelsOutput(
            channels=channels, count=len(channels), next_cursor=cursor, output_meta=FREE
        )

    async def unload(self):
        await self.client.aclose()

    async def on_cancel(self):
        return True


def _channel(value: str) -> str:
    """Slack takes an id or a name; a leading hash is not part of either."""
    return value.strip().lstrip("#")
