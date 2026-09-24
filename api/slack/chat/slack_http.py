"""HTTP helper for the slack/chat app.

Slack's Web API answers 200 with {"ok": false, "error": "..."} for most
failures, so the status code alone proves nothing — every response is checked
for ok, and the error code is turned into something a caller can act on.

print() is used instead of a module-level logger, which would not reach task
logs from a helper.
"""

import asyncio
import os
from typing import Any, Dict, Optional

import httpx

RETRY_STATUSES = (429, 500, 502, 503, 504)
MAX_ATTEMPTS = 4

DEFAULT_BASE_URL = "https://slack.com/api"

# Slack error codes worth explaining rather than echoing.
ERROR_HELP = {
    "not_in_channel": "the bot is not in that channel — invite it with /invite @yourbot",
    "channel_not_found": "no such channel, or the bot cannot see it; private channels "
    "require the bot to be a member",
    "invalid_auth": "the SLACK_BOT_TOKEN secret is not a valid token",
    "token_revoked": "the SLACK_BOT_TOKEN secret has been revoked; reinstall the app",
    "account_inactive": "the token belongs to a deactivated workspace or user",
    "missing_scope": "the bot token lacks a required OAuth scope — chat:write to post, "
    "channels:read for list_channels, chat:write.customize to override name or icon",
    "cant_update_message": "only the message's author can edit it, and it must not be too old",
    "message_not_found": "no message with that ts in that channel",
    "is_archived": "the channel is archived",
    "msg_too_long": "the message exceeds Slack's 40,000 character limit",
    "rate_limited": "Slack is rate limiting this token",
}


def get_bot_token() -> str:
    """The workspace bot token, from the team's SLACK_BOT_TOKEN secret.

    Deliberately not an app input: a bot token is a credential, and the
    platform already stores one per team.
    """
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "SLACK_BOT_TOKEN is not set. This app reads the workspace bot token from your "
            "team's secret of that name — create a Slack app, install it to the workspace, "
            "copy the Bot User OAuth Token (starts with xoxb-) and set it with "
            "`belt secrets set SLACK_BOT_TOKEN <token>`. A secret whose record exists but "
            "holds an empty value is not injected at all."
        )
    return token.strip()


def resolve_base_url(base_url: Optional[str]) -> str:
    candidate = (base_url or os.environ.get("SLACK_API_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
    if not candidate.startswith(("http://", "https://")):
        raise ValueError(f"base_url must start with http:// or https://, got {candidate!r}")
    return candidate


class SlackClient:
    def __init__(self, cancelled=None, timeout: float = 60.0):
        self._client = httpx.AsyncClient(timeout=timeout)
        self._cancelled = cancelled or (lambda: False)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def call(
        self,
        method_name: str,
        payload: Dict[str, Any],
        *,
        base_url: Optional[str] = None,
        method: str = "POST",
    ) -> Dict[str, Any]:
        """Call a Web API method and return its body, raising on ok: false."""
        url = f"{resolve_base_url(base_url)}/{method_name}"
        headers = {"Authorization": f"Bearer {get_bot_token()}"}

        for attempt in range(1, MAX_ATTEMPTS + 1):
            if method == "GET":
                response = await self._client.get(url, headers=headers, params=payload)
            else:
                response = await self._client.post(url, headers=headers, json=payload)

            if (
                response.status_code in RETRY_STATUSES
                and attempt < MAX_ATTEMPTS
                and not self._cancelled()
            ):
                delay = _retry_after(response, attempt)
                print(
                    f"slack {method_name} returned {response.status_code}; "
                    f"retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.0f}s"
                )
                await asyncio.sleep(delay)
                continue

            if response.status_code >= 400:
                raise RuntimeError(
                    f"Slack {method_name} HTTP {response.status_code}: {response.text[:600]}"
                )

            body = _json(response, method_name)
            if body.get("ok"):
                return body

            error = str(body.get("error") or "unknown_error")
            # Slack answers 200 + ok:false for its own rate limiting too.
            if error == "ratelimited" and attempt < MAX_ATTEMPTS and not self._cancelled():
                delay = _retry_after(response, attempt)
                print(f"slack {method_name} rate limited; retry in {delay:.0f}s")
                await asyncio.sleep(delay)
                continue

            raise RuntimeError(_explain(method_name, error, body))

        raise RuntimeError(f"Slack {method_name} exhausted {MAX_ATTEMPTS} attempts")

    async def permalink(
        self, channel: str, ts: str, *, base_url: Optional[str] = None
    ) -> str:
        """Best-effort permalink. A missing link must not fail a successful post."""
        if not channel or not ts:
            return ""
        try:
            body = await self.call(
                "chat.getPermalink",
                {"channel": channel, "message_ts": ts},
                base_url=base_url,
                method="GET",
            )
            return str(body.get("permalink") or "")
        except RuntimeError as exc:
            print(f"permalink lookup failed, continuing without it: {exc}")
            return ""


def _retry_after(response: httpx.Response, attempt: int) -> float:
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return min(2 ** (attempt - 1), 30)


def _json(response: httpx.Response, method_name: str) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        raise RuntimeError(
            f"Slack {method_name} returned a non-JSON body: {response.text[:400]}"
        ) from None
    if not isinstance(body, dict):
        raise RuntimeError(f"Slack {method_name} returned {type(body).__name__}, expected an object")
    return body


def _explain(method_name: str, error: str, body: Dict[str, Any]) -> str:
    message = f"Slack {method_name} failed: {error}"
    if error in ERROR_HELP:
        message += f" — {ERROR_HELP[error]}"
    needed = body.get("needed")
    if needed:
        message += f" (needed scope: {needed})"
    warnings = body.get("response_metadata") or {}
    messages = warnings.get("messages")
    if messages:
        message += f" | {'; '.join(str(m) for m in messages)[:300]}"
    return message
