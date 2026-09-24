"""PagerDuty Events API v2 — page a human, and take it back.

Paging as a deliberate act. Nothing here depends on alert routing: a caller
that has decided something is real triggers an incident and says why; a caller
that has decided it is not resolves one. Both sides matter — an incident that
nobody resolves keeps escalating long after the condition cleared.

The routing key is the team's own PAGERDUTY_KEY secret, never a request field.
Free to run — every function reports empty usage metas so pricing zeroes it.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, OutputMeta
from pydantic import BaseModel, Field

from .pagerduty_http import PagerDutyClient, parse_rfc3339, rfc3339

FREE = OutputMeta(inputs=[], outputs=[])

DEDUP_HELP = (
    "Groups events about one problem. A trigger with a dedup_key that is already open "
    "updates that incident instead of opening a second one, and acknowledge/resolve need "
    "the same key. Choose it from the condition (e.g. 'log-error-burst:api'), not from "
    "the moment — a fresh key per firing is how one outage becomes twenty pages. "
    "Omitted on a trigger, PagerDuty generates one and returns it."
)
SEVERITY_HELP = (
    "How bad this is for the affected system. critical and error page under most "
    "escalation policies; warning and info usually only appear in the incident feed."
)
SOURCE_HELP = (
    "Where the problem is observed — a hostname, service or app ref. PagerDuty groups "
    "and displays incidents by this."
)


class EventInput(BaseAppInput):
    """Shared by every call: which PagerDuty region to send to."""

    base_url: Optional[str] = Field(
        default=None,
        description="Events API base, for the EU service region "
        "(https://events.eu.pagerduty.com). Defaults to the app's configured "
        "PAGERDUTY_EVENTS_URL, or the US endpoint. The routing key always comes from "
        "your team's PAGERDUTY_KEY secret, never from the request.",
    )


class Link(BaseModel):
    href: str = Field(description="URL to open from the incident, e.g. a dashboard or runbook.")
    text: str = Field(default="", description="Link label shown in PagerDuty.")


class TriggerAlertInput(EventInput):
    summary: str = Field(
        max_length=1024,
        description="One line a woken person can act on: what is broken, where, how big. "
        "This is the incident title and often the whole push notification.",
        examples=["11 tasks stuck across 4 teams — scheduler dispatch exhausted retries"],
    )
    source: str = Field(description=SOURCE_HELP, examples=["api.inference.sh"])
    severity: Literal["critical", "error", "warning", "info"] = Field(
        default="error", description=SEVERITY_HELP
    )
    dedup_key: Optional[str] = Field(default=None, max_length=255, description=DEDUP_HELP)
    component: Optional[str] = Field(
        default=None, description="Part of the source that failed, e.g. scheduler or postgres."
    )
    group: Optional[str] = Field(
        default=None, description="Logical grouping, e.g. prod-us-east or the cluster name."
    )
    class_name: Optional[str] = Field(
        default=None,
        description="Kind of problem, e.g. dispatch-failure or latency. Sent as PagerDuty's "
        "`class` field.",
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Evidence shown on the incident: counts, ids, the verdict and why. "
        "This is where a triage agent puts its reasoning so the human does not start cold.",
    )
    links: List[Link] = Field(
        default_factory=list, description="Dashboards, runbooks or log queries to open."
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="When the problem started, RFC3339. Defaults to now. Use the condition's "
        "real start so the incident timeline is not skewed by triage time.",
    )


class TriggerAlertOutput(BaseAppOutput):
    dedup_key: str = Field(
        description="Key identifying this incident — keep it; acknowledge and resolve need it."
    )
    status: str = Field(description="PagerDuty's status, 'success' when accepted.")
    message: str = Field(description="PagerDuty's response message.")


class AlertActionInput(EventInput):
    dedup_key: str = Field(
        max_length=255,
        description="The key returned by trigger_alert, identifying the incident to act on.",
    )


class AcknowledgeAlertOutput(BaseAppOutput):
    dedup_key: str = Field(description="The incident that was acknowledged.")
    status: str = Field(description="PagerDuty's status, 'success' when accepted.")
    message: str = Field(description="PagerDuty's response message.")


class ResolveAlertOutput(BaseAppOutput):
    dedup_key: str = Field(description="The incident that was resolved.")
    status: str = Field(description="PagerDuty's status, 'success' when accepted.")
    message: str = Field(description="PagerDuty's response message.")


class SendChangeEventInput(EventInput):
    summary: str = Field(
        max_length=1024,
        description="What changed, one line. Change events never page — they land on the "
        "service timeline so the next incident shows what shipped just before it.",
        examples=["deployed api-v1048"],
    )
    source: Optional[str] = Field(
        default=None, description="Who or what made the change, e.g. a CI job or a person."
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Extra context: version, commit, who approved it."
    )
    links: List[Link] = Field(
        default_factory=list, description="Links to the build, commit or release notes."
    )
    timestamp: Optional[str] = Field(
        default=None, description="When the change happened, RFC3339. Defaults to now."
    )


class SendChangeEventOutput(BaseAppOutput):
    status: str = Field(description="PagerDuty's status, 'success' when accepted.")
    message: str = Field(description="PagerDuty's response message.")


class App(BaseApp):
    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.client = PagerDutyClient(cancelled=self._cancelled)
        self.logger.info("pagerduty app initialized")

    def _cancelled(self) -> bool:
        context = getattr(self, "context", None)
        return bool(getattr(context, "cancel_requested", False))

    async def trigger_alert(self, input_data: TriggerAlertInput) -> TriggerAlertOutput:
        """Open or update a PagerDuty incident. This pages whoever is on call."""
        payload: Dict[str, Any] = {
            "summary": input_data.summary,
            "source": input_data.source,
            "severity": input_data.severity,
            "timestamp": _when(input_data.timestamp),
        }
        for field, value in (
            ("component", input_data.component),
            ("group", input_data.group),
            ("class", input_data.class_name),
            ("custom_details", input_data.details),
        ):
            if value:
                payload[field] = value

        body: Dict[str, Any] = {"event_action": "trigger", "payload": payload}
        if input_data.dedup_key:
            body["dedup_key"] = input_data.dedup_key
        if input_data.links:
            body["links"] = [{"href": l.href, "text": l.text} for l in input_data.links]

        self.logger.info(
            f"triggering {input_data.severity} incident from {input_data.source}: "
            f"{input_data.summary[:120]}"
        )
        result = await self.client.enqueue(body, base_url=input_data.base_url)
        dedup_key = str(result.get("dedup_key") or input_data.dedup_key or "")
        self.logger.info(f"triggered incident dedup_key={dedup_key}")

        return TriggerAlertOutput(
            dedup_key=dedup_key,
            status=str(result.get("status") or ""),
            message=str(result.get("message") or ""),
            output_meta=FREE,
        )

    async def acknowledge_alert(self, input_data: AlertActionInput) -> AcknowledgeAlertOutput:
        """Acknowledge an incident: someone is on it, stop escalating."""
        self.logger.info(f"acknowledging {input_data.dedup_key}")
        result = await self.client.enqueue(
            {"event_action": "acknowledge", "dedup_key": input_data.dedup_key},
            base_url=input_data.base_url,
        )
        return AcknowledgeAlertOutput(
            dedup_key=input_data.dedup_key,
            status=str(result.get("status") or ""),
            message=str(result.get("message") or ""),
            output_meta=FREE,
        )

    async def resolve_alert(self, input_data: AlertActionInput) -> ResolveAlertOutput:
        """Resolve an incident: the condition is over, close it and stop the escalation."""
        self.logger.info(f"resolving {input_data.dedup_key}")
        result = await self.client.enqueue(
            {"event_action": "resolve", "dedup_key": input_data.dedup_key},
            base_url=input_data.base_url,
        )
        return ResolveAlertOutput(
            dedup_key=input_data.dedup_key,
            status=str(result.get("status") or ""),
            message=str(result.get("message") or ""),
            output_meta=FREE,
        )

    async def send_change_event(self, input_data: SendChangeEventInput) -> SendChangeEventOutput:
        """Record a deploy or config change on the service timeline. Never pages."""
        payload: Dict[str, Any] = {
            "summary": input_data.summary,
            "timestamp": _when(input_data.timestamp),
        }
        if input_data.source:
            payload["source"] = input_data.source
        if input_data.details:
            payload["custom_details"] = input_data.details

        body: Dict[str, Any] = {"payload": payload}
        if input_data.links:
            body["links"] = [{"href": l.href, "text": l.text} for l in input_data.links]

        self.logger.info(f"sending change event: {input_data.summary[:120]}")
        result = await self.client.enqueue_change(body, base_url=input_data.base_url)
        return SendChangeEventOutput(
            status=str(result.get("status") or ""),
            message=str(result.get("message") or ""),
            output_meta=FREE,
        )

    async def unload(self):
        await self.client.aclose()

    async def on_cancel(self):
        return True


def _when(timestamp: Optional[str]) -> str:
    if timestamp:
        return rfc3339(parse_rfc3339(timestamp, "timestamp"))
    return rfc3339(datetime.now(timezone.utc))
