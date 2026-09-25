"""Grafana: alerts, silences, annotations, datasources, and logs via Loki.

Logs live here rather than in their own app because querying Loki goes through
Grafana's datasource proxy — same host, same token, same permissions. A
separate Loki app would have been a Grafana client wearing another name.

Point it at any Grafana: the URL and token both come from the team's own
GRAFANA_URL and GRAFANA_API_KEY secrets, never from the request, and datasources
are resolved by uid or name, never by an id that only means something on one
instance.

Free to run — every function reports empty usage metas so pricing zeroes it.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, OutputMeta
from pydantic import BaseModel, Field, model_validator

from .grafana_http import (
    GrafanaClient,
    from_nanos,
    millis,
    parse_rfc3339,
    resolve_window,
    rfc3339,
    to_nanos,
)

ALERTMANAGER_PREFIX = "/api/alertmanager/grafana/api/v2"

FREE = OutputMeta(inputs=[], outputs=[])

DATASOURCE_HELP = (
    "Loki datasource, by uid or name as it appears in Grafana (numeric ids also work). "
    "Run list_datasources to see what this Grafana has."
)
SINCE_HELP = "Window ending now: a number followed by s, m, h, d or w. Ignored when start is given."


class GrafanaInput(BaseAppInput):
    """Shared by every function. The Grafana URL and token come from secrets and app env."""


class WindowInput(GrafanaInput):
    """Log functions take a window: relative `since`, or explicit RFC3339 bounds."""

    since: str = Field(default="15m", description=SINCE_HELP)
    start: Optional[str] = Field(
        default=None,
        description="Absolute window start, RFC3339 (2026-09-24T08:00:00Z). Overrides since.",
    )
    end: Optional[str] = Field(
        default=None, description="Absolute window end, RFC3339. Defaults to now."
    )
    datasource: str = Field(default="loki", description=DATASOURCE_HELP)


# --- datasources -------------------------------------------------------------


class ListDatasourcesInput(GrafanaInput):
    pass


class Datasource(BaseModel):
    uid: str = Field(description="Stable id — pass this as `datasource` on log functions.")
    name: str = Field(description="Display name in Grafana.")
    type: str = Field(description="Datasource type, e.g. loki, prometheus, tempo.")
    id: int = Field(description="Numeric id, unique to this Grafana instance only.")


class ListDatasourcesOutput(BaseAppOutput):
    datasources: List[Datasource] = Field(description="Datasources this token can see.")
    count: int = Field(description="Number of datasources returned.")


# --- alerts ------------------------------------------------------------------


class Matcher(BaseModel):
    """One label condition. A silence matches an alert when every matcher does."""

    name: str = Field(description="Label name, e.g. alertname, severity or app_ref.")
    value: str = Field(description="Value to match against.")
    is_regex: bool = Field(default=False, description="Treat value as a regular expression.")
    is_equal: bool = Field(
        default=True, description="False inverts the match (label must NOT equal value)."
    )


class ListAlertsInput(GrafanaInput):
    active: bool = Field(default=True, description="Include alerts that are currently firing.")
    silenced: bool = Field(default=False, description="Include alerts muted by a silence.")
    inhibited: bool = Field(default=False, description="Include alerts suppressed by another.")
    filters: List[Matcher] = Field(
        default_factory=list,
        description="Narrow the result to alerts whose labels match all of these. "
        "Leave empty for every alert.",
    )
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum alerts to return.")


class Alert(BaseModel):
    labels: Dict[str, str] = Field(description="Alert labels, including alertname and severity.")
    annotations: Dict[str, str] = Field(description="Alert annotations, such as summary.")
    state: str = Field(description="active, suppressed or unprocessed.")
    starts_at: str = Field(description="When the alert started firing, RFC3339.")
    updated_at: str = Field(description="When Grafana last updated it, RFC3339.")
    silenced_by: List[str] = Field(description="Ids of silences currently muting this alert.")
    fingerprint: str = Field(description="Grafana's id for this alert instance.")


class ListAlertsOutput(BaseAppOutput):
    alerts: List[Alert] = Field(description="Matching alerts.")
    count: int = Field(description="Number of alerts returned.")


# --- silences ----------------------------------------------------------------


class ListSilencesInput(GrafanaInput):
    include_expired: bool = Field(
        default=False, description="Include silences that have already ended."
    )
    filters: List[Matcher] = Field(
        default_factory=list,
        description="Only silences whose matchers include these label conditions.",
    )


class Silence(BaseModel):
    id: str = Field(description="Silence id — pass this to delete_silence.")
    matchers: List[Matcher] = Field(description="The label conditions this silence mutes.")
    starts_at: str = Field(description="When the silence began, RFC3339.")
    ends_at: str = Field(description="When it expires, RFC3339.")
    state: str = Field(description="active, pending or expired.")
    comment: str = Field(description="Why it was created.")
    created_by: str = Field(description="Who created it.")


class ListSilencesOutput(BaseAppOutput):
    silences: List[Silence] = Field(description="Matching silences.")
    count: int = Field(description="Number of silences returned.")


class CreateSilenceInput(GrafanaInput):
    matchers: List[Matcher] = Field(
        min_length=1,
        description="Label conditions the silence mutes. Match on the logical alert "
        "identity (alertname, severity, grafana_folder) rather than instance labels "
        "such as container id — a redeploy changes those and the silence stops matching.",
    )
    duration_minutes: int = Field(
        default=45, ge=1, le=10080, description="How long to mute, from now (1 minute to 7 days)."
    )
    comment: str = Field(
        description="Why this is being silenced. Shows in Grafana and is the only "
        "record a human reads later."
    )
    created_by: str = Field(
        default="inference-sh", description="Who to record as the author of the silence."
    )
    starts_at: Optional[str] = Field(
        default=None, description="Start the silence at this RFC3339 time instead of now."
    )


class CreateSilenceOutput(BaseAppOutput):
    silence_id: str = Field(description="Id of the created silence — keep it to delete early.")
    starts_at: str = Field(description="When the silence starts, RFC3339.")
    ends_at: str = Field(description="When it expires, RFC3339.")


class DeleteSilenceInput(GrafanaInput):
    silence_id: str = Field(
        description="Silence to expire immediately, from create_silence or list_silences. "
        "Deleting a silence lets its alerts notify again."
    )


class DeleteSilenceOutput(BaseAppOutput):
    silence_id: str = Field(description="The silence that was expired.")
    deleted: bool = Field(description="True when Grafana accepted the deletion.")


# --- annotations -------------------------------------------------------------


class CreateAnnotationInput(GrafanaInput):
    text: str = Field(description="Annotation body. Markdown renders in Grafana.")
    tags: List[str] = Field(
        default_factory=list,
        description="Tags to file it under, e.g. oncall-false-alarm. Dashboards filter on these.",
    )
    time: Optional[str] = Field(
        default=None, description="When the annotation applies, RFC3339. Defaults to now."
    )
    time_end: Optional[str] = Field(
        default=None, description="End of the annotated region, RFC3339. Omit for a point in time."
    )

    @model_validator(mode="after")
    def _check_range(self):
        if self.time_end and not self.time:
            raise ValueError("time_end needs time; give both ends of the region")
        if self.time and self.time_end:
            if parse_rfc3339(self.time, "time") > parse_rfc3339(self.time_end, "time_end"):
                raise ValueError("time must not be later than time_end")
        return self


class CreateAnnotationOutput(BaseAppOutput):
    annotation_id: int = Field(description="Id of the created annotation.")
    message: str = Field(description="Grafana's confirmation message.")


# --- logs (Loki through the datasource proxy) --------------------------------

QUERY_HELP = (
    "LogQL query. A stream selector in braces is required, optionally followed by "
    'filters: |= "text" contains, != excludes, |~ "re" regex. Chain filters to AND them. '
    'Example: {job=~".+"} |= "error" |~ "scheduler|task".'
)


class QueryLogsInput(WindowInput):
    query: str = Field(description=QUERY_HELP, examples=['{job=~".+"} |= "error"'])
    limit: int = Field(
        default=100, ge=1, le=5000, description="Maximum log lines to return (1 to 5000)."
    )
    direction: Literal["backward", "forward"] = Field(
        default="backward",
        description="backward returns newest first (find the last occurrence); "
        "forward returns oldest first (find when something started).",
    )
    max_line_length: int = Field(
        default=2000,
        ge=100,
        le=20000,
        description="Log lines longer than this are truncated, so one huge line cannot "
        "crowd out the rest of the result.",
    )


class LogLine(BaseModel):
    timestamp: str = Field(description="RFC3339 timestamp of the log line, UTC.")
    line: str = Field(description="The log line, truncated to max_line_length.")
    truncated: bool = Field(description="Whether this line was cut short.")
    labels: Dict[str, str] = Field(description="Stream labels this line came from.")


class QueryLogsOutput(BaseAppOutput):
    lines: List[LogLine] = Field(description="Matching log lines, ordered by direction.")
    count: int = Field(description="Number of lines returned.")
    streams: int = Field(description="Number of distinct label streams matched.")
    window_start: str = Field(description="Window start actually queried, RFC3339 UTC.")
    window_end: str = Field(description="Window end actually queried, RFC3339 UTC.")
    lines_scanned: int = Field(
        description="Lines Loki scanned. Zero with no results means the selector matched "
        "no stream; nonzero means it scanned data and nothing matched the filters."
    )


class QueryLogsInstantInput(GrafanaInput):
    query: str = Field(
        description="LogQL query evaluated at a single instant. Metric queries belong "
        'here: sum(count_over_time({job=~".+"} |= "error" [15m])).',
        examples=['sum(count_over_time({job=~".+"} |= "error" [15m]))'],
    )
    time: Optional[str] = Field(
        default=None, description="Instant to evaluate at, RFC3339. Defaults to now."
    )
    limit: int = Field(default=100, ge=1, le=5000, description="Maximum entries to return.")
    datasource: str = Field(default="loki", description=DATASOURCE_HELP)


class InstantResult(BaseModel):
    labels: Dict[str, str] = Field(description="Labels identifying this result.")
    value: Optional[str] = Field(default=None, description="Sample value, for metric queries.")
    line: Optional[str] = Field(default=None, description="Log line, for log queries.")
    timestamp: str = Field(description="RFC3339 timestamp of the sample or line, UTC.")


class QueryLogsInstantOutput(BaseAppOutput):
    result_type: str = Field(description="Loki's result type: vector, streams or matrix.")
    results: List[InstantResult] = Field(description="Results at the requested instant.")
    count: int = Field(description="Number of results returned.")


class LogLabelsInput(WindowInput):
    pass


class LogLabelsOutput(BaseAppOutput):
    labels: List[str] = Field(description="Label names present in the window.")
    count: int = Field(description="Number of label names returned.")


class LogLabelValuesInput(WindowInput):
    label: str = Field(
        description="Label whose values to list, e.g. job, component or container.",
        examples=["component"],
    )


class LogLabelValuesOutput(BaseAppOutput):
    label: str = Field(description="The label that was queried.")
    values: List[str] = Field(description="Values seen for that label in the window.")
    count: int = Field(description="Number of values returned.")


class LogSeriesInput(WindowInput):
    selector: str = Field(
        description="Stream selector in braces; returns the label sets that match it.",
        examples=['{job=~".+"}'],
    )


class LogSeriesOutput(BaseAppOutput):
    series: List[Dict[str, str]] = Field(description="Label sets matching the selector.")
    count: int = Field(description="Number of label sets returned.")


class App(BaseApp):
    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.client = GrafanaClient(cancelled=self._cancelled)
        self.logger.info("grafana app initialized")

    def _cancelled(self) -> bool:
        context = getattr(self, "context", None)
        return bool(getattr(context, "cancel_requested", False))

    # --- datasources ---------------------------------------------------------

    async def list_datasources(self, input_data: ListDatasourcesInput) -> ListDatasourcesOutput:
        """List the datasources this Grafana exposes — start here to find a uid."""
        payload = await self.client.request(
            "GET", "/api/datasources"
        )
        found = [
            Datasource(
                uid=str(item.get("uid") or ""),
                name=str(item.get("name") or ""),
                type=str(item.get("type") or ""),
                id=int(item.get("id") or 0),
            )
            for item in (payload or [])
        ]
        self.logger.info(f"list_datasources returned {len(found)} datasources")
        return ListDatasourcesOutput(datasources=found, count=len(found), output_meta=FREE)

    # --- alerts --------------------------------------------------------------

    async def list_alerts(self, input_data: ListAlertsInput) -> ListAlertsOutput:
        """List alerts Grafana currently knows about."""
        params: List[tuple] = [
            ("active", str(input_data.active).lower()),
            ("silenced", str(input_data.silenced).lower()),
            ("inhibited", str(input_data.inhibited).lower()),
        ]
        params += [("filter", _filter_expression(m)) for m in input_data.filters]

        payload = await self.client.request(
            "GET", f"{ALERTMANAGER_PREFIX}/alerts", params=params
        )
        alerts = [_alert(item) for item in (payload or [])][: input_data.limit]
        self.logger.info(f"list_alerts returned {len(alerts)} alerts")
        return ListAlertsOutput(alerts=alerts, count=len(alerts), output_meta=FREE)

    # --- silences ------------------------------------------------------------

    async def list_silences(self, input_data: ListSilencesInput) -> ListSilencesOutput:
        """List silences, active by default."""
        params = [("filter", _filter_expression(m)) for m in input_data.filters]
        payload = await self.client.request(
            "GET",
            f"{ALERTMANAGER_PREFIX}/silences",
            params=params or None,
        )

        silences = []
        for item in payload or []:
            silence = _silence(item)
            if not input_data.include_expired and silence.state == "expired":
                continue
            silences.append(silence)

        self.logger.info(f"list_silences returned {len(silences)} silences")
        return ListSilencesOutput(silences=silences, count=len(silences), output_meta=FREE)

    async def create_silence(self, input_data: CreateSilenceInput) -> CreateSilenceOutput:
        """Mute every alert matching the given labels for a while."""
        starts = (
            parse_rfc3339(input_data.starts_at, "starts_at")
            if input_data.starts_at
            else datetime.now(timezone.utc)
        )
        ends = starts + timedelta(minutes=input_data.duration_minutes)

        body = {
            "matchers": [
                {"name": m.name, "value": m.value, "isRegex": m.is_regex, "isEqual": m.is_equal}
                for m in input_data.matchers
            ],
            "startsAt": rfc3339(starts),
            "endsAt": rfc3339(ends),
            "createdBy": input_data.created_by,
            "comment": input_data.comment,
        }

        payload = await self.client.request(
            "POST", f"{ALERTMANAGER_PREFIX}/silences", json_body=body
        )
        silence_id = (payload or {}).get("silenceID") or (payload or {}).get("id") or ""
        self.logger.info(
            f"created silence {silence_id} for {input_data.duration_minutes}m: {input_data.comment}"
        )
        return CreateSilenceOutput(
            silence_id=silence_id,
            starts_at=rfc3339(starts),
            ends_at=rfc3339(ends),
            output_meta=FREE,
        )

    async def delete_silence(self, input_data: DeleteSilenceInput) -> DeleteSilenceOutput:
        """Expire a silence now, so its alerts can notify again."""
        await self.client.request(
            "DELETE",
            f"{ALERTMANAGER_PREFIX}/silence/{input_data.silence_id}",
        )
        self.logger.info(f"deleted silence {input_data.silence_id}")
        return DeleteSilenceOutput(
            silence_id=input_data.silence_id, deleted=True, output_meta=FREE
        )

    # --- annotations ---------------------------------------------------------

    async def create_annotation(self, input_data: CreateAnnotationInput) -> CreateAnnotationOutput:
        """Leave a note on the Grafana timeline."""
        body: Dict[str, Any] = {"text": input_data.text, "tags": input_data.tags}
        if input_data.time:
            body["time"] = millis(parse_rfc3339(input_data.time, "time"))
        if input_data.time_end:
            body["timeEnd"] = millis(parse_rfc3339(input_data.time_end, "time_end"))

        payload = await self.client.request(
            "POST", "/api/annotations", json_body=body
        )
        annotation_id = int((payload or {}).get("id") or 0)
        self.logger.info(f"created annotation {annotation_id} tags={input_data.tags}")
        return CreateAnnotationOutput(
            annotation_id=annotation_id,
            message=str((payload or {}).get("message") or ""),
            output_meta=FREE,
        )

    # --- logs ----------------------------------------------------------------

    async def query_logs(self, input_data: QueryLogsInput) -> QueryLogsOutput:
        """Query logs over a time range (LogQL query_range)."""
        start, end = resolve_window(input_data.since, input_data.start, input_data.end)
        prefix = await self.client.datasource_path(input_data.datasource)
        self.logger.info(
            f"query_logs {input_data.query!r} {start.isoformat()} to {end.isoformat()} "
            f"limit={input_data.limit} direction={input_data.direction}"
        )

        payload = await self.client.request(
            "GET",
            f"{prefix}/loki/api/v1/query_range",
            params={
                "query": input_data.query,
                "start": to_nanos(start),
                "end": to_nanos(end),
                "limit": input_data.limit,
                "direction": input_data.direction,
            },
        )

        data = (payload or {}).get("data") or {}
        streams = data.get("result") or []
        lines: List[LogLine] = []
        for stream in streams:
            labels = stream.get("stream") or {}
            for entry in stream.get("values") or []:
                nanos, text = entry[0], entry[1]
                lines.append(
                    LogLine(
                        timestamp=from_nanos(nanos),
                        line=text[: input_data.max_line_length],
                        truncated=len(text) > input_data.max_line_length,
                        labels=labels,
                    )
                )

        lines.sort(key=lambda item: item.timestamp, reverse=input_data.direction == "backward")
        lines = lines[: input_data.limit]

        scanned = _lines_scanned(data)
        self.logger.info(f"query_logs returned {len(lines)} lines, scanned {scanned}")

        return QueryLogsOutput(
            lines=lines,
            count=len(lines),
            streams=len(streams),
            window_start=start.isoformat(),
            window_end=end.isoformat(),
            lines_scanned=scanned,
            output_meta=FREE,
        )

    async def query_logs_instant(
        self, input_data: QueryLogsInstantInput
    ) -> QueryLogsInstantOutput:
        """Evaluate a log query at a single instant — use this for metric queries."""
        prefix = await self.client.datasource_path(input_data.datasource)
        params: Dict[str, Any] = {"query": input_data.query, "limit": input_data.limit}
        if input_data.time:
            params["time"] = to_nanos(parse_rfc3339(input_data.time, "time"))

        self.logger.info(f"query_logs_instant {input_data.query!r}")
        payload = await self.client.request(
            "GET", f"{prefix}/loki/api/v1/query", params=params
        )

        data = (payload or {}).get("data") or {}
        result_type = data.get("resultType") or "unknown"
        results: List[InstantResult] = []

        for item in data.get("result") or []:
            labels = item.get("metric") or item.get("stream") or {}
            if "value" in item:
                nanos, value = item["value"][0], item["value"][1]
                results.append(
                    InstantResult(labels=labels, value=str(value), timestamp=from_nanos(nanos))
                )
            for entry in item.get("values") or []:
                nanos, value = entry[0], entry[1]
                if result_type == "streams":
                    results.append(
                        InstantResult(labels=labels, line=value, timestamp=from_nanos(nanos))
                    )
                else:
                    results.append(
                        InstantResult(labels=labels, value=str(value), timestamp=from_nanos(nanos))
                    )

        self.logger.info(f"query_logs_instant returned {len(results)} results ({result_type})")
        return QueryLogsInstantOutput(
            result_type=result_type,
            results=results[: input_data.limit],
            count=min(len(results), input_data.limit),
            output_meta=FREE,
        )

    async def log_labels(self, input_data: LogLabelsInput) -> LogLabelsOutput:
        """List log label names present in the window."""
        start, end = resolve_window(input_data.since, input_data.start, input_data.end)
        prefix = await self.client.datasource_path(input_data.datasource)
        payload = await self.client.request(
            "GET",
            f"{prefix}/loki/api/v1/labels",
            params={"start": to_nanos(start), "end": to_nanos(end)},
        )
        values = (payload or {}).get("data") or []
        self.logger.info(f"log_labels returned {len(values)} names")
        return LogLabelsOutput(labels=values, count=len(values), output_meta=FREE)

    async def log_label_values(self, input_data: LogLabelValuesInput) -> LogLabelValuesOutput:
        """List the values a log label takes in the window."""
        start, end = resolve_window(input_data.since, input_data.start, input_data.end)
        prefix = await self.client.datasource_path(input_data.datasource)
        payload = await self.client.request(
            "GET",
            f"{prefix}/loki/api/v1/label/{input_data.label}/values",
            params={"start": to_nanos(start), "end": to_nanos(end)},
        )
        values = (payload or {}).get("data") or []
        self.logger.info(f"log_label_values {input_data.label} returned {len(values)} values")
        return LogLabelValuesOutput(
            label=input_data.label, values=values, count=len(values), output_meta=FREE
        )

    async def log_series(self, input_data: LogSeriesInput) -> LogSeriesOutput:
        """List the log label sets matching a stream selector."""
        start, end = resolve_window(input_data.since, input_data.start, input_data.end)
        prefix = await self.client.datasource_path(input_data.datasource)
        payload = await self.client.request(
            "GET",
            f"{prefix}/loki/api/v1/series",
            params={
                "match[]": input_data.selector,
                "start": to_nanos(start),
                "end": to_nanos(end),
            },
        )
        found = (payload or {}).get("data") or []
        self.logger.info(f"log_series returned {len(found)} label sets")
        return LogSeriesOutput(series=found, count=len(found), output_meta=FREE)

    async def unload(self):
        await self.client.aclose()

    async def on_cancel(self):
        return True


def _filter_expression(matcher: Matcher) -> str:
    """Alertmanager filters look like alertname="Log error burst"."""
    if matcher.is_regex:
        operator = "=~" if matcher.is_equal else "!~"
    else:
        operator = "=" if matcher.is_equal else "!="
    escaped = matcher.value.replace('"', '\\"')
    return f'{matcher.name}{operator}"{escaped}"'


def _alert(item: Dict[str, Any]) -> Alert:
    status = item.get("status") or {}
    return Alert(
        labels=item.get("labels") or {},
        annotations=item.get("annotations") or {},
        state=str(status.get("state") or "unknown"),
        starts_at=str(item.get("startsAt") or ""),
        updated_at=str(item.get("updatedAt") or ""),
        silenced_by=[str(s) for s in (status.get("silencedBy") or [])],
        fingerprint=str(item.get("fingerprint") or ""),
    )


def _silence(item: Dict[str, Any]) -> Silence:
    status = item.get("status") or {}
    matchers = [
        Matcher(
            name=str(m.get("name") or ""),
            value=str(m.get("value") or ""),
            is_regex=bool(m.get("isRegex")),
            is_equal=bool(m.get("isEqual", True)),
        )
        for m in (item.get("matchers") or [])
    ]
    return Silence(
        id=str(item.get("id") or ""),
        matchers=matchers,
        starts_at=str(item.get("startsAt") or ""),
        ends_at=str(item.get("endsAt") or ""),
        state=str(status.get("state") or "unknown"),
        comment=str(item.get("comment") or ""),
        created_by=str(item.get("createdBy") or ""),
    )


def _lines_scanned(data: Dict[str, Any]) -> int:
    summary = ((data.get("stats") or {}).get("summary")) or {}
    try:
        return int(summary.get("totalLinesProcessed") or 0)
    except (TypeError, ValueError):
        return 0
