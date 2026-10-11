import os
import logging
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, function
from pydantic import Field, field_validator
from typing import List, Literal, Optional
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


# --- Input models ---

class ListEventsInput(BaseAppInput):
    calendar_id: str = Field(default="primary", description="Calendar ID to list events from")
    time_min: str = Field(default="", description="Lower bound (inclusive) for event start time (ISO 8601)")
    time_max: str = Field(default="", description="Upper bound (exclusive) for event end time (ISO 8601)")
    max_results: int = Field(default=10, description="Maximum number of events to return")
    query: str = Field(default="", description="Free text search query")


class GetEventInput(BaseAppInput):
    event_id: str = Field(description="The event ID to retrieve")
    calendar_id: str = Field(default="primary", description="Calendar ID the event belongs to")


class CreateEventInput(BaseAppInput):
    summary: str = Field(description="Title of the event")
    start: str = Field(description="Start time in ISO 8601 format (dateTime or date)")
    end: str = Field(description="End time in ISO 8601 format (dateTime or date)")
    calendar_id: str = Field(default="primary", description="Calendar ID to create the event in")
    description: str = Field(default="", description="Description of the event")
    location: str = Field(default="", description="Location of the event")
    attendees: List[str] = Field(default=[], description="List of attendee email addresses")
    timezone: str = Field(default="", description="Timezone for the event (e.g. America/New_York)")


class UpdateEventInput(BaseAppInput):
    event_id: str = Field(description="The event ID to update")
    calendar_id: str = Field(default="primary", description="Calendar ID the event belongs to")
    summary: str = Field(default="", description="New title for the event")
    start: str = Field(default="", description="New start time in ISO 8601 format")
    end: str = Field(default="", description="New end time in ISO 8601 format")
    description: str = Field(default="", description="New description for the event")
    location: str = Field(default="", description="New location for the event")


class DeleteEventInput(BaseAppInput):
    event_id: str = Field(description="The event ID to delete")
    calendar_id: str = Field(default="primary", description="Calendar ID the event belongs to")


class ListCalendarsInput(BaseAppInput):
    pass


AclRole = Literal["freeBusyReader", "reader", "writer", "owner"]


def _validate_email(value: str) -> str:
    value = value.strip()
    if "@" not in value:
        raise ValueError(f"email must be an email address (got {value!r})")
    return value


class CreateCalendarInput(BaseAppInput):
    summary: str = Field(description="Name of the new calendar")
    description: str = Field(default="", description="Description of the calendar")
    timezone: str = Field(default="", description="Timezone of the calendar (e.g. America/New_York). Defaults to the account timezone")


class ShareCalendarInput(BaseAppInput):
    calendar_id: str = Field(description="Calendar ID to share")
    email: str = Field(description="Email address of the person to share the calendar with")
    role: AclRole = Field(
        default="reader",
        description="Access level: freeBusyReader (free/busy only), reader (see event details), writer (edit events), owner (manage sharing)",
    )
    send_notifications: bool = Field(default=True, description="Whether to email the person about the new access")

    @field_validator("email")
    @classmethod
    def _check_email(cls, v: str) -> str:
        return _validate_email(v)


class UnshareCalendarInput(BaseAppInput):
    calendar_id: str = Field(description="Calendar ID to stop sharing")
    email: str = Field(description="Email address of the person whose access to remove")

    @field_validator("email")
    @classmethod
    def _check_email(cls, v: str) -> str:
        return _validate_email(v)


class ListCalendarSharesInput(BaseAppInput):
    calendar_id: str = Field(default="primary", description="Calendar ID to list sharing rules for")


class DeleteCalendarInput(BaseAppInput):
    calendar_id: str = Field(description="ID of the secondary calendar to delete. The primary calendar cannot be deleted")


# --- Output models ---

class AttendeeInfo(BaseAppOutput):
    email: str = Field(default="", description="Attendee email")
    response_status: str = Field(default="", description="Attendee response status")


class EventInfo(BaseAppOutput):
    id: str = Field(default="", description="Event ID")
    summary: str = Field(default="", description="Event title")
    start: str = Field(default="", description="Event start time")
    end: str = Field(default="", description="Event end time")
    location: str = Field(default="", description="Event location")
    description: str = Field(default="", description="Event description")
    status: str = Field(default="", description="Event status")
    attendees: List[AttendeeInfo] = Field(default=[], description="Event attendees")


class ListEventsOutput(BaseAppOutput):
    events: List[EventInfo] = Field(default=[], description="List of calendar events")


class GetEventOutput(BaseAppOutput):
    id: str = Field(default="", description="Event ID")
    summary: str = Field(default="", description="Event title")
    start: str = Field(default="", description="Event start time")
    end: str = Field(default="", description="Event end time")
    location: str = Field(default="", description="Event location")
    description: str = Field(default="", description="Event description")
    status: str = Field(default="", description="Event status")
    html_link: str = Field(default="", description="Link to the event in Google Calendar")
    attendees: List[AttendeeInfo] = Field(default=[], description="Event attendees")
    creator: str = Field(default="", description="Event creator email")
    organizer: str = Field(default="", description="Event organizer email")
    created: str = Field(default="", description="Event creation time")
    updated: str = Field(default="", description="Event last updated time")


class CreateEventOutput(BaseAppOutput):
    id: str = Field(default="", description="Created event ID")
    summary: str = Field(default="", description="Created event title")
    html_link: str = Field(default="", description="Link to the event in Google Calendar")


class UpdateEventOutput(BaseAppOutput):
    id: str = Field(default="", description="Updated event ID")
    summary: str = Field(default="", description="Updated event title")
    html_link: str = Field(default="", description="Link to the event in Google Calendar")


class DeleteEventOutput(BaseAppOutput):
    success: bool = Field(default=False, description="Whether the deletion was successful")


class CalendarInfo(BaseAppOutput):
    id: str = Field(default="", description="Calendar ID")
    summary: str = Field(default="", description="Calendar name")
    primary: bool = Field(default=False, description="Whether this is the primary calendar")
    timezone: str = Field(default="", description="Calendar timezone")


class ListCalendarsOutput(BaseAppOutput):
    calendars: List[CalendarInfo] = Field(default=[], description="List of calendars")


class CreateCalendarOutput(BaseAppOutput):
    id: str = Field(default="", description="Created calendar ID")
    summary: str = Field(default="", description="Created calendar name")
    timezone: str = Field(default="", description="Created calendar timezone")


class ShareCalendarOutput(BaseAppOutput):
    rule_id: str = Field(default="", description="ID of the created access rule")
    role: str = Field(default="", description="Access level granted")


class UnshareCalendarOutput(BaseAppOutput):
    success: bool = Field(default=False, description="Whether the access was removed")
    rule_id: str = Field(default="", description="ID of the removed access rule")


class CalendarShareInfo(BaseAppOutput):
    rule_id: str = Field(default="", description="Access rule ID")
    role: str = Field(default="", description="Access level (none, freeBusyReader, reader, writer, owner)")
    scope_type: str = Field(default="", description="Who the rule applies to: default (public), user, group, or domain")
    scope_value: str = Field(default="", description="Email address or domain the rule applies to (empty for public)")


class ListCalendarSharesOutput(BaseAppOutput):
    shares: List[CalendarShareInfo] = Field(default=[], description="Access rules on the calendar")


class DeleteCalendarOutput(BaseAppOutput):
    success: bool = Field(default=False, description="Whether the deletion was successful")


# --- Helpers ---

def _parse_event_time(time_obj: dict) -> str:
    """Extract a readable time string from a Google Calendar event time object."""
    return time_obj.get("dateTime", time_obj.get("date", ""))


def _build_time_body(time_str: str, timezone: str = "") -> dict:
    """Build a Google Calendar time object from an ISO 8601 string."""
    if len(time_str) <= 10:
        # Date-only format (YYYY-MM-DD)
        return {"date": time_str}
    body = {"dateTime": time_str}
    if timezone:
        body["timeZone"] = timezone
    return body


def _parse_attendees(attendees_raw: list) -> List[AttendeeInfo]:
    """Parse attendees from Google Calendar API response."""
    return [
        AttendeeInfo(
            email=a.get("email", ""),
            response_status=a.get("responseStatus", ""),
        )
        for a in (attendees_raw or [])
    ]


def _list_acl_rules(service, calendar_id: str) -> list:
    """Return every ACL rule on a calendar, following pagination."""
    rules = []
    page_token = None
    while True:
        kwargs = {"calendarId": calendar_id}
        if page_token:
            kwargs["pageToken"] = page_token
        result = service.acl().list(**kwargs).execute()
        rules.extend(result.get("items", []))
        page_token = result.get("nextPageToken")
        if not page_token:
            return rules


def _build_service():
    """Build the Google Calendar API service using OAuth token."""
    token = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("GOOGLE_OAUTH_ACCESS_TOKEN is not set")
    creds = Credentials(token=token)
    return build("calendar", "v3", credentials=creds)


# --- App ---

class App(BaseApp):

    async def setup(self):
        self.logger = logging.getLogger(__name__)

    @function(
        credentials=[{"key": "google.calendar.events"}],
        annotations={"title": "List events", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_events(self, input_data: ListEventsInput) -> ListEventsOutput:
        """List events from a Google Calendar."""
        self.logger.info(f"Listing events from calendar={input_data.calendar_id}, max_results={input_data.max_results}")
        service = _build_service()

        kwargs = {
            "calendarId": input_data.calendar_id,
            "maxResults": input_data.max_results,
            "singleEvents": True,
            "orderBy": "startTime",
        }
        if input_data.time_min:
            kwargs["timeMin"] = input_data.time_min
        if input_data.time_max:
            kwargs["timeMax"] = input_data.time_max
        if input_data.query:
            kwargs["q"] = input_data.query

        result = service.events().list(**kwargs).execute()
        items = result.get("items", [])

        events = [
            EventInfo(
                id=item.get("id", ""),
                summary=item.get("summary", ""),
                start=_parse_event_time(item.get("start", {})),
                end=_parse_event_time(item.get("end", {})),
                location=item.get("location", ""),
                description=item.get("description", ""),
                status=item.get("status", ""),
                attendees=_parse_attendees(item.get("attendees")),
            )
            for item in items
        ]

        self.logger.info(f"Found {len(events)} events")
        return ListEventsOutput(events=events)

    @function(
        credentials=[{"key": "google.calendar.events"}],
        annotations={"title": "Get event", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_event(self, input_data: GetEventInput) -> GetEventOutput:
        """Get full details of a single event."""
        self.logger.info(f"Getting event={input_data.event_id} from calendar={input_data.calendar_id}")
        service = _build_service()

        event = service.events().get(
            calendarId=input_data.calendar_id,
            eventId=input_data.event_id,
        ).execute()

        return GetEventOutput(
            id=event.get("id", ""),
            summary=event.get("summary", ""),
            start=_parse_event_time(event.get("start", {})),
            end=_parse_event_time(event.get("end", {})),
            location=event.get("location", ""),
            description=event.get("description", ""),
            status=event.get("status", ""),
            html_link=event.get("htmlLink", ""),
            attendees=_parse_attendees(event.get("attendees")),
            creator=event.get("creator", {}).get("email", ""),
            organizer=event.get("organizer", {}).get("email", ""),
            created=event.get("created", ""),
            updated=event.get("updated", ""),
        )

    @function(
        credentials=[{"key": "google.calendar.events"}],
        annotations={"title": "Create event", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def create_event(self, input_data: CreateEventInput) -> CreateEventOutput:
        """Create a new calendar event."""
        self.logger.info(f"Creating event '{input_data.summary}' in calendar={input_data.calendar_id}")
        service = _build_service()

        event_body = {
            "summary": input_data.summary,
            "start": _build_time_body(input_data.start, input_data.timezone),
            "end": _build_time_body(input_data.end, input_data.timezone),
        }
        if input_data.description:
            event_body["description"] = input_data.description
        if input_data.location:
            event_body["location"] = input_data.location
        if input_data.attendees:
            event_body["attendees"] = [{"email": email} for email in input_data.attendees]

        created = service.events().insert(
            calendarId=input_data.calendar_id,
            body=event_body,
        ).execute()

        self.logger.info(f"Created event id={created.get('id')}")
        return CreateEventOutput(
            id=created.get("id", ""),
            summary=created.get("summary", ""),
            html_link=created.get("htmlLink", ""),
        )

    @function(
        credentials=[{"key": "google.calendar.events"}],
        annotations={"title": "Update event", "readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def update_event(self, input_data: UpdateEventInput) -> UpdateEventOutput:
        """Update an existing calendar event."""
        self.logger.info(f"Updating event={input_data.event_id} in calendar={input_data.calendar_id}")
        service = _build_service()

        # Fetch existing event first
        existing = service.events().get(
            calendarId=input_data.calendar_id,
            eventId=input_data.event_id,
        ).execute()

        # Apply non-empty fields
        if input_data.summary:
            existing["summary"] = input_data.summary
        if input_data.description:
            existing["description"] = input_data.description
        if input_data.location:
            existing["location"] = input_data.location
        if input_data.start:
            existing["start"] = _build_time_body(input_data.start)
        if input_data.end:
            existing["end"] = _build_time_body(input_data.end)

        updated = service.events().update(
            calendarId=input_data.calendar_id,
            eventId=input_data.event_id,
            body=existing,
        ).execute()

        self.logger.info(f"Updated event id={updated.get('id')}")
        return UpdateEventOutput(
            id=updated.get("id", ""),
            summary=updated.get("summary", ""),
            html_link=updated.get("htmlLink", ""),
        )

    @function(
        credentials=[{"key": "google.calendar.events"}],
        annotations={"title": "Delete event", "readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def delete_event(self, input_data: DeleteEventInput) -> DeleteEventOutput:
        """Delete a calendar event."""
        self.logger.info(f"Deleting event={input_data.event_id} from calendar={input_data.calendar_id}")
        service = _build_service()

        service.events().delete(
            calendarId=input_data.calendar_id,
            eventId=input_data.event_id,
        ).execute()

        self.logger.info(f"Deleted event={input_data.event_id}")
        return DeleteEventOutput(success=True)

    @function(
        credentials=[{"key": "google.calendar"}],
        annotations={"title": "List calendars", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_calendars(self, input_data: ListCalendarsInput) -> ListCalendarsOutput:
        """List all calendars accessible to the user."""
        self.logger.info("Listing calendars")
        service = _build_service()

        result = service.calendarList().list().execute()
        items = result.get("items", [])

        calendars = [
            CalendarInfo(
                id=item.get("id", ""),
                summary=item.get("summary", ""),
                primary=item.get("primary", False),
                timezone=item.get("timeZone", ""),
            )
            for item in items
        ]

        self.logger.info(f"Found {len(calendars)} calendars")
        return ListCalendarsOutput(calendars=calendars)

    @function(
        credentials=[{"key": "google.calendar"}],
        annotations={"title": "Create calendar", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def create_calendar(self, input_data: CreateCalendarInput) -> CreateCalendarOutput:
        """Create a new secondary calendar owned by the user."""
        self.logger.info(f"Creating calendar '{input_data.summary}'")
        service = _build_service()

        calendar_body = {"summary": input_data.summary}
        if input_data.description:
            calendar_body["description"] = input_data.description
        if input_data.timezone:
            calendar_body["timeZone"] = input_data.timezone

        created = service.calendars().insert(body=calendar_body).execute()

        self.logger.info(f"Created calendar id={created.get('id')}")
        return CreateCalendarOutput(
            id=created.get("id", ""),
            summary=created.get("summary", ""),
            timezone=created.get("timeZone", ""),
        )

    @function(
        credentials=[{"key": "google.calendar"}],
        annotations={"title": "Share calendar", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def share_calendar(self, input_data: ShareCalendarInput) -> ShareCalendarOutput:
        """Share a calendar with a person by email, granting the given access level."""
        self.logger.info(f"Sharing calendar={input_data.calendar_id} with {input_data.email} as {input_data.role}")
        service = _build_service()

        rule = service.acl().insert(
            calendarId=input_data.calendar_id,
            body={
                "role": input_data.role,
                "scope": {"type": "user", "value": input_data.email},
            },
            sendNotifications=input_data.send_notifications,
        ).execute()

        self.logger.info(f"Created ACL rule id={rule.get('id')}")
        return ShareCalendarOutput(
            rule_id=rule.get("id", ""),
            role=rule.get("role", ""),
        )

    @function(
        credentials=[{"key": "google.calendar"}],
        annotations={"title": "Unshare calendar", "readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    )
    async def unshare_calendar(self, input_data: UnshareCalendarInput) -> UnshareCalendarOutput:
        """Remove a person's access to a calendar."""
        self.logger.info(f"Removing {input_data.email} from calendar={input_data.calendar_id}")
        service = _build_service()

        email = input_data.email.lower()
        match = next(
            (
                rule for rule in _list_acl_rules(service, input_data.calendar_id)
                if rule.get("scope", {}).get("type") == "user"
                and rule.get("scope", {}).get("value", "").lower() == email
            ),
            None,
        )
        if match is None:
            raise ValueError(f"{input_data.email} has no access rule on calendar {input_data.calendar_id}")

        service.acl().delete(
            calendarId=input_data.calendar_id,
            ruleId=match["id"],
        ).execute()

        self.logger.info(f"Deleted ACL rule id={match['id']}")
        return UnshareCalendarOutput(success=True, rule_id=match["id"])

    @function(
        credentials=[{"key": "google.calendar"}],
        annotations={"title": "List calendar shares", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_calendar_shares(self, input_data: ListCalendarSharesInput) -> ListCalendarSharesOutput:
        """List who a calendar is shared with and at what access level."""
        self.logger.info(f"Listing ACL rules for calendar={input_data.calendar_id}")
        service = _build_service()

        shares = [
            CalendarShareInfo(
                rule_id=rule.get("id", ""),
                role=rule.get("role", ""),
                scope_type=rule.get("scope", {}).get("type", ""),
                scope_value=rule.get("scope", {}).get("value", ""),
            )
            for rule in _list_acl_rules(service, input_data.calendar_id)
        ]

        self.logger.info(f"Found {len(shares)} ACL rules")
        return ListCalendarSharesOutput(shares=shares)

    @function(
        credentials=[{"key": "google.calendar"}],
        annotations={"title": "Delete calendar", "readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def delete_calendar(self, input_data: DeleteCalendarInput) -> DeleteCalendarOutput:
        """Permanently delete a secondary calendar and all of its events."""
        if input_data.calendar_id.strip().lower() == "primary":
            raise ValueError(
                "The primary calendar cannot be deleted. To empty it, clear it instead "
                "(Google Calendar settings > Delete all events) or remove events with delete_event. "
                "To delete a secondary calendar, pass its ID from list_calendars."
            )

        self.logger.info(f"Deleting calendar={input_data.calendar_id}")
        service = _build_service()

        service.calendars().delete(calendarId=input_data.calendar_id).execute()

        self.logger.info(f"Deleted calendar={input_data.calendar_id}")
        return DeleteCalendarOutput(success=True)
