import os
import tempfile
import base64
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, function
from pydantic import Field
from typing import List, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


# --- list_messages ---

class ListMessagesInput(BaseAppInput):
    query: str = Field(default="", description="Gmail search query (e.g. 'is:unread', 'from:alice@example.com')")
    max_results: int = Field(default=10, description="Maximum number of messages to return", ge=1, le=100)


class MessageSummary(BaseAppOutput):
    id: str = Field(description="Message ID")
    thread_id: str = Field(description="Thread ID")
    subject: str = Field(default="", description="Email subject")
    sender: str = Field(default="", description="From address")
    date: str = Field(default="", description="Date header")
    snippet: str = Field(default="", description="Short snippet of the message")


class ListMessagesOutput(BaseAppOutput):
    messages: List[MessageSummary] = Field(default_factory=list, description="List of message summaries")


# --- get_message ---

class GetMessageInput(BaseAppInput):
    message_id: str = Field(description="The ID of the message to retrieve")


class AttachmentInfo(BaseAppOutput):
    attachment_id: str = Field(description="Attachment ID, for get_attachment")
    filename: str = Field(default="", description="File name")
    mime_type: str = Field(default="", description="MIME type")
    size: int = Field(default=0, description="Size in bytes")


class GetMessageOutput(BaseAppOutput):
    id: str = Field(description="Message ID")
    thread_id: str = Field(description="Thread ID")
    subject: str = Field(default="", description="Email subject")
    sender: str = Field(default="", description="From address")
    to: str = Field(default="", description="To address")
    date: str = Field(default="", description="Date header")
    body: str = Field(default="", description="Message body text")
    labels: List[str] = Field(default_factory=list, description="Label IDs on this message")
    attachments: List[AttachmentInfo] = Field(default_factory=list, description="Files attached to the message; download one with get_attachment")


# --- get_attachment ---

class GetAttachmentInput(BaseAppInput):
    message_id: str = Field(description="The ID of the message the attachment is on")
    attachment_id: str = Field(description="The attachment ID, from get_message's attachments")


class GetAttachmentOutput(BaseAppOutput):
    file: File = Field(description="The attachment's content")
    filename: str = Field(default="", description="File name")
    mime_type: str = Field(default="", description="MIME type")


# --- send_message ---

class SendMessageInput(BaseAppInput):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body (plain text)")
    cc: str = Field(default="", description="CC recipients (comma-separated)")
    bcc: str = Field(default="", description="BCC recipients (comma-separated)")


class SendMessageOutput(BaseAppOutput):
    id: str = Field(description="Sent message ID")
    thread_id: str = Field(description="Thread ID of sent message")


# --- create_draft ---

class CreateDraftInput(BaseAppInput):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body (plain text)")
    cc: str = Field(default="", description="CC recipients (comma-separated)")
    bcc: str = Field(default="", description="BCC recipients (comma-separated)")


class CreateDraftOutput(BaseAppOutput):
    id: str = Field(description="Draft ID")
    message_id: str = Field(description="Message ID of the draft")


# --- modify_message ---

class ModifyMessageInput(BaseAppInput):
    message_id: str = Field(description="The ID of the message to modify")
    add_labels: List[str] = Field(default_factory=list, description="Label IDs to add")
    remove_labels: List[str] = Field(default_factory=list, description="Label IDs to remove")


class ModifyMessageOutput(BaseAppOutput):
    id: str = Field(description="Message ID")
    labels: List[str] = Field(default_factory=list, description="Updated label IDs")


# --- list_labels ---

class ListLabelsInput(BaseAppInput):
    pass


class LabelInfo(BaseAppOutput):
    id: str = Field(description="Label ID")
    name: str = Field(description="Label name")
    type: str = Field(default="", description="Label type (system or user)")


class ListLabelsOutput(BaseAppOutput):
    labels: List[LabelInfo] = Field(default_factory=list, description="List of labels")


# --- Helpers ---

def _get_header(headers: list, name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _decode_body(payload: dict) -> str:
    """Extract plain text body from message payload."""
    # Direct body on the payload
    if payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

    # Multipart: look for text/plain first, then text/html
    parts = payload.get("parts", [])
    for mime_type in ["text/plain", "text/html"]:
        for part in parts:
            if part.get("mimeType") == mime_type and part.get("body", {}).get("data"):
                return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
            # Nested multipart
            for sub in part.get("parts", []):
                if sub.get("mimeType") == mime_type and sub.get("body", {}).get("data"):
                    return base64.urlsafe_b64decode(sub["body"]["data"]).decode("utf-8", errors="replace")

    return ""


def _attachments(payload: dict) -> List[AttachmentInfo]:
    """Every part with a file name and an attachment ID, at any depth."""
    out: List[AttachmentInfo] = []
    stack = [payload]
    while stack:
        part = stack.pop(0)
        body = part.get("body", {})
        if part.get("filename") and body.get("attachmentId"):
            out.append(AttachmentInfo(
                attachment_id=body["attachmentId"],
                filename=part["filename"],
                mime_type=part.get("mimeType", ""),
                size=body.get("size", 0),
            ))
        stack.extend(part.get("parts", []))
    return out


def _build_mime_message(to: str, subject: str, body: str, cc: str = "", bcc: str = "") -> str:
    """Build a MIME message and return base64url-encoded raw string."""
    message = MIMEMultipart()
    message["to"] = to
    message["subject"] = subject
    if cc:
        message["cc"] = cc
    if bcc:
        message["bcc"] = bcc
    message.attach(MIMEText(body, "plain"))
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    return raw


# --- App ---

class App(BaseApp):
    service: object = None

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        token = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
        if not token:
            raise ValueError(
                "GOOGLE_OAUTH_ACCESS_TOKEN not found. "
                "Please ensure the Google Gmail integration is connected in Settings."
            )
        creds = Credentials(token=token)
        self.service = build("gmail", "v1", credentials=creds)
        self.logger.info("Gmail service initialized")

    @function(
        credentials=[{"key": "google.gmail.readonly"}],
        annotations={"title": "List messages", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_messages(self, input_data: ListMessagesInput) -> ListMessagesOutput:
        """List Gmail messages matching a query."""
        self.logger.info(f"Listing messages with query='{input_data.query}' max_results={input_data.max_results}")

        results = self.service.users().messages().list(
            userId="me",
            q=input_data.query or None,
            maxResults=input_data.max_results,
        ).execute()

        message_ids = results.get("messages", [])
        summaries = []

        for msg_ref in message_ids:
            msg = self.service.users().messages().get(
                userId="me",
                id=msg_ref["id"],
                format="metadata",
                metadataHeaders=["Subject", "From", "Date"],
            ).execute()

            headers = msg.get("payload", {}).get("headers", [])
            summaries.append(MessageSummary(
                id=msg["id"],
                thread_id=msg.get("threadId", ""),
                subject=_get_header(headers, "Subject"),
                sender=_get_header(headers, "From"),
                date=_get_header(headers, "Date"),
                snippet=msg.get("snippet", ""),
            ))

        self.logger.info(f"Found {len(summaries)} messages")
        return ListMessagesOutput(messages=summaries)

    @function(
        credentials=[{"key": "google.gmail.readonly"}],
        annotations={"title": "Get message", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_message(self, input_data: GetMessageInput) -> GetMessageOutput:
        """Get a single Gmail message with full body."""
        self.logger.info(f"Getting message id={input_data.message_id}")

        msg = self.service.users().messages().get(
            userId="me",
            id=input_data.message_id,
            format="full",
        ).execute()

        payload = msg.get("payload", {})
        headers = payload.get("headers", [])

        return GetMessageOutput(
            id=msg["id"],
            thread_id=msg.get("threadId", ""),
            subject=_get_header(headers, "Subject"),
            sender=_get_header(headers, "From"),
            to=_get_header(headers, "To"),
            date=_get_header(headers, "Date"),
            body=_decode_body(payload),
            labels=msg.get("labelIds", []),
            attachments=_attachments(payload),
        )

    @function(
        credentials=[{"key": "google.gmail.readonly"}],
        annotations={"title": "Download attachment", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def get_attachment(self, input_data: GetAttachmentInput) -> GetAttachmentOutput:
        """Download a file attached to a Gmail message."""
        self.logger.info(f"Getting attachment from message id={input_data.message_id}")

        msg = self.service.users().messages().get(
            userId="me",
            id=input_data.message_id,
            format="full",
        ).execute()
        # Gmail gives an attachment a new ID on every read, so match it by
        # ID first and fall back to the only attachment when there is one.
        found = _attachments(msg.get("payload", {}))
        info = next((a for a in found if a.attachment_id == input_data.attachment_id), None)
        if info is None and len(found) == 1:
            info = found[0]
        attachment_id = info.attachment_id if info else input_data.attachment_id

        data = self.service.users().messages().attachments().get(
            userId="me",
            messageId=input_data.message_id,
            id=attachment_id,
        ).execute().get("data", "")

        filename = os.path.basename((info.filename if info else "") or "attachment")
        path = os.path.join(tempfile.mkdtemp(), filename)
        with open(path, "wb") as fh:
            fh.write(base64.urlsafe_b64decode(data))

        self.logger.info(f"Attachment saved to {path}")
        return GetAttachmentOutput(
            file=File(path=path),
            filename=filename,
            mime_type=info.mime_type if info else "",
        )

    @function(
        credentials=[{"key": "google.gmail.send"}],
        annotations={"title": "Send email", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def send_message(self, input_data: SendMessageInput) -> SendMessageOutput:
        """Send an email via Gmail."""
        self.logger.info(f"Sending message to={input_data.to} subject='{input_data.subject}'")

        raw = _build_mime_message(
            to=input_data.to,
            subject=input_data.subject,
            body=input_data.body,
            cc=input_data.cc,
            bcc=input_data.bcc,
        )

        result = self.service.users().messages().send(
            userId="me",
            body={"raw": raw},
        ).execute()

        self.logger.info(f"Message sent id={result['id']}")
        return SendMessageOutput(
            id=result["id"],
            thread_id=result.get("threadId", ""),
        )

    @function(
        credentials=[{"key": "google.gmail.compose"}],
        annotations={"title": "Create draft", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    )
    async def create_draft(self, input_data: CreateDraftInput) -> CreateDraftOutput:
        """Create a draft email in Gmail."""
        self.logger.info(f"Creating draft to={input_data.to} subject='{input_data.subject}'")

        raw = _build_mime_message(
            to=input_data.to,
            subject=input_data.subject,
            body=input_data.body,
            cc=input_data.cc,
            bcc=input_data.bcc,
        )

        result = self.service.users().drafts().create(
            userId="me",
            body={"message": {"raw": raw}},
        ).execute()

        self.logger.info(f"Draft created id={result['id']}")
        return CreateDraftOutput(
            id=result["id"],
            message_id=result.get("message", {}).get("id", ""),
        )

    @function(
        credentials=[{"key": "google.gmail.modify"}],
        annotations={"title": "Modify message labels", "readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
    )
    async def modify_message(self, input_data: ModifyMessageInput) -> ModifyMessageOutput:
        """Add or remove labels on a Gmail message."""
        self.logger.info(
            f"Modifying message id={input_data.message_id} "
            f"add={input_data.add_labels} remove={input_data.remove_labels}"
        )

        result = self.service.users().messages().modify(
            userId="me",
            id=input_data.message_id,
            body={
                "addLabelIds": input_data.add_labels,
                "removeLabelIds": input_data.remove_labels,
            },
        ).execute()

        self.logger.info(f"Message modified id={result['id']} labels={result.get('labelIds', [])}")
        return ModifyMessageOutput(
            id=result["id"],
            labels=result.get("labelIds", []),
        )

    @function(
        credentials=[{"key": "google.gmail.labels"}],
        annotations={"title": "List labels", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    )
    async def list_labels(self, input_data: ListLabelsInput) -> ListLabelsOutput:
        """List all Gmail labels."""
        self.logger.info("Listing labels")

        results = self.service.users().labels().list(userId="me").execute()
        labels = results.get("labels", [])

        label_list = [
            LabelInfo(
                id=label["id"],
                name=label.get("name", ""),
                type=label.get("type", ""),
            )
            for label in labels
        ]

        self.logger.info(f"Found {len(label_list)} labels")
        return ListLabelsOutput(labels=label_list)

    async def unload(self):
        """Cleanup resources."""
        self.service = None
