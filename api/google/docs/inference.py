import os
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import List, Optional, Any
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


# --- get_document ---

class GetDocumentInput(BaseAppInput):
    document_id: str = Field(description="The ID of the Google Doc to retrieve")


class GetDocumentOutput(BaseAppOutput):
    id: str = Field(description="Document ID")
    title: str = Field(description="Document title")
    body_text: str = Field(description="Plain text content of the document body")


# --- create_document ---

class CreateDocumentInput(BaseAppInput):
    title: str = Field(description="Title for the new document")
    body: str = Field(default="", description="Optional body text to insert after creation")


class CreateDocumentOutput(BaseAppOutput):
    id: str = Field(description="ID of the created document")
    title: str = Field(description="Title of the created document")
    url: str = Field(description="URL of the created document")


# --- insert_text ---

class InsertTextInput(BaseAppInput):
    document_id: str = Field(description="The ID of the Google Doc")
    text: str = Field(description="Text to insert")
    index: int = Field(default=1, description="Character index at which to insert text (1 = start of doc)")


class InsertTextOutput(BaseAppOutput):
    document_id: str = Field(description="The ID of the updated document")


# --- replace_text ---

class ReplaceTextInput(BaseAppInput):
    document_id: str = Field(description="The ID of the Google Doc")
    find: str = Field(description="Text to find")
    replace_with: str = Field(description="Replacement text")
    match_case: bool = Field(default=True, description="Whether the search is case-sensitive")


class ReplaceTextOutput(BaseAppOutput):
    document_id: str = Field(description="The ID of the updated document")
    occurrences_changed: int = Field(description="Number of occurrences replaced")


# --- append_text ---

class AppendTextInput(BaseAppInput):
    document_id: str = Field(description="The ID of the Google Doc")
    text: str = Field(description="Text to append to the end of the document")


class AppendTextOutput(BaseAppOutput):
    document_id: str = Field(description="The ID of the updated document")


# --- batch_update ---

class BatchUpdateInput(BaseAppInput):
    document_id: str = Field(description="The ID of the Google Doc")
    requests: List[dict] = Field(default=[], description="List of request dicts for batchUpdate (advanced use)")


class BatchUpdateOutput(BaseAppOutput):
    document_id: str = Field(description="The ID of the updated document")
    replies: List[Any] = Field(description="Replies from the batchUpdate call")


def _extract_text(doc: dict) -> str:
    """Extract plain text from a Google Docs document body."""
    text_parts = []
    body = doc.get("body", {})
    for element in body.get("content", []):
        paragraph = element.get("paragraph")
        if paragraph:
            for elem in paragraph.get("elements", []):
                text_run = elem.get("textRun")
                if text_run:
                    text_parts.append(text_run.get("content", ""))
    return "".join(text_parts)


def _get_end_index(doc: dict) -> int:
    """Get the end index of the document body content."""
    body = doc.get("body", {})
    content = body.get("content", [])
    if content:
        last_element = content[-1]
        return last_element.get("endIndex", 1) - 1
    return 1


def _build_service():
    """Build the Google Docs API service using OAuth access token."""
    token = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
    if not token:
        raise ValueError(
            "GOOGLE_OAUTH_ACCESS_TOKEN not found. "
            "Please ensure the Google Docs integration is connected in Settings."
        )
    creds = Credentials(token=token)
    return build("docs", "v1", credentials=creds)


class App(BaseApp):

    async def get_document(self, input_data: GetDocumentInput) -> GetDocumentOutput:
        """Retrieve a Google Doc and return its ID, title, and plain text body."""
        self.logger.info(f"Getting document: {input_data.document_id}")
        service = _build_service()
        doc = service.documents().get(documentId=input_data.document_id).execute()
        body_text = _extract_text(doc)
        self.logger.info(f"Retrieved document '{doc.get('title')}' ({len(body_text)} chars)")
        return GetDocumentOutput(
            id=doc["documentId"],
            title=doc.get("title", ""),
            body_text=body_text,
        )

    async def create_document(self, input_data: CreateDocumentInput) -> CreateDocumentOutput:
        """Create a new Google Doc, optionally inserting body text."""
        self.logger.info(f"Creating document: {input_data.title}")
        service = _build_service()
        doc = service.documents().create(body={"title": input_data.title}).execute()
        doc_id = doc["documentId"]
        self.logger.info(f"Created document {doc_id}")

        if input_data.body:
            self.logger.info(f"Inserting body text ({len(input_data.body)} chars)")
            service.documents().batchUpdate(
                documentId=doc_id,
                body={
                    "requests": [
                        {
                            "insertText": {
                                "location": {"index": 1},
                                "text": input_data.body,
                            }
                        }
                    ]
                },
            ).execute()

        url = f"https://docs.google.com/document/d/{doc_id}/edit"
        return CreateDocumentOutput(
            id=doc_id,
            title=doc.get("title", input_data.title),
            url=url,
        )

    async def insert_text(self, input_data: InsertTextInput) -> InsertTextOutput:
        """Insert text at a given index in a Google Doc."""
        self.logger.info(f"Inserting text into {input_data.document_id} at index {input_data.index}")
        service = _build_service()
        service.documents().batchUpdate(
            documentId=input_data.document_id,
            body={
                "requests": [
                    {
                        "insertText": {
                            "location": {"index": input_data.index},
                            "text": input_data.text,
                        }
                    }
                ]
            },
        ).execute()
        self.logger.info("Text inserted successfully")
        return InsertTextOutput(document_id=input_data.document_id)

    async def replace_text(self, input_data: ReplaceTextInput) -> ReplaceTextOutput:
        """Find and replace text in a Google Doc."""
        self.logger.info(
            f"Replacing '{input_data.find}' with '{input_data.replace_with}' "
            f"in {input_data.document_id} (match_case={input_data.match_case})"
        )
        service = _build_service()
        result = service.documents().batchUpdate(
            documentId=input_data.document_id,
            body={
                "requests": [
                    {
                        "replaceAllText": {
                            "containsText": {
                                "text": input_data.find,
                                "matchCase": input_data.match_case,
                            },
                            "replaceText": input_data.replace_with,
                        }
                    }
                ]
            },
        ).execute()

        occurrences = 0
        for reply in result.get("replies", []):
            replace_result = reply.get("replaceAllText", {})
            occurrences = replace_result.get("occurrencesChanged", 0)

        self.logger.info(f"Replaced {occurrences} occurrences")
        return ReplaceTextOutput(
            document_id=input_data.document_id,
            occurrences_changed=occurrences,
        )

    async def append_text(self, input_data: AppendTextInput) -> AppendTextOutput:
        """Append text to the end of a Google Doc."""
        self.logger.info(f"Appending text to {input_data.document_id}")
        service = _build_service()

        doc = service.documents().get(documentId=input_data.document_id).execute()
        end_index = _get_end_index(doc)
        self.logger.info(f"Document end index: {end_index}")

        service.documents().batchUpdate(
            documentId=input_data.document_id,
            body={
                "requests": [
                    {
                        "insertText": {
                            "location": {"index": end_index},
                            "text": input_data.text,
                        }
                    }
                ]
            },
        ).execute()
        self.logger.info("Text appended successfully")
        return AppendTextOutput(document_id=input_data.document_id)

    async def batch_update(self, input_data: BatchUpdateInput) -> BatchUpdateOutput:
        """Execute a raw batchUpdate for advanced use cases."""
        self.logger.info(
            f"Batch update on {input_data.document_id} with {len(input_data.requests)} requests"
        )
        service = _build_service()
        result = service.documents().batchUpdate(
            documentId=input_data.document_id,
            body={"requests": input_data.requests},
        ).execute()
        replies = result.get("replies", [])
        self.logger.info(f"Batch update complete, {len(replies)} replies")
        return BatchUpdateOutput(
            document_id=input_data.document_id,
            replies=replies,
        )
