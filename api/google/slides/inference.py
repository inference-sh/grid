import os
import uuid
import logging

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import List, Optional, Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


# --- get_presentation ---

class GetPresentationInput(BaseAppInput):
    presentation_id: str = Field(description="The ID of the Google Slides presentation")


class SlideInfo(BaseAppOutput):
    object_id: str = Field(description="The object ID of the slide")
    texts: List[str] = Field(default_factory=list, description="Text content extracted from the slide")


class GetPresentationOutput(BaseAppOutput):
    id: str = Field(description="Presentation ID")
    title: str = Field(description="Presentation title")
    slides: List[SlideInfo] = Field(default_factory=list, description="List of slides with extracted text")
    num_slides: int = Field(description="Total number of slides")


# --- create_presentation ---

class CreatePresentationInput(BaseAppInput):
    title: str = Field(description="Title for the new presentation")


class CreatePresentationOutput(BaseAppOutput):
    id: str = Field(description="Presentation ID")
    title: str = Field(description="Presentation title")
    url: str = Field(description="URL to open the presentation")


# --- add_slide ---

class AddSlideInput(BaseAppInput):
    presentation_id: str = Field(description="The ID of the presentation")
    layout: str = Field(default="BLANK", description="Predefined layout: BLANK, TITLE, TITLE_AND_BODY, TITLE_ONLY, etc.")


class AddSlideOutput(BaseAppOutput):
    presentation_id: str = Field(description="The presentation ID")
    slide_id: str = Field(description="The object ID of the new slide")


# --- add_text ---

class AddTextInput(BaseAppInput):
    presentation_id: str = Field(description="The ID of the presentation")
    slide_index: int = Field(description="Zero-based index of the slide to add text to")
    text: str = Field(description="The text content to add")
    x: float = Field(default=100, description="X position in points")
    y: float = Field(default=100, description="Y position in points")
    width: float = Field(default=400, description="Width of the text box in points")
    height: float = Field(default=50, description="Height of the text box in points")


class AddTextOutput(BaseAppOutput):
    presentation_id: str = Field(description="The presentation ID")


# --- delete_slide ---

class DeleteSlideInput(BaseAppInput):
    slide_id: str = Field(description="The object ID of the slide to delete")
    presentation_id: str = Field(description="The ID of the presentation")


class DeleteSlideOutput(BaseAppOutput):
    presentation_id: str = Field(description="The presentation ID")
    success: bool = Field(description="Whether the slide was deleted successfully")


# --- batch_update ---

class BatchUpdateInput(BaseAppInput):
    presentation_id: str = Field(description="The ID of the presentation")
    requests: List[Any] = Field(default_factory=list, description="List of Slides API batch update request objects")


class BatchUpdateOutput(BaseAppOutput):
    presentation_id: str = Field(description="The presentation ID")
    replies: List[Any] = Field(default_factory=list, description="List of replies from the batch update")


# --- Helpers ---

def _extract_slide_texts(slide: dict) -> List[str]:
    """Extract all text content from a slide's page elements."""
    texts = []
    for element in slide.get("pageElements", []):
        shape = element.get("shape")
        if not shape:
            continue
        text_content = shape.get("text")
        if not text_content:
            continue
        for text_element in text_content.get("textElements", []):
            text_run = text_element.get("textRun")
            if text_run and text_run.get("content"):
                content = text_run["content"].strip()
                if content:
                    texts.append(content)
    return texts


# --- App ---

class App(BaseApp):
    service: object = None

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        token = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
        if not token:
            raise ValueError(
                "GOOGLE_OAUTH_ACCESS_TOKEN not found. "
                "Please ensure the Google Slides integration is connected in Settings."
            )
        creds = Credentials(token=token)
        self.service = build("slides", "v1", credentials=creds)
        self.logger.info("Google Slides service initialized")

    async def get_presentation(self, input_data: GetPresentationInput) -> GetPresentationOutput:
        """Get a presentation with slide content."""
        self.logger.info(f"Getting presentation id={input_data.presentation_id}")

        presentation = self.service.presentations().get(
            presentationId=input_data.presentation_id
        ).execute()

        slides = []
        for slide in presentation.get("slides", []):
            slides.append(SlideInfo(
                object_id=slide.get("objectId", ""),
                texts=_extract_slide_texts(slide),
            ))

        self.logger.info(f"Presentation '{presentation.get('title', '')}' has {len(slides)} slides")
        return GetPresentationOutput(
            id=presentation["presentationId"],
            title=presentation.get("title", ""),
            slides=slides,
            num_slides=len(slides),
        )

    async def create_presentation(self, input_data: CreatePresentationInput) -> CreatePresentationOutput:
        """Create a new Google Slides presentation."""
        self.logger.info(f"Creating presentation title='{input_data.title}'")

        presentation = self.service.presentations().create(
            body={"title": input_data.title}
        ).execute()

        presentation_id = presentation["presentationId"]
        url = f"https://docs.google.com/presentation/d/{presentation_id}/edit"

        self.logger.info(f"Created presentation id={presentation_id}")
        return CreatePresentationOutput(
            id=presentation_id,
            title=presentation.get("title", input_data.title),
            url=url,
        )

    async def add_slide(self, input_data: AddSlideInput) -> AddSlideOutput:
        """Add a new slide to a presentation."""
        self.logger.info(f"Adding slide to presentation={input_data.presentation_id} layout={input_data.layout}")

        slide_id = f"slide_{uuid.uuid4().hex[:8]}"

        requests = [{
            "createSlide": {
                "objectId": slide_id,
                "slideLayoutReference": {
                    "predefinedLayout": input_data.layout,
                },
            }
        }]

        self.service.presentations().batchUpdate(
            presentationId=input_data.presentation_id,
            body={"requests": requests},
        ).execute()

        self.logger.info(f"Added slide id={slide_id}")
        return AddSlideOutput(
            presentation_id=input_data.presentation_id,
            slide_id=slide_id,
        )

    async def add_text(self, input_data: AddTextInput) -> AddTextOutput:
        """Add a text box to a slide in a presentation."""
        self.logger.info(
            f"Adding text to presentation={input_data.presentation_id} "
            f"slide_index={input_data.slide_index}"
        )

        # Get the presentation to find the slide object ID by index
        presentation = self.service.presentations().get(
            presentationId=input_data.presentation_id
        ).execute()

        slides = presentation.get("slides", [])
        if input_data.slide_index < 0 or input_data.slide_index >= len(slides):
            raise ValueError(
                f"slide_index {input_data.slide_index} out of range. "
                f"Presentation has {len(slides)} slides."
            )

        slide_object_id = slides[input_data.slide_index]["objectId"]
        textbox_id = f"textbox_{uuid.uuid4().hex[:8]}"

        requests = [
            {
                "createShape": {
                    "objectId": textbox_id,
                    "shapeType": "TEXT_BOX",
                    "elementProperties": {
                        "pageObjectId": slide_object_id,
                        "size": {
                            "width": {"magnitude": input_data.width, "unit": "PT"},
                            "height": {"magnitude": input_data.height, "unit": "PT"},
                        },
                        "transform": {
                            "scaleX": 1,
                            "scaleY": 1,
                            "translateX": input_data.x,
                            "translateY": input_data.y,
                            "unit": "PT",
                        },
                    },
                }
            },
            {
                "insertText": {
                    "objectId": textbox_id,
                    "text": input_data.text,
                    "insertionIndex": 0,
                }
            },
        ]

        self.service.presentations().batchUpdate(
            presentationId=input_data.presentation_id,
            body={"requests": requests},
        ).execute()

        self.logger.info(f"Added text box id={textbox_id} on slide {slide_object_id}")
        return AddTextOutput(
            presentation_id=input_data.presentation_id,
        )

    async def delete_slide(self, input_data: DeleteSlideInput) -> DeleteSlideOutput:
        """Delete a slide from a presentation."""
        self.logger.info(
            f"Deleting slide id={input_data.slide_id} from presentation={input_data.presentation_id}"
        )

        requests = [{
            "deleteObject": {
                "objectId": input_data.slide_id,
            }
        }]

        try:
            self.service.presentations().batchUpdate(
                presentationId=input_data.presentation_id,
                body={"requests": requests},
            ).execute()
            self.logger.info(f"Deleted slide id={input_data.slide_id}")
            return DeleteSlideOutput(
                presentation_id=input_data.presentation_id,
                success=True,
            )
        except Exception as e:
            self.logger.error(f"Failed to delete slide: {e}")
            return DeleteSlideOutput(
                presentation_id=input_data.presentation_id,
                success=False,
            )

    async def batch_update(self, input_data: BatchUpdateInput) -> BatchUpdateOutput:
        """Execute a raw batch update on a presentation for advanced use."""
        self.logger.info(
            f"Batch update on presentation={input_data.presentation_id} "
            f"with {len(input_data.requests)} requests"
        )

        result = self.service.presentations().batchUpdate(
            presentationId=input_data.presentation_id,
            body={"requests": input_data.requests},
        ).execute()

        replies = result.get("replies", [])
        self.logger.info(f"Batch update completed with {len(replies)} replies")
        return BatchUpdateOutput(
            presentation_id=input_data.presentation_id,
            replies=replies,
        )

    async def unload(self):
        """Cleanup resources."""
        self.service = None
