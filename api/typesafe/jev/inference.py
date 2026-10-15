"""
Jev — TypeSafe's System One model.

One request evaluates one `state` against any number of typed questions and returns a
typed, calibrated answer for each. Three question types (the primitives):

  choice  which of these options?   -> choice, probabilities, confidence
  score   which level on a scale?   -> score, probabilities, confidence, legend
  noul    is this true?             -> noul (probability of yes, 0 to 1)

The TypeSafe API takes one `questions` map keyed by id, discriminated by `type`. Here each
primitive has its own typed list (`choices`, `scores`, `nouls`) and every question carries
its `id`; answers come back keyed by the same ids. Ids share one namespace.

`state`, `instructions` and every option / level / criteria description accept a string,
an object or an array: Jev is trained to read JSON structure.

`images` go up as multipart file parts listed in `assets`, each under a fresh UUID reference.
With images the state sent is `{"images": [ref, ...], "text": state}`, so questions can name
`images[0]`. Images are a TypeSafe "danger zone" feature: off unless the request carries
`TypeSafe-Danger-Zone-Images: true`, and the format may change between releases. Each image
is rotated upright, downsized (never upscaled) into TypeSafe's limits, and padded when under
its minimum, because the API rejects, never resizes, an image outside them. Image tokens are billed as input tokens.

API: POST https://api.typesafe.ai/v1/systemone — https://docs.typesafe.ai/api, /images
"""

import asyncio
import io
import json
import logging
import math
import os
import tempfile
import uuid
from typing import Any, Dict, List, Tuple

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta
from inferencesh.images import ImageLimits, fit_image
from inferencesh.models.decision import DecisionOutput, DecisionVisionInput, Structured
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

API_BASE = "https://api.typesafe.ai/v1"
RETRY_STATUSES = (429, 529)
MAX_ATTEMPTS = 6

# TypeSafe image limits (https://docs.typesafe.ai/images#limits): PNG or JPEG, 16 MiB per
# file, each side 256 to 1568 px, total pixels at least 78,400 (280 x 280). The request body,
# images included, is at most 32 MiB.
MAX_IMAGES = 8
MAX_SIDE = 1568
MIN_SIDE = 256
MIN_PIXELS = 78_400
MAX_IMAGE_BYTES = 16 * 1024 * 1024
EXIF_ORIENTATION = 0x0112
IMAGES_HEADER = "TypeSafe-Danger-Zone-Images"

# ── Input and output ─────────────────────────────────────────────────────────

class AppInput(DecisionVisionInput):
    state: Structured = Field(
        default="",
        description="The content to evaluate: a string, or a JSON object / array of related context (messages, records, a policy). Every question sees the same state. May be empty when `images` carries the content. Send only what the questions need; unrelated detail costs accuracy.",
        examples=["Our API started returning 500 errors 20 minutes ago and we can't process any customer orders."],
    )
    images: List[File] = Field(
        default_factory=list,
        max_length=MAX_IMAGES,
        description="Up to 8 images (PNG or JPEG, or any format Pillow reads) the questions are about. Sent as `{\"images\": [...], \"text\": state}`, so questions can name `images[0]`. Downsized to 1568 px on the long side; one under 280 x 280 is padded. Preview feature upstream: formats and accuracy may change.",
    )
    model: str = Field(
        default="jev-latest",
        description="Model name or alias: `jev-latest`, `jev-preview`, or a pinned version such as `jev-1.13.0`. Pin a version if you tuned thresholds against it.",
    )


class AppOutput(DecisionOutput):
    model: str = Field(description="The versioned model that answered, e.g. `jev-1.13.0`.")
    input_tokens: int = Field(default=0, description="Input tokens (billed).")
    output_tokens: int = Field(default=0, description="Output tokens (free upstream).")


# ── list_models ──────────────────────────────────────────────────────────────

class ListModelsInput(BaseAppInput):
    pass


class ModelCard(BaseModel):
    name: str = Field(description="Model id or alias, as accepted by `model`.")
    description: str = Field(default="", description="What the model is for.")
    release_date: str = Field(default="", description="When the model or alias was released.")


class ListModelsOutput(BaseAppOutput):
    models: List[ModelCard] = Field(description="Names this account can send in `model`. Pinned versions are accepted even when not listed.")


# ── Images ───────────────────────────────────────────────────────────────────

TYPESAFE_IMAGES = ImageLimits(max_long_edge=MAX_SIDE, max_bytes=MAX_IMAGE_BYTES, mime_types=("image/png", "image/jpeg"))


def pad_size(width: int, height: int) -> Tuple[int, int]:
    """The smallest canvas that holds the image and meets TypeSafe's minimums."""
    w, h = max(width, MIN_SIDE), max(height, MIN_SIDE)
    if w * h < MIN_PIXELS:
        grow = math.sqrt(MIN_PIXELS / (w * h))
        w, h = min(MAX_SIDE, math.ceil(w * grow)), min(MAX_SIDE, math.ceil(h * grow))
        if w * h < MIN_PIXELS:
            w = max(w, math.ceil(MIN_PIXELS / h))
    return w, h


def prepare_image(path: str) -> Tuple[bytes, str]:
    """The image as (bytes, mime) inside TypeSafe's limits, never upscaled. The API strips
    EXIF without applying it, so an EXIF-rotated photo (and an animated one, which it
    refuses) is first saved upright as one still frame. Then fit_image downsizes only, and
    sends a fitting image as is. An image under the minimum size is centered on a white
    canvas rather than enlarged."""
    with Image.open(path) as img:
        rotated = img.getexif().get(EXIF_ORIENTATION, 1) not in (0, 1)
        animated = getattr(img, "is_animated", False)
        if rotated or animated:
            img.seek(0)
            upright = ImageOps.exif_transpose(img)
            fmt = "JPEG" if img.format == "JPEG" else "PNG"
            if fmt == "JPEG":
                upright = upright.convert("RGB")
            with tempfile.NamedTemporaryFile(suffix="." + fmt.lower(), delete=False) as tmp:
                upright.save(tmp, fmt, quality=95)
            try:
                return pad_image(*fit_image(tmp.name, TYPESAFE_IMAGES))
            finally:
                os.unlink(tmp.name)
    return pad_image(*fit_image(path, TYPESAFE_IMAGES))


def pad_image(data: bytes, mime: str) -> Tuple[bytes, str]:
    with Image.open(io.BytesIO(data)) as img:
        w, h = pad_size(*img.size)
        if (w, h) == img.size:
            return data, mime
        img = img.convert("RGBA" if "A" in img.getbands() or img.mode == "P" else "RGB")
        canvas = Image.new(img.mode, (w, h), "white")
        canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
        buf = io.BytesIO()
        canvas.save(buf, "PNG")
        return buf.getvalue(), "image/png"


def build_request(input_data: AppInput) -> Tuple[Dict[str, Any], Dict[str, Tuple[str, bytes, str]]]:
    """The /systemone JSON body and the image file parts. Each image gets a fresh UUID
    reference: references match as plain text anywhere in the request, so they must not
    occur in it by accident."""
    body: Dict[str, Any] = {"model": input_data.model, "questions": input_data.questions()}
    if not input_data.images:
        body["state"] = input_data.state
        return body, {}
    refs: List[str] = []
    assets: Dict[str, Dict[str, str]] = {}
    parts: Dict[str, Tuple[str, bytes, str]] = {}
    for i, image in enumerate(input_data.images):
        ref, part = str(uuid.uuid4()), f"image-{i}"
        refs.append(ref)
        assets[ref] = {"type": "image", "part": part}
        data, mime = prepare_image(image.path)
        parts[part] = (f"{part}.{'png' if mime == 'image/png' else 'jpg'}", data, mime)
    state: Dict[str, Any] = {"images": refs}
    if input_data.state != "":
        state["text"] = input_data.state
    body["state"] = state
    body["assets"] = assets
    return body, parts


# ── App ──────────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("TYPESAFE_KEY")
    if not key:
        raise RuntimeError(
            "TYPESAFE_KEY is not set. A secret whose record exists but holds an empty "
            "value is not injected at all — check that `belt secrets get TYPESAFE_KEY "
            "--json` reports a non-empty masked_value, and re-set it if it does not."
        )
    return key.strip()


class App(BaseApp):
    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.client = httpx.AsyncClient(base_url=API_BASE, timeout=120)
        self._cancel = False

    async def on_cancel(self):
        self._cancel = True
        return True

    def _cancelled(self) -> bool:
        ctx = getattr(self, "context", None)
        return self._cancel or (ctx is not None and ctx.cancel_requested)

    async def _request(self, method: str, path: str, headers: Dict[str, str] = None, **kwargs) -> Dict[str, Any]:
        """One API call. 429 / 529 are retried with backoff, honoring retry-after."""
        headers = {**(headers or {}), "Authorization": f"Bearer {get_api_key()}"}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            resp = await self.client.request(method, path, headers=headers, **kwargs)
            if resp.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS and not self._cancelled():
                try:
                    delay = float(resp.headers.get("retry-after", ""))
                except ValueError:
                    delay = min(2 ** (attempt - 1), 30)
                self.logger.info(f"TypeSafe returned {resp.status_code}; retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.0f}s")
                await asyncio.sleep(delay)
                continue
            if resp.status_code == 401:
                raise RuntimeError("TypeSafe rejected the API key (401). Check the TYPESAFE_KEY secret.")
            if resp.status_code >= 400:
                raise RuntimeError(f"TypeSafe API error {resp.status_code}: {resp.text[:1000]}")
            return resp.json()
        raise RuntimeError("unreachable")

    async def run(self, input_data: AppInput) -> AppOutput:
        """Evaluate the state against every question in one request."""
        self._cancel = False
        self.logger.info(
            f"model={input_data.model} choices={len(input_data.choices)} "
            f"scores={len(input_data.scores)} nouls={len(input_data.nouls)}"
        )

        body, parts = build_request(input_data)
        if parts:
            self.logger.info(f"images={len(parts)} ({sum(len(p[1]) for p in parts.values()) // 1024} KiB as JPEG)")
            # The JSON body is the first part, named `request`; each image is its own file part.
            files = {"request": (None, json.dumps(body), "application/json"), **parts}
            data = await self._request("POST", "/systemone", files=files, headers={IMAGES_HEADER: "true"})
        else:
            data = await self._request("POST", "/systemone", json=body)

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        self.logger.info(f"answered by {data.get('model')}: {input_tokens} in / {output_tokens} out tokens")
        return AppOutput.from_answers(
            data.get("answers") or {},
            input_data,
            model=data.get("model") or input_data.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def list_models(self, input_data: ListModelsInput) -> ListModelsOutput:
        """List the model names and aliases this account can use. Free."""
        data = await self._request("GET", "/models")
        return ListModelsOutput(
            models=[
                ModelCard(
                    name=m.get("name", ""),
                    description=m.get("description") or "",
                    release_date=str(m.get("release_date") or ""),
                )
                for m in data.get("models") or []
            ],
            output_meta=OutputMeta(inputs=[], outputs=[]),
        )

    async def unload(self):
        await self.client.aclose()
