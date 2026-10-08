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

API: POST https://api.typesafe.ai/v1/systemone — https://docs.typesafe.ai/api
"""

import asyncio
import logging
import os
from typing import Any, Dict, List

import httpx
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, OutputMeta
from inferencesh.models.decision import DecisionInput, DecisionOutput, Structured
from pydantic import BaseModel, Field

API_BASE = "https://api.typesafe.ai/v1"
RETRY_STATUSES = (429, 529)
MAX_ATTEMPTS = 6

# ── Input and output ─────────────────────────────────────────────────────────

class AppInput(DecisionInput):
    state: Structured = Field(
        description="The content to evaluate: a string, or a JSON object / array of related context (messages, records, a policy). Text only. Every question sees the same state. Send only what the questions need; unrelated detail costs accuracy.",
        examples=["Our API started returning 500 errors 20 minutes ago and we can't process any customer orders."],
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

    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """One API call. 429 / 529 are retried with backoff, honoring retry-after."""
        headers = {"Authorization": f"Bearer {get_api_key()}"}
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
        questions = input_data.questions()
        self.logger.info(
            f"model={input_data.model} choices={len(input_data.choices)} "
            f"scores={len(input_data.scores)} nouls={len(input_data.nouls)}"
        )

        data = await self._request(
            "POST", "/systemone",
            json={"state": input_data.state, "model": input_data.model, "questions": questions},
        )

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
