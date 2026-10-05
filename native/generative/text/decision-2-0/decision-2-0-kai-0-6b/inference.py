"""
Decision-2.0-Kai-0.6B — the smallest and fastest Decision 2.0 model.

0.60B parameters, 8,192 input tokens. Schemas and the model call are shared by every
decision-2-0-* app: see decision_helper.py.
"""

import asyncio
import logging

from inferencesh import BaseApp

from .decision_helper import AppInput, AppOutput, decide, load_model

MODEL_ID = "vllm-sr/Decision-2.0-Kai-0.6B"
# The repository runs its own code (trust_remote_code). Review the diff before moving this.
REVISION = "cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764"
# Padded tokens in one forward. Bounds peak VRAM; a larger request runs as several forwards.
BATCH_TOKENS = 65536


class App(BaseApp):
    async def setup(self, metadata):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.model = await asyncio.to_thread(load_model, MODEL_ID, REVISION, BATCH_TOKENS, self.logger)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Evaluate the state against every question."""
        return await asyncio.to_thread(decide, self.model, input_data, self.logger)

    async def unload(self):
        import torch

        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()
