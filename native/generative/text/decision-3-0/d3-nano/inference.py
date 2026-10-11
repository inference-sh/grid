"""
d3-nano — The 2B Decision 3.0 model from vLLM Semantic Router.

2.21B parameters, text, images and videos in. Schemas and the model call are shared by every
d3 app: see d3_helper.py.
"""

import asyncio
import logging

from inferencesh import BaseApp

from .d3_helper import AppInput, AppOutput, decide, load_model

MODEL_ID = "vllm-sr/d3-nano"
# The repository runs its own code (trust_remote_code). Review the diff before moving this.
REVISION = "3aeb1c22a5632aac72353c36b3d0111c40e7dbf4"
# Padded tokens in one forward. Bounds peak VRAM; a larger request runs as several forwards.
BATCH_TOKENS = 32768


class App(BaseApp):
    async def setup(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.runtime = await asyncio.to_thread(load_model, MODEL_ID, REVISION, self.logger)

    async def run(self, input_data: AppInput) -> AppOutput:
        """Evaluate the state, images and videos against every question."""
        return await asyncio.to_thread(decide, self.runtime, input_data, BATCH_TOKENS, self.logger)

    async def unload(self):
        import torch

        if hasattr(self, "runtime"):
            del self.runtime
        torch.cuda.empty_cache()
