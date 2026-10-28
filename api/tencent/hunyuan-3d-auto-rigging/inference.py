"""Hunyuan 3D Auto Rigging on Tencent Cloud International (`hunyuan` service, SubmitAutoRiggingJob).

Rigs and skins a character model, optionally with one of 48 preset
animations. Tencent fetches the model from a URL (there is no upload action),
so the app passes the input file's own URL. Signing and polling are in
tencent_helper.py.

Tencent bills 10 credits per job and does not report them on the
international API; the output reports them in `output_meta`.
"""

import logging
import os
from typing import List, Literal, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, RawMeta, function
from pydantic import Field

from .tencent_helper import TencentClient, result_files

CREDITS = 10
MAX_BYTES = 60 * 1024 * 1024
FORMATS = {".fbx": "FBX", ".obj": "OBJ"}

# MotionType 1 to 48, in Tencent's order. Tencent names 31 and 45 both "Turn Right".
ANIMATIONS = [
    "roundhouse-kick", "left-hook", "charged-attack", "charged-punch", "two-hit-combo", "two-hit-combo-2",
    "backstep", "hit-reaction-1", "hit-reaction-2", "hit-reaction-3", "knockdown-1", "knockdown-2",
    "landing", "dejection", "throat-slash", "jab", "continuous-strikes", "kick", "side-kick", "tai-chi",
    "backflip", "crouching-turn", "walk-1", "walk-2", "walk-3", "idle-1", "idle-2", "street-dance",
    "the-twist", "turn-left", "turn-right", "jog", "jog-2", "run", "sprint-1", "sprint-2", "sprint-3",
    "jump-in-place-1", "sliding-tackle", "long-jump-forward", "long-jump-forward-2", "vault",
    "intimidation", "fall-forward", "turn-right-2", "jump-in-place-2", "turn-around", "send-shockwave",
]
Animation = Literal[tuple(ANIMATIONS)]  # type: ignore[valid-type]


class AppInput(BaseAppInput):
    model: File = Field(
        description="The character to rig, standing in a T-pose with its arms out. FBX or OBJ, up to 60 MB.",
    )
    animation: Optional[Animation] = Field(default=None, description="A preset animation to apply to the rigged character.")


class AppOutput(BaseAppOutput):
    model: File = Field(description="The rigged character.")
    preview: Optional[File] = Field(default=None, description="A preview render.")
    files: List[File] = Field(default_factory=list, description="Every other file Tencent returned.")


def model_type(file: File) -> str:
    for name in (file.filename, file.path, (file.uri or "").split("?")[0]):
        ext = os.path.splitext(name or "")[1].lower()
        if ext in FORMATS:
            return FORMATS[ext]
    raise ValueError("the model must be an FBX or OBJ file")


class App(BaseApp):
    async def setup(self, config):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.tencent = TencentClient()

    @function(annotations={"title": "Rig a character", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
    async def run(self, input_data: AppInput) -> AppOutput:
        """Rig and skin a T-posed character model, optionally with a preset animation."""
        kind = model_type(input_data.model)
        url = input_data.model.uri or ""
        if not url.startswith(("http://", "https://")):
            raise ValueError("the model needs a URL Tencent can fetch; pass it as a URL or upload it to inference.sh")
        size = os.path.getsize(input_data.model.path)
        if size > MAX_BYTES:
            raise ValueError("the model is %d MB; Tencent takes at most 60 MB" % (size // 2**20))

        params = {"File3D": {"Url": url, "Type": kind}}
        if input_data.animation:
            params["MotionType"] = ANIMATIONS.index(input_data.animation) + 1
        self.logger.info("auto rigging %s (%d KB), animation %s", kind, size // 1024, input_data.animation or "none")
        job = await self.tencent.run_job("SubmitAutoRiggingJob", "DescribeAutoRiggingJob", params, "Hunyuan 3D auto rigging")
        model, preview, others = await result_files(job, "Hunyuan 3D auto rigging")

        self.logger.info("usage: credits=%d animation=%s", CREDITS, input_data.animation or "")
        meta = RawMeta(extra={"credits": CREDITS, "animation": input_data.animation or ""})
        return AppOutput(model=model, preview=preview, files=others, output_meta=OutputMeta(inputs=[], outputs=[meta]))

    async def unload(self):
        await self.tencent.close()
