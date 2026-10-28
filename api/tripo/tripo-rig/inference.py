"""Tripo Auto Rig: rig a 3D model on Tripo's own API, optionally with preset animations.

The model is uploaded (`POST /files`) and rigged (`POST /animations/rig`).
Without a rig type, Tripo's free rig check (`POST /animations/rig-check`)
picks it first and refuses models it cannot rig. With animations, each is
retargeted onto the rig (`POST /animations/retarget`, one task for all).

Tripo bills 25 credits per rig and 10 per animation ($0.01 each), reported
per task as `credits_consumed`; the output reports their sum in `output_meta`.
"""

import logging
import mimetypes
import os
from typing import List, Literal, Optional

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, RawMeta, function
from pydantic import Field, model_validator

from .tripo_helper import TripoClient, UpstreamError, credits, download, output_url

MAX_BYTES = 150 * 1024 * 1024
FORMATS = {".glb": "model/gltf-binary", ".fbx": "application/octet-stream", ".obj": "text/plain", ".stl": "model/stl"}
# v3.0 is Tripo's choice for humanoids and the only one for `others` and the engine specs; v2.5 for creatures.
RIG_MODEL_HUMANOID = "v3.0-20260909"
RIG_MODEL_CREATURE = "v2.5-20260210"

RigType = Literal["biped", "quadruped", "hexapod", "octopod", "avian", "serpentine", "aquatic", "others"]
# The presets of rig models v2.5 and v3.0; each fits the rig type in its name, plain ones are biped. Avian has none.
Preset = Literal[
    "preset:idle", "preset:walk", "preset:run", "preset:dive", "preset:climb", "preset:jump",
    "preset:slash", "preset:shoot", "preset:hurt", "preset:fall", "preset:turn",
    "preset:quadruped:walk", "preset:hexapod:walk", "preset:octopod:walk",
    "preset:serpentine:march", "preset:aquatic:march",
]


class AppInput(BaseAppInput):
    model: File = Field(description="The 3D model to rig: GLB, FBX, OBJ or STL, up to 150 MB.")
    rig_type: Optional[RigType] = Field(
        default=None, description="The kind of body. Empty: Tripo's rig check decides, and refuses a model it cannot rig.",
    )
    spec: Literal["tripo", "mixamo", "actorcore", "unity", "unreal", "vrm"] = Field(
        default="tripo", description="Bone naming for a biped: Tripo's own, or that of Mixamo, ActorCore, Unity, Unreal or VRM.",
    )
    out_format: Literal["glb", "fbx"] = Field(default="glb", description="File format of the rigged model.")
    animations: List[Preset] = Field(
        default_factory=list,
        max_length=5,
        description="Preset animations to put on the rig, up to five (10 credits each). Plain presets are for bipeds; the others name their rig type.",
    )
    animate_in_place: bool = Field(default=False, description="Keep animations in place instead of moving through the scene.")

    @model_validator(mode="after")
    def check_inputs(self):
        if self.spec != "tripo" and self.rig_type not in (None, "biped"):
            raise ValueError("spec applies to bipeds only")
        if self.rig_type == "avian" and self.animations:
            raise ValueError("Tripo has no animations for avian rigs")
        return self


class AppOutput(BaseAppOutput):
    model: File = Field(description="The rigged model.")
    animated: Optional[File] = Field(default=None, description="The rigged model with the animations, when any were asked for.")
    rig_type: str = Field(description="The rig type used.")


def upload_name(file: File) -> str:
    for name in (file.filename, file.path, (file.uri or "").split("?")[0]):
        ext = os.path.splitext(name or "")[1].lower()
        if ext in FORMATS:
            return "model" + ext
    guessed = mimetypes.guess_extension(file.content_type or "") or ""
    if guessed in FORMATS:
        return "model" + guessed
    raise ValueError("the model must be a GLB, FBX, OBJ or STL file")


class App(BaseApp):
    async def setup(self, config):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.tripo = TripoClient()

    @function(annotations={"title": "Rig a 3D model", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
    async def run(self, input_data: AppInput) -> AppOutput:
        """Rig a 3D model with a skeleton and skin weights, optionally with preset animations."""
        size = os.path.getsize(input_data.model.path)
        if size > MAX_BYTES:
            raise ValueError("the model is %d MB; Tripo takes at most 150 MB" % (size // 2**20))
        name = upload_name(input_data.model)
        with open(input_data.model.path, "rb") as f:
            token = await self.tripo.upload(name, f.read(), FORMATS[os.path.splitext(name)[1]])

        rig_type = input_data.rig_type
        if rig_type is None:
            check = (await self.tripo.run("/animations/rig-check", {"input": token})).get("output") or {}
            if not check.get("riggable"):
                raise UpstreamError("Tripo's rig check says this model cannot be rigged; set rig_type to rig it anyway")
            rig_type = check.get("rig_type") or "biped"
            self.logger.info("rig check: %s", rig_type)
        if input_data.spec != "tripo" and rig_type != "biped":
            raise ValueError("spec applies to bipeds only; the rig check found a %s" % rig_type)
        if rig_type == "avian" and input_data.animations:
            raise ValueError("Tripo has no animations for avian rigs, and the rig check found an avian")

        rig_model = RIG_MODEL_HUMANOID if rig_type in ("biped", "others") or input_data.spec != "tripo" else RIG_MODEL_CREATURE
        body = {"input": token, "model": rig_model, "rig_type": rig_type, "out_format": input_data.out_format}
        if rig_type == "biped":
            body["spec"] = input_data.spec
        self.logger.info("rigging %s as %s with %s, spec %s", name, rig_type, rig_model, input_data.spec)
        rig = await self.tripo.run("/animations/rig", body)
        total = credits(rig)
        model = File(path=await download(output_url(rig), "." + input_data.out_format))

        animated = None
        if input_data.animations:
            body = {
                "input": rig["task_id"],
                "animations": list(input_data.animations),
                "out_format": input_data.out_format,
                "animate_in_place": input_data.animate_in_place,
            }
            self.logger.info("retargeting %s", ", ".join(input_data.animations))
            retarget = await self.tripo.run("/animations/retarget", body)
            total += credits(retarget)
            animated = File(path=await download(output_url(retarget), "." + input_data.out_format))

        self.logger.info("usage: credits=%s rig_type=%s animations=%d", total, rig_type, len(input_data.animations))
        meta = RawMeta(
            extra={"credits": total, "rig_type": rig_type, "rig_model": rig_model, "animations": len(input_data.animations)}
        )
        return AppOutput(model=model, animated=animated, rig_type=rig_type, output_meta=OutputMeta(inputs=[], outputs=[meta]))

    async def unload(self):
        await self.tripo.close()
