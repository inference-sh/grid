"""TokenRig (SkinTokens, VAST-AI-Research): automatic rigging of a 3D model.

One autoregressive pass predicts the skeleton and the skin weights together
(https://github.com/VAST-AI-Research/SkinTokens, MIT, vendored in
`skintokens/`). Meshes are read and written by Blender's Python module in a
separate local process (`skintokens.server.bpy_server`), as upstream does,
because bpy does not work from the threads the data loader uses.

Checkpoints come from the Hugging Face repo VAST-AI/SkinTokens; the model
code finds them, and its skeleton configs, by relative path (`experiments/...`,
`models/Qwen3-0.6B`, `configs/...`), so setup links them into a working
directory and runs from there.
"""

import logging
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Literal, Optional

import requests
import torch
from huggingface_hub import snapshot_download
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "VAST-AI/SkinTokens"
TOKENRIG_CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
VAE_CKPT = "experiments/skin_vae_2_10_32768/last.ckpt"
# Only the config: the weights are part of the TokenRig checkpoint.
LLM_REPO = "Qwen/Qwen3-0.6B"
LLM_DIR = "models/Qwen3-0.6B"

BPY_START_TIMEOUT = 120
INPUT_EXTENSIONS = (".glb", ".gltf", ".fbx", ".obj", ".dae", ".blend")


class AppInput(BaseAppInput):
    model: File = Field(description="The 3D model to rig: GLB, glTF, FBX, OBJ, DAE or .blend.")
    output_format: Literal["glb", "fbx"] = Field(default="glb", description="File format of the rigged model.")
    keep_original: bool = Field(
        default=True,
        description="Put the rig onto the original model, keeping its textures and materials. Off exports the mesh the model rigged.",
    )
    smooth_skin: bool = Field(default=False, description="Clean up the skin weights with a voxel pass.")
    num_beams: int = Field(default=10, ge=1, le=20, description="Beam search width. Higher is slower and often better.")
    top_k: int = Field(default=5, ge=1, le=200, description="Sampling top-k.")
    top_p: float = Field(default=0.95, ge=0.1, le=1.0, description="Sampling top-p.")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature.")
    repetition_penalty: float = Field(default=2.0, ge=0.5, le=3.0, description="Repetition penalty.")
    seed: Optional[int] = Field(default=None, ge=0, description="Seed. Empty: random.")


class AppOutput(BaseAppOutput):
    model: File = Field(description="The rigged model, with its skeleton and skin weights.")


def input_path(file: File, workdir: str) -> str:
    """The input under a name with its real extension: Blender picks the importer by extension."""
    ext = os.path.splitext(file.filename or file.path)[1].lower()
    if ext not in INPUT_EXTENSIONS:
        ext = mimetypes.guess_extension(file.content_type or "") or ""
    if ext not in INPUT_EXTENSIONS:
        raise ValueError("unsupported model file %r: use one of %s" % (file.filename or file.path, ", ".join(INPUT_EXTENSIONS)))
    fd, path = tempfile.mkstemp(suffix=ext, dir=workdir)
    os.close(fd)
    shutil.copyfile(file.path, path)
    return path


class App(BaseApp):
    async def setup(self, config):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        if APP_DIR not in sys.path:
            sys.path.insert(0, APP_DIR)

        repo = snapshot_download(REPO_ID, allow_patterns=[TOKENRIG_CKPT, VAE_CKPT])
        llm = snapshot_download(LLM_REPO, ignore_patterns=["*.bin", "*.safetensors"])
        self.workdir = tempfile.mkdtemp(prefix="tokenrig_")
        os.symlink(os.path.join(repo, "experiments"), os.path.join(self.workdir, "experiments"))
        os.makedirs(os.path.join(self.workdir, "models"))
        os.symlink(llm, os.path.join(self.workdir, LLM_DIR))
        os.symlink(os.path.join(APP_DIR, "configs"), os.path.join(self.workdir, "configs"))
        os.chdir(self.workdir)

        self.bpy = self._start_bpy_server()

        from skintokens.data.transform import Transform
        from skintokens.server.spec import get_model
        from skintokens.tokenizer.parse import get_tokenizer

        self.model = get_model(TOKENRIG_CKPT).eval()
        self.tokenizer = get_tokenizer(**self.model.tokenizer_config)
        self.transform = Transform.parse(**self.model.transform_config["predict_transform"])
        self.logger.info("TokenRig loaded from %s", TOKENRIG_CKPT)

    def _start_bpy_server(self) -> subprocess.Popen:
        from skintokens.server.spec import BPY_SERVER

        code = "import sys; sys.path.insert(0, %r); from skintokens.server.bpy_server import run; run()" % APP_DIR
        proc = subprocess.Popen([sys.executable, "-c", code], cwd=self.workdir, start_new_session=True)
        deadline = time.monotonic() + BPY_START_TIMEOUT
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("bpy server exited with code %d" % proc.returncode)
            try:
                if requests.get(BPY_SERVER + "/ping", timeout=1).ok:
                    self.logger.info("bpy server ready (pid %d)", proc.pid)
                    return proc
            except requests.RequestException:
                pass
            time.sleep(0.5)
        proc.kill()
        raise RuntimeError("bpy server did not start in %ds" % BPY_START_TIMEOUT)

    def _bpy(self, endpoint: str, payload: dict) -> None:
        """One export or transfer on the bpy server; the payload goes by file, as upstream does."""
        from skintokens.server.spec import BPY_SERVER, bytes_to_object, object_to_bytes

        fd, payload_path = tempfile.mkstemp(suffix=".pt", dir=self.workdir)
        with os.fdopen(fd, "wb") as f:
            f.write(object_to_bytes(payload))
        response = requests.post("%s/%s" % (BPY_SERVER, endpoint), data=object_to_bytes({"payload_path": payload_path}))
        response.raise_for_status()
        result = bytes_to_object(response.content)
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError("bpy %s failed: %s" % (endpoint, result.get("traceback") or result["error"]))
        if result != "ok":
            raise RuntimeError("bpy %s failed: %s" % (endpoint, result))

    async def run(self, input_data: AppInput) -> AppOutput:
        """Rig a 3D model: predict its skeleton and skin weights."""
        from skintokens.data.dataset import DatasetConfig, RigDatasetModule
        from skintokens.data.vertex_group import voxel_skin

        if self.bpy.poll() is not None:
            self.logger.warning("bpy server exited (code %s); restarting", self.bpy.returncode)
            self.bpy = self._start_bpy_server()

        source = input_path(input_data.model, self.workdir)
        if input_data.seed is not None:
            torch.manual_seed(input_data.seed)
        self.logger.info(
            "rigging %s: beams %d, keep original %s, smooth skin %s",
            os.path.basename(source), input_data.num_beams, input_data.keep_original, input_data.smooth_skin,
        )

        dataset_config = DatasetConfig.parse(
            shuffle=False,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            persistent_workers=False,
            datapath={"data_name": None, "loader": "bpy_server", "filepaths": {"articulation": [source]}},
        ).split_by_cls()
        module = RigDatasetModule(
            predict_dataset_config=dataset_config,
            predict_transform=self.transform,
            tokenizer=self.tokenizer,
            process_fn=self.model._process_fn,
        )
        batch = next(iter(module.predict_dataloader()["articulation"]))
        batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # Rig from scratch: a skeleton already in the file is not used.
        batch.pop("skeleton_tokens", None)
        batch.pop("skeleton_mask", None)
        batch["generate_kwargs"] = dict(
            max_length=2048,
            top_k=input_data.top_k,
            top_p=input_data.top_p,
            temperature=input_data.temperature,
            repetition_penalty=input_data.repetition_penalty,
            num_return_sequences=1,
            num_beams=input_data.num_beams,
            do_sample=True,
        )

        started = time.monotonic()
        with torch.no_grad():
            asset = self.model.predict_step(batch, skeleton_tokens=None, make_asset=True)["results"][0].asset
        if asset is None:
            raise RuntimeError("TokenRig produced no rig for this model")
        self.logger.info("predicted %d joints in %.1fs", len(asset.joints), time.monotonic() - started)

        if input_data.smooth_skin:
            voxel = asset.voxel(resolution=196)
            asset.skin *= voxel_skin(
                grid=0,
                grid_coords=voxel.coords,
                joints=asset.joints,
                vertices=asset.vertices,
                faces=asset.faces,
                mode="square",
                voxel_size=voxel.voxel_size,
            )
            asset.normalize_skin()

        fd, out_path = tempfile.mkstemp(suffix="." + input_data.output_format)
        os.close(fd)
        if input_data.keep_original:
            self._bpy("transfer", dict(source_asset=asset, target_path=source, export_path=out_path, group_per_vertex=4))
        else:
            self._bpy("export", dict(asset=asset, filepath=out_path, group_per_vertex=4))
        os.remove(source)
        if not os.path.getsize(out_path):
            raise RuntimeError("the rigged model came back empty")
        return AppOutput(model=File(path=out_path))

    async def unload(self):
        if getattr(self, "bpy", None) and self.bpy.poll() is None:
            self.bpy.kill()
