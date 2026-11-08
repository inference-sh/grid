"""Pika H3 Flex on our own GPUs, from the open weights.

Text-to-video and first/last-frame image-to-video with synchronized stereo
audio, through the Diffusers modular pipeline (fl2va adapter). Follows the
loading recipe of the official examples (examples/_common.py in the HF repo):
every component is managed by a ComponentsManager that swaps it between CPU
and the GPU, and the 62 GB transformer streams its blocks from pinned host
memory one at a time, so one 80 GB card runs every supported shape.
"""
import asyncio
import logging
import math
import os
import subprocess
import json
import tempfile
import time
from typing import Literal, Optional

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, BaseAppSetup, File, OutputMeta, VideoMeta, ImageMeta
from pydantic import Field

MODEL_REPO = "Pika-Labs/Pika-H3-Flex"
MODEL_REVISION = "ca45d99f9b1d1859ffb9dc1cd6bfaaa994257f56"   # HF main on 2026-11-08; adapters are sha256-pinned upstream
WORKFLOW = "fl2va"
FPS = 24

# The canvases the model was validated on (diffusers docs, "Generation"). (width, height).
BUCKETS = {
    "480p": {"21:9": (960, 416), "16:9": (832, 480), "3:2": (704, 480), "4:3": (640, 480), "5:4": (608, 480),
             "1:1": (480, 480), "4:5": (480, 608), "3:4": (480, 640), "2:3": (480, 704), "9:16": (480, 832),
             "9:21": (416, 960)},
    "768p": {"21:9": (1536, 672), "16:9": (1344, 768), "3:2": (1152, 768), "4:3": (1024, 768), "5:4": (960, 768),
             "1:1": (768, 768), "4:5": (768, 960), "3:4": (768, 1024), "2:3": (768, 1152), "9:16": (768, 1344),
             "9:21": (672, 1536)},
}
ASPECTS = ["21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"]

Resolution = Literal["480p", "768p"]
AspectRatio = Literal["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"]


class AppSetup(BaseAppSetup):
    resident_transformer: bool = Field(
        default=False,
        description="Keep the 62 GB transformer on the GPU instead of streaming its blocks from host memory. Faster, but 768p clips over 5 seconds run out of memory on an 80 GB card.",
    )


class AppInput(BaseAppInput):
    """Pika H3 Flex, self-hosted. Text-to-video, or image-to-video from a first and/or last frame, with audio."""

    prompt: str = Field(
        min_length=1,
        max_length=8000,
        description="What happens in the video, in plain prose. Describe the shots in order with timing, the camera, the subjects and the sound (dialogue in quotes, ambience, music or 'no music'). There is no prompt rewriter here, so be concrete.",
        examples=["A red fox trots along a frozen riverbank at dawn, breath steaming. 0-3s: low tracking shot beside the fox. 3-5s: it stops and looks straight into the lens. Sound: crunching snow, distant crows, no music."],
    )
    image: Optional[File] = Field(
        default=None,
        description="First frame (JPEG, PNG or WebP). Center-cropped to the canvas aspect ratio and held as frame one.",
    )
    last_frame_image: Optional[File] = Field(
        default=None,
        description="Last frame. Can be given alone to generate up to it, or with image for first-and-last-frame.",
    )
    duration: int = Field(
        default=5, ge=5, le=15,
        description="Requested length in whole seconds, 5 to 15. Snapped up to the model's frame grid (5s gives 124 frames, 5.17s).",
    )
    aspect_ratio: AspectRatio = Field(
        default="auto",
        description="Canvas aspect ratio. auto follows the first keyframe when one is given, otherwise 16:9.",
    )
    resolution: Resolution = Field(
        default="768p",
        description="Native generation tier. 480p is about twice as fast.",
    )
    soundscape: Optional[str] = Field(
        default=None,
        description="Overall ambience and physical sounds (the model's overall_soundscape field). Defaults to natural sound matching the scene.",
    )
    music: Optional[str] = Field(
        default=None,
        description="Background score (the model's non_diegetic_music field). Default none.",
    )
    format_prompt: bool = Field(
        default=True,
        description="Wrap the prompt in Pika's structured template (keyframe alignment line, integrated_multimodal_description, overall_soundscape, non_diegetic_music). Off, or a prompt that already contains integrated_multimodal_description, is sent verbatim.",
    )
    seed: int = Field(default=-1, ge=-1, description="Seed, or -1 for a random one.")


class AppOutput(BaseAppOutput):
    video: File = Field(description="MP4 with H.264 video at 24 fps and AAC stereo audio at 32 kHz.")
    seed: int = Field(description="Seed used.")
    width: int = Field(description="Canvas width in pixels.")
    height: int = Field(description="Canvas height in pixels.")
    frames: int = Field(description="Frames generated.")


def pick_bucket(resolution: str, aspect_ratio: str, keyframe_size: Optional[tuple]) -> tuple:
    """Return (width, height, ratio_label) for the request."""
    table = BUCKETS[resolution]
    if aspect_ratio != "auto":
        w, h = table[aspect_ratio]
        return w, h, aspect_ratio
    if keyframe_size:
        kw, kh = keyframe_size
        target = kw / kh
        label = min(ASPECTS, key=lambda a: abs(math.log((table[a][0] / table[a][1]) / target)))
    else:
        label = "16:9"
    w, h = table[label]
    return w, h, label


def format_prompt(prompt: str, mode: str, seconds: float, soundscape: Optional[str], music: Optional[str]) -> str:
    """Put a plain prompt into the template from docs/prompt-guidance.md (T2VA and FL2VA)."""
    if "integrated_multimodal_description" in prompt:
        return prompt
    body = prompt.strip()
    if not body.lstrip().startswith("[Shot"):
        body = "[Shot 1] " + body
    end = f"{seconds:.2f}"
    if mode == "image-to-video":
        head = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
        body = body if "<Picture 1>" in body else body.replace("[Shot 1] ", "[Shot 1] Begin from <Picture 1>. ", 1)
    elif mode == "last-frame":
        head = (f"How the reference pictures align with the target video \u2014 <Picture 1> (from [Shot 1]) aligns with the "
                f"{end}-second mark of the target video.\n\n")
        body = body if "<Picture 1>" in body else body + " End with the subjects and framing aligned with <Picture 1>."
    elif mode == "first-last-frame":
        head = (f"How the reference pictures align with the target video \u2014 <Picture 1> (from [Shot 1]) aligns with the "
                f"0.00-second mark of the target video; <Picture 2> (from [Shot 1]) aligns with the {end}-second mark of the target video.\n\n")
        if "<Picture 1>" not in body:
            body = body.replace("[Shot 1] ", "[Shot 1] Begin aligned with <Picture 1>. ", 1) + " End aligned with <Picture 2>."
    else:
        head = ""
    sound = (soundscape or "").strip() or "Natural ambient sound and physical sounds matching the scene."
    score = (music or "").strip() or "N/A"
    return f"{head}integrated_multimodal_description: {body}\n\noverall_soundscape: {sound}\n\nnon_diegetic_music: {score}\n"


def snap_frames(duration: int) -> int:
    """The frame count the pipeline will produce: 24 fps frames rounded up to the 17n+5 grid the video VAE
    decodes. Only for logging and the prompt; the pipeline is given the raw duration * 24 like Pika's own
    examples, because it range-checks that raw count (120..360) before it rounds up itself."""
    frames = duration * FPS
    while frames % 17 != 5:
        frames += 1
    return frames


class App(BaseApp):

    async def setup(self, config: AppSetup):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.resident_transformer = bool(getattr(config, "resident_transformer", False))
        self.cancelled = False

        import torch
        from accelerate import Accelerator
        from huggingface_hub import snapshot_download

        self.device = Accelerator().device
        if self.device.type != "cuda":
            raise RuntimeError(f"Pika H3 Flex needs a CUDA GPU, got {self.device}")
        self.device_str = f"cuda:{self.device.index or 0}"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True

        t0 = time.monotonic()
        token = os.environ.get("HF_TOKEN") or None
        allow = ["modular_model_index.json", "*.json", "text_encoder/*", "tokenizer/*", "processor/*", "vae/*",
                 "audio_vae/*", "transformer/*", "scheduler/*", "audio_scheduler/*", f"{WORKFLOW}/*"]
        self.logger.info(f"Downloading {MODEL_REPO}@{MODEL_REVISION[:8]} ({WORKFLOW} components, about 150 GB)...")
        await asyncio.to_thread(snapshot_download, MODEL_REPO, revision=MODEL_REVISION, allow_patterns=allow, token=token)
        self.logger.info(f"Weights ready in {time.monotonic() - t0:.0f}s")

        self.pipe = await asyncio.to_thread(self._build_pipeline)
        self.logger.info(f"Pipeline ready in {time.monotonic() - t0:.0f}s (resident_transformer={self.resident_transformer})")

    def _build_pipeline(self):
        import torch
        from diffusers import ComponentsManager, ModularPipeline

        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device=self.device_str)
        # Keep the full auto blocks so a text-only request takes the t2va path and a keyframe request the
        # fl2va path (pruning the blocks with workflow= would run the keyframe conditioning step on text-only
        # requests and fail). Only the component loading is restricted to the fl2va set, which is the t2va set
        # too, so the 62GB transformer_ref partition is never pulled.
        pipe = ModularPipeline.from_pretrained(MODEL_REPO, revision=MODEL_REVISION, components_manager=manager)
        pipe.load_components(
            workflow=WORKFLOW,
            pretrained_model_name_or_path=MODEL_REPO, revision=MODEL_REVISION,
            dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16},
        )
        pipe.load_lora_weights(MODEL_REPO, revision=MODEL_REVISION, weight_name=f"{WORKFLOW}/adapter.safetensors")

        transformer = pipe.transformer

        def release_other_components(module, inputs):
            # From the official example: once group offload has its first block on the GPU, the manager
            # can mistake the whole transformer for resident, so push every other component off first.
            for hook in manager.model_hooks:
                if hook.model is not module and hook.model.device.type == "cuda":
                    hook.offload()

        transformer.register_forward_pre_hook(release_other_components, prepend=True)
        if not self.resident_transformer:
            transformer.enable_group_offload(
                onload_device=torch.device(self.device_str), offload_device=torch.device("cpu"),
                offload_type="block_level", num_blocks_per_group=1, use_stream=True, record_stream=False,
            )
        self.manager = manager
        return pipe

    def _generate(self, prompt, image, last_image, width, height, num_frames, seed, out_path):
        import torch
        from diffusers.utils.export_utils import encode_video

        kwargs = {}
        if image is not None:
            kwargs["image"] = image
        if last_image is not None:
            kwargs["last_image"] = last_image
        torch.cuda.reset_peak_memory_stats()
        result = self.pipe(
            prompt=prompt, height=height, width=width, num_frames=num_frames,
            generator=torch.Generator(self.device_str).manual_seed(seed),
            output=["videos", "audio", "sampling_rate"], **kwargs,
        )
        torch.cuda.synchronize()
        frames = result["videos"][0]
        audio = result["audio"][0].detach().float().cpu()
        sr = int(result["sampling_rate"])
        if not torch.isfinite(audio).all():
            raise RuntimeError("generated audio contains NaN or Inf")
        encode_video(frames, fps=FPS, output_path=out_path, audio=audio, audio_sample_rate=sr)
        peak = torch.cuda.max_memory_allocated() / 2**30
        return len(frames), sr, peak

    async def run(self, input_data: AppInput) -> AppOutput:
        from PIL import Image, ImageOps
        import random

        self.cancelled = False
        image = last_image = None
        keyframe_size = None
        if input_data.image:
            image = ImageOps.exif_transpose(Image.open(input_data.image.path)).convert("RGB")
            keyframe_size = image.size
        if input_data.last_frame_image:
            last_image = ImageOps.exif_transpose(Image.open(input_data.last_frame_image.path)).convert("RGB")
            keyframe_size = keyframe_size or last_image.size

        width, height, ratio = pick_bucket(input_data.resolution, input_data.aspect_ratio, keyframe_size)
        num_frames = snap_frames(input_data.duration)
        seed = input_data.seed if input_data.seed >= 0 else random.randint(0, 2**31 - 1)
        mode = "first-last-frame" if image is not None and last_image is not None else \
            "image-to-video" if image is not None else "last-frame" if last_image is not None else "text-to-video"
        self.logger.info(f"Mode {mode}: {width}x{height} ({ratio}, {input_data.resolution}), {num_frames} frames "
                         f"({num_frames / FPS:.2f}s), seed {seed}")

        prompt = input_data.prompt
        if input_data.format_prompt:
            prompt = format_prompt(prompt, mode, num_frames / FPS, input_data.soundscape, input_data.music)
        self.logger.info("Prompt:\n" + prompt[:1500])

        workdir = tempfile.mkdtemp(prefix="pika-")
        out_path = os.path.join(workdir, "video.mp4")
        t0 = time.monotonic()
        frames, sr, peak = await asyncio.to_thread(
            self._generate, prompt, image, last_image, width, height, input_data.duration * FPS, seed, out_path)
        gen_s = time.monotonic() - t0
        self.logger.info(f"Generated {frames} frames in {gen_s:.1f}s, peak GPU {peak:.1f} GiB, audio {sr} Hz")

        probe = self._probe(out_path)
        self.logger.info(f"Output: {probe}")

        input_metas = [ImageMeta() for f in (input_data.image, input_data.last_frame_image) if f]
        output_meta = OutputMeta(
            inputs=input_metas,
            outputs=[VideoMeta(
                width=width, height=height, resolution=input_data.resolution, seconds=frames / FPS, fps=FPS,
                extra={"mode": mode, "model": "pika-h3-flex", "aspect_ratio": ratio, "frames": frames, "seed": seed,
                       "requested_duration": input_data.duration, "generation_seconds": round(gen_s, 1),
                       "peak_gpu_gib": round(peak, 1), "resident_transformer": self.resident_transformer},
            )],
        )
        return AppOutput(video=File(path=out_path), seed=seed, width=width, height=height, frames=frames,
                         output_meta=output_meta)

    @staticmethod
    def _probe(path: str) -> dict:
        try:
            data = json.loads(subprocess.run(
                ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                capture_output=True, text=True, timeout=30).stdout)
            out = {"seconds": float((data.get("format") or {}).get("duration") or 0)}
            for s in data.get("streams", []):
                if s.get("codec_type") == "video":
                    out.update(width=s.get("width"), height=s.get("height"), fps=s.get("r_frame_rate"), frames=s.get("nb_frames"))
                elif s.get("codec_type") == "audio":
                    out.update(sample_rate=s.get("sample_rate"), channels=s.get("channels"))
            return out
        except Exception as e:
            return {"error": str(e)}

    async def on_cancel(self):
        self.cancelled = True
        self.logger.info("Cancellation requested; the current denoising pass runs to completion")
        return True

    async def unload(self):
        import gc
        import torch
        self.pipe = None
        self.manager = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
