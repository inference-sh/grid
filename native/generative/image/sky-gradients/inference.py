import asyncio
import json
import logging
import math
import os
import queue
import random
import threading
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, ImageMeta, OutputMeta, VideoMeta
from PIL import Image
from pydantic import Field, model_validator

from .sky import (DEFAULT, SUN_AMP, SUN_MAX, SUN_MID, SUN_MIN, clamp, css_from_pixels, eff,
                  facing_name, grid_dims, phase_name, randomize)

AURORA_PALETTES = {"sunset_blend": 7, "rose": 1, "peach": 2, "lavender": 3, "orchid": 4, "coral": 5, "plum": 6,
                   "classic_green": 0}
PALETTE_NAMES = {v: k for k, v in AURORA_PALETTES.items()}

# input field -> state key used by the sky gradients page (and its Recipe JSON)
FIELD_KEYS = {
    "sun_elevation": "sunEl", "facing": "look", "framing": "framing", "view_width": "fovH",
    "horizon_height": "horizon", "frame_top": "elTop", "frame_bottom": "elBot", "horizon_emphasis": "curve",
    "below_horizon": "below", "altitude_km": "alt",
    "haze": "haze", "haze_particle_size": "mieG", "air_density": "rayleigh", "ozone": "ozone",
    "volcanic_aerosol": "volcanic", "wildfire_smoke": "smoke", "multiple_scattering": "ms",
    "show_sun": "showSun", "sun_size": "sunSize", "sun_glow": "bloom",
    "aurora": "aurora", "aurora_height": "auroraEl", "aurora_palette": "auroraPal", "aurora_hue_shift": "auroraHue",
    "aurora_color_variation": "auroraVar", "aurora_upper_color": "auroraRed", "aurora_spikes": "auroraSpike",
    "aurora_motion": "auroraMotion", "light_leak": "leak", "leak_angle": "leakAngle", "leak_motion": "leakMotion",
    "shape_seed": "seed",
    "auto_exposure": "autoExp", "exposure_ev": "ev", "contrast": "contrast", "saturation": "sat",
    "warmth": "warmth", "lift": "lift", "grain": "grain", "grain_size": "grainSize", "vignette": "vignette",
}
SUN_KEYS, VIEW_KEYS = {"sunEl"}, {"look", "framing", "fovH", "horizon", "elTop", "elBot", "curve", "alt", "below"}
AIR_KEYS = {"haze", "mieG", "rayleigh", "ozone", "volcanic", "smoke"}
FILM_KEYS = {"warmth", "sat", "lift", "vignette"}


class SkyParams(BaseAppInput):
    width: int = Field(default=1920, ge=1, le=16384, description="Output width in pixels.")
    height: int = Field(default=1080, ge=1, le=16384, description="Output height in pixels.")
    randomize: Literal["new", "nearby", "off"] = Field(
        default="new",
        description="new: a fresh random sky. nearby: small changes around the given settings or recipe. "
                    "off: exactly the given settings. "
                    "When randomizing, settings left at their default are randomized and changed ones are kept.")
    random_seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1,
                                       description="Seed for randomize; the same seed and settings give the same sky.")
    spice: bool = Field(default=True, description="When randomizing, sometimes add aurora and light leaks.")
    recipe: Optional[str] = Field(default=None,
                                  description="Recipe JSON from the sky gradients page or a previous output. "
                                              "Fields you pass override it.")

    # sun & view
    sun_elevation: float = Field(default=-2.2, ge=SUN_MIN, le=SUN_MAX,
                                           description="Sun elevation in degrees. Kept to the sunset window.")
    facing: float = Field(default=0.0, ge=-180, le=180,
                                    description="View direction in degrees from the sun: 0 faces the sun, 180 faces away.")
    framing: Literal["natural", "stretch"] = Field(default="natural", description="natural: round like a lens. stretch: free frame top and bottom.")
    view_width: float = Field(default=60.0, ge=4, le=360, description="Horizontal view in degrees.")
    horizon_height: float = Field(default=0.3, ge=0, le=0.8,
                                            description="Horizon height in the frame, natural framing.")
    frame_top: float = Field(default=50.0, ge=2, le=90, description="Top of frame in degrees, stretch framing.")
    frame_bottom: float = Field(default=-1.0, ge=-20, le=10, description="Bottom of frame in degrees, stretch framing.")
    horizon_emphasis: float = Field(default=1.8, ge=0.5, le=4,
                                              description="Gives the horizon band more room, stretch framing.")
    below_horizon: Literal["none", "mirror", "extend", "ink"] = Field(default="mirror", description="none: horizon at the bottom edge. mirror: still water. extend: fade the horizon down. "
                                  "ink: night ink.")
    altitude_km: float = Field(default=0.05, ge=0, le=15, description="Viewer height in km.")

    # air
    haze: float = Field(default=0.3, ge=0, le=1, description="Aerosol haze.")
    haze_particle_size: float = Field(default=0.8, ge=0.5, le=0.95, description="Haze phase asymmetry.")
    air_density: float = Field(default=1.0, ge=0.3, le=2.5, description="Rayleigh scattering multiplier.")
    ozone: float = Field(default=1.0, ge=0, le=3, description="Ozone multiplier.")
    volcanic_aerosol: float = Field(default=0.0, ge=0, le=1.5, description="Stratospheric aerosol, purple light.")
    wildfire_smoke: float = Field(default=0.0, ge=0, le=1.5, description="Smoke.")
    multiple_scattering: float = Field(default=0.35, ge=0, le=1.5, description="Multiple scattering.")

    # sun disc
    show_sun: bool = Field(default=False, description="Draw the sun disc.")
    sun_size: float = Field(default=1.2, ge=0.3, le=6, description="Sun disc size multiplier.")
    sun_glow: float = Field(default=0.6, ge=0, le=3, description="Sun glow.")

    # spice
    aurora: float = Field(default=0.0, ge=0, le=2, description="Aurora strength; strongest with the sun well down.")
    aurora_height: float = Field(default=6.0, ge=-6, le=40, description="Aurora elevation in degrees.")
    aurora_palette: Literal["sunset_blend", "rose", "peach", "lavender", "orchid", "coral", "plum", "classic_green"] = Field(default="sunset_blend", description="Aurora colours.")
    aurora_hue_shift: float = Field(default=0.0, ge=-40, le=40, description="Aurora hue shift in degrees.")
    aurora_color_variation: float = Field(default=0.7, ge=0, le=1.5, description="Aurora colour variation.")
    aurora_upper_color: float = Field(default=0.3, ge=0, le=1.5, description="Aurora upper colour.")
    aurora_spikes: float = Field(default=0.2, ge=0, le=1, description="Aurora ray spikes; 0 keeps soft curtains.")
    aurora_motion: bool = Field(default=True, description="Aurora moves in video.")
    light_leak: float = Field(default=0.0, ge=0, le=1.5, description="Film light leak.")
    leak_angle: float = Field(default=200.0, ge=0, le=360, description="Leak direction in degrees.")
    leak_motion: bool = Field(default=True, description="Leak moves in video.")
    shape_seed: int = Field(default=7, ge=0, le=999, description="Aurora, leak and grain shape seed.")
    time_offset: float = Field(default=0, ge=0, le=100000, description="Start time in seconds for aurora and leak shapes.")

    # film
    auto_exposure: bool = Field(default=True, description="Auto exposure.")
    exposure_ev: float = Field(default=0.0, ge=-4, le=4, description="Exposure in EV.")
    contrast: float = Field(default=1.02, ge=0.6, le=1.6, description="Contrast.")
    saturation: float = Field(default=1.06, ge=0, le=2, description="Saturation.")
    warmth: float = Field(default=0.12, ge=-1, le=1, description="Warmth.")
    lift: float = Field(default=0.35, ge=0.1, le=1, description="Lift blacks to ink.")
    grain: float = Field(default=0.0, ge=0, le=1.5, description="Film grain.")
    grain_size: float = Field(default=0.5, ge=0.5, le=4, description="Grain size.")
    vignette: float = Field(default=0.0, ge=0, le=1, description="Vignette.")


class ImageInput(SkyParams):
    pass


class VideoInput(SkyParams):
    width: int = Field(default=1920, ge=2, le=8192, description="Output width in pixels. Odd sizes are rounded up to even for H.264.")
    height: int = Field(default=1080, ge=2, le=8192, description="Output height in pixels. Odd sizes are rounded up to even for H.264.")
    duration: float = Field(default=8, ge=1, le=60, description="Video length in seconds.")
    fps: int = Field(default=30, ge=1, le=60, description="Frames per second.")
    sun_motion: Literal["cycle", "sweep", "still"] = Field(
        default="cycle",
        description="cycle: sun eases down into twilight and back up, looping. sweep: sun moves once from "
                    "sun_elevation to sun_end. still: sun fixed; only aurora, leak and grain move.")
    sun_cycles: int = Field(default=1, ge=1, le=8, description="Full sun loops per video (cycle).")
    sun_end: Optional[float] = Field(default=None, ge=SUN_MIN, le=SUN_MAX,
                                     description="Final sun elevation for sweep (default: the far end of the window).")
    seamless_loop: bool = Field(default=True,
                                description="Fade the end of aurora and leak motion into the start so the video loops "
                                            "without a jump (cycle and still).")
    animate_grain: bool = Field(default=False, description="New grain every frame, like film. Noise does not compress: "
                                                          "files get about 10x larger; raise crf to compensate.")
    crf: int = Field(default=18, ge=0, le=40, description="H.264 quality; lower is better and larger.")

    @model_validator(mode="after")
    def _limits(self):
        if self.width * self.height > 8192 * 4320:
            raise ValueError("width x height must be at most 8192 x 4320 pixels for video")
        if self.duration * self.fps > 3600:
            raise ValueError("duration x fps must be at most 3600 frames")
        return self


class SkyOutput(BaseAppOutput):
    recipe: Dict[str, Any] = Field(description="Every setting used, in the sky gradients page's Recipe format. "
                                               "Pass it back as recipe to reproduce or vary this sky.")
    random_seed: Optional[int] = Field(default=None, description="Seed used when randomizing.")
    description: str = Field(description="Sky phase and view, e.g. 'civil twilight, facing the sun'.")
    renderer: str = Field(default="", description="GL renderer that drew the frames.")


class ImageOutput(SkyOutput):
    image: File = Field(description="PNG image.")
    css_gradient: str = Field(description="CSS linear-gradient approximating the sky top to bottom.")
    palette: List[str] = Field(description="Seven hex colours from top to bottom.")


class VideoOutput(SkyOutput):
    video: File = Field(description="H.264 MP4 video.")
    width: int = Field(description="Video width in pixels.")
    height: int = Field(description="Video height in pixels.")
    frames: int = Field(description="Number of frames.")
    timings: Dict[str, Any] = Field(default_factory=dict, description="Seconds spent per stage.")


def resolve_state(inp: SkyParams):
    """Merge defaults, recipe, randomizer and explicit fields into the page's state dict."""
    base = dict(DEFAULT)
    if inp.recipe:
        try:
            rec = json.loads(inp.recipe)
        except json.JSONDecodeError as e:
            raise ValueError("recipe is not valid JSON: %s" % e)
        if not isinstance(rec, dict):
            raise ValueError("recipe must be a JSON object")
        base.update({k: v for k, v in rec.items() if k in DEFAULT or k == "expPow"})
    # a setting counts as chosen when it differs from its default (forms send every field)
    explicit = {}
    for f, key in FIELD_KEYS.items():
        v = getattr(inp, f)
        if v != type(inp).model_fields[f].default:
            explicit[key] = AURORA_PALETTES[v] if key == "auroraPal" else v
    base.update(explicit)
    seed = None
    if inp.randomize != "off":
        seed = inp.random_seed if inp.random_seed is not None else random.SystemRandom().randrange(2**31)
        S = randomize(base, random.Random(seed), near=inp.randomize == "nearby", spice=inp.spice,
                      lock_sun="sunEl" in explicit, lock_view=VIEW_KEYS <= explicit.keys(),
                      lock_air=AIR_KEYS <= explicit.keys(), lock_film=FILM_KEYS <= explicit.keys())
        S.update(explicit)
    else:
        S = base
    S["sunEl"] = clamp(float(S["sunEl"]), SUN_MIN, SUN_MAX)
    for k in ("auroraPal", "seed"):
        S[k] = int(S[k])
    for k, v in DEFAULT.items():
        if isinstance(v, float):
            S[k] = float(S[k])
    return S, seed


def recipe_of(S):
    out = {}
    for k in DEFAULT:
        v = S[k]
        out[k] = round(v, 4) if isinstance(v, float) else v
    return out


class App(BaseApp):
    async def setup(self, config=None):
        self.logger = logging.getLogger(__name__)
        # one thread owns the GL context; every render runs there
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.cancelled = False
        self.renderer = await self._gl(self._make_renderer)
        self.nvenc = self._probe_nvenc()
        print("renderer: %s, tile %d" % (self.renderer.renderer_name, self.renderer.tile), flush=True)

    @staticmethod
    def _probe_nvenc():
        try:
            p = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=256x256:d=0.1", "-frames:v", "2",
                                "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=30)
            return p.returncode == 0
        except Exception:
            return False

    def _make_renderer(self):
        from .render import SkyRenderer
        return SkyRenderer()

    async def _gl(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.pool, fn, *args)

    async def render_image(self, input_data: ImageInput) -> ImageOutput:
        """Render a still sky gradient as a PNG at any size."""
        self.cancelled = False
        S, seed = resolve_state(input_data)
        w, h = input_data.width, input_data.height
        self.logger.info("image %dx%d seed=%s recipe=%s", w, h, seed, json.dumps(recipe_of(S)))
        t0 = time.time()
        img, E, css, pal = await self._gl(self._still, S, w, h, input_data.time_offset)
        self.logger.info("rendered in %.2fs", time.time() - t0)
        path = os.path.join(tempfile.mkdtemp(), "sky_%dx%d.png" % (w, h))
        Image.fromarray(img).save(path, compress_level=4 if w * h > 16_000_000 else 6)
        return ImageOutput(
            image=File(path=path), recipe=recipe_of(S), random_seed=seed,
            description="%s, %s" % (phase_name(S["sunEl"]), facing_name(S["look"])),
            renderer=self.renderer.renderer_name, css_gradient=css, palette=pal,
            output_meta=OutputMeta(outputs=[ImageMeta(width=w, height=h, resolution_mp=w * h / 1e6)]),
        )

    def _still(self, S, w, h, time_base):
        r = self.renderer
        img, E = r.render_still(S, w, h, time_base=time_base)
        # CSS stops from a small render of the same view without grain or vignette
        cs = eff({**S, "grain": 0.0, "vignette": 0.0}, w / h)
        G = r.sky_grid(cs, 12, 96, 24)
        r.set_uniforms(cs, G, 48, 192, time_base=time_base)
        css, pal = css_from_pixels(r.draw(48, 192))
        return img, E, css, pal

    async def render_video(self, input_data: VideoInput) -> VideoOutput:
        """Render an animated sky gradient as an MP4: the sun cycles or sweeps while aurora, leaks and grain move."""
        self.cancelled = False
        S, seed = resolve_state(input_data)
        w, h = input_data.width + input_data.width % 2, input_data.height + input_data.height % 2
        fps, D = input_data.fps, float(input_data.duration)
        n = max(1, int(round(D * fps)))
        self.logger.info("video %dx%d %d frames @%dfps motion=%s seed=%s recipe=%s", w, h, n, fps,
                         input_data.sun_motion, seed, json.dumps(recipe_of(S)))
        path = os.path.join(tempfile.mkdtemp(), "sky_%dx%d.mp4" % (w, h))
        t0 = time.time()
        timings = await self._gl(self._video, S, input_data, w, h, n, path)
        timings["total"] = round(time.time() - t0, 3)
        self.logger.info("video done in %.1fs (%.0f ms/frame)", time.time() - t0, (time.time() - t0) * 1000 / n)
        return VideoOutput(
            video=File(path=path), recipe=recipe_of(S), random_seed=seed,
            description="%s, %s" % (phase_name(S["sunEl"]), facing_name(S["look"])),
            width=w, height=h, frames=n, renderer=self.renderer.renderer_name, timings=timings,
            output_meta=OutputMeta(outputs=[VideoMeta(width=w, height=h, resolution_mp=w * h / 1e6,
                                                      seconds=n / fps, fps=fps)]),
        )

    def _sun_path(self, S, inp: VideoInput, n, fps):
        D = n / fps
        el0 = S["sunEl"]
        if inp.sun_motion == "cycle":
            # start at the given sun, heading down first (as the page's Play does)
            ph0 = math.acos(clamp((el0 - SUN_MID) / SUN_AMP, -1, 1))
            return [SUN_MID + SUN_AMP * math.cos(ph0 + 2 * math.pi * inp.sun_cycles * (i / fps) / D) for i in range(n)]
        if inp.sun_motion == "sweep":
            el1 = inp.sun_end if inp.sun_end is not None else (SUN_MIN if el0 - SUN_MIN > SUN_MAX - el0 else SUN_MAX)
            return [el0 + (el1 - el0) * (0.5 - 0.5 * math.cos(math.pi * i / max(n - 1, 1))) for i in range(n)]
        return [el0] * n

    def _video(self, S, inp: VideoInput, w, h, n, path):
        r, fps = self.renderer, inp.fps
        suns = self._sun_path(S, inp, n, fps)
        cols, rows, steps = grid_dims(w / h, "final")
        tm = {}
        # 1. every frame's sky grid on the GPU (small syncs for ambient light and auto exposure)
        ts = time.time()
        frames = []
        for el in suns:
            if self.cancelled:
                raise RuntimeError("cancelled")
            E = eff({**S, "sunEl": el}, w / h)
            frames.append((E, r.sky_grid(E, cols, rows, steps)))
        tm["sky"] = time.time() - ts

        loop_len = n / fps if (inp.seamless_loop and inp.sun_motion != "sweep") else 0.0
        fade = min(2.0, 0.3 * loop_len)
        if self.nvenc:
            codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", str(max(inp.crf, 1)),
                     "-b:v", "0", "-spatial-aq", "1"]
        else:
            codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(inp.crf), "-x264-params", "aq-mode=3"]
        tm["encoder"] = codec[1]
        color = ["-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
        cmd = (["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p"] + color +
               ["-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "-"] + codec + color + ["-movflags", "+faststart", path])
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        nbytes = r.video_begin(w, h)
        # a writer thread feeds ffmpeg while the GPU draws the next frames
        free, full = queue.Queue(), queue.Queue(maxsize=4)
        for _ in range(5):
            free.put(np.empty(nbytes, np.uint8))
        werr = []

        def writer():
            try:
                while True:
                    b = full.get()
                    if b is None:
                        break
                    proc.stdin.write(b)
                    free.put(b)
            except BaseException as e:
                werr.append(e)
                free.put(None)

        def collect(slot):
            b = free.get()
            if b is None or werr:
                raise RuntimeError("ffmpeg stopped: %s" % (werr[0] if werr else "?"))
            r.video_collect(slot, b)
            full.put(b)

        wt = threading.Thread(target=writer, daemon=True)
        wt.start()
        td = time.time()
        try:
            # 2. draw + YUV on the GPU; frame i reads back while frame i+1 draws
            for i, (E, G) in enumerate(frames):
                if self.cancelled:
                    raise RuntimeError("cancelled")
                r.upload_grid(G)
                r.set_uniforms(E, G, w, h, t=i / fps, loop_len=loop_len, fade=fade,
                               grain_t=(i + 1) if inp.animate_grain else 0.0, time_base=inp.time_offset)
                r.video_frame(i % 2)
                if i > 0:
                    collect((i - 1) % 2)
            collect((n - 1) % 2)
            tm["draw"] = time.time() - td
            full.put(None)
            wt.join()
            if werr:
                raise RuntimeError("ffmpeg stopped: %s" % werr[0])
            proc.stdin.close()
            te = time.time()
            if proc.wait() != 0:
                raise RuntimeError("ffmpeg failed: %s" % proc.stderr.read().decode(errors="replace")[-2000:])
            tm["encode_tail"] = time.time() - te
            return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in tm.items()}
        except BaseException:
            proc.kill()
            proc.wait()
            try:
                full.put_nowait(None)
            except queue.Full:
                pass
            raise
        finally:
            r.video_end()

    async def unload(self):
        if getattr(self, "renderer", None) is not None:
            await self._gl(self.renderer.release)
        self.pool.shutdown(wait=False)

    async def on_cancel(self):
        self.cancelled = True
        return True
