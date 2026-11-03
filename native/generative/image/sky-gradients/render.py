"""Headless GL renderer (moderngl over EGL; llvmpipe on CPU machines).

Frames larger than one framebuffer are drawn in tiles: the shader takes a pixel
offset, so tiles join without seams at any width and height.
"""
import math

import moderngl
import numpy as np

from .shaders import FRAG, SKY_FRAG, VERT, YUV_FRAG
from .sky import R, coeffs, dip_deg, dir_from, eff, get_lut, grid_dims, lut_sample


class SkyRenderer:
    def __init__(self, max_tile=4096):
        self.ctx = self._context()
        info = self.ctx.info
        self.renderer_name = info.get("GL_RENDERER", "?")
        lim = min(int(info.get("GL_MAX_RENDERBUFFER_SIZE", 4096)), int(info["GL_MAX_VIEWPORT_DIMS"][0]),
                  int(info["GL_MAX_VIEWPORT_DIMS"][1]))
        self.tile = min(max_tile, lim)
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.vao = self.ctx.vertex_array(self.prog, [])
        self.sky_prog = self.ctx.program(vertex_shader=VERT, fragment_shader=SKY_FRAG)
        self.sky_vao = self.ctx.vertex_array(self.sky_prog, [])
        self.amb_tex = self.ctx.texture((12, 1), 4, dtype="f4")
        self.amb_fbo = self.ctx.framebuffer(color_attachments=[self.amb_tex])
        self.lut_tex, self.lut_key = None, None
        self.grid_fbo = None
        self.tex = None
        self.yuv_prog = self.ctx.program(vertex_shader=VERT, fragment_shader=YUV_FRAG)
        self.yuv_vao = self.ctx.vertex_array(self.yuv_prog, [])
        self._vid = None
        self._fbos = {}

    @staticmethod
    def _context():
        """Prefer a hardware EGL device; fall back to Mesa llvmpipe."""
        soft = None
        for idx in range(8):
            try:
                ctx = moderngl.create_standalone_context(backend="egl", require=330, device_index=idx)
            except Exception:
                break
            if "llvmpipe" not in ctx.info.get("GL_RENDERER", "").lower():
                if soft is not None:
                    soft.release()
                return ctx
            if soft is None:
                soft = ctx
            else:
                ctx.release()
        return soft if soft is not None else moderngl.create_standalone_context(backend="egl", require=330)

    def _set(self, name, value):
        if name in self.prog:
            self.prog[name].value = value

    def _fbo(self, w, h):
        fbo = self._fbos.get((w, h))
        if fbo is None:
            if len(self._fbos) > 4:
                for f in self._fbos.values():
                    f.release()
                self._fbos.clear()
            fbo = self.ctx.simple_framebuffer((w, h), components=4)
            self._fbos[(w, h)] = fbo
        return fbo

    def upload_grid(self, G):
        size = (G["cols"], G["rows"])
        if self.tex is None or self.tex.size != size:
            if self.tex is not None:
                self.tex.release()
            self.tex = self.ctx.texture(size, 4, dtype="f4")
            self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tex.write(np.ascontiguousarray(G["data"], dtype=np.float32).tobytes())

    def sky_grid(self, S, cols, rows, N):
        """Compute the HDR sky grid on the GPU; it becomes the texture the next draw samples.
        Same maths as sky.compute_grid (float32 here)."""
        c = coeffs(S)
        lut = get_lut(c)
        if self.lut_key is not id(lut):
            if self.lut_tex is None:
                self.lut_tex = self.ctx.texture((lut.shape[1], lut.shape[0]), 3, dtype="f4")
                self.lut_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.lut_tex.write(np.ascontiguousarray(lut, np.float32).tobytes())
            self.lut_key = id(lut)
        sd = dir_from(0.0, S["sunEl"])
        dip = dip_deg(S)
        p = self.sky_prog
        vals = {"uAlt": S["alt"], "uMieG": S["mieG"], "uMs": S["ms"], "uMieS": c["mieS"], "uMieE": c["mieE"],
                "uBetaR": c["betaR"], "uOzA": c["ozA"], "uStratS": c["stratS"], "uStratE": c["stratE"],
                "uSmokeE": c["smokeE"], "uSmokeS": c["smokeS"], "uSd": sd,
                "uLook": S["look"], "uFov": S["fovH"], "uElBot": S["elBot"], "uElTop": S["elTop"],
                "uCurve": S["curve"], "uDip": dip}
        for k, v in vals.items():
            if k in p:
                p[k].value = tuple(float(x) for x in v) if np.ndim(v) else float(v)
        for k, v in (("uBelow", {"mirror": 1, "extend": 2}.get(S["below"], 0)), ("uN", int(N)),
                     ("uGridSize", (cols, rows)), ("uLut", 0)):
            if k in p:
                p[k].value = v
        self.lut_tex.use(0)
        # ambient light from 12 directions
        p["uMode"].value = 0
        self.amb_fbo.use()
        self.amb_fbo.viewport = (0, 0, 12, 1)
        self.sky_vao.render(moderngl.TRIANGLES, vertices=3)
        amb = np.frombuffer(self.amb_fbo.read(components=4, dtype="f4"), np.float32).reshape(12, 4)[:, :3].sum(0) / 240
        if "uAmb" in p:
            p["uAmb"].value = tuple(float(x) for x in amb)
        # the grid
        if self.grid_fbo is None or self.grid_fbo.size != (cols, rows):
            if self.grid_fbo is not None:
                self.grid_fbo.release()
                self.tex.release()
            self.tex = self.ctx.texture((cols, rows), 4, dtype="f4")
            self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.grid_fbo = self.ctx.framebuffer(color_attachments=[self.tex])
        p["uMode"].value = 1
        self.grid_fbo.use()
        self.grid_fbo.viewport = (0, 0, cols, rows)
        self.sky_vao.render(moderngl.TRIANGLES, vertices=3)
        data = np.frombuffer(self.grid_fbo.read(components=4, dtype="f4"), np.float32).reshape(rows, cols, 4)
        rad = data[..., :3].astype(np.float64)
        lum = 0.2126 * rad[..., 0] + 0.7152 * rad[..., 1] + 0.0722 * rad[..., 2]
        mask = data[..., 3] > 0
        avg = math.exp(np.log(1e-7 + lum[mask]).sum() / max(int(mask.sum()), 1)) + 2e-5
        if S["autoExp"]:
            exposure = 0.3 / avg ** S.get("expPow", 0.86) * 2 ** S["ev"]
        else:
            exposure = 1.2 * 2 ** S["ev"]
        sun_col = lut_sample(lut, np.array([0.0, R + S["alt"], 0.0]), sd) * 20
        return dict(data=data, cols=cols, rows=rows, exposure=exposure, sunCol=sun_col, dip=dip)

    def set_uniforms(self, S, G, w, h, t=0.0, loop_len=0.0, fade=0.0, grain_t=0.0, time_base=0.0):
        """S is the effective state (after eff())."""
        u = self._set
        u("uGridSize", (G["cols"], G["rows"]))
        u("uRes", (float(w), float(h)))
        u("uAz0", S["look"] - S["fovH"] / 2)
        u("uAz1", S["look"] + S["fovH"] / 2)
        u("uElBot", S["elBot"]); u("uElTop", S["elTop"]); u("uCurve", S["curve"])
        u("uSunCol", tuple(float(x) for x in G["sunCol"]))
        u("uSunEl", S["sunEl"]); u("uSunSize", S["sunSize"]); u("uBloom", S["bloom"])
        u("uShowSun", 1.0 if (S["showSun"] and S["sunEl"] > G["dip"] - 0.3) else 0.0)
        span = S["elTop"] - S["elBot"]
        ts = (S["sunEl"] - S["elBot"]) / span
        vS = ts ** (1 / S["curve"]) if ts >= 0 else ts * 0.5
        rel = ((0 - S["look"]) % 360 + 540) % 360 - 180
        uS = 0.5 + rel / S["fovH"]
        u("uSunPx", (uS * w, vS * h))
        u("uPxPerDeg", w / S["fovH"])
        u("uExposure", float(G["exposure"]))
        u("uAurora", S["aurora"]); u("uAuroraEl", S["auroraEl"]); u("uAuroraRed", S["auroraRed"])
        u("uTime", float(t))
        u("uAurPal", float(S["auroraPal"])); u("uAurHue", float(S["auroraHue"])); u("uAurVar", S["auroraVar"])
        u("uAurSpike", S["auroraSpike"])
        u("uLeakMove", 1.0 if S["leakMotion"] else 0.0)
        u("uAurMove", 1.0 if S["auroraMotion"] else 0.0)
        u("uDip", float(G["dip"])); u("uMirror", 1.0 if S["below"] == "mirror" else 0.0)
        u("uGrain", S["grain"])
        u("uGrainSize", S["grainSize"] * max(1.0, w / 1600))  # same scaling as the page's image export
        u("uLeak", S["leak"]); u("uLeakAngle", math.radians(S["leakAngle"])); u("uVignette", S["vignette"])
        u("uSat", S["sat"]); u("uWarmth", S["warmth"]); u("uContrast", S["contrast"]); u("uLift", S["lift"])
        u("uSeed", float(S["seed"]))
        u("uLoopLen", float(loop_len)); u("uFade", float(fade)); u("uGrainT", float(grain_t)); u("uTimeBase", float(time_base))

    def draw(self, w, h, out=None, components=3):
        """Render the current uniforms to an (h, w, components) uint8 array, top row first."""
        if out is None:
            out = np.empty((h, w, components), np.uint8)
        self.tex.use(0)
        self._set("uGrid", 0)
        T = self.tile
        for y0 in range(0, h, T):
            th = min(T, h - y0)
            for x0 in range(0, w, T):
                tw = min(T, w - x0)
                fbo = self._fbo(tw, th)
                fbo.use()
                fbo.viewport = (0, 0, tw, th)
                self._set("uOffset", (float(x0), float(y0)))
                self.vao.render(moderngl.TRIANGLES, vertices=3)
                buf = np.frombuffer(fbo.read(components=components, alignment=1), np.uint8).reshape(th, tw, components)
                # GL rows run bottom-up; tile y0 counts from the bottom edge
                out[h - y0 - th:h - y0, x0:x0 + tw] = buf[::-1]
        return out

    # ---- video: whole frame on the GPU, YUV conversion there, async readback via two PBOs ----
    def video_begin(self, w, h):
        self.video_end()
        ftex = self.ctx.texture((w, h), 4, dtype="f2")   # float frame keeps the dither for YUV quantization
        ytex = self.ctx.texture((w, h * 3 // 2), 1, dtype="f1")
        self._vid = dict(w=w, h=h, ftex=ftex, ffbo=self.ctx.framebuffer(color_attachments=[ftex]),
                         ytex=ytex, yfbo=self.ctx.framebuffer(color_attachments=[ytex]),
                         pbos=[self.ctx.buffer(reserve=w * h * 3 // 2) for _ in range(2)])
        return w * h * 3 // 2

    def video_frame(self, slot):
        """Draw the current uniforms, convert to I420 and start an async readback into PBO slot."""
        v = self._vid
        w, h = v["w"], v["h"]
        self.tex.use(0)
        self._set("uGrid", 0)
        self._set("uOffset", (0.0, 0.0))
        v["ffbo"].use()
        v["ffbo"].viewport = (0, 0, w, h)
        self.vao.render(moderngl.TRIANGLES, vertices=3)
        v["ftex"].use(1)
        self.yuv_prog["uFrame"].value = 1
        self.yuv_prog["uSize"].value = (w, h)
        v["yfbo"].use()
        v["yfbo"].viewport = (0, 0, w, h * 3 // 2)
        self.yuv_vao.render(moderngl.TRIANGLES, vertices=3)
        v["yfbo"].read_into(v["pbos"][slot], components=1, alignment=1)

    def video_collect(self, slot, out):
        """Wait for PBO slot and copy its I420 bytes into out (a uint8 array)."""
        self._vid["pbos"][slot].read_into(out)

    def video_end(self):
        if self._vid:
            for k in ("ffbo", "yfbo", "ftex", "ytex"):
                self._vid[k].release()
            for b in self._vid["pbos"]:
                b.release()
            self._vid = None

    def render_still(self, S, w, h, time_base=0.0, quality="final"):
        E = eff(S, w / h)
        cols, rows, n = grid_dims(w / h, quality)
        G = self.sky_grid(E, cols, rows, n)
        self.set_uniforms(E, G, w, h, time_base=time_base)
        return self.draw(w, h), E

    def release(self):
        for f in self._fbos.values():
            f.release()
        self._fbos.clear()
        self.ctx.release()
