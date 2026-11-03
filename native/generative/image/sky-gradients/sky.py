"""Sky physics, parameters and randomiser, ported from the sky gradients page.

Single-scattering atmosphere (Rayleigh, Mie haze, ozone, volcanic and smoke layers)
computed on a small HDR grid; the shader interpolates it to any output size.
"""
import math
import random

import numpy as np

R, RT, LUT_W, LUT_H = 6360.0, 6460.0, 128, 40
SUN_MIN, SUN_MAX = -9.0, 0.5
SUN_MID, SUN_AMP = (SUN_MIN + SUN_MAX) / 2, (SUN_MAX - SUN_MIN) / 2

DEFAULT = dict(
    sunEl=-2.2, look=0.0, fovH=60.0, framing="natural", horizon=0.3, elTop=50.0, elBot=-1.0, curve=1.8,
    alt=0.05, below="mirror",
    haze=0.3, mieG=0.8, rayleigh=1.0, ozone=1.0, volcanic=0.0, smoke=0.0, ms=0.35,
    showSun=False, sunSize=1.2, bloom=0.6,
    aurora=0.0, auroraEl=6.0, auroraRed=0.3, auroraPal=7, auroraHue=0.0, auroraVar=0.7, auroraSpike=0.2,
    auroraMotion=True, leakMotion=True,
    autoExp=True, ev=0.0, contrast=1.02, sat=1.06, warmth=0.12, lift=0.35, grain=0.0, grainSize=0.5,
    leak=0.0, leakAngle=200.0, vignette=0.0, seed=7,
)


def clamp(x, a, b):
    return min(b, max(a, x))


# ---------------- physics ----------------
def coeffs(S):
    mie = 0.4 + 40 * S["haze"] ** 2
    v, k, ry, oz = S["volcanic"] * 0.035, S["smoke"] * 0.25, S["rayleigh"], S["ozone"]
    return dict(
        betaR=np.array([5.802e-3, 13.558e-3, 33.1e-3]) * ry,
        mieS=3.996e-3 * mie, mieE=4.44e-3 * mie,
        ozA=np.array([0.65e-3, 1.881e-3, 0.085e-3]) * oz,
        stratS=np.array([v * 0.92, v, v * 1.06]), stratE=np.array([v * 0.95, v * 1.03, v * 1.09]),
        smokeE=np.array([k * 0.68, k, k * 1.45]), smokeS=np.array([k * 0.68 * 0.86, k * 0.86, k * 1.45 * 0.86]),
    )


def _dens(hh):
    z = (hh - 20) / 4
    return (np.exp(-hh / 8), np.exp(-hh / 1.2), np.maximum(0, 1 - np.abs(hh - 25) / 15), np.exp(-z * z),
            np.exp(-hh / 2.5))


def compute_lut(c):
    """Sun transmittance by height and elevation, shape (LUT_H, LUT_W, 3)."""
    N = 28
    yy = np.arange(LUT_H)[:, None] / (LUT_H - 1)
    r0 = R + 100 * yy * yy
    v = np.arange(LUT_W)[None, :] / (LUT_W - 1) * 1.37 - 0.37
    e = np.sign(v) * v * v * np.pi / 2
    se, ce = np.sin(e), np.cos(e)
    b = r0 * se
    cc = (r0 - RT) * (r0 + RT)
    t_exit = -b + np.sqrt(b * b - cc)
    tc = -b
    ext = lambda dr, dm, dO, ds, dk: (c["betaR"][:, None, None] * dr + c["mieE"] * dm + c["ozA"][:, None, None] * dO
                                       + c["stratE"][:, None, None] * ds + c["smokeE"][:, None, None] * dk)

    def seg(t0, t1, dense):
        L = t1 - t0
        od = np.zeros((3,) + np.broadcast(L, r0).shape)
        for k in range(N):
            s = (k + 0.5) / N
            if dense:
                t, dt = t0 + s * s * L, 2 * s / N * L
            else:
                t, dt = t0 + (1 - (1 - s) ** 2) * L, 2 * (1 - s) / N * L
            px, py = t * ce, r0 + t * se
            hh = np.maximum(np.sqrt(px * px + py * py) - R, 0)
            od += ext(*_dens(hh)) * dt
        return od

    split = (tc > 0) & (tc < t_exit)
    tcs = np.clip(tc, 0, None)
    od = np.where(split, seg(0.0, tcs, False) + seg(tcs, t_exit, True), seg(0.0, t_exit, True))
    return np.exp(-od).transpose(1, 2, 0).copy()


_lut_cache = {}


def get_lut(c):
    key = tuple(np.round(np.concatenate([np.ravel(np.asarray(x, dtype=float)) for x in c.values()]), 12))
    lut = _lut_cache.get(key)
    if lut is None:
        lut = compute_lut(c)
        if len(_lut_cache) > 40:
            _lut_cache.pop(next(iter(_lut_cache)))
        _lut_cache[key] = lut
    return lut


def lut_sample(lut, P, sd):
    r = np.linalg.norm(P, axis=-1)
    h = r - R
    mu = np.clip(np.sum(P * sd, axis=-1) / r, -1, 1)
    e = np.arcsin(mu)
    v = np.sign(e) * np.sqrt(np.abs(e) / (np.pi / 2))
    x = np.clip((v + 0.37) / 1.37, 0, 1) * (LUT_W - 1)
    y = np.sqrt(np.clip(h / 100, 0, 1)) * (LUT_H - 1)
    x0 = np.minimum(np.floor(x), LUT_W - 2).astype(np.int64)
    y0 = np.minimum(np.floor(y), LUT_H - 2).astype(np.int64)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    a, b, cc, d = lut[y0, x0], lut[y0, x0 + 1], lut[y0 + 1, x0], lut[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (cc * (1 - fx) + d * fx) * fy


def hg(c, g):
    return (1 - g * g) / (4 * math.pi * np.power(np.maximum(1 + g * g - 2 * g * c, 1e-4), 1.5))


def sky(S, c, lut, sd, rd, amb, N=24):
    """Radiance along each ray in rd (M, 3). sd and amb are (3,) or per ray (M, 3)."""
    alt = S["alt"]
    r0 = R + alt
    b = r0 * rd[:, 1]
    cc = (alt - 100) * (2 * R + alt + 100)
    t_max = -b + np.sqrt(np.maximum(b * b - cc, 0))
    mu = np.sum(rd * sd, axis=-1)
    phR = 3 / (16 * math.pi) * (1 + mu * mu)
    phM, phS, phK = hg(mu, S["mieG"]), hg(mu, 0.72), hg(mu, 0.55)
    iso, am = S["ms"] / (4 * math.pi), S["ms"] * 2.2
    M = rd.shape[0]
    L = np.zeros((M, 3))
    T = np.ones((M, 3))
    tp = 0.0
    for i in range(N):
        s = (i + 1) / N
        t = t_max * s * s
        dt, tm, tp = t - tp, 0.5 * (t + tp), t
        P = np.stack([rd[:, 0] * tm, r0 + rd[:, 1] * tm, rd[:, 2] * tm], axis=-1)
        h = np.maximum(np.linalg.norm(P, axis=-1) - R, 0)
        dr, dm, dO, ds, dk = (x[:, None] for x in _dens(h))
        TS = lut_sample(lut, P, sd)
        sR, sM, sS, sK = c["betaR"] * dr, c["mieS"] * dm, c["stratS"] * ds, c["smokeS"] * dk
        ext = sR + c["mieE"] * dm + c["ozA"] * dO + c["stratE"] * ds + c["smokeE"] * dk
        Sv = ((sR * phR[:, None] + sM * phM[:, None] + sS * phS[:, None] + sK * phK[:, None]
               + (sR + sM + sS + sK) * iso) * TS + (sR + sM + sS) * amb * am)
        st = np.exp(-ext * dt[:, None])
        L += T * Sv * (1 - st) / np.maximum(ext, 1e-7)
        T *= st
    return L * 20


def dir_from(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack(np.broadcast_arrays(np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)), axis=-1)


def dip_deg(S):
    return -math.degrees(math.acos(R / (R + S["alt"])))


def eff(S, ar):
    """Natural framing keeps one degree the same size across and up."""
    none = S["below"] == "none"
    if S["framing"] == "natural":
        span = S["fovH"] / ar
        el_bot = dip_deg(S) + 0.02 if none else -S["horizon"] * span
        return {**S, "elBot": el_bot, "elTop": el_bot + span, "curve": 1.0}
    if none:
        return {**S, "elBot": max(S["elBot"], dip_deg(S) + 0.02)}
    return dict(S)


def compute_grid(S, cols, rows, N=24):
    """HDR sky radiance grid (rows, cols, 4), row 0 at the bottom of the frame."""
    return compute_grids(S, [S["sunEl"]], cols, rows, N)[0]


def compute_grids(S, suns, cols, rows, N=24, max_rays=400_000):
    """One grid per sun elevation in suns, computed in batched passes (only the sun differs)."""
    out = []
    per = max(1, max_rays // (cols * rows))
    for i in range(0, len(suns), per):
        out += _grids(S, suns[i:i + per], cols, rows, N)
    return out


def _grids(S, suns, cols, rows, N):
    K = len(suns)
    c = coeffs(S)
    lut = get_lut(c)
    sds = dir_from(0.0, np.asarray(suns, float))                       # (K, 3)
    amb_dirs = dir_from(np.array([0, 90, 180, 270] * 3, float), np.repeat([8.0, 35.0, 75.0], 4))
    amb = sky(S, c, lut, np.repeat(sds, 12, axis=0), np.tile(amb_dirs, (K, 1)), np.zeros(3))
    amb = amb.reshape(K, 12, 3).sum(1) / 240                          # (K, 3)
    dip = dip_deg(S)
    v = (np.arange(rows) + 0.5) / rows
    el = S["elBot"] + (S["elTop"] - S["elBot"]) * np.power(v, S["curve"])
    e, f, a = el.copy(), np.ones(rows), np.ones(rows)
    low = el < dip + 0.05
    depth = np.radians(dip - el)
    if S["below"] == "mirror":
        cosi = np.sin(np.maximum(depth, 0))
        fm = 0.02 + 0.98 * np.power(1 - cosi, 5)
        e = np.where(low, dip + 0.05 + (dip - el), e)
        f = np.where(low, 0.15 + 0.85 * fm, f)
    elif S["below"] == "extend":
        e = np.where(low, dip + 0.05, e)
        f = np.where(low, 1 / (1 + np.degrees(depth) * 0.12), f)
    else:
        e = np.where(low, dip + 0.05, e)
        a = np.where(low, 0.0, a)
    az = S["look"] - S["fovH"] / 2 + S["fovH"] * (np.arange(cols) + 0.5) / cols
    rd = dir_from(az[None, :], e[:, None]).reshape(-1, 3)
    M = rd.shape[0]
    rad_all = sky(S, c, lut, np.repeat(sds, M, axis=0), np.tile(rd, (K, 1)), np.repeat(amb, M, axis=0), N)
    rad_all = rad_all.reshape(K, rows, cols, 3) * f[None, :, None, None]
    mask = np.broadcast_to(a[:, None] > 0, (rows, cols))
    sun_cols = lut_sample(lut, np.broadcast_to(np.array([0.0, R + S["alt"], 0.0]), (K, 3)), sds) * 20
    grids = []
    for k in range(K):
        rad = rad_all[k]
        data = np.concatenate([rad, np.broadcast_to(a[:, None, None], (rows, cols, 1))], axis=-1)
        lum = 0.2126 * rad[..., 0] + 0.7152 * rad[..., 1] + 0.0722 * rad[..., 2]
        avg = math.exp(np.log(1e-7 + lum[mask]).sum() / max(mask.sum(), 1)) + 2e-5
        if S["autoExp"]:
            exposure = 0.3 / avg ** S.get("expPow", 0.86) * 2 ** S["ev"]
        else:
            exposure = 1.2 * 2 ** S["ev"]
        grids.append(dict(data=data.astype(np.float32), cols=cols, rows=rows, exposure=exposure,
                          sunCol=sun_cols[k], dip=dip))
    return grids


def grid_dims(ar, quality="final"):
    if quality == "final":
        return max(24, round(40 * min(ar, 2.5))), 160, 64
    return max(12, round(40 * min(ar, 2))), 112, 48


# ---------------- naming ----------------
def phase_name(el):
    if el > 6: return "daylight"
    if el > -0.83: return "golden hour"
    if el > -4: return "civil twilight"
    if el > -6: return "blue hour"
    if el > -12: return "nautical twilight"
    if el > -18: return "astronomical twilight"
    return "night"


def facing_name(look):
    a = abs(((look % 360) + 540) % 360 - 180)
    return "facing the sun" if a < 35 else "facing away from the sun" if a > 145 else "side-lit"


# ---------------- randomiser ----------------
def randomize(base, rng: random.Random, near=False, spice=True, lock_sun=False, lock_air=False, lock_view=False,
              lock_film=False):
    rnd = lambda a, b: a + rng.random() * (b - a)
    pick = lambda xs: xs[int(rng.random() * len(xs))]
    s = dict(base)
    if not lock_sun:
        s["sunEl"] = clamp(base["sunEl"] + rnd(-1.5, 1.5), SUN_MIN, SUN_MAX) if near else rnd(-8, 0.2)
    if not lock_view:
        if near:
            s["look"] = clamp(base["look"] + rnd(-25, 25), -180, 180)
            s["curve"] = clamp(base["curve"] + rnd(-0.3, 0.3), 0.5, 4)
            s["elTop"] = clamp(base["elTop"] + rnd(-10, 10), 4, 90)
            s["fovH"] = clamp(base["fovH"] * rnd(0.8, 1.25), 10, 200)
            s["horizon"] = clamp(base["horizon"] + rnd(-0.08, 0.08), 0, 0.8)
        else:
            if rng.random() < 0.65:
                look = rnd(-30, 30)
            elif rng.random() < 0.6:
                look = pick([-180, 180]) + rnd(-25, 25)
            else:
                look = rnd(-120, 120)
            s["look"] = ((look + 540) % 360) - 180
            s["framing"] = "natural"
            s["fovH"] = pick([30, 45, 60, 80, 100, 130])
            s["horizon"] = rnd(0.05, 0.4)
            s["elTop"] = pick([12, 20, 30, 45, 60, 85])
            s["elBot"] = rnd(-3, 0.5)
            s["curve"] = rnd(1, 2.6)
            s["alt"] = rnd(3, 12) if rng.random() < 0.15 else 0.05
            s["below"] = pick(["none", "none", "mirror", "mirror", "extend"])
    if not lock_air:
        if near:
            s["haze"] = clamp(base["haze"] + rnd(-0.1, 0.1), 0, 1)
            s["ozone"] = clamp(base["ozone"] + rnd(-0.3, 0.3), 0, 3)
        else:
            s["haze"], s["mieG"], s["rayleigh"], s["ozone"] = rnd(0.05, 0.42), rnd(0.7, 0.9), rnd(0.8, 1.4), rnd(0.7, 1.8)
            s["volcanic"] = rnd(0.3, 1) if rng.random() < 0.18 else 0.0
            s["smoke"] = rnd(0.2, 0.9) if rng.random() < 0.12 else 0.0
    # near sunrise/sunset the sky away from the sun is still daytime blue: keep those views toward the sun and low
    if not lock_view and s["sunEl"] > -1.5:
        a = abs(((s["look"] % 360) + 540) % 360 - 180)
        if 50 < a < 140:
            s["look"] = rnd(-40, 40)
        if a >= 140 and s["sunEl"] > -0.5:
            s["look"] = rnd(-40, 40)
        s["elTop"] = min(s["elTop"], 30)
        s["fovH"] = min(s["fovH"], 80)
    if not lock_film:
        # grain is left as given (default 0): the randomizer does not add it
        s["warmth"], s["sat"], s["lift"] = rnd(-0.15, 0.4), rnd(0.9, 1.25), rnd(0.25, 0.6)
        s["vignette"] = rnd(0.1, 0.5) if rng.random() < 0.3 else 0.0
    if spice:
        s["leak"] = rnd(0.15, 0.7) if rng.random() < 0.3 else 0.0
        s["leakAngle"] = round(rnd(0, 360))
        s["aurora"] = rnd(0.4, 1.3) if (s["sunEl"] < -3 and rng.random() < 0.35) else 0.0
        s["auroraEl"] = rnd(3, 14)
        s["auroraRed"] = rnd(0.5, 1.3) if rng.random() < 0.35 else rnd(0, 0.4)
        s["auroraPal"] = pick([1, 2, 3, 4, 5, 6, 7, 7])
        s["auroraHue"] = round(rnd(-15, 15)) if rng.random() < 0.4 else 0
        s["auroraVar"] = rnd(0.4, 1.2)
    s["seed"] = int(rnd(0, 99))
    return s


# ---------------- CSS extraction ----------------
def _hex(c):
    return "#" + "".join("%02x" % int(round(clamp(v, 0, 255))) for v in c)


def css_from_pixels(px):
    """px: (h, w, 3) uint8 top-down. Returns (css gradient, 7-colour palette)."""
    h, w = px.shape[:2]
    cols = px[:, 4:w - 4, :].astype(np.float64).mean(axis=1)
    N = len(cols)
    stops = [0, N - 1]

    def err(i, a, b):
        t = (i - a) / (b - a)
        return float(np.max(np.abs(cols[a] * (1 - t) + cols[b] * t - cols[i])))

    for _ in range(12):
        best, be = -1, 2.5
        for k in range(len(stops) - 1):
            for i in range(stops[k] + 1, stops[k + 1]):
                e = err(i, stops[k], stops[k + 1])
                if e > be:
                    be, best = e, i
        if best < 0:
            break
        stops = sorted(stops + [best])

    def pct(i):
        p = "%.1f" % (i / (N - 1) * 100)
        return p[:-2] if p.endswith(".0") else p

    css = "linear-gradient(180deg, %s)" % ", ".join("%s %s%%" % (_hex(cols[i]), pct(i)) for i in stops)
    palette = [_hex(cols[int(i / 6 * (N - 1) + 0.5)]) for i in range(7)]
    return css, palette
