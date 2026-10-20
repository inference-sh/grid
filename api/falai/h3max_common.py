"""
Shared helpers for the fal.ai MiniMax H3 Max family of apps.

Symlink this file from each app folder (`ln -sr h3max_common.py <app>/h3max_common.py`)
and import it with a relative import. Do not copy it.
"""

import json
import logging
import subprocess
from typing import Dict, Iterable, Optional, Tuple

FPS = 24

# (resolution, aspect_ratio) -> (width, height). H3 Max output sizes per fal.
DIMENSIONS = {
    ("480P", "16:9"): (854, 480), ("480P", "9:16"): (480, 854),
    ("480P", "4:3"): (640, 480), ("480P", "3:4"): (480, 640),
    ("480P", "1:1"): (480, 480), ("480P", "21:9"): (1120, 480),
    ("768P", "16:9"): (1365, 768), ("768P", "9:16"): (768, 1365),
    ("768P", "4:3"): (1024, 768), ("768P", "3:4"): (768, 1024),
    ("768P", "1:1"): (768, 768), ("768P", "21:9"): (1792, 768),
    ("1080P", "16:9"): (1920, 1080), ("1080P", "9:16"): (1080, 1920),
    ("1080P", "4:3"): (1440, 1080), ("1080P", "3:4"): (1080, 1440),
    ("1080P", "1:1"): (1080, 1080), ("1080P", "21:9"): (2520, 1080),
    ("2K", "16:9"): (2560, 1440), ("2K", "9:16"): (1440, 2560),
    ("2K", "4:3"): (1920, 1440), ("2K", "3:4"): (1440, 1920),
    ("2K", "1:1"): (1440, 1440), ("2K", "21:9"): (3360, 1440),
}


def dims_for(resolution: str, aspect_ratio: str) -> Tuple[int, int]:
    """Nominal output size for a resolution and aspect ratio; 16:9 when the ratio is adaptive/auto."""
    res = resolution.upper()
    ratio = aspect_ratio if aspect_ratio in ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16") else "16:9"
    return DIMENSIONS.get((res, ratio), DIMENSIONS.get((res, "16:9"), (1365, 768)))


def uris(files: Optional[Iterable]) -> list:
    return [f.uri for f in files] if files else []


def probe_video(path: str, logger: Optional[logging.Logger] = None) -> Dict[str, float]:
    """Width, height, fps and seconds of a video file via ffprobe. Empty dict when ffprobe is unavailable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", path],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(out.stdout or "{}")
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.warning(f"ffprobe failed: {e}")
        return {}
    info: Dict[str, float] = {}
    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            info["width"] = int(s.get("width") or 0)
            info["height"] = int(s.get("height") or 0)
            num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
            try:
                info["fps"] = round(float(num) / float(den or 1))
            except (ValueError, ZeroDivisionError):
                info["fps"] = 0
            break
    try:
        info["seconds"] = float(data.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        info["seconds"] = 0.0
    return info
