"""A live voice loop: audio frames in, an effect, audio frames out.

The client sends microphone audio as binary frames (PCM s16le mono, 16 kHz,
any frame size) and gets each frame back with the effect applied, so what
you say comes back in your ear as you say it. A test app for streams: no
model, no downloads.

Protocol on the socket (the app's, not the platform's):

    app -> client   {"type": "audio_format", "encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}
    client -> app   <binary PCM s16le mono 16 kHz>            (a frame every ~20 ms)
    app -> client   <binary PCM s16le mono 16 kHz>            (the same frame, with the effect)
    client -> app   {"type": "effect", "effect": "echo"}       (switch the effect mid-stream)
    client closes  ->  the function returns with what it heard
"""

import json
import time
from typing import Literal, Optional

import numpy as np
from pydantic import Field

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, OutputMeta, AudioMeta

try:
    from inferencesh import Socket
except ImportError:  # SDK older than streams; the kernel only needs the parameter name
    from typing import Any as Socket

SAMPLE_RATE = 16000

Effect = Literal["none", "robot", "echo", "chipmunk", "deep"]


class LoopInput(BaseAppInput):
    effect: Effect = Field(default="robot", description="Effect applied to the audio on its way back")
    gain: float = Field(default=1.0, ge=0.0, le=4.0, description="Output gain")


class LoopOutput(BaseAppOutput):
    frames: int = Field(description="Audio frames received")
    seconds: float = Field(description="Audio received, in seconds")
    peak: float = Field(description="Loudest input sample, 0..1")
    effect: str = Field(description="Effect in use when the stream ended")


class App(BaseApp):
    async def setup(self, metadata):
        pass

    async def stream(self, input_data: LoopInput, socket: Socket) -> LoopOutput:
        fx = _Effects(SAMPLE_RATE)
        effect = input_data.effect
        frames, samples, peak = 0, 0, 0.0
        await socket.send({"type": "audio_format", "encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE, "channels": 1})

        async for frame in socket:
            if not isinstance(frame, bytes):
                effect = _effect_of(frame) or effect
                await socket.send({"type": "effect", "effect": effect})
                continue
            audio = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
            frames += 1
            samples += len(audio)
            peak = max(peak, float(np.max(np.abs(audio))) if len(audio) else 0.0)
            out = fx.apply(effect, audio) * input_data.gain
            await socket.send((np.clip(out, -1.0, 1.0) * 32767).astype("<i2").tobytes())

        seconds = samples / SAMPLE_RATE
        return LoopOutput(
            frames=frames, seconds=round(seconds, 3), peak=round(peak, 3), effect=effect,
            output_meta=OutputMeta(inputs=[AudioMeta(seconds=seconds)], outputs=[AudioMeta(seconds=seconds)]),
        )


class _Effects:
    """Stateful effects over a stream of frames: the state carries across frames
    so the output is continuous whatever the frame size."""

    def __init__(self, rate: int):
        self.rate = rate
        self.phase = 0.0                              # robot: ring modulator phase
        self.delay = np.zeros(int(rate * 0.3), dtype=np.float32)  # echo: 300 ms line
        self.delay_pos = 0
        self.resample_tail = np.zeros(0, dtype=np.float32)        # pitch: leftover input

    def apply(self, effect: str, audio: np.ndarray) -> np.ndarray:
        if effect == "robot":
            t = (np.arange(len(audio)) + self.phase) / self.rate
            self.phase += len(audio)
            return audio * np.sin(2 * np.pi * 50 * t).astype(np.float32)
        if effect == "echo":
            out = np.empty_like(audio)
            for i, x in enumerate(audio):
                d = self.delay[self.delay_pos]
                y = x + 0.5 * d
                self.delay[self.delay_pos] = y
                self.delay_pos = (self.delay_pos + 1) % len(self.delay)
                out[i] = y
            return out
        if effect in ("chipmunk", "deep"):
            # Pitch by resampling; keeps the frame length so playback stays
            # continuous, at the cost of dropping or repeating a little audio.
            factor = 1.35 if effect == "chipmunk" else 0.75
            src = np.concatenate([self.resample_tail, audio])
            n = len(audio)
            need = int(n * factor)
            if len(src) < need:
                src = np.pad(src, (0, need - len(src)), mode="edge")
            idx = np.linspace(0, need - 1, n)
            out = np.interp(idx, np.arange(len(src)), src).astype(np.float32)
            self.resample_tail = src[need:][-n:] if factor < 1 else np.zeros(0, dtype=np.float32)
            return out
        return audio


def _effect_of(frame: str) -> Optional[str]:
    try:
        msg = json.loads(frame)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(msg, dict) and msg.get("type") == "effect" and msg.get("effect") in ("none", "robot", "echo", "chipmunk", "deep"):
        return msg["effect"]
    return None
