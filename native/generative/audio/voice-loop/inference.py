"""A live voice loop: audio frames in, an effect, audio frames out.

The caller sends microphone audio and gets each frame back with the effect
applied, so what you say comes back in your ear as you say it. A test app for
streams: no model, no downloads.

What the socket carries is declared by the models, the same way the request
body is: ``audio`` is a live field in both directions, and the ordinary
fields can be changed while the stream runs.

    caller -> app   <binary PCM s16le mono 16 kHz>     an item of LoopInput.audio (a frame every ~20 ms)
    caller -> app   {"effect": "echo"}                  change LoopInput.effect mid-stream
    app -> caller   {"effect": "robot"}                 LoopOutput.effect; also the first frame, so the
                                                        caller knows the app is there
    app -> caller   <binary PCM s16le mono 16 kHz>     an item of LoopOutput.audio
    caller closes  ->  the function returns with what it heard
"""

from typing import Literal

import numpy as np
from pydantic import Field

from inferencesh import AudioMeta, BaseApp, BaseAppInput, BaseAppOutput, Live, OutputMeta, PCM16, Socket, Stream

SAMPLE_RATE = 16000

Effect = Literal["none", "robot", "echo", "chipmunk", "deep"]


class LoopInput(BaseAppInput):
    effect: Effect = Field(default="robot", description="Effect applied to the audio on its way back")
    gain: float = Field(default=1.0, ge=0.0, le=4.0, description="Output gain")
    audio: Stream[PCM16(SAMPLE_RATE)] = Field(description="Microphone audio, a frame every 20 ms or so")


class LoopOutput(BaseAppOutput):
    audio: Stream[PCM16(SAMPLE_RATE)] = Field(description="The same audio with the effect applied")
    effect: Effect = Field(default="robot", description="Effect in use")
    frames: int = Field(default=0, description="Audio frames received")
    seconds: float = Field(default=0, description="Audio received, in seconds")
    peak: float = Field(default=0, description="Loudest input sample, 0..1")


class App(BaseApp):
    async def setup(self, metadata):
        pass

    async def stream(self, input_data: LoopInput, socket: Socket) -> LoopOutput:
        fx = _Effects(SAMPLE_RATE)
        live = Live(socket, input_data, LoopOutput)
        frames, samples, peak = 0, 0, 0.0
        await live.send(effect=input_data.effect)

        async for update in live:
            if update.field == "effect":
                await live.send(effect=input_data.effect)
            if update.field != "audio":
                continue
            audio = np.frombuffer(update.value, dtype="<i2").astype(np.float32) / 32768.0
            frames += 1
            samples += len(audio)
            peak = max(peak, float(np.max(np.abs(audio))) if len(audio) else 0.0)
            out = fx.apply(input_data.effect, audio) * input_data.gain
            await live.send(audio=(np.clip(out, -1.0, 1.0) * 32767).astype("<i2").tobytes())

        seconds = samples / SAMPLE_RATE
        return LoopOutput(
            frames=frames, seconds=round(seconds, 3), peak=round(peak, 3), effect=input_data.effect,
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
