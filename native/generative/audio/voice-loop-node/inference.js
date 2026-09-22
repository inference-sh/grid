/**
 * A live voice loop: audio frames in, an effect, audio frames out. The Node
 * twin of infsh/voice-loop, so the two kernels are held to the same contract.
 *
 * The caller sends microphone audio and gets each frame back with the effect
 * applied, so what you say comes back in your ear as you say it. A test app
 * for streams: no model, no downloads.
 *
 * What the socket carries is declared by the schemas, the same way the request
 * body is: `audio` is a live field in both directions, and the ordinary fields
 * can be changed while the stream runs.
 *
 *   caller -> app   <binary PCM s16le mono 16 kHz>   an item of StreamInput.audio (a frame every ~20 ms)
 *   caller -> app   {"effect": "echo"}                change StreamInput.effect mid-stream
 *   app -> caller   {"effect": "robot"}               StreamOutput.effect; also the first frame, so the
 *                                                     caller knows the app is there
 *   app -> caller   <binary PCM s16le mono 16 kHz>   an item of StreamOutput.audio
 *   caller closes  ->  the function returns with what it heard
 */
import { z } from "zod";
import { Live, audioMeta, createStreamSchema, pcm16 } from "@inferencesh/app";

const SAMPLE_RATE = 16000;

const Effect = z.enum(["none", "robot", "echo", "chipmunk", "deep"]);

export const StreamInput = z.object({
  effect: Effect.default("robot").describe("Effect applied to the audio on its way back"),
  gain: z.number().min(0).max(4).default(1).describe("Output gain"),
  audio: createStreamSchema(z, pcm16(z, SAMPLE_RATE)).describe("Microphone audio, a frame every 20 ms or so"),
});

export const StreamOutput = z.object({
  audio: createStreamSchema(z, pcm16(z, SAMPLE_RATE)).describe("The same audio with the effect applied"),
  effect: Effect.default("robot").describe("Effect in use"),
  frames: z.number().default(0).describe("Audio frames received"),
  seconds: z.number().default(0).describe("Audio received, in seconds"),
  peak: z.number().default(0).describe("Loudest input sample, 0..1"),
  output_meta: z.any().optional(),
});

export class App {
  async stream(input, socket) {
    const fx = new Effects(SAMPLE_RATE);
    const live = new Live(socket, input, StreamInput, StreamOutput);
    let frames = 0;
    let samples = 0;
    let peak = 0;
    await live.send({ effect: input.effect });

    for await (const update of live) {
      if (update.field === "effect") {
        await live.send({ effect: input.effect });
      }
      if (update.field !== "audio") continue;
      const audio = pcmToFloat(update.value);
      frames += 1;
      samples += audio.length;
      for (const x of audio) peak = Math.max(peak, Math.abs(x));
      const out = fx.apply(input.effect, audio);
      await live.send({ audio: floatToPCM(out, input.gain) });
    }

    const seconds = samples / SAMPLE_RATE;
    return {
      frames,
      seconds: round3(seconds),
      peak: round3(peak),
      effect: input.effect,
      output_meta: { inputs: [audioMeta({ seconds })], outputs: [audioMeta({ seconds })] },
    };
  }
}

function pcmToFloat(buf) {
  const samples = new Int16Array(buf.buffer, buf.byteOffset, buf.byteLength >> 1);
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) out[i] = samples[i] / 32768;
  return out;
}

function floatToPCM(audio, gain) {
  const out = new Int16Array(audio.length);
  for (let i = 0; i < audio.length; i++) {
    const x = Math.max(-1, Math.min(1, audio[i] * gain));
    out[i] = Math.round(x * 32767);
  }
  return Buffer.from(out.buffer, out.byteOffset, out.byteLength);
}

function round3(x) {
  return Math.round(x * 1000) / 1000;
}

/**
 * Stateful effects over a stream of frames: the state carries across frames
 * so the output is continuous whatever the frame size.
 */
class Effects {
  constructor(rate) {
    this.rate = rate;
    this.phase = 0; // robot: ring modulator phase
    this.delay = new Float32Array(Math.floor(rate * 0.3)); // echo: 300 ms line
    this.delayPos = 0;
    this.resampleTail = new Float32Array(0); // pitch: leftover input
  }

  apply(effect, audio) {
    if (effect === "robot") {
      const out = new Float32Array(audio.length);
      for (let i = 0; i < audio.length; i++) {
        out[i] = audio[i] * Math.sin((2 * Math.PI * 50 * (i + this.phase)) / this.rate);
      }
      this.phase += audio.length;
      return out;
    }
    if (effect === "echo") {
      const out = new Float32Array(audio.length);
      for (let i = 0; i < audio.length; i++) {
        const y = audio[i] + 0.5 * this.delay[this.delayPos];
        this.delay[this.delayPos] = y;
        this.delayPos = (this.delayPos + 1) % this.delay.length;
        out[i] = y;
      }
      return out;
    }
    if (effect === "chipmunk" || effect === "deep") {
      // Pitch by resampling; keeps the frame length so playback stays
      // continuous, at the cost of dropping or repeating a little audio.
      const factor = effect === "chipmunk" ? 1.35 : 0.75;
      const n = audio.length;
      const need = Math.floor(n * factor);
      let src = concat(this.resampleTail, audio);
      if (src.length < need) src = padEdge(src, need);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const pos = (i * (need - 1)) / Math.max(1, n - 1);
        const lo = Math.floor(pos);
        const hi = Math.min(lo + 1, src.length - 1);
        out[i] = src[lo] + (src[hi] - src[lo]) * (pos - lo);
      }
      this.resampleTail = factor < 1 ? src.subarray(need).slice(-n) : new Float32Array(0);
      return out;
    }
    return audio;
  }
}

function concat(a, b) {
  const out = new Float32Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function padEdge(src, length) {
  const out = new Float32Array(length);
  out.set(src, 0);
  out.fill(src.length ? src[src.length - 1] : 0, src.length);
  return out;
}
