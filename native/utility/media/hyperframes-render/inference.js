import { z } from "zod";
import { File, videoMeta } from "@inferencesh/app";
import puppeteer from "puppeteer-core";
import { spawn } from "node:child_process";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";

export const RunInput = z.object({
  html: z
    .string()
    .describe(
      "Hyperframes HTML composition. Use class='clip' with data-start, data-duration, " +
        "data-track-index attributes. Register GSAP timelines on window.__timelines (paused)."
    ),
  width: z.number().default(1920).describe("Video width in pixels"),
  height: z.number().default(1080).describe("Video height in pixels"),
  fps: z.number().default(30).describe("Frames per second"),
  duration: z
    .number()
    .default(0)
    .describe(
      "Video duration in seconds. 0 = auto-detect from clip data-start + data-duration attributes."
    ),
  codec: z
    .enum(["h264", "h265", "vp8", "vp9", "gif", "prores"])
    .default("h264")
    .describe("Output codec"),
  scale_factor: z
    .number()
    .default(1)
    .describe("Device scale factor (2 for retina-quality output)"),
  transparent: z
    .boolean()
    .default(false)
    .describe(
      "Render with alpha channel (only works with vp8/vp9/prores codecs)"
    ),
  audio_url: z
    .string()
    .optional()
    .describe("URL to an audio file to mux into the output"),
});

export const RunOutput = z.object({
  video: z.any().describe("Rendered video file"),
});

// --- FFmpeg codec configs ---

function ffmpegArgs(codec, transparent, fps) {
  const input = ["-f", "image2pipe", "-framerate", `${fps}`, "-i", "pipe:0"];

  switch (codec) {
    case "h264":
      return {
        input,
        output: ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18", "-movflags", "+faststart"],
        ext: "mp4",
      };
    case "h265":
      return {
        input,
        output: ["-c:v", "libx265", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20", "-movflags", "+faststart"],
        ext: "mp4",
      };
    case "vp8":
      return {
        input,
        output: [
          "-c:v", "libvpx",
          ...(transparent ? ["-pix_fmt", "yuva420p", "-auto-alt-ref", "0"] : ["-pix_fmt", "yuv420p"]),
          "-b:v", "2M", "-crf", "10",
        ],
        ext: "webm",
      };
    case "vp9":
      return {
        input,
        output: [
          "-c:v", "libvpx-vp9",
          ...(transparent ? ["-pix_fmt", "yuva420p", "-auto-alt-ref", "0"] : ["-pix_fmt", "yuv420p"]),
          "-b:v", "2M", "-crf", "18",
        ],
        ext: "webm",
      };
    case "prores":
      return {
        input,
        output: [
          "-c:v", "prores_ks",
          "-profile:v", transparent ? "4444" : "3",
          "-pix_fmt", transparent ? "yuva444p10le" : "yuv422p10le",
        ],
        ext: "mov",
      };
    case "gif":
      return {
        input,
        output: ["-vf", `split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse`],
        ext: "gif",
      };
  }
}

// --- Page scripts ---

// Auto-detect duration from clip attributes
const detectDuration = `() => {
  let maxEnd = 0;
  for (const el of document.querySelectorAll('.clip[data-start][data-duration]')) {
    const start = parseFloat(el.dataset.start) || 0;
    const dur = parseFloat(el.dataset.duration) || 0;
    maxEnd = Math.max(maxEnd, start + dur);
  }
  // Also check GSAP timeline durations
  const timelines = window.__timelines || {};
  for (const tl of Object.values(timelines)) {
    if (tl && typeof tl.duration === 'function') {
      maxEnd = Math.max(maxEnd, tl.duration());
    }
  }
  return maxEnd;
}`;

// Seek all animations + handle Hyperframes clip visibility and track z-index
const seekTo = `(targetTime) => {
  // 1. GSAP registered timelines
  const timelines = window.__timelines || {};
  for (const tl of Object.values(timelines)) {
    if (tl && typeof tl.seek === 'function') tl.seek(targetTime);
  }
  if (window.gsap && window.gsap.globalTimeline) {
    window.gsap.globalTimeline.seek(targetTime);
  }

  // 2. Web Animations API (covers CSS animations + WAAPI)
  for (const anim of document.getAnimations()) {
    anim.pause();
    anim.currentTime = targetTime * 1000;
  }

  // 3. Hyperframes clip visibility — show/hide based on data-start/data-duration
  for (const el of document.querySelectorAll('.clip[data-start][data-duration]')) {
    const start = parseFloat(el.dataset.start) || 0;
    const dur = parseFloat(el.dataset.duration) || 0;
    const visible = targetTime >= start && targetTime < start + dur;
    el.style.visibility = visible ? 'visible' : 'hidden';
    el.style.pointerEvents = visible ? 'auto' : 'none';
  }

  // 4. Track layering — apply z-index from data-track-index
  for (const el of document.querySelectorAll('[data-track-index]')) {
    el.style.zIndex = el.dataset.trackIndex;
  }

  // 5. Force synchronous layout flush
  document.body.offsetHeight;
}`;

// Wait for fonts + images + video metadata
const waitReady = `async () => {
  await document.fonts.ready;
  await Promise.all([
    ...Array.from(document.images).map(img =>
      img.complete ? Promise.resolve() :
        new Promise(r => { img.onload = r; img.onerror = r; })
    ),
    ...Array.from(document.querySelectorAll('video')).map(v =>
      v.readyState >= 1 ? Promise.resolve() :
        new Promise(r => { v.onloadedmetadata = r; v.onerror = r; })
    ),
  ]);
}`;

// Pause everything before frame-by-frame capture
const pauseAll = `() => {
  const timelines = window.__timelines || {};
  for (const tl of Object.values(timelines)) {
    if (tl && typeof tl.pause === 'function') tl.pause();
  }
  if (window.gsap && window.gsap.globalTimeline) {
    window.gsap.globalTimeline.pause();
  }
  for (const anim of document.getAnimations()) {
    anim.pause();
  }
  // Hide all clips initially
  for (const el of document.querySelectorAll('.clip[data-start][data-duration]')) {
    el.style.visibility = 'hidden';
  }
}`;

export class App {
  async setup() {
    console.log("Launching browser...");
    this.browser = await puppeteer.launch({
      executablePath: "/usr/bin/chromium",
      headless: true,
      args: [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--disable-extensions",
      ],
    });
    console.log("Browser ready");
  }

  async run(inputData) {
    const {
      html,
      width,
      height,
      fps,
      duration: durationInput,
      codec,
      scale_factor,
      transparent,
      audio_url,
    } = inputData;

    // --- page setup ---
    const page = await this.browser.newPage();
    await page.setViewport({ width, height, deviceScaleFactor: scale_factor });
    await page.setContent(html, { waitUntil: "networkidle0", timeout: 30000 });

    console.log("Waiting for fonts and assets...");
    await page.evaluate(`(${waitReady})()`);

    // Auto-detect duration if not specified
    let duration = durationInput;
    if (!duration || duration <= 0) {
      duration = await page.evaluate(`(${detectDuration})()`);
      if (!duration || duration <= 0) duration = 5;
      console.log(`Auto-detected duration: ${duration}s`);
    }

    // Pause all animations so we drive time manually
    await page.evaluate(`(${pauseAll})()`);

    const totalFrames = Math.round(duration * fps);
    const actualWidth = width * scale_factor;
    const actualHeight = height * scale_factor;

    console.log(
      `Rendering ${totalFrames} frames ${actualWidth}x${actualHeight} @ ${fps}fps codec=${codec}` +
        (transparent ? " transparent" : "") +
        (audio_url ? " +audio" : "")
    );

    // --- FFmpeg setup ---
    const ff = ffmpegArgs(codec, transparent, fps);
    const outputPath = join(tmpdir(), `${randomUUID()}.${ff.ext}`);

    const args = [
      ...ff.input,
      ...(audio_url ? ["-i", audio_url] : []),
      ...ff.output,
      ...(audio_url ? ["-c:a", "aac", "-shortest"] : []),
      "-y",
      outputPath,
    ];

    const proc = spawn("ffmpeg", args, { stdio: ["pipe", "pipe", "pipe"] });

    let stderr = "";
    proc.stderr.on("data", (d) => (stderr += d.toString()));

    const done = new Promise((resolve, reject) => {
      proc.on("close", (code) =>
        code === 0
          ? resolve()
          : reject(new Error(`FFmpeg exited ${code}: ${stderr.slice(-500)}`))
      );
      proc.on("error", reject);
    });

    // --- frame capture loop ---
    const logEvery = Math.max(1, Math.floor(totalFrames / 10));
    const screenshotOpts = {
      type: "png",
      omitBackground: transparent,
      captureBeyondViewport: false,
      optimizeForSpeed: true,
      encoding: "binary",
    };

    for (let f = 0; f < totalFrames; f++) {
      const targetTime = f / fps;
      await page.evaluate(`(${seekTo})(${targetTime})`);

      const buf = await page.screenshot(screenshotOpts);
      if (!proc.stdin.writable) break;

      const ok = proc.stdin.write(buf);
      if (!ok) await new Promise((r) => proc.stdin.once("drain", r));

      if ((f + 1) % logEvery === 0) {
        console.log(`Captured ${f + 1}/${totalFrames} frames`);
      }
    }

    console.log(`All ${totalFrames} frames captured, encoding...`);
    proc.stdin.end();
    await done;
    await page.close();

    console.log("Render complete");

    const resolution =
      actualHeight <= 720 ? "720p" : actualHeight <= 1080 ? "1080p" : "4k";

    return {
      video: File.fromPath(outputPath),
      output_meta: {
        outputs: [videoMeta({ resolution, seconds: duration, fps })],
      },
    };
  }

  async unload() {
    if (this.browser) await this.browser.close();
  }
}
