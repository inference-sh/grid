import { z } from "zod";
import { File, videoMeta } from "@inferencesh/app";
import { bundle } from "@remotion/bundler";
import {
  renderMedia,
  selectComposition,
  ensureBrowser,
} from "@remotion/renderer";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";

export const RunInput = z.object({
  title: z.string().describe("Main title text"),
  subtitle: z.string().default("").describe("Optional subtitle"),
  background_color: z
    .string()
    .default("#000000")
    .describe("Background color (hex)"),
  text_color: z.string().default("#ffffff").describe("Text color (hex)"),
  width: z.number().default(1920).describe("Video width in pixels"),
  height: z.number().default(1080).describe("Video height in pixels"),
  fps: z.number().default(30).describe("Frames per second"),
  duration_seconds: z
    .number()
    .default(5)
    .describe("Video duration in seconds"),
});

export const RunOutput = z.object({
  video: z.any().describe("Rendered MP4 video file"),
});

export class App {
  async setup() {
    // Pre-download browser for Remotion
    await ensureBrowser();

    // Bundle the Remotion project once — reused across all renders
    this.bundleLocation = await bundle({
      entryPoint: join(import.meta.dirname, "src", "index.ts"),
      onProgress: (progress) => {
        if (progress % 25 === 0) {
          console.log(`Bundling: ${progress}%`);
        }
      },
    });
    console.log("Bundle ready at:", this.bundleLocation);
  }

  async run(input) {
    const {
      title,
      subtitle,
      background_color,
      text_color,
      width,
      height,
      fps,
      duration_seconds,
    } = input;

    const durationInFrames = Math.round(duration_seconds * fps);

    const inputProps = { title, subtitle, background_color, text_color };

    const composition = await selectComposition({
      serveUrl: this.bundleLocation,
      id: "TextReveal",
      inputProps,
    });

    // Override dimensions, fps, and duration from user input
    composition.width = width;
    composition.height = height;
    composition.fps = fps;
    composition.durationInFrames = durationInFrames;

    const outputPath = join(tmpdir(), `${randomUUID()}.mp4`);

    await renderMedia({
      composition,
      serveUrl: this.bundleLocation,
      codec: "h264",
      outputLocation: outputPath,
      inputProps,
    });

    const resolution = height <= 720 ? "720p" : height <= 1080 ? "1080p" : "4k";

    return {
      video: File.fromPath(outputPath),
      output_meta: {
        outputs: [
          videoMeta({ resolution, seconds: duration_seconds, fps }),
        ],
      },
    };
  }
}
