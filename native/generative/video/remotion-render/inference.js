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
import { writeFileSync, mkdirSync, rmSync } from "node:fs";

export const RunInput = z.object({
  code: z
    .string()
    .describe(
      "React component TSX code. Must export default a component. " +
        "Can import from 'remotion' (useCurrentFrame, useVideoConfig, spring, interpolate, AbsoluteFill, Sequence, etc.) " +
        "and 'react'. Receives inputProps as component props."
    ),
  composition_id: z
    .string()
    .default("Main")
    .describe("Composition ID to render"),
  props: z
    .record(z.any())
    .default({})
    .describe("Props passed to the component"),
  width: z.number().default(1920).describe("Video width in pixels"),
  height: z.number().default(1080).describe("Video height in pixels"),
  fps: z.number().default(30).describe("Frames per second"),
  duration_seconds: z
    .number()
    .default(5)
    .describe("Video duration in seconds"),
  codec: z
    .enum(["h264", "h265", "vp8", "vp9", "gif", "prores"])
    .default("h264")
    .describe("Output codec"),
});

export const RunOutput = z.object({
  video: z.any().describe("Rendered video file"),
});

// Normalize user-submitted code: fix smart quotes, unescape double-encoded JSON strings
function normalizeCode(code) {
  // Replace smart/curly quotes with straight quotes (common from copy-paste)
  let result = code
    .replace(/[\u2018\u2019]/g, "'")   // ' ' → '
    .replace(/[\u201C\u201D]/g, '"');   // " " → "

  // If the string has no real newlines, it was likely double-escaped in transit
  // (e.g. JSON-stringified twice). Unescape common sequences.
  if (!result.includes("\n")) {
    result = result
      .replace(/\\n/g, "\n")
      .replace(/\\t/g, "\t")
      .replace(/\\"/g, '"');
  }

  return result;
}

export class App {
  async setup() {
    console.log("Downloading Chrome Headless Shell...");
    await ensureBrowser();
    console.log("Browser ready");
    this.appDir = import.meta.dirname;
  }

  async run(input) {
    const {
      code,
      composition_id,
      props,
      width,
      height,
      fps,
      duration_seconds,
      codec,
    } = input;

    const durationInFrames = Math.round(duration_seconds * fps);

    // Write user code into a subdirectory of the app so webpack
    // resolves remotion/react from the app's node_modules naturally
    const renderDir = join(this.appDir, `.render-${randomUUID()}`);
    mkdirSync(renderDir, { recursive: true });

    try {
    const codeContent = normalizeCode(code);
    writeFileSync(join(renderDir, "UserComp.tsx"), codeContent);

    const entryCode = `
import { registerRoot } from "remotion";
import React from "react";
import { Composition } from "remotion";
import UserComp from "./UserComp";

const Root: React.FC = () => {
  return (
    <Composition
      id="${composition_id}"
      component={UserComp}
      width={${width}}
      height={${height}}
      fps={${fps}}
      durationInFrames={${durationInFrames}}
      defaultProps={${JSON.stringify(props)}}
    />
  );
};

registerRoot(Root);
`;
    writeFileSync(join(renderDir, "index.tsx"), entryCode);

    console.log(`Bundling composition "${composition_id}" (${width}x${height} @ ${fps}fps, ${duration_seconds}s)...`);
    const bundleLocation = await bundle({
      entryPoint: join(renderDir, "index.tsx"),
    });
    console.log("Bundle complete");

    const composition = await selectComposition({
      serveUrl: bundleLocation,
      id: composition_id,
      inputProps: props,
    });

    const ext = codec === "gif" ? "gif" : codec === "prores" ? "mov" : "mp4";
    const outputPath = join(tmpdir(), `${randomUUID()}.${ext}`);

    console.log(`Rendering ${durationInFrames} frames (${codec})...`);
    await renderMedia({
      composition,
      serveUrl: bundleLocation,
      codec,
      outputLocation: outputPath,
      inputProps: props,
    });
    console.log("Render complete");

    const resolution =
      height <= 720 ? "720p" : height <= 1080 ? "1080p" : "4k";

    return {
      video: File.fromPath(outputPath),
      output_meta: {
        outputs: [videoMeta({ resolution, seconds: duration_seconds, fps })],
      },
    };
    } finally {
      // Clean up temp render directory
      rmSync(renderDir, { recursive: true, force: true });
    }
  }
}
