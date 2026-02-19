import { z } from "zod";
import { File } from "@inferencesh/app";
import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Marked } from "marked";
import { gfmHeadingId } from "marked-gfm-heading-id";
import { markedHighlight } from "marked-highlight";
import hljs from "highlight.js";

export const RunInput = z.object({
  markdown: z.string().describe("Markdown text to convert"),
  full_page: z
    .boolean()
    .default(false)
    .describe("Wrap output in a full HTML page with default styles"),
});

export const RunOutput = z.object({
  html: z.string().describe("Converted HTML string"),
  file: z.any().optional().describe("HTML file"),
});

export class App {
  async setup() {
    this.marked = new Marked(
      gfmHeadingId(),
      markedHighlight({
        emptyLangClass: "hljs",
        langPrefix: "hljs language-",
        highlight(code, lang) {
          if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
          }
          return hljs.highlightAuto(code).value;
        },
      })
    );
  }

  async run(inputData) {
    const html = await this.marked.parse(inputData.markdown);

    let output = html;
    if (inputData.full_page) {
      output = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; color: #24292f; }
pre { background: #f6f8fa; padding: 1rem; border-radius: 6px; overflow-x: auto; }
code { font-size: 0.9em; }
blockquote { border-left: 4px solid #d0d7de; margin: 0; padding: 0 1rem; color: #656d76; }
table { border-collapse: collapse; } th, td { border: 1px solid #d0d7de; padding: 6px 13px; }
img { max-width: 100%; }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.0/styles/github.min.css">
</head>
<body>
${html}
</body>
</html>`;
    }

    const outputPath = join(tmpdir(), "output.html");
    writeFileSync(outputPath, output);

    return {
      html: output,
      file: File.fromPath(outputPath),
    };
  }
}
