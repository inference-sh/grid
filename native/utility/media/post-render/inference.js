import { z } from "zod";
import { File, imageMeta } from "@inferencesh/app";
import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import satori from "satori";
import { Resvg } from "@resvg/resvg-js";

const AuthorSchema = z.object({
  name: z.string().describe("Display name"),
  username: z.string().describe("Handle without @"),
  profile_image_url: z.string().describe("Avatar URL"),
  verified: z.boolean().default(false),
});

export const RunInput = z.object({
  text: z.string().describe("Post text content"),
  author: AuthorSchema.describe("Author info from x/user-get"),
  created_at: z.string().optional().describe("Post timestamp ISO string"),
  like_count: z.number().default(0),
  retweet_count: z.number().default(0),
  reply_count: z.number().default(0),
  view_count: z.number().default(0),
  bookmark_count: z.number().default(0),
  media_urls: z
    .array(z.string())
    .default([])
    .describe("Image URLs embedded in the post"),
  theme: z.enum(["dark", "light"]).default("dark"),
  width: z.number().default(600).describe("Card width in pixels"),
  padding: z.number().default(32).describe("Outer padding around the card"),
  background_color: z.string().optional().describe("Outer background color (defaults to match theme)"),
});

export const RunOutput = z.object({
  image: z.any().describe("Rendered post card PNG"),
});

function formatCount(n) {
  if (n >= 1_000_000)
    return (n / 1_000_000).toFixed(1).replace(/\.0$/, "") + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1).replace(/\.0$/, "") + "K";
  return String(n);
}

function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  const h = d.getHours();
  const m = d.getMinutes().toString().padStart(2, "0");
  const ampm = h >= 12 ? "PM" : "AM";
  const h12 = h % 12 || 12;
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  return `${h12}:${m} ${ampm} · ${months[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
}

// Metric item with label
function metricItem(count, label, color, countColor) {
  if (count === 0) return null;
  return {
    type: "div",
    props: {
      style: { display: "flex", alignItems: "center", gap: 4, marginRight: 16 },
      children: [
        {
          type: "span",
          props: {
            style: { color: countColor || "#e7e9ea", fontWeight: 700, fontSize: 14 },
            children: formatCount(count),
          },
        },
        {
          type: "span",
          props: {
            style: { color, fontSize: 14 },
            children: label,
          },
        },
      ],
    },
  };
}

function postCard(input) {
  const dark = input.theme === "dark";
  const bg = dark ? "#000000" : "#ffffff";
  const fg = dark ? "#e7e9ea" : "#0f1419";
  const secondary = dark ? "#71767b" : "#536471";
  const border = dark ? "#2f3336" : "#eff3f4";
  const metricLabel = dark ? "#71767b" : "#536471";

  const children = [];

  // Header: avatar + name/handle + X logo
  children.push({
    type: "div",
    props: {
      style: { display: "flex", alignItems: "center", marginBottom: 12 },
      children: [
        {
          type: "img",
          props: {
            src: input.author.profile_image_url.replace("_normal", "_bigger"),
            width: 42,
            height: 42,
            style: { borderRadius: 21, marginRight: 10 },
          },
        },
        {
          type: "div",
          props: {
            style: { display: "flex", flexDirection: "column", flex: 1 },
            children: [
              {
                type: "span",
                props: {
                  style: { color: fg, fontWeight: 700, fontSize: 15 },
                  children: input.author.name,
                },
              },
              {
                type: "span",
                props: {
                  style: { color: secondary, fontSize: 14 },
                  children: `@${input.author.username}`,
                },
              },
            ],
          },
        },
        // X logo
        {
          type: "svg",
          props: {
            width: 22,
            height: 22,
            viewBox: "0 0 1200 1227",
            children: {
              type: "path",
              props: {
                d: "M714.163 519.284L1160.89 0H1055.03L667.137 450.887L357.328 0H0L468.492 681.821L0 1226.37H105.866L515.491 750.218L842.672 1226.37H1200L714.137 519.284H714.163ZM569.165 687.828L521.697 619.934L144.011 79.6944H306.615L611.412 515.685L658.88 583.579L1055.08 1150.3H892.476L569.165 687.854V687.828Z",
                fill: fg,
              },
            },
          },
        },
      ],
    },
  });

  // Post text
  children.push({
    type: "div",
    props: {
      style: {
        color: fg,
        fontSize: 17,
        lineHeight: 1.4,
        marginBottom: 12,
        whiteSpace: "pre-wrap",
        letterSpacing: "-0.01em",
      },
      children: input.text,
    },
  });

  // Media images
  if (input.media_urls.length > 0) {
    children.push({
      type: "div",
      props: {
        style: {
          display: "flex",
          gap: 2,
          borderRadius: 16,
          overflow: "hidden",
          marginBottom: 12,
          border: `1px solid ${border}`,
        },
        children: input.media_urls.slice(0, 4).map((url) => ({
          type: "img",
          props: {
            src: url,
            style: { flex: 1, objectFit: "cover", maxHeight: 300 },
          },
        })),
      },
    });
  }

  // Timestamp
  children.push({
    type: "div",
    props: {
      style: {
        color: secondary,
        fontSize: 15,
        paddingBottom: 12,
        borderBottom: `1px solid ${border}`,
      },
      children: formatDate(input.created_at),
    },
  });

  // Metrics row
  const metrics = [
    metricItem(input.reply_count, "Replies", metricLabel, fg),
    metricItem(input.retweet_count, "Reposts", metricLabel, fg),
    metricItem(input.like_count, "Likes", metricLabel, fg),
    metricItem(input.bookmark_count, "Bookmarks", metricLabel, fg),
    metricItem(input.view_count, "Views", metricLabel, fg),
  ].filter(Boolean);

  if (metrics.length > 0) {
    children.push({
      type: "div",
      props: {
        style: {
          display: "flex",
          paddingTop: 12,
          paddingBottom: 12,
          borderBottom: `1px solid ${border}`,
        },
        children: metrics,
      },
    });
  }

  const outerBg = input.background_color || (dark ? "#1a1a1a" : "#f0f0f0");
  const pad = input.padding;

  const card = {
    type: "div",
    props: {
      style: {
        display: "flex",
        flexDirection: "column",
        backgroundColor: bg,
        borderRadius: 16,
        border: `1px solid ${border}`,
        padding: "16px 16px",
        width: "100%",
        fontFamily: "Inter, sans-serif",
      },
      children,
    },
  };

  if (pad > 0) {
    return {
      type: "div",
      props: {
        style: {
          display: "flex",
          backgroundColor: outerBg,
          padding: pad,
          width: "100%",
        },
        children: [card],
      },
    };
  }

  return card;
}

export class App {
  fontRegular = null;
  fontBold = null;

  async setup() {
    console.log("Loading fonts...");
    const [regular, bold] = await Promise.all([
      fetch(
        "https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-400-normal.ttf"
      ).then((r) => r.arrayBuffer()),
      fetch(
        "https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-700-normal.ttf"
      ).then((r) => r.arrayBuffer()),
    ]);
    this.fontRegular = regular;
    this.fontBold = bold;
    console.log("Fonts loaded");
  }

  async run(inputData) {
    console.log(`Rendering post by @${inputData.author.username}`);

    const markup = postCard(inputData);

    const svg = await satori(markup, {
      width: inputData.width,
      fonts: [
        { name: "Inter", data: this.fontRegular, weight: 400, style: "normal" },
        { name: "Inter", data: this.fontBold, weight: 700, style: "normal" },
      ],
    });

    const resvg = new Resvg(svg, {
      fitTo: { mode: "width", value: inputData.width * 2 },
    });
    const pngData = resvg.render();
    const pngBuffer = pngData.asPng();

    const outputPath = join(tmpdir(), `post-${Date.now()}.png`);
    writeFileSync(outputPath, pngBuffer);

    console.log(`Rendered ${pngBuffer.length} bytes`);

    return {
      image: File.fromPath(outputPath),
      output_meta: {
        outputs: [imageMeta({ width: inputData.width, height: 0, count: 1 })],
      },
    };
  }
}
