import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  spring,
  interpolate,
} from "remotion";
import { z } from "zod";

export const textRevealSchema = z.object({
  title: z.string(),
  subtitle: z.string().default(""),
  background_color: z.string().default("#000000"),
  text_color: z.string().default("#ffffff"),
});

type Props = z.infer<typeof textRevealSchema>;

export const TextReveal: React.FC<Props> = ({
  title,
  subtitle,
  background_color,
  text_color,
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  // Title animation: scale + opacity spring
  const titleProgress = spring({
    frame,
    fps,
    config: { damping: 20, stiffness: 80 },
  });

  const titleScale = interpolate(titleProgress, [0, 1], [0.8, 1]);
  const titleOpacity = titleProgress;

  // Subtitle animation: fade in after title settles
  const subtitleDelay = Math.round(fps * 0.6);
  const subtitleProgress = spring({
    frame: Math.max(0, frame - subtitleDelay),
    fps,
    config: { damping: 20, stiffness: 60 },
  });

  const subtitleOpacity = subtitleProgress;
  const subtitleY = interpolate(subtitleProgress, [0, 1], [20, 0]);

  // Fade out near the end
  const fadeOutStart = durationInFrames - Math.round(fps * 0.5);
  const fadeOut = interpolate(
    frame,
    [fadeOutStart, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <AbsoluteFill
      style={{
        backgroundColor: background_color,
        justifyContent: "center",
        alignItems: "center",
        opacity: fadeOut,
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 24,
          padding: "0 80px",
        }}
      >
        <div
          style={{
            color: text_color,
            fontSize: 80,
            fontWeight: 700,
            fontFamily:
              '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
            textAlign: "center",
            transform: `scale(${titleScale})`,
            opacity: titleOpacity,
            lineHeight: 1.2,
          }}
        >
          {title}
        </div>
        {subtitle && (
          <div
            style={{
              color: text_color,
              fontSize: 36,
              fontWeight: 400,
              fontFamily:
                '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
              textAlign: "center",
              opacity: subtitleOpacity,
              transform: `translateY(${subtitleY}px)`,
            }}
          >
            {subtitle}
          </div>
        )}
      </div>
    </AbsoluteFill>
  );
};
