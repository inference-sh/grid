import React from "react";
import { Composition } from "remotion";
import { TextReveal, textRevealSchema } from "./TextReveal";

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="TextReveal"
        component={TextReveal}
        schema={textRevealSchema}
        defaultProps={{
          title: "Hello World",
          subtitle: "",
          background_color: "#000000",
          text_color: "#ffffff",
        }}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={150}
        calculateMetadata={({ props, defaultProps }) => {
          const fps = 30;
          const duration = 5;
          return {
            fps,
            durationInFrames: Math.round(duration * fps),
          };
        }}
      />
    </>
  );
};
