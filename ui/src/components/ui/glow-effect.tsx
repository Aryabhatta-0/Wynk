import type * as React from "react";
import { cn } from "@/lib/utils";

// same presets as motion-primitives' GlowEffect, in px
const BLUR = { softest: 2, soft: 4, medium: 12, strong: 16, stronger: 24, strongest: 24, none: 0 } as const;

export interface GlowEffectProps {
  className?: string;
  style?: React.CSSProperties;
  colors?: string[];
  mode?: "rotate" | "static";
  blur?: keyof typeof BLUR;
  duration?: number;
  scale?: number;
}

/** A conic glow turning behind its parent (CSS only, so it stays smooth under load). */
export function GlowEffect({
  className,
  style,
  colors = ["#FF5733", "#33FF57", "#3357FF", "#F1C40F"],
  mode = "rotate",
  blur = "medium",
  duration = 5,
  scale = 1,
}: GlowEffectProps) {
  return (
    <div
      aria-hidden="true"
      className={cn("glow-effect pointer-events-none absolute inset-0 h-full w-full", mode === "rotate" && "glow-rotate", className)}
      style={
        {
          ...style,
          "--glow-colors": [...colors, colors[0]].join(", "),
          "--glow-duration": `${duration}s`,
          filter: `blur(${BLUR[blur]}px)`,
          transform: `scale(${scale})`,
          willChange: "transform",
        } as React.CSSProperties
      }
    />
  );
}
