import { cn } from "../../lib/cn";

const BLUR = { soft: 8, medium: 16, strong: 32, stronger: 48, strongest: 64 } as const;

interface GlowEffectProps {
  colors: string[];
  mode?: "rotate" | "static";
  blur?: keyof typeof BLUR;
  duration?: number; // seconds per turn
  scale?: number;
  className?: string;
}

/** A conic glow that turns slowly behind its parent (CSS only; still under reduced motion). */
export function GlowEffect({ colors, mode = "rotate", blur = "medium", duration = 5, scale = 1, className }: GlowEffectProps) {
  return (
    <div
      aria-hidden="true"
      className={cn("glow-effect pointer-events-none absolute inset-0", mode === "rotate" && "glow-rotate", className)}
      style={
        {
          "--glow-colors": colors.join(", "),
          "--glow-duration": `${duration}s`,
          filter: `blur(${BLUR[blur] / 4}px)`,
          transform: `scale(${scale})`,
        } as React.CSSProperties
      }
    />
  );
}
