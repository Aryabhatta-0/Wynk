import { GHOST_BODY, GHOST_EYES, GHOST_VIEWBOX } from "../ghost/mark";
import { cn } from "../lib/cn";

// which eye is which: the eye path starting further right is the right eye
const eyeX = (d: string) => Number(d.slice(1).split(" ")[0]);
const [LEFT_EYE, RIGHT_EYE] = [...GHOST_EYES].sort((a, b) => eyeX(a) - eyeX(b));

/**
 * The ghost as a vector, eyes kept separate so they can wink (`[data-eye]`).
 * `body` and `eyes` take any CSS colour.
 */
export function GhostFigure({ className, body = "#fbf6fa", eyes = "#170d16" }: { className?: string; body?: string; eyes?: string }) {
  return (
    <svg viewBox={GHOST_VIEWBOX} className={cn("overflow-visible", className)} aria-hidden="true" focusable="false">
      <path d={GHOST_BODY} fill={body} />
      <path data-eye="left" d={LEFT_EYE} fill={eyes} style={{ transformBox: "fill-box", transformOrigin: "center" }} />
      <path data-eye="right" d={RIGHT_EYE} fill={eyes} style={{ transformBox: "fill-box", transformOrigin: "center" }} />
    </svg>
  );
}

/** The logo tile: the white ghost on a plum square, as in the favicon. */
export function GhostTile({ className }: { className?: string }) {
  return (
    <span className={cn("grid place-items-center rounded-[28%] bg-plum", className)}>
      <GhostFigure className="h-[64%] w-auto" />
    </span>
  );
}
