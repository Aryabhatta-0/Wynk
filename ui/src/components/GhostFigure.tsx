import { GHOST_BODY, GHOST_EYES, GHOST_VIEWBOX } from "../ghost/mark";
import { cn } from "@/lib/utils";

/** The logo: the ghost's first pose drawn as a magenta outline on white. */
export function GhostLogo({ className }: { className?: string }) {
  return (
    <svg viewBox={GHOST_VIEWBOX} className={cn("overflow-visible", className)} aria-hidden="true" focusable="false">
      <path d={GHOST_BODY} fill="#ffffff" stroke="var(--color-magenta)" strokeWidth={1.75} strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
      {GHOST_EYES.map((d) => (
        <path key={d.slice(0, 24)} d={d} fill="var(--color-magenta-ink)" />
      ))}
    </svg>
  );
}
