import { useEffect, useRef } from "react";
import "../ghost/ghost-animation.js";
import type { GhostAnimation } from "../ghost/ghost-animation.js";

declare module "react" {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace JSX {
    interface IntrinsicElements {
      "ghost-animation": React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
        speed?: string;
      };
    }
  }
}

/** The ghost from the supplied design, moving while the run moves (it is the agent at work). */
export function Ghost({ active, className }: { active: boolean; className?: string }) {
  const ref = useRef<GhostAnimation>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (active && !reduce) el.play();
    else el.pause();
  }, [active]);
  return <ghost-animation ref={ref} className={className} aria-hidden="true" />;
}
