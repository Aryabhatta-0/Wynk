import { type Ref, useEffect, useImperativeHandle, useRef } from "react";
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

export interface GhostHandle {
  wink: (duration?: number) => void;
}

/** The animated ghost from the supplied design. Colours come from the --ghost-* CSS variables. */
export function Ghost({ active, className, ref }: { active: boolean; className?: string; ref?: Ref<GhostHandle> }) {
  const el = useRef<GhostAnimation>(null);
  useImperativeHandle(ref, () => ({ wink: (d) => el.current?.wink(d) }), []);
  useEffect(() => {
    const g = el.current;
    if (!g) return;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (active && !reduce) g.play();
    else g.pause();
  }, [active]);
  return <ghost-animation ref={el} className={className} aria-hidden="true" />;
}
