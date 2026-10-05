import {
  ChatCenteredText,
  Check,
  DownloadSimple,
  Funnel,
  Highlighter,
  type Icon,
  PenNib,
  SealCheck,
  TreeStructure,
  X,
} from "@phosphor-icons/react";
import { useId, useLayoutEffect, useRef, useState } from "react";
import { STAGE_LABEL, type Stage, type StageStatus } from "../lib/chat";
import { cn } from "@/lib/utils";

const STAGE_ICON: Record<Stage["kind"], Icon> = {
  GATHER: DownloadSimple,
  FILTER: Funnel,
  EXTRACT: Highlighter,
  REASON: TreeStructure,
  VERIFY: SealCheck,
  SYNTHESIZE: PenNib,
};

interface Props {
  stages: Stage[];
  /** per-stage status; omitted means every stage finished (a static workflow) */
  status?: StageStatus[];
  answered?: boolean; // the answer node lights when the run produced its answer
  failed?: boolean;
  /** end-point labels: a chat run reads Question → Answer, a dataset workflow Input → Output */
  from?: string;
  to?: string;
}

interface Line {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/*
  One ant's workflow, drawn like the supplied beam visualizer: white circular nodes joined by
  beams. Question -> stages -> answer. A pulse runs along the beam into the stage that is
  working; finished beams turn solid; a failed stage breaks the chain.
*/
export function WorkflowGraph({ stages, status = stages.map(() => "ok"), answered = true, failed = false, from = "Question", to = "Answer" }: Props) {
  const nodeCount = stages.length + 2;
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const wrap = useRef<HTMLDivElement>(null);
  const dots = useRef<(HTMLSpanElement | null)[]>([]);
  const [lines, setLines] = useState<Line[]>([]);

  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const measure = () => {
      const box = el.getBoundingClientRect();
      const c = dots.current.slice(0, nodeCount).map((d) => {
        const r = d!.getBoundingClientRect();
        return { x: r.left - box.left + r.width / 2, y: r.top - box.top + r.height / 2, r: r.width / 2 };
      });
      setLines(
        c.slice(1).map((b, i) => {
          const a = c[i];
          const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
          const ux = (b.x - a.x) / len;
          const uy = (b.y - a.y) / len;
          return { x1: a.x + ux * (a.r + 4), y1: a.y + uy * (a.r + 4), x2: b.x - ux * (b.r + 4), y2: b.y - uy * (b.r + 4) };
        }),
      );
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [nodeCount, stages]);

  // node i of the chain: 0 = question, 1..n = stages, n + 1 = answer
  const nodeState = (i: number): StageStatus => {
    if (i === 0) return "ok";
    if (i === nodeCount - 1) return answered ? "ok" : failed ? "skipped" : "idle";
    return status[i - 1] ?? "idle";
  };
  // edge i joins node i -> node i + 1
  const edgeState = (i: number): "idle" | "flow" | "done" => {
    const next = nodeState(i + 1);
    if (next === "running") return "flow";
    if (next === "ok" || next === "failed") return "done";
    return "idle";
  };

  return (
    <div ref={wrap} className="relative flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between sm:gap-1">
      <svg className="pointer-events-none absolute inset-0 size-full overflow-visible" aria-hidden="true">
        {lines.map((l, i) => {
          const state = edgeState(i);
          return (
            <g key={i}>
              {/* user-space gradient: a bounding-box gradient never paints a perfectly flat line */}
              <linearGradient id={`${uid}-edge-${i}`} gradientUnits="userSpaceOnUse" {...l}>
                <stop offset="0" stopColor="#ef75d3" stopOpacity="0.15" />
                <stop offset="0.75" stopColor="#c23a9f" />
                <stop offset="1" stopColor="#a3248a" />
              </linearGradient>
              <line {...l} stroke="#e7d9e3" strokeWidth={2} strokeLinecap="round" />
              {state === "done" && <line {...l} stroke="#c23a9f" strokeOpacity={0.7} strokeWidth={2} strokeLinecap="round" />}
              {state === "flow" && (
                <line {...l} className="beam-flow" stroke={`url(#${uid}-edge-${i})`} strokeWidth={3} strokeLinecap="round" pathLength={1} />
              )}
            </g>
          );
        })}
      </svg>

      <Node refEl={(d) => (dots.current[0] = d)} icon={ChatCenteredText} state="ok" label={from} />
      {stages.map((s, i) => (
        <Node
          key={`${i}-${s.kind}`}
          refEl={(d) => (dots.current[i + 1] = d)}
          icon={STAGE_ICON[s.kind]}
          state={status[i] ?? "idle"}
          label={STAGE_LABEL[s.kind]}
          detail={s.options.map((o) => o.replaceAll("_", " ")).join(", ")}
        />
      ))}
      <Node
        refEl={(d) => (dots.current[nodeCount - 1] = d)}
        icon={Check}
        state={nodeState(nodeCount - 1)}
        label={to}
        accent
      />
    </div>
  );
}

function Node({
  refEl,
  icon: IconCmp,
  state,
  label,
  detail,
  accent,
}: {
  refEl: (d: HTMLSpanElement | null) => void;
  icon: Icon;
  state: StageStatus;
  label: string;
  detail?: string;
  accent?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-3 transition-opacity duration-200 sm:w-0 sm:flex-1 sm:flex-col sm:gap-2 sm:text-center",
        state === "skipped" && "opacity-40",
      )}
    >
      <span
        ref={refEl}
        className={cn(
          "relative z-10 grid size-12 shrink-0 place-items-center rounded-full border-2 shadow-[0_0_20px_-12px_rgba(0,0,0,0.8)] transition-[border-color,box-shadow,background-color,color] duration-200 ease-out",
          // one background class only: with both, the stylesheet order (not this list) picks the winner
          !(state === "ok" && accent) && "bg-white",
          state === "idle" && "border-line text-ink-soft",
          state === "running" && "border-magenta text-magenta-ink shadow-[0_0_0_5px_rgb(239_117_211/0.18)]",
          state === "ok" && !accent && "border-magenta-ink text-ink",
          state === "ok" && accent && "border-magenta-ink bg-magenta-ink text-white",
          state === "failed" && "border-dashed border-ink-soft text-ink",
          state === "skipped" && "border-line text-ink-soft",
        )}
      >
        {state === "failed" ? <X size={20} weight="bold" /> : <IconCmp size={20} weight={state === "ok" && accent ? "bold" : "duotone"} />}
      </span>
      <span className="min-w-0 sm:px-0.5">
        <span className="block text-sm font-semibold">{label}</span>
        <span className="block text-xs leading-snug text-ink-soft">
          {state === "failed" ? "failed" : state === "skipped" ? "not reached" : state === "running" ? "working" : (detail ?? "")}
        </span>
      </span>
    </div>
  );
}
