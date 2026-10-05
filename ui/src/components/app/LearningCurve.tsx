import { useLayoutEffect, useRef, useState } from "react";
import type { CurvePoint } from "@/api";
import { int } from "@/lib/format";

interface Props {
  points: CurvePoint[];
  baseline: number | null;
  /** x-axis extent: the candidate budget, so progress through it is visible */
  maxEvaluated: number;
  metricLabel: string;
  format: (v: number) => string;
  lowerIsBetter: boolean;
  /** quality is bounded to 0..1 */
  bounded?: boolean;
}

const H = 232;
const M = { top: 14, right: 104, bottom: 30, left: 56 };

/*
  Best feasible value of the objective metric so far, against candidates evaluated. Wynk is
  solid magenta, random search dashed blue (the dash is the secondary encoding for colour-blind
  readers), the fixed baseline a dotted reference line. Hover or arrow keys read every series.
*/
export function LearningCurve({ points, baseline, maxEvaluated, metricLabel, format, lowerIsBetter, bounded }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const plot = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  const [hover, setHover] = useState<number | null>(null);

  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(320, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  if (!points.some((p) => p.wynk !== null || p.random !== null)) {
    return (
      <div ref={wrap}>
        <div className="grid h-[232px] place-items-center rounded-lg border border-dashed border-line-strong px-6 text-center text-sm text-ink-soft">
          No feasible candidate yet. The curve starts at the first one that meets every hard constraint.
        </div>
      </div>
    );
  }

  const values = points.flatMap((p) => [p.wynk, p.random]).filter((v): v is number => v !== null);
  if (baseline !== null) values.push(baseline);
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  const pad = (hi - lo || Math.abs(hi) * 0.1 || 0.05) * 0.15;
  lo -= pad;
  hi += pad;
  if (bounded) {
    lo = Math.max(0, lo);
    hi = Math.min(1, hi);
  }
  const xMax = Math.max(maxEvaluated, points.at(-1)?.evaluated ?? 1);
  const iw = width - M.left - M.right;
  const ih = H - M.top - M.bottom;
  const x = (v: number) => M.left + (v / xMax) * iw;
  const y = (v: number) => M.top + (1 - (v - lo) / (hi - lo || 1)) * ih;

  const path = (key: "wynk" | "random") => {
    let d = "";
    let prev: number | null = null;
    for (const p of points) {
      const v = p[key];
      if (v === null) continue;
      // best-so-far is a step function: hold the previous value until it improves
      d += d ? ` H${x(p.evaluated).toFixed(1)}` : `M${x(p.evaluated).toFixed(1)},${y(v).toFixed(1)}`;
      if (prev !== null && prev !== v) d += ` V${y(v).toFixed(1)}`;
      prev = v;
    }
    return d;
  };

  const yTicks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  const xTicks = niceTicks(xMax);
  const last = points.at(-1)!;
  const ends = endLabels(
    [
      { key: "wynk", label: "Wynk (ACO)", v: lastValue(points, "wynk") },
      { key: "random", label: "Random search", v: lastValue(points, "random") },
    ].filter((e): e is { key: string; label: string; v: number } => e.v !== null),
    y,
  );

  const pick = (clientX: number) => {
    const rect = plot.current!.getBoundingClientRect();
    const ev = ((clientX - rect.left - M.left) / iw) * xMax;
    let best = 0;
    points.forEach((p, i) => {
      if (Math.abs(p.evaluated - ev) < Math.abs(points[best].evaluated - ev)) best = i;
    });
    setHover(best);
  };
  const hp = hover !== null ? points[hover] : null;
  const readout = (p: CurvePoint) =>
    `${int(p.evaluated)} candidates: Wynk ${p.wynk === null ? "none feasible" : format(p.wynk)}, random search ${p.random === null ? "none feasible" : format(p.random)}${baseline === null ? "" : `, fixed baseline ${format(baseline)}`}`;

  return (
    <div ref={wrap}>
      <ul className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink" aria-label="Legend">
        <LegendItem dash={undefined} color="var(--color-series-wynk)" label="Wynk (ACO)" />
        <LegendItem dash="5 4" color="var(--color-series-random)" label="Random search" />
        {baseline !== null && <LegendItem dash="1.5 3" color="var(--color-series-base)" label="Fixed baseline" />}
        <li className="text-ink-soft">{lowerIsBetter ? "Lower is better" : "Higher is better"}</li>
      </ul>
      <div
        ref={plot}
        className="relative touch-none rounded-md"
        // a scrubber over the curve's points: arrow keys move it, the value text reads every series
        role="slider"
        aria-roledescription="chart scrubber"
        aria-label={`Learning curve: best ${metricLabel} so far against candidates evaluated`}
        aria-valuemin={1}
        aria-valuemax={points.length}
        aria-valuenow={(hover ?? points.length - 1) + 1}
        aria-valuetext={readout(hover !== null ? points[hover] : last)}
        tabIndex={0}
        onPointerMove={(e) => pick(e.clientX)}
        onPointerLeave={() => setHover(null)}
        onBlur={() => setHover(null)}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") setHover((h) => Math.min(points.length - 1, (h ?? -1) + 1));
          else if (e.key === "ArrowLeft") setHover((h) => Math.max(0, (h ?? points.length) - 1));
          else if (e.key === "Escape") setHover(null);
          else return;
          e.preventDefault();
        }}
      >
        <svg width={width} height={H} viewBox={`0 0 ${width} ${H}`} className="block max-w-full" aria-hidden="true">
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={M.left} x2={M.left + iw} y1={y(t)} y2={y(t)} stroke="var(--color-line)" />
              <text x={M.left - 8} y={y(t)} dy="0.32em" textAnchor="end" className="fill-ink-soft text-[11px] tnum">
                {format(t)}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={t} x={x(t)} y={H - 10} textAnchor="middle" className="fill-ink-soft text-[11px] tnum">
              {int(t)}
            </text>
          ))}
          <text x={M.left + iw} y={H - 10} dx={8} className="fill-ink-soft text-[11px]">
            candidates
          </text>

          {baseline !== null && (
            <line
              x1={M.left}
              x2={M.left + iw}
              y1={y(baseline)}
              y2={y(baseline)}
              stroke="var(--color-series-base)"
              strokeWidth={1.5}
              strokeDasharray="1.5 3"
            />
          )}
          <path d={path("random")} fill="none" stroke="var(--color-series-random)" strokeWidth={2} strokeDasharray="5 4" strokeLinejoin="round" />
          <path d={path("wynk")} fill="none" stroke="var(--color-series-wynk)" strokeWidth={2} strokeLinejoin="round" />

          {ends.map((e) => (
            <g key={e.key}>
              <circle
                cx={x(last.evaluated)}
                cy={y(e.v)}
                r={3.5}
                fill={e.key === "wynk" ? "var(--color-series-wynk)" : "var(--color-series-random)"}
                stroke="#fff"
                strokeWidth={2}
              />
              <text x={M.left + iw + 10} y={e.y} dy="0.32em" className="fill-ink text-[11px] font-medium">
                {e.label}
              </text>
            </g>
          ))}

          {hp && (
            <g pointerEvents="none">
              <line x1={x(hp.evaluated)} x2={x(hp.evaluated)} y1={M.top} y2={M.top + ih} stroke="var(--color-ink)" strokeOpacity={0.35} />
              {hp.random !== null && (
                <circle cx={x(hp.evaluated)} cy={y(hp.random)} r={4} fill="var(--color-series-random)" stroke="#fff" strokeWidth={2} />
              )}
              {hp.wynk !== null && (
                <circle cx={x(hp.evaluated)} cy={y(hp.wynk)} r={4} fill="var(--color-series-wynk)" stroke="#fff" strokeWidth={2} />
              )}
            </g>
          )}
        </svg>
        {hp && (
          <div
            className="pointer-events-none absolute top-2 z-10 min-w-40 rounded-lg border border-line-strong bg-white px-3 py-2 text-xs shadow-[0_8px_24px_-12px_rgba(28,15,26,0.35)]"
            style={x(hp.evaluated) > width / 2 ? { right: width - x(hp.evaluated) + 10 } : { left: x(hp.evaluated) + 10 }}
            aria-live="polite"
          >
            <div className="mb-1 font-semibold">{int(hp.evaluated)} candidates</div>
            <Row color="var(--color-series-wynk)" label="Wynk (ACO)" value={hp.wynk === null ? "none feasible" : format(hp.wynk)} />
            <Row color="var(--color-series-random)" label="Random search" value={hp.random === null ? "none feasible" : format(hp.random)} />
            {baseline !== null && <Row color="var(--color-series-base)" label="Fixed baseline" value={format(baseline)} />}
          </div>
        )}
      </div>
    </div>
  );
}

function LegendItem({ color, dash, label }: { color: string; dash: string | undefined; label: string }) {
  return (
    <li className="flex items-center gap-1.5">
      <svg width="18" height="6" aria-hidden="true">
        <line x1="1" x2="17" y1="3" y2="3" stroke={color} strokeWidth={2} strokeDasharray={dash} strokeLinecap="round" />
      </svg>
      {label}
    </li>
  );
}

function Row({ color, label, value }: { color: string; label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 py-0.5">
      <span className="flex items-center gap-1.5 text-ink-soft">
        <span className="size-2 rounded-full" style={{ background: color }} aria-hidden="true" />
        {label}
      </span>
      <span className="font-medium text-ink tnum">{value}</span>
    </div>
  );
}

function lastValue(points: CurvePoint[], key: "wynk" | "random"): number | null {
  for (let i = points.length - 1; i >= 0; i--) if (points[i][key] !== null) return points[i][key];
  return null;
}

/** Keep end labels at least 14px apart. */
function endLabels<T extends { v: number }>(items: T[], y: (v: number) => number): (T & { y: number })[] {
  const placed = items.map((it) => ({ ...it, y: y(it.v) })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < placed.length; i++) if (placed[i].y - placed[i - 1].y < 14) placed[i].y = placed[i - 1].y + 14;
  return placed;
}

function niceTicks(max: number): number[] {
  const raw = max / 5;
  const mag = 10 ** Math.floor(Math.log10(raw || 1));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let t = 0; t <= max + 1e-9; t += step) out.push(Math.round(t));
  return out;
}
