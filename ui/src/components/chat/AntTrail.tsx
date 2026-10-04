import type { Stage, StageStatus } from "@/lib/chat";

// every option the construction graph offers at each stage (core/stages.py)
const OPTIONS: Record<Stage["kind"], string[]> = {
  GATHER: ["fetch", "api", "jev"],
  FILTER: ["keyword chunk", "section select"],
  EXTRACT: ["direct", "schema guided", "cot"],
  REASON: ["single", "decompose"],
  VERIFY: ["schema check", "evidence span", "self consistency"],
  SYNTHESIZE: ["direct", "cite evidence"],
};

const W = 300;
const H = 124;
const START = { x: 8, y: 58 };
const END = { x: W - 8, y: 58 };

interface Pt {
  x: number;
  y: number;
}

/** Small deterministic noise in [0, 1): stands in for the pheromone on edges this ant did not take. */
function noise(...k: number[]) {
  const s = Math.sin(k.reduce((a, b, i) => a + b * (i + 1) * 12.9898, 0)) * 43758.5453;
  return s - Math.floor(s);
}

interface Props {
  antId: number;
  stages: Stage[];
  status: StageStatus[];
  trail: number;
  chosen: boolean;
}

/*
  Ant colony optimisation, inside one card: the construction graph (every option at every
  stage), faint edges weighted by pheromone, the ant walking its path stage by stage. While a
  stage runs, the ant sniffs the candidate edges before committing to one. Walked edges turn
  solid; the chosen ant's path thickens as it is reinforced.
*/
export function AntTrail({ antId, stages, status, trail, chosen }: Props) {
  const n = stages.length;
  const colX = (i: number) => 48 + (i * (W - 96)) / Math.max(1, n - 1);
  const layers: Pt[][] = stages.map((s, i) => {
    const opts = OPTIONS[s.kind];
    const ys = opts.length === 3 ? [18, 58, 98] : [36, 80];
    return opts.map((_, j) => ({ x: colX(i), y: ys[j] }));
  });
  const pick = stages.map((s) => Math.max(0, OPTIONS[s.kind].indexOf(s.options[0])));
  const path: Pt[] = [START, ...layers.map((l, i) => l[pick[i]]), END];

  const failedAt = status.indexOf("failed");
  const runningAt = status.indexOf("running");
  const doneCount = status.filter((s) => s === "ok").length;
  // which path node the ant stands on
  const at = failedAt >= 0 ? failedAt + 1 : runningAt >= 0 ? runningAt + 1 : doneCount === n ? n + 1 : doneCount;
  const antPt = path[Math.min(at, path.length - 1)];
  const prevPt = path[Math.max(0, at - 1)];

  const all: { a: Pt; b: Pt; s: number; key: string }[] = [];
  const from = [[START], ...layers];
  const to = [...layers, [END]];
  from.forEach((la, i) =>
    la.forEach((a, j) =>
      to[i].forEach((b, k) => all.push({ a, b, s: noise(antId, i, j, k), key: `${i}-${j}-${k}` })),
    ),
  );

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block w-full overflow-visible" role="img" aria-label={`Ant ${antId} choosing a workflow`}>
      {/* the colony's memory: every edge, as strong as its pheromone */}
      {all.map(({ a, b, s, key }) => (
        <line key={key} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#1c0f1a" strokeOpacity={0.05 + s * 0.12} strokeWidth={0.6 + s * 0.9} />
      ))}

      {/* the ant's path: faint ahead of it, solid where it has walked */}
      {path.slice(1).map((b, i) => {
        const a = path[i];
        const walked = i < at && !(failedAt >= 0 && i >= failedAt + 1);
        const dead = failedAt >= 0 && i > failedAt;
        return (
          <line
            key={`p${i}`}
            className="trail-edge"
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            stroke={walked ? "#a3248a" : "#ffffff"}
            strokeOpacity={dead ? 0.15 : walked ? 1 : 0.55 + trail * 0.4}
            strokeWidth={walked ? (chosen ? 3.6 : 2.2) : 1.4}
            strokeLinecap="round"
            strokeDasharray={walked || dead ? undefined : "3 4"}
          />
        );
      })}

      {/* sniffing: while a stage runs, the ant tastes every option before choosing */}
      {runningAt >= 0 &&
        layers[runningAt].map((b, j) => (
          <line
            key={`s${runningAt}-${j}`}
            className="trail-sniff"
            x1={prevPt.x}
            y1={prevPt.y}
            x2={b.x}
            y2={b.y}
            stroke="#ffffff"
            strokeWidth={j === pick[runningAt] ? 2.4 : 1.4}
            strokeLinecap="round"
            pathLength={1}
            style={{ animationDelay: `${j * 0.12}s` }}
          />
        ))}

      {/* option nodes */}
      {layers.map((l, i) =>
        l.map((p, j) => {
          const onPath = j === pick[i];
          const reached = onPath && (status[i] === "ok" || status[i] === "running");
          return (
            <circle
              key={`n${i}-${j}`}
              cx={p.x}
              cy={p.y}
              r={onPath ? 4 : 3}
              fill={reached ? "#a3248a" : "#ffffff"}
              stroke={onPath ? "#a3248a" : "#e9d3e2"}
              strokeWidth={1.2}
            />
          );
        }),
      )}
      <circle cx={START.x} cy={START.y} r={3.5} fill="#1c0f1a" />
      <circle cx={END.x} cy={END.y} r={4.5} fill={doneCount === n ? "#a3248a" : "#ffffff"} stroke="#a3248a" strokeWidth={1.4} />

      {/* the ant */}
      <g className="trail-ant" style={{ transform: `translate(${antPt.x}px, ${antPt.y}px)` }}>
        {runningAt >= 0 && <circle r={9} fill="#ef75d3" fillOpacity={0.25} className="trail-pulse" />}
        {failedAt >= 0 ? (
          <g stroke="#1c0f1a" strokeWidth={2.2} strokeLinecap="round">
            <line x1={-4.5} y1={-4.5} x2={4.5} y2={4.5} />
            <line x1={4.5} y1={-4.5} x2={-4.5} y2={4.5} />
          </g>
        ) : (
          <circle r={5.5} fill="#a3248a" stroke="#ffffff" strokeWidth={2} />
        )}
      </g>

      {/* stage names under their columns */}
      {stages.map((s, i) => (
        <text key={`t${i}`} x={colX(i)} y={H - 1} textAnchor="middle" className="fill-ink-soft" style={{ fontSize: 8.5 }}>
          {s.kind.toLowerCase()}
        </text>
      ))}
    </svg>
  );
}
