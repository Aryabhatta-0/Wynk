import { ChartLineUp, Info, Table } from "@phosphor-icons/react";
import { useState } from "react";
import { Link, useParams } from "react-router";
import { type Experiment, type Measurement, type StopReason, api, errorMessage } from "@/api";
import { CandidateTable } from "@/components/app/CandidateTable";
import { LearningCurve } from "@/components/app/LearningCurve";
import { SearchView } from "@/components/app/SearchView";
import { ErrorState, LoadingState, Meter, Metric, Panel, StateBadge, StateLegend, StatusBadge } from "@/components/app/ui";
import { WorkflowGraph } from "@/components/WorkflowGraph";
import {
  EVALUATOR_LABEL,
  OBJECTIVE_LABEL,
  TASK_LABEL,
  change,
  costPer1k,
  duration,
  int,
  metricName as metricOf,
  pct,
  points,
  secs,
  usd,
} from "@/lib/format";
import { balancedFormula, curveMetric } from "@/lib/objective";
import { useResource } from "@/lib/useResource";
import { cn } from "@/lib/utils";

const live = (e: Experiment) => e.status === "queued" || e.status === "running";

const STOP_TEXT: Record<StopReason, string> = {
  generations: "all generations ran",
  candidates: "candidate limit reached",
  spend: "spend limit reached",
  duration: "time limit reached",
};

const PHASES = [
  { id: "searching", label: "Search", hint: "candidates on optimization split" },
  { id: "validating", label: "Validate", hint: "finalists on validation split" },
  { id: "testing", label: "Held-out test", hint: "champion measured once" },
  { id: "done", label: "Done", hint: "" },
] as const;

export function ExperimentPage() {
  const { experimentId = "" } = useParams();
  const exp = useResource(() => api().getExperiment(experimentId), [experimentId], { pollMs: 1000, shouldPoll: live });
  const [cancelError, setCancelError] = useState<string | null>(null);

  if (exp.state === "loading") return <LoadingState label="Loading experiment" rows={6} />;
  if (exp.state === "error") return <ErrorState title="Could not load this experiment" message={exp.error} onRetry={exp.reload} />;

  const e = exp.data;
  const { budget } = e.config;
  const objective = e.config.preferences.objective;
  const metricName = metricOf(e.config.evaluation);
  const best = e.candidates.find((c) => c.id === (e.championId ?? e.bestId)) ?? null;
  const metric = curveMetric(objective, metricName);

  const cancel = async () => {
    setCancelError(null);
    try {
      await api().cancelExperiment(e.id);
      exp.reload();
    } catch (err) {
      setCancelError(errorMessage(err));
    }
  };

  return (
    <div className="space-y-4" data-testid="experiment" data-status={e.status}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <Link to=".." relative="path" className="text-xs text-ink-soft hover:text-ink">
            ← Experiments
          </Link>
          <h2 className="mt-1 flex flex-wrap items-center gap-2 text-xl font-bold">
            {e.name} <StatusBadge status={e.status} />
          </h2>
          <p className="mt-0.5 text-xs text-ink-soft">
            {TASK_LABEL[e.config.taskType]} · objective: {OBJECTIVE_LABEL[objective].toLowerCase()} · evaluator:{" "}
            {EVALUATOR_LABEL[e.config.evaluation.evaluator].toLowerCase()} · models: {e.config.models.join(", ")}
          </p>
          {e.identityHash && (
            <p
              className="mt-0.5 font-mono text-[11px] text-ink-soft"
              title="Experiment identity: changes when the dataset, splits, task contract, models, evaluator, objective or limits change"
            >
              identity {e.identityHash.slice(0, 16)}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {live(e) && (
            <button type="button" className="btn btn-quiet btn-sm" onClick={cancel}>
              Cancel run
            </button>
          )}
          <Link to="results" className={cn("btn btn-sm", e.status === "completed" ? "btn-primary" : "btn-quiet")}>
            Compare results
          </Link>
        </div>
      </div>

      {cancelError && <ErrorState title="Could not cancel" message={cancelError} />}
      {e.status === "failed" && <ErrorState title="The run failed" message={e.failure ?? "No reason was reported."} />}
      {e.status === "cancelled" && (
        <Notice>Cancelled after {int(e.progress.evaluated)} candidates. Results below are partial and there is no champion.</Notice>
      )}
      {e.status === "completed" && !e.championId && (
        <Notice tone="bad">
          Finished without a champion: no validated workflow met every hard constraint on the validation split. Loosen a constraint or allow more
          models and run again.
        </Notice>
      )}

      <Panel bodyClassName="p-0">
        <div className="grid divide-line xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)] xl:divide-x">
          <div className="space-y-3 p-4">
            <ol className="flex flex-wrap gap-1.5" aria-label="Run phases">
              {PHASES.map((p, i) => {
                const at = PHASES.findIndex((x) => x.id === e.progress.phase);
                const state = e.status === "queued" ? "todo" : i < at || e.progress.phase === "done" ? "done" : i === at ? "now" : "todo";
                return (
                  <li
                    key={p.id}
                    aria-current={state === "now" ? "step" : undefined}
                    title={p.hint}
                    className={cn(
                      "rounded-md border px-2 py-1 text-xs",
                      state === "done" && "border-line text-magenta-ink",
                      state === "now" && "border-magenta-ink bg-magenta-wash font-semibold text-magenta-ink",
                      state === "todo" && "border-line text-ink-soft",
                    )}
                  >
                    {p.label}
                  </li>
                );
              })}
            </ol>
            {e.status === "queued" ? (
              <p className="text-sm text-ink-soft">Queued. The run starts when a worker is free.</p>
            ) : (
              <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
                <Progress
                  label="Candidates evaluated"
                  value={e.progress.evaluated}
                  max={budget.maxCandidates}
                  text={`${int(e.progress.evaluated)} / ${int(budget.maxCandidates)}`}
                />
                <Progress
                  label="Generation"
                  value={e.progress.generation}
                  max={budget.maxGenerations}
                  text={`${e.progress.generation} / ${budget.maxGenerations}`}
                />
                <Progress
                  label="Spend"
                  value={e.progress.spendUsd}
                  max={budget.maxSpendUsd}
                  text={`${usd(e.progress.spendUsd)} / ${usd(budget.maxSpendUsd)}`}
                />
                <Progress
                  label="Search time"
                  value={e.progress.elapsedSec}
                  max={budget.maxDurationMin * 60}
                  text={`${duration(e.progress.elapsedSec)} / ${budget.maxDurationMin} min`}
                />
              </div>
            )}
            {e.progress.stopReason && (
              <p className="text-xs text-ink-soft" data-testid="stop-reason">
                Search stopped: {STOP_TEXT[e.progress.stopReason]}.
              </p>
            )}
          </div>
          <div className="grid grid-cols-3 divide-x divide-line border-t border-line xl:border-t-0">
            <Metric
              testId="best-quality"
              label={`Best ${metricName}`}
              value={best ? pct(best.optimization.quality) : "—"}
              {...versus(best?.optimization, e.baseline, "quality", true, points, "no feasible candidate yet")}
            />
            <Metric
              label="Cost / 1k"
              value={best ? costPer1k(best.optimization.costPerExample).replace(" / 1k", "") : "—"}
              {...versus(best?.optimization, e.baseline, "costPerExample", false, change)}
            />
            <Metric
              label="Mean latency"
              value={best ? secs(best.optimization.meanLatencyS) : "—"}
              {...versus(best?.optimization, e.baseline, "meanLatencyS", false, change)}
            />
          </div>
        </div>
        <p className="border-t border-line px-4 py-2 text-[11px] text-ink-soft">
          Best so far is the top feasible candidate by the objective, measured on the optimization split. Validation and held-out test numbers are on
          the results page.
        </p>
      </Panel>

      <CurvePanel e={e} metric={metric} />

      <Panel
        title={e.championId ? "Champion workflow" : "Current best workflow"}
        meta={best ? `genome ${best.genomeHash} · generation ${best.generation}` : undefined}
        actions={best && <StateBadge state={best.state} />}
      >
        {best ? (
          <div className="space-y-3">
            <div className="rounded-xl border border-line px-4 py-5">
              <WorkflowGraph stages={best.stages} from="Input" to="Output" />
            </div>
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-ink-soft">
              <span>
                Model <span className="font-mono text-ink">{best.model}</span>
              </span>
              <StateLegend />
            </div>
          </div>
        ) : (
          <p className="text-sm text-ink-soft">
            {e.status === "queued" ? "Appears once the search starts." : "No candidate has met every hard constraint yet."}
          </p>
        )}
      </Panel>

      <SearchView generations={e.generations} edges={e.edges} maxGenerations={budget.maxGenerations} />

      <Panel
        title="Candidates"
        meta={`${int(e.candidates.length)} evaluated · ranked by ${OBJECTIVE_LABEL[objective].toLowerCase()}, infeasible last`}
        bodyClassName="p-0"
      >
        {e.candidates.length ? (
          <CandidateTable candidates={e.candidates} bestId={e.bestId} metric={metricName} />
        ) : (
          <p className="p-4 text-sm text-ink-soft">No candidates yet.</p>
        )}
      </Panel>
    </div>
  );
}

function CurvePanel({ e, metric }: { e: Experiment; metric: ReturnType<typeof curveMetric> }) {
  const [table, setTable] = useState(false);
  const baseline = e.curveBaseline;
  return (
    <Panel
      title="Learning curve"
      meta={`Best feasible ${metric.label} so far, optimization split`}
      actions={
        <div className="flex rounded-full border border-line p-0.5" role="group" aria-label="Learning curve view">
          <button
            type="button"
            aria-pressed={!table}
            onClick={() => setTable(false)}
            className={cn("rounded-full px-2 py-0.5 text-xs", !table && "bg-wash font-semibold")}
          >
            <ChartLineUp size={13} className="mr-1 inline" aria-hidden="true" />
            Chart
          </button>
          <button
            type="button"
            aria-pressed={table}
            onClick={() => setTable(true)}
            className={cn("rounded-full px-2 py-0.5 text-xs", table && "bg-wash font-semibold")}
          >
            <Table size={13} className="mr-1 inline" aria-hidden="true" />
            Table
          </button>
        </div>
      }
    >
      {table ? (
        <div className="max-h-[260px] overflow-auto">
          <table className="data-table" aria-label="Learning curve data">
            <thead>
              <tr>
                <th className="num">Candidates</th>
                <th className="num">Wynk (ACO)</th>
                <th className="num">Random search</th>
                <th className="num">Fixed baseline</th>
              </tr>
            </thead>
            <tbody>
              {e.curve.map((p) => (
                <tr key={p.evaluated}>
                  <td className="num">{int(p.evaluated)}</td>
                  <td className="num">{p.wynk === null ? "—" : metric.format(p.wynk)}</td>
                  <td className="num">{p.random === null ? "—" : metric.format(p.random)}</td>
                  <td className="num">{baseline === null ? "—" : metric.format(baseline)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <LearningCurve
          points={e.curve}
          baseline={baseline}
          maxEvaluated={e.config.budget.maxCandidates}
          metricLabel={metric.label}
          format={metric.format}
          lowerIsBetter={metric.lowerIsBetter}
          bounded={metric.bounded}
        />
      )}
      <p className="mt-2 text-[11px] text-ink-soft">
        {e.config.preferences.balanced && <>Balanced utility = {balancedFormula(e.config.preferences.balanced)}. </>}
        Random search evaluates the same number of candidates as Wynk, sampled uniformly from the same space. Only candidates that meet every hard
        constraint count.
      </p>
    </Panel>
  );
}

function Progress({ label, value, max, text }: { label: string; value: number; max: number; text: string }) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-xs">
        <span className="text-ink-soft">{label}</span>
        <span className="font-medium tnum">{text}</span>
      </div>
      <Meter value={value} max={max} label={label} />
    </div>
  );
}

function Notice({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "bad" }) {
  return (
    <div
      role="status"
      className={cn(
        "flex items-start gap-2.5 rounded-[10px] border px-4 py-3 text-sm",
        tone === "bad" ? "border-bad/30 bg-bad-wash" : "border-line-strong bg-wash",
      )}
    >
      <Info size={17} weight="fill" className={cn("mt-0.5 shrink-0", tone === "bad" ? "text-bad" : "text-magenta-ink")} aria-hidden="true" />
      <p>{children}</p>
    </div>
  );
}

type Delta = (now: number, before: number) => string;

/** A best-vs-baseline line for a metric tile; silent when either side was not measured. */
function versus(
  best: Measurement | undefined,
  base: Measurement | null,
  key: "quality" | "costPerExample" | "meanLatencyS",
  higherIsBetter: boolean,
  delta: Delta,
  empty?: string,
): { detail?: string; tone: "neutral" | "better" | "worse" } {
  const a = best?.[key] ?? null;
  const b = base?.[key] ?? null;
  if (a === null || b === null) return { detail: best ? undefined : empty, tone: "neutral" };
  return { detail: `${delta(a, b)} vs baseline`, tone: (higherIsBetter ? a >= b : a <= b) ? "better" : "worse" };
}
