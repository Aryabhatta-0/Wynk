import { CheckCircle, Equals, Info, Lock, MinusCircle, RocketLaunch, XCircle } from "@phosphor-icons/react";
import { Fragment } from "react";
import { Link, useParams } from "react-router";
import { type ConstraintKey, type Experiment, type MethodResult, type ResultsComparison, type SplitId, api } from "@/api";
import { Check, ErrorState, LoadingState, Panel, StateBadge, StatusBadge } from "@/components/app/ui";
import { WorkflowChain } from "@/components/app/WorkflowChain";
import { WorkflowGraph } from "@/components/WorkflowGraph";
import { CONSTRAINT_LABEL, METHOD_LABEL, SPLIT_LABEL, TASK_METRIC, constraintValue, int, ms, pct, usd } from "@/lib/format";
import { useResource } from "@/lib/useResource";
import { cn } from "@/lib/utils";
import { type Fact, whyFacts } from "@/lib/why";

const SPLITS: { id: SplitId; role: string }[] = [
  { id: "optimization", role: "The search scores every candidate here. Numbers can be optimistic: this split shaped the choice." },
  { id: "validation", role: "Finalists are re-measured here and the champion is picked. Not used by the search itself." },
  { id: "test", role: "Measured once, after the champion is fixed. Used only for reporting, never for choosing." },
];

const SHORT_CONSTRAINT: Record<ConstraintKey, string> = {
  minQuality: "quality",
  maxCostPer1k: "cost",
  maxLatencyP95Ms: "p95",
  allowedModels: "model",
};

const live = (d: [Experiment, ResultsComparison]) => d[0].status === "queued" || d[0].status === "running";

export function ResultsPage() {
  const { experimentId = "" } = useParams();
  const data = useResource(() => Promise.all([api().getExperiment(experimentId), api().getResults(experimentId)]), [experimentId], {
    pollMs: 1500,
    shouldPoll: live,
  });

  if (data.state === "loading") return <LoadingState label="Loading results" rows={5} />;
  if (data.state === "error") return <ErrorState title="Could not load results" message={data.error} onRetry={data.reload} />;

  const [e, r] = data.data;
  const metric = TASK_METRIC[e.config.taskType];
  const champion = e.candidates.find((c) => c.id === e.championId) ?? null;
  const validatedCount = e.candidates.filter((c) => c.state !== "candidate").length;
  const facts = champion ? whyFacts({ results: r, taskType: e.config.taskType, objective: e.config.preferences.objective, validatedCount }) : [];

  return (
    <div className="space-y-4" data-testid="results">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <Link to=".." relative="path" className="text-xs text-ink-soft hover:text-ink">
            ← Experiment
          </Link>
          <h2 className="mt-1 flex flex-wrap items-center gap-2 text-xl font-bold">
            Results: {e.name} <StatusBadge status={e.status} />
          </h2>
          <p className="mt-0.5 text-xs text-ink-soft">Fixed baseline, random search and Wynk compared on the same splits of the same dataset.</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <button type="button" className="btn btn-quiet btn-sm" disabled aria-describedby="deploy-note">
            <RocketLaunch size={14} aria-hidden="true" /> Deploy champion
          </button>
          <span id="deploy-note" className="rounded-full bg-wash px-2 py-0.5 text-[11px] text-ink-soft">
            Deployment is coming soon
          </span>
        </div>
      </div>

      {e.status !== "completed" && (
        <div role="status" className="flex items-start gap-2.5 rounded-[10px] border border-line-strong bg-wash px-4 py-3 text-sm">
          <Info size={17} weight="fill" className="mt-0.5 shrink-0 text-magenta-ink" aria-hidden="true" />
          <p>
            {e.status === "running" || e.status === "queued"
              ? "The run is still going. Validation appears when finalists are re-measured; held-out test appears once, after the champion is fixed."
              : "This run did not complete, so there is no champion and no held-out test result."}
          </p>
        </div>
      )}

      <div className="grid gap-3 md:grid-cols-3">
        {SPLITS.map((s) => {
          const locked = s.id === "test" && r.testLocked;
          return (
            <section key={s.id} className={cn("panel px-4 py-3", s.id === "test" && "border-magenta-ink/40")}>
              <div className="flex items-center justify-between gap-2">
                <h3 className="font-sans text-sm font-semibold tracking-normal">{SPLIT_LABEL[s.id]}</h3>
                <span className="text-xs text-ink-soft tnum">{int(r.splitSizes[s.id])} rows</span>
              </div>
              <p className="mt-1 text-xs leading-snug text-ink-soft">{s.role}</p>
              {locked && (
                <p className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-ink-soft">
                  <Lock size={12} weight="bold" aria-hidden="true" /> Locked until a champion is fixed
                </p>
              )}
            </section>
          );
        })}
      </div>

      <Panel title="Comparison" meta={`${metric}, cost per 1,000 examples and p95 latency, by split`} bodyClassName="p-0">
        <ComparisonTable results={r} metric={metric} />
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <Panel title="Champion" actions={champion && <StateBadge state="champion" />}>
          {champion ? (
            <div className="space-y-3">
              <div className="rounded-xl border border-line px-4 py-5">
                <WorkflowGraph stages={champion.stages} from="Input" to="Output" />
              </div>
              <dl className="grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
                <div>
                  <dt className="inline text-ink-soft">Model </dt>
                  <dd className="inline font-mono">{champion.model}</dd>
                </div>
                <div>
                  <dt className="inline text-ink-soft">Genome </dt>
                  <dd className="inline font-mono">{champion.genomeHash}</dd>
                </div>
                <div>
                  <dt className="inline text-ink-soft">Path </dt>
                  <dd className="inline">candidate in generation {champion.generation} → validated → champion</dd>
                </div>
              </dl>
            </div>
          ) : (
            <p className="text-sm text-ink-soft">
              {e.status === "completed"
                ? "No champion: no validated workflow met every hard constraint on the validation split."
                : "A champion is chosen after validation, when the run completes."}
            </p>
          )}
        </Panel>

        <Panel title="Why this workflow?" meta={r.testLocked ? "From validation measurements" : "From measurements only"}>
          {facts.length ? <Facts facts={facts} /> : <p className="text-sm text-ink-soft">Measured facts appear once a champion is chosen.</p>}
          <p className="mt-3 border-t border-line pt-2 text-[11px] leading-snug text-ink-soft">
            Every line restates a measurement or a constraint check. Wynk does not claim why a stage helps; the search only shows that this
            combination measured best.
          </p>
        </Panel>
      </div>
    </div>
  );
}

function ComparisonTable({ results, metric }: { results: ResultsComparison; metric: string }) {
  const order: MethodResult["method"][] = ["fixed_baseline", "random_search", "wynk_aco"];
  const methods = order.map((id) => results.methods.find((m) => m.method === id)).filter((m): m is MethodResult => !!m);
  const best = (split: SplitId) => Math.max(...methods.map((m) => m.splits[split]?.quality ?? -1));

  return (
    <div className="overflow-x-auto">
      <table className="data-table min-w-[900px] [&_td]:px-2 [&_th]:px-2" aria-label="Method comparison">
        <thead>
          <tr>
            <th rowSpan={2}>Method</th>
            {SPLITS.map((s) => (
              <th key={s.id} colSpan={3} className={cn("border-l border-line text-center", s.id === "test" && "bg-magenta-wash/50 text-magenta-ink")}>
                {SPLIT_LABEL[s.id]}
              </th>
            ))}
            <th rowSpan={2}>Constraints</th>
          </tr>
          <tr>
            {SPLITS.map((s) => (
              <Fragment key={s.id}>
                <th className={cn("num border-l border-line", s.id === "test" && "bg-magenta-wash/50")}>{metric}</th>
                <th className={cn("num", s.id === "test" && "bg-magenta-wash/50")}>Cost / 1k</th>
                <th className={cn("num", s.id === "test" && "bg-magenta-wash/50")}>p95</th>
              </Fragment>
            ))}
          </tr>
        </thead>
        <tbody>
          {methods.map((m) => (
            <tr key={m.method} className={cn(m.method === "wynk_aco" && "bg-magenta-wash/30")} data-testid={`method-${m.method}`}>
              <td className="w-[230px] min-w-[210px] !pl-3">
                <div className="font-semibold">{METHOD_LABEL[m.method]}</div>
                <div className="text-[11px] text-ink-soft">
                  {m.method === "fixed_baseline" ? "one hand-written workflow" : `best of ${int(m.evaluated)} candidates`}
                </div>
                <WorkflowChain stages={m.stages} model={m.model} className="mt-1" />
              </td>
              {SPLITS.map((s) => {
                const v = m.splits[s.id];
                if (!v)
                  return (
                    <td key={s.id} colSpan={3} className="border-l border-line text-center text-xs text-ink-soft">
                      {s.id === "test" && results.testLocked ? (
                        <span className="inline-flex items-center gap-1">
                          <Lock size={12} aria-hidden="true" /> locked
                        </span>
                      ) : (
                        "not measured yet"
                      )}
                    </td>
                  );
                const top = v.quality === best(s.id);
                return (
                  <Fragment key={s.id}>
                    <td className={cn("num border-l border-line", top && "font-semibold")}>{pct(v.quality)}</td>
                    <td className="num">{usd(v.costPer1k)}</td>
                    <td className="num">{ms(v.latencyP95Ms)}</td>
                  </Fragment>
                );
              })}
              <td>
                {m.constraints.every((c) => c.ok) ? (
                  <Check ok>All {m.constraints.length} met</Check>
                ) : (
                  <ul className="space-y-0.5">
                    {m.constraints
                      .filter((c) => !c.ok)
                      .map((c) => (
                        <li
                          key={c.key}
                          title={`${CONSTRAINT_LABEL[c.key]}: ${constraintValue(c.key, c.observed)}, limit ${constraintValue(c.key, c.limit)}`}
                        >
                          <Check ok={false}>
                            {SHORT_CONSTRAINT[c.key]} {constraintValue(c.key, c.observed)}
                          </Check>
                        </li>
                      ))}
                  </ul>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="border-t border-line px-3 py-2 text-[11px] text-ink-soft">
        Bold marks the highest {metric} in each split. Constraints are checked on held-out test when it is unlocked, otherwise on validation (or
        optimization while the search runs).
      </p>
    </div>
  );
}

function Facts({ facts }: { facts: Fact[] }) {
  const icon = {
    better: <CheckCircle size={15} weight="fill" className="text-good" aria-label="better" />,
    worse: <XCircle size={15} weight="fill" className="text-bad" aria-label="worse" />,
    same: <Equals size={15} weight="bold" className="text-ink-soft" aria-label="no difference" />,
    neutral: <MinusCircle size={15} className="text-ink-soft" aria-hidden="true" />,
  };
  return (
    <ul className="space-y-2 text-[13px] leading-snug" data-testid="why-facts">
      {facts.map((f) => (
        <li key={f.id} className="flex items-start gap-2">
          <span className="mt-0.5 shrink-0">{icon[f.tone]}</span>
          <span className="tnum">{f.text}</span>
        </li>
      ))}
    </ul>
  );
}
