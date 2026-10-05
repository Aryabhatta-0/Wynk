import { Flask, Plus } from "@phosphor-icons/react";
import { Link } from "react-router";
import { type Experiment, api } from "@/api";
import { EmptyState, ErrorState, LoadingState, Panel, StatusBadge } from "@/components/app/ui";
import { OBJECTIVE_LABEL, TASK_LABEL, int, metricName, pct, relative } from "@/lib/format";
import { useResource } from "@/lib/useResource";
import { useProject } from "./ProjectLayout";

const anyLive = (es: Experiment[]) => es.some((e) => e.status === "queued" || e.status === "running");

export function ExperimentsPage() {
  const { project } = useProject();
  const experiments = useResource(() => api().listExperiments(project.id), [project.id], { pollMs: 2000, shouldPoll: anyLive });
  const add = (
    <Link to="new" className="btn btn-primary btn-sm">
      <Plus size={14} weight="bold" aria-hidden="true" /> New experiment
    </Link>
  );

  return (
    <Panel
      title="Experiments"
      meta="Optimization runs on this project's datasets"
      actions={experiments.state === "ready" && experiments.data.length > 0 && add}
      bodyClassName="p-0"
    >
      {experiments.state === "loading" && (
        <div className="p-4">
          <LoadingState label="Loading experiments" />
        </div>
      )}
      {experiments.state === "error" && (
        <div className="p-4">
          <ErrorState title="Could not load experiments" message={experiments.error} onRetry={experiments.reload} />
        </div>
      )}
      {experiments.state === "ready" &&
        (experiments.data.length === 0 ? (
          <EmptyState icon={Flask} title="No experiments yet" action={add}>
            Configure an experiment on a mapped dataset: choose the task, hard constraints, objective and budget, and Wynk searches for the best
            workflow.
          </EmptyState>
        ) : (
          <table className="data-table" aria-label="Experiments">
            <thead>
              <tr>
                <th>Experiment</th>
                <th>Status</th>
                <th>Task</th>
                <th>Objective</th>
                <th className="num">Candidates</th>
                <th className="num">Best quality</th>
                <th>Champion</th>
                <th className="num">Started</th>
              </tr>
            </thead>
            <tbody>
              {experiments.data.map((e) => {
                const best = e.candidates.find((c) => c.id === (e.championId ?? e.bestId));
                return (
                  <tr key={e.id}>
                    <td>
                      <Link to={e.id} className="font-semibold hover:text-magenta-ink">
                        {e.name}
                      </Link>
                    </td>
                    <td>
                      <StatusBadge status={e.status} />
                    </td>
                    <td>{TASK_LABEL[e.config.taskType]}</td>
                    <td>{OBJECTIVE_LABEL[e.config.preferences.objective]}</td>
                    <td className="num">
                      {int(e.progress.evaluated)} / {int(e.config.budget.maxCandidates)}
                    </td>
                    <td className="num">{best ? `${pct(best.optimization.quality)} ${metricName(e.config.evaluation)}` : "—"}</td>
                    <td>
                      {e.championId ? (
                        <Link to={`${e.id}/results`} className="btn btn-text text-[13px]">
                          View results
                        </Link>
                      ) : (
                        <span className="text-ink-soft">—</span>
                      )}
                    </td>
                    <td className="num text-ink-soft">{relative(e.createdAt)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ))}
    </Panel>
  );
}
