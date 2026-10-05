import { RocketLaunch, TreeStructure } from "@phosphor-icons/react";
import { Link } from "react-router";
import { api } from "@/api";
import { EmptyState, ErrorState, LoadingState, Panel, StateBadge, StateLegend } from "@/components/app/ui";
import { WorkflowChain } from "@/components/app/WorkflowChain";
import { OBJECTIVE_LABEL, costPer1k, pct, secs } from "@/lib/format";
import { useResource } from "@/lib/useResource";
import { useProject } from "./ProjectLayout";

export function WorkflowsPage() {
  const { project } = useProject();
  const workflows = useResource(() => api().listWorkflows(project.id), [project.id]);

  return (
    <Panel title="Workflows" meta="Validated workflows and champions from this project's experiments" actions={<StateLegend />} bodyClassName="p-0">
      {workflows.state === "loading" && (
        <div className="p-4">
          <LoadingState label="Loading workflows" />
        </div>
      )}
      {workflows.state === "error" && (
        <div className="p-4">
          <ErrorState title="Could not load workflows" message={workflows.error} onRetry={workflows.reload} />
        </div>
      )}
      {workflows.state === "ready" &&
        (workflows.data.length === 0 ? (
          <EmptyState icon={TreeStructure} title="No validated workflows yet">
            Workflows appear here once an experiment re-measures its finalists on the validation split. The best one that meets every constraint
            becomes champion.
          </EmptyState>
        ) : (
          <table className="data-table" aria-label="Workflows">
            <thead>
              <tr>
                <th>State</th>
                <th>Workflow</th>
                <th>Experiment</th>
                <th className="num">Val. quality</th>
                <th className="num">Cost / 1k</th>
                <th className="num">Mean latency</th>
                <th>Deployment</th>
              </tr>
            </thead>
            <tbody>
              {workflows.data.map(({ candidate: c, experimentId, experimentName, datasetName, objective }) => (
                <tr key={c.id}>
                  <td>
                    <StateBadge state={c.state} />
                  </td>
                  <td>
                    <WorkflowChain stages={c.stages} model={c.model} />
                  </td>
                  <td>
                    <Link to={`../experiments/${experimentId}/results`} className="font-medium hover:text-magenta-ink">
                      {experimentName}
                    </Link>
                    <div className="text-[11px] text-ink-soft">
                      {datasetName} · {OBJECTIVE_LABEL[objective].toLowerCase()}
                    </div>
                  </td>
                  <td className="num">{c.validation ? pct(c.validation.quality) : "—"}</td>
                  <td className="num">{costPer1k((c.validation ?? c.optimization).costPerExample).replace(" / 1k", "")}</td>
                  <td className="num">{secs((c.validation ?? c.optimization).meanLatencyS)}</td>
                  <td>
                    <span className="inline-flex items-center gap-1 text-xs text-ink-soft" title="Deployment is not available yet">
                      <RocketLaunch size={13} aria-hidden="true" /> Coming soon
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ))}
    </Panel>
  );
}
