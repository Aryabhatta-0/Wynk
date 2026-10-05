import { Database, Plus } from "@phosphor-icons/react";
import { Link } from "react-router";
import { api, latestVersion } from "@/api";
import { EmptyState, ErrorState, LoadingState, Panel } from "@/components/app/ui";
import { date, int } from "@/lib/format";
import { useResource } from "@/lib/useResource";
import { useProject } from "./ProjectLayout";

export function DatasetsPage() {
  const { project } = useProject();
  const datasets = useResource(() => api().listDatasets(project.id), [project.id]);
  const add = (
    <Link to="new" className="btn btn-primary btn-sm">
      <Plus size={14} weight="bold" aria-hidden="true" /> Add dataset
    </Link>
  );

  return (
    <Panel
      title="Datasets"
      meta="Registered datasets: every version is immutable"
      actions={datasets.state === "ready" && datasets.data.length > 0 && add}
      bodyClassName="p-0"
    >
      {datasets.state === "loading" && (
        <div className="p-4">
          <LoadingState label="Loading datasets" />
        </div>
      )}
      {datasets.state === "error" && (
        <div className="p-4">
          <ErrorState title="Could not load datasets" message={datasets.error} onRetry={datasets.reload} />
        </div>
      )}
      {datasets.state === "ready" &&
        (datasets.data.length === 0 ? (
          <EmptyState icon={Database} title="No datasets in this project" action={add}>
            Upload a CSV or JSONL file of examples. The server inspects it, then you map which columns are inputs, which are the expected output and
            which add context, and register it.
          </EmptyState>
        ) : (
          <table className="data-table" aria-label="Datasets">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Format</th>
                <th className="num">Rows</th>
                <th className="num">Columns</th>
                <th className="num">Version</th>
                <th>Row ids</th>
                <th className="num">Registered</th>
              </tr>
            </thead>
            <tbody>
              {datasets.data.map((d) => {
                const v = latestVersion(d);
                return (
                  <tr key={d.id} data-testid="dataset-row">
                    <td>
                      <Link to={d.id} className="font-semibold hover:text-magenta-ink">
                        {d.name}
                      </Link>
                      <div className="font-mono text-[11px] text-ink-soft">{d.id}</div>
                    </td>
                    <td className="font-mono text-[12px] uppercase">{v.format}</td>
                    <td className="num">{int(v.rowCount)}</td>
                    <td className="num">{v.columns.length}</td>
                    <td className="num">
                      v{v.version}
                      {d.versions.length > 1 && <span className="text-ink-soft"> of {d.versions.length}</span>}
                    </td>
                    <td className="text-[12px] text-ink-soft">{v.mapping.id ? <span className="font-mono">{v.mapping.id}</span> : "generated"}</td>
                    <td className="num text-ink-soft">{date(v.createdAt)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ))}
    </Panel>
  );
}
