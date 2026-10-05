import { Database, Plus } from "@phosphor-icons/react";
import { Link } from "react-router";
import { api } from "@/api";
import { Check, EmptyState, ErrorState, LoadingState, Panel } from "@/components/app/ui";
import { bytes, date, int } from "@/lib/format";
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
      meta="Examples the search scores workflows against"
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
            Add a CSV, JSONL or Parquet file of examples, then map which columns are inputs, which one is the expected output, and which add context.
          </EmptyState>
        ) : (
          <table className="data-table" aria-label="Datasets">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Format</th>
                <th className="num">Rows</th>
                <th className="num">Columns</th>
                <th>Mapping</th>
                <th className="num">Added</th>
                <th>
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {datasets.data.map((d) => (
                <tr key={d.id}>
                  <td>
                    <Link to={d.id} className="font-semibold hover:text-magenta-ink">
                      {d.name}
                    </Link>
                    <div className="font-mono text-[11px] text-ink-soft">
                      {d.fileName} · {bytes(d.sizeBytes)}
                    </div>
                  </td>
                  <td className="font-mono text-[12px] uppercase">{d.format}</td>
                  <td className="num">{d.rowCount === null ? "—" : int(d.rowCount)}</td>
                  <td className="num">{d.columns.length}</td>
                  <td>{d.status === "ready" ? <Check ok>Complete</Check> : <Check ok={false}>Needs mapping</Check>}</td>
                  <td className="num text-ink-soft">{date(d.createdAt)}</td>
                  <td className="num">
                    {d.status === "ready" ? (
                      <Link to={`../experiments/new?dataset=${d.id}`} className="btn btn-quiet btn-sm">
                        Configure experiment
                      </Link>
                    ) : (
                      <Link to={d.id} className="btn btn-quiet btn-sm">
                        Finish mapping
                      </Link>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ))}
    </Panel>
  );
}
