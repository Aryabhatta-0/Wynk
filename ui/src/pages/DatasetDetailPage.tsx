import { useState } from "react";
import { Link, useParams } from "react-router";
import { type ColumnMapping, api, errorMessage } from "@/api";
import { DatasetPreview, MappingEditor } from "@/components/app/DatasetSchema";
import { Check, ErrorState, LoadingState, Panel } from "@/components/app/ui";
import { bytes, date, int } from "@/lib/format";
import { useResource } from "@/lib/useResource";
import { validateMapping } from "@/lib/validation";

export function DatasetDetailPage() {
  const { datasetId = "" } = useParams();
  const dataset = useResource(() => api().getDataset(datasetId), [datasetId]);
  // unsaved edits; null means "show what is saved"
  const [mapping, setMapping] = useState<ColumnMapping | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  if (dataset.state === "loading") return <LoadingState label="Loading dataset" />;
  if (dataset.state === "error") return <ErrorState title="Could not load this dataset" message={dataset.error} onRetry={dataset.reload} />;

  const d = dataset.data;
  const m = mapping ?? d.mapping;
  const errors = validateMapping(m, d.columns);
  const dirty = JSON.stringify(m) !== JSON.stringify(d.mapping);

  const save = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      const saved = await api().updateMapping(d.id, m);
      setMapping(saved.mapping);
      dataset.reload();
    } catch (err) {
      setSaveError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <Link to=".." relative="path" className="text-xs text-ink-soft hover:text-ink">
            ← Datasets
          </Link>
          <h2 className="mt-1 text-xl font-bold">{d.name}</h2>
          <p className="mt-0.5 font-mono text-xs text-ink-soft">
            {d.fileName} · {d.format.toUpperCase()} · {bytes(d.sizeBytes)} · {int(d.rowCount)} rows · version {d.version} · added {date(d.createdAt)}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {d.status === "ready" ? <Check ok>Mapping complete</Check> : <Check ok={false}>Needs mapping</Check>}
          {d.status === "ready" && !dirty ? (
            <Link to={`../../experiments/new?dataset=${d.id}`} relative="path" className="btn btn-primary btn-sm">
              Configure experiment
            </Link>
          ) : (
            <button type="button" className="btn btn-primary btn-sm" disabled>
              Configure experiment
            </button>
          )}
        </div>
      </div>

      <Panel
        title="Schema and column roles"
        actions={
          <button type="button" className="btn btn-quiet btn-sm" disabled={!dirty || saving || errors.length > 0} onClick={save}>
            {saving ? "Saving…" : "Save mapping"}
          </button>
        }
      >
        <MappingEditor columns={d.columns} preview={d.preview} mapping={m} onChange={setMapping} errors={errors} />
        {saveError && <p className="mt-2 text-xs text-bad">{saveError}</p>}
      </Panel>

      <Panel title="Preview" meta={`First ${d.preview.length} rows`} bodyClassName="p-0">
        <DatasetPreview columns={d.columns} rows={d.preview} mapping={m} />
      </Panel>
    </div>
  );
}
