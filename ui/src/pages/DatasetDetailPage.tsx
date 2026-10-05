import { Info } from "@phosphor-icons/react";
import { useState } from "react";
import { Link, useLocation, useNavigate, useParams, useSearchParams } from "react-router";
import {
  type ColumnMapping,
  type Dataset,
  type DatasetSplitSet,
  type DatasetUpload,
  type DatasetVersion,
  EXPERIMENT_BACKEND_MISSING,
  api,
  latestVersion,
} from "@/api";
import { DatasetPreview, MappingEditor } from "@/components/app/DatasetSchema";
import { DatasetSteps } from "@/components/app/FlowSteps";
import { ApiErrorState, Fact, Field, LoadingState, Panel } from "@/components/app/ui";
import { date, int } from "@/lib/format";
import { type Resource, useResource } from "@/lib/useResource";
import { validateMapping } from "@/lib/validation";

/*
  A registered dataset: its versions, the selected version's identity and roles, its durable splits
  (the last step of the dataset flow) and the upload's preview. Every number is the server's.
*/
export function DatasetDetailPage() {
  const { datasetId = "" } = useParams();
  const [params] = useSearchParams();
  const dataset = useResource(() => api().getDataset(datasetId), [datasetId]);

  if (dataset.state === "loading") return <LoadingState label="Loading dataset" />;
  if (dataset.state === "error") return <ApiErrorState title="Could not load this dataset" error={dataset.cause} onRetry={dataset.reload} />;

  const d = dataset.data;
  const requested = Number(params.get("version"));
  const v = d.versions.find((x) => x.version === requested) ?? latestVersion(d);
  return <VersionView key={`${d.id}@${v.version}`} dataset={d} version={v} reload={dataset.reload} />;
}

function VersionView({ dataset: d, version: v, reload }: { dataset: Dataset; version: DatasetVersion; reload: () => void }) {
  const navigate = useNavigate();
  const registered = (useLocation().state as { registered?: boolean } | null)?.registered;
  const upload = useResource(() => api().getUpload(v.uploadId), [v.uploadId]);
  const splits = useResource(() => api().listSplits(d.id, v.version), [d.id, v.version]);
  const experiments = api().experiments === "simulated";

  return (
    <div className="space-y-4" data-testid="dataset" data-version={v.version}>
      <DatasetSteps current={splits.state === "ready" && splits.data.length > 0 ? "done" : "Create splits"} />
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <Link to=".." relative="path" className="text-xs text-ink-soft hover:text-ink">
            ← Datasets
          </Link>
          <h2 className="mt-1 text-xl font-bold">{v.name}</h2>
          <p className="mt-0.5 font-mono text-xs text-ink-soft">{d.id}</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {d.versions.length > 1 && (
            <label className="flex items-center gap-2 text-xs text-ink-soft">
              Version
              <select
                className="input h-8 w-auto py-0 text-[13px]"
                value={v.version}
                onChange={(e) => navigate(`?version=${e.target.value}`)}
                aria-label="Dataset version"
              >
                {[...d.versions].reverse().map((x) => (
                  <option key={x.version} value={x.version}>
                    v{x.version}
                    {x.version === d.latestVersion ? " (latest)" : ""}
                  </option>
                ))}
              </select>
            </label>
          )}
          {experiments ? (
            <Link to={`../../experiments/new?dataset=${encodeURIComponent(d.id)}`} relative="path" className="btn btn-primary btn-sm">
              Configure experiment
            </Link>
          ) : (
            <button type="button" className="btn btn-quiet btn-sm" disabled title={EXPERIMENT_BACKEND_MISSING}>
              Configure experiment · needs experiment backend
            </button>
          )}
        </div>
      </div>

      {registered !== undefined && (
        <p role="status" className="flex items-center gap-1.5 text-xs text-ink-soft">
          <Info size={14} weight="fill" className="text-magenta-ink" aria-hidden="true" />
          {registered
            ? `Registered version ${v.version}. Next, create its splits.`
            : `This upload with these roles was already registered as version ${v.version}; nothing new was created.`}
        </p>
      )}

      <Panel title={`Version ${v.version}`} meta={`registered ${date(v.createdAt)}`}>
        <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2 xl:grid-cols-4" data-testid="version-facts">
          <Fact label="Format">{v.format.toUpperCase()}</Fact>
          <Fact label="Rows" testId="version-rows">
            {int(v.rowCount)}
          </Fact>
          <Fact label="Columns">{v.columns.length}</Fact>
          <Fact label="Row ids">
            {v.rowIdSource === "column" ? (
              <>
                from <span className="font-mono">{v.mapping.id}</span>
              </>
            ) : (
              <>
                generated <span className="font-mono text-xs text-ink-soft">({v.rowIdScheme})</span>
              </>
            )}
          </Fact>
          <Fact label="Content hash (sha256)" wide testId="version-content-hash">
            <span className="font-mono text-[12px] break-all">{v.contentHash}</span>
          </Fact>
          <Fact label="Identity hash" wide testId="version-identity-hash">
            <span className="font-mono text-[12px] break-all">{v.identityHash}</span>
          </Fact>
        </dl>
      </Panel>

      <SplitsPanel dataset={d} version={v} splits={splits} />

      <RolesPanel dataset={d} version={v} upload={upload.state === "ready" ? upload.data : null} onRegistered={reload} />

      {upload.state === "loading" && <LoadingState label="Loading preview" rows={2} />}
      {upload.state === "error" && <ApiErrorState title="Could not load the upload's preview" error={upload.cause} onRetry={upload.reload} />}
      {upload.state === "ready" && (
        <Panel title="Preview" meta={`First ${upload.data.preview.length} rows of ${upload.data.fileName ?? "the upload"}`} bodyClassName="p-0">
          <DatasetPreview columns={v.columns} rows={upload.data.preview} mapping={v.mapping} />
        </Panel>
      )}
    </div>
  );
}

function SplitsPanel({
  dataset: d,
  version: v,
  splits,
}: {
  dataset: Dataset;
  version: DatasetVersion;
  splits: Resource<DatasetSplitSet[]>;
}) {
  const [validationPct, setValidationPct] = useState("20");
  const [testPct, setTestPct] = useState("20");
  const [seed, setSeed] = useState("0");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [result, setResult] = useState<{ hash: string; created: boolean } | null>(null);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    setResult(null);
    try {
      const { value, created } = await api().createSplits(d.id, v.version, {
        validationPct: Number(validationPct),
        testPct: Number(testPct),
        seed: Number(seed),
      });
      setResult({ hash: value.splitsHash, created });
      splits.reload();
    } catch (err) {
      setError(err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Panel title="Splits" meta="Deterministic: the same version, seed and fractions always give the same rows">
      {splits.state === "loading" && <LoadingState label="Loading splits" rows={1} />}
      {splits.state === "error" && <ApiErrorState title="Could not load splits" error={splits.cause} onRetry={splits.reload} />}
      {splits.state === "ready" &&
        (splits.data.length === 0 ? (
          <p className="text-sm text-ink-soft">No splits for this version yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table" aria-label="Splits">
              <thead>
                <tr>
                  <th>Splits</th>
                  <th className="num">Seed</th>
                  <th className="num">Validation</th>
                  <th className="num">Test</th>
                  <th className="num">Optimization rows</th>
                  <th className="num">Validation rows</th>
                  <th className="num">Test rows</th>
                  <th className="num">Created</th>
                </tr>
              </thead>
              <tbody>
                {splits.data.map((s) => (
                  <tr key={s.splitsHash} data-testid="splits-row" data-splits-hash={s.splitsHash}>
                    <td className="font-mono text-[12px]" title={s.splitsHash}>
                      {s.splitsHash.slice(0, 16)}
                      <span className="ml-1.5 text-ink-soft">{s.method}</span>
                    </td>
                    <td className="num">{s.plan ? s.plan.seed : "—"}</td>
                    <td className="num">{s.plan ? `${s.plan.validationPct}%` : "—"}</td>
                    <td className="num">{s.plan ? `${s.plan.testPct}%` : "—"}</td>
                    <td className="num" data-testid="size-optimization">
                      {int(s.sizes.optimization)}
                    </td>
                    <td className="num" data-testid="size-validation">
                      {int(s.sizes.validation)}
                    </td>
                    <td className="num" data-testid="size-test">
                      {int(s.sizes.test)}
                    </td>
                    <td className="num text-ink-soft">{date(s.createdAt)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}

      <form onSubmit={create} className="mt-4 grid gap-3 border-t border-line pt-4 sm:grid-cols-[repeat(3,minmax(0,9rem))_auto] sm:items-end" noValidate>
        <Field label="Validation (%)">
          {(p) => <input {...p} className="input" type="number" min={0} max={99} value={validationPct} onChange={(e) => setValidationPct(e.target.value)} />}
        </Field>
        <Field label="Test (%)">
          {(p) => <input {...p} className="input" type="number" min={0} max={99} value={testPct} onChange={(e) => setTestPct(e.target.value)} />}
        </Field>
        <Field label="Seed">{(p) => <input {...p} className="input" type="number" min={0} value={seed} onChange={(e) => setSeed(e.target.value)} />}</Field>
        <div>
          <button type="submit" className="btn btn-primary btn-sm" disabled={saving}>
            {saving ? "Creating…" : "Create splits"}
          </button>
        </div>
      </form>
      <p className="mt-2 text-xs text-ink-soft">Optimization gets the remaining rows. The server assigns rows by a seeded hash of each row id.</p>
      {result && (
        <p role="status" className="mt-2 text-xs text-ink-soft">
          {result.created ? "Created splits " : "These splits already existed: "}
          <span className="font-mono">{result.hash.slice(0, 16)}</span>
        </p>
      )}
      {error !== null && (
        <div className="mt-3">
          <ApiErrorState error={error} />
        </div>
      )}
    </Panel>
  );
}

/** Roles of this version. Changing them registers a new version of the same upload. */
function RolesPanel({
  dataset: d,
  version: v,
  upload,
  onRegistered,
}: {
  dataset: Dataset;
  version: DatasetVersion;
  upload: DatasetUpload | null;
  onRegistered: () => void;
}) {
  const navigate = useNavigate();
  const [mapping, setMapping] = useState<ColumnMapping>(v.mapping);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<unknown>(null);
  // null counts are the upload's (whole file); types and nullability are the version's
  const columns = v.columns.map((c) => ({ ...c, nullCount: upload?.columns.find((u) => u.name === c.name)?.nullCount }));
  const problems = validateMapping(mapping, v.columns);
  const dirty = JSON.stringify(mapping) !== JSON.stringify(v.mapping);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const { value, created } = await api().registerDataset(v.uploadId, { datasetId: d.id, name: v.name, mapping });
      onRegistered();
      navigate(`?version=${value.version}`, { state: { registered: created } });
    } catch (err) {
      setError(err);
      setSaving(false);
    }
  };

  return (
    <Panel
      title="Schema and column roles"
      meta="Changing roles registers a new version; existing versions never change"
      actions={
        <button type="button" className="btn btn-quiet btn-sm" disabled={!dirty || saving || problems.length > 0} onClick={save}>
          {saving ? "Registering…" : "Register as new version"}
        </button>
      }
    >
      <MappingEditor columns={columns} preview={upload?.preview ?? []} mapping={mapping} onChange={setMapping} errors={problems} />
      {error !== null && (
        <div className="mt-3">
          <ApiErrorState error={error} />
        </div>
      )}
    </Panel>
  );
}
