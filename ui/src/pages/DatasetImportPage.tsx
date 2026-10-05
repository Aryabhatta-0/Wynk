import { FileArrowUp, Info } from "@phosphor-icons/react";
import { useRef, useState } from "react";
import { Link, useLocation, useNavigate, useSearchParams } from "react-router";
import { type ColumnMapping, type DatasetUpload, api } from "@/api";
import { DatasetPreview, MappingEditor } from "@/components/app/DatasetSchema";
import { useShell } from "@/components/app/AppShell";
import { DatasetSteps } from "@/components/app/FlowSteps";
import { ApiErrorState, Fact, Field, LoadingState, Panel } from "@/components/app/ui";
import { slugFor, suggestMapping, uploadFormat } from "@/lib/dataset";
import { bytes, date, int } from "@/lib/format";
import { useResource } from "@/lib/useResource";
import { cn } from "@/lib/utils";
import { validateMapping } from "@/lib/validation";
import { useProject } from "./ProjectLayout";

const EXAMPLE_CSV = `ticket_id,subject,body,customer_tier,queue
T-1,Charged twice,"My card shows two charges for March.",pro,billing
T-2,Cannot log in,"SSO redirects back to the login page.",enterprise,technical
T-3,Update company name,"We rebranded; please change the account name.",pro,account
T-4,Where is my order,"Tracking has not moved for a week.",free,shipping
T-5,Refund request,"Downgraded mid-cycle, expected a prorated refund.",pro,billing
T-6,Webhook failures,"Deliveries to our endpoint time out since Monday.",enterprise,technical
`;

/** Navigation state after an upload: false when the server already had these exact bytes. */
interface UploadedState {
  created?: boolean;
}

/*
  Upload → Inspect → Map columns → Register. The file's bytes go to the server, which inspects
  them; the upload id is kept in the address (?upload=) so a reload resumes from the server's
  record. Registration leads to the dataset page, where splits are created.
*/
export function DatasetImportPage() {
  const [params] = useSearchParams();
  const uploadId = params.get("upload");
  return uploadId ? <InspectAndRegister key={uploadId} uploadId={uploadId} /> : <UploadStep />;
}

function UploadStep() {
  const { project } = useProject();
  const navigate = useNavigate();
  const fileInput = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState<string | null>(null);
  const [error, setError] = useState<{ fileName: string; error: unknown } | null>(null);

  const send = async (name: string, data: Blob) => {
    setError(null);
    setUploading(name);
    try {
      const { value, created } = await api().uploadDataset(project.id, { name, format: uploadFormat(name), bytes: data });
      navigate(`?upload=${encodeURIComponent(value.id)}`, { state: { created } satisfies UploadedState });
    } catch (err) {
      setError({ fileName: name, error: err });
      setUploading(null);
    }
  };

  return (
    <div className="space-y-4">
      <DatasetSteps current="Upload" />
      <Panel title="Add dataset" meta="CSV or JSONL">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files[0];
            if (file) send(file.name, file);
          }}
          className={cn(
            "flex flex-col items-center gap-2 rounded-xl border border-dashed px-6 py-8 text-center transition-colors",
            dragging ? "border-magenta-ink bg-magenta-wash" : "border-line-strong",
          )}
        >
          <FileArrowUp size={26} weight="duotone" className="text-magenta-ink" aria-hidden="true" />
          {uploading ? (
            <p className="text-sm" role="status">
              Uploading {uploading}…
            </p>
          ) : (
            <p className="text-sm">
              Drop a file here or{" "}
              <button type="button" className="btn btn-text" onClick={() => fileInput.current?.click()}>
                choose one
              </button>
            </p>
          )}
          <p className="max-w-[64ch] text-xs text-ink-soft">
            The file is uploaded to {api().mode === "live" ? "the Wynk API" : "the mock adapter (it stays in this browser tab)"}, which reads every row
            and reports the column types, missing values, row count and content hash. Nothing is registered until you confirm the column roles.
          </p>
          <input
            ref={fileInput}
            type="file"
            accept=".csv,.jsonl,.ndjson"
            className="sr-only"
            aria-label="Dataset file"
            data-testid="dataset-file"
            disabled={uploading !== null}
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) send(file.name, file);
            }}
          />
          <button
            type="button"
            className="btn btn-quiet btn-sm mt-1"
            disabled={uploading !== null}
            onClick={() => send("support_tickets_example.csv", new Blob([EXAMPLE_CSV], { type: "text/csv" }))}
          >
            Use an example CSV
          </button>
        </div>
      </Panel>
      {error && <ApiErrorState error={error.error} />}
      {error && (
        <p className="text-xs text-ink-soft" data-testid="upload-error-file">
          File: <span className="font-mono">{error.fileName}</span>. Nothing was stored.
        </p>
      )}
    </div>
  );
}

function InspectAndRegister({ uploadId }: { uploadId: string }) {
  const upload = useResource(() => api().getUpload(uploadId), [uploadId]);
  if (upload.state === "loading") return <LoadingState label="Loading the inspected upload" rows={4} />;
  if (upload.state === "error")
    return (
      <div className="space-y-3">
        <ApiErrorState title="Could not load this upload" error={upload.cause} onRetry={upload.reload} />
        <Link to="." className="btn btn-quiet btn-sm">
          Upload a different file
        </Link>
      </div>
    );
  return <MapAndRegister upload={upload.data} />;
}

function MapAndRegister({ upload }: { upload: DatasetUpload }) {
  const { project, reloadProject } = useProject();
  const { refreshProjects } = useShell();
  const navigate = useNavigate();
  const created = (useLocation().state as UploadedState | null)?.created;
  const suggestedName = (upload.fileName ?? "dataset").replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ");
  const [mapping, setMapping] = useState<ColumnMapping>(() => suggestMapping(upload.columns));
  const [name, setName] = useState(suggestedName);
  const [datasetId, setDatasetId] = useState<string | null>(null); // null: follow the name
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const id = datasetId ?? slugFor(name);
  const problems = validateMapping(mapping, upload.columns);
  const source = api().mode === "live" ? "the Wynk API" : "the mock adapter";

  const register = async () => {
    setSaving(true);
    setError(null);
    try {
      const { value, created: isNew } = await api().registerDataset(upload.id, { datasetId: id, name, mapping });
      reloadProject();
      refreshProjects();
      navigate(`/projects/${project.id}/datasets/${encodeURIComponent(value.datasetId)}?version=${value.version}`, {
        state: { registered: isNew },
      });
    } catch (err) {
      setError(err);
      setSaving(false);
    }
  };

  return (
    <div className="space-y-4">
      <DatasetSteps current="Map columns" />
      <Panel
        title="Inspection"
        meta={`by ${source}`}
        actions={
          <Link to="." className="btn btn-quiet btn-sm">
            Upload a different file
          </Link>
        }
      >
        {created === false && (
          <p role="status" className="mb-3 flex items-center gap-1.5 text-xs text-ink-soft">
            <Info size={14} weight="fill" className="text-magenta-ink" aria-hidden="true" /> These exact bytes were already uploaded to this project;
            the existing upload is reused.
          </p>
        )}
        <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2 xl:grid-cols-4" data-testid="inspection">
          <Fact label="File">{upload.fileName ?? "—"}</Fact>
          <Fact label="Format">{upload.format.toUpperCase()}</Fact>
          <Fact label="Size">{bytes(upload.sizeBytes)}</Fact>
          <Fact label="Rows" testId="inspection-rows">
            {int(upload.rowCount)}
          </Fact>
          <Fact label="Columns">{upload.columns.length}</Fact>
          <Fact label="Uploaded">{date(upload.createdAt)}</Fact>
          <Fact label="Content hash (sha256)" wide testId="inspection-hash">
            <span className="font-mono text-[12px] break-all">{upload.contentHash}</span>
          </Fact>
        </dl>
      </Panel>

      <Panel title="Schema and column roles" meta="Types, nullability and missing counts are over every row, as the server inferred them">
        <MappingEditor columns={upload.columns} preview={upload.preview} mapping={mapping} onChange={setMapping} errors={problems} />
      </Panel>

      <Panel
        title="Preview"
        meta={`First ${upload.preview.length} of ${int(upload.rowCount)} rows, as returned by ${source}`}
        bodyClassName="p-0"
      >
        <DatasetPreview columns={upload.columns} rows={upload.preview} mapping={mapping} />
      </Panel>

      <Panel title="Register">
        <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] md:items-start">
          <Field label="Dataset name">
            {(p) => <input {...p} className="input" value={name} onChange={(e) => setName(e.target.value)} maxLength={200} />}
          </Field>
          <Field label="Dataset id" hint="Lowercase letters, digits, _ . or -. Registering the same id again adds a version.">
            {(p) => <input {...p} className="input font-mono" value={id} onChange={(e) => setDatasetId(e.target.value)} maxLength={128} />}
          </Field>
          <div className="flex gap-2 md:pt-6">
            <button type="button" className="btn btn-primary btn-sm" disabled={saving || problems.length > 0 || !name.trim()} onClick={register}>
              {saving ? "Registering…" : "Register dataset"}
            </button>
          </div>
        </div>
        <p className="mt-2 text-xs text-ink-soft" data-testid="row-id-source">
          Row ids:{" "}
          {mapping.id ? (
            <>
              taken from <span className="font-mono text-ink">{mapping.id}</span>; every value must be present and unique.
            </>
          ) : (
            "generated by the server from each row's content (no id column chosen)."
          )}
        </p>
        {error !== null && (
          <div className="mt-3">
            <ApiErrorState error={error} />
          </div>
        )}
      </Panel>
    </div>
  );
}
