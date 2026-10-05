import { FileArrowUp, Info } from "@phosphor-icons/react";
import { useRef, useState } from "react";
import { Link, useNavigate } from "react-router";
import { type ColumnMapping, type DatasetFormat, api, errorMessage } from "@/api";
import { DatasetPreview, MappingEditor } from "@/components/app/DatasetSchema";
import { useShell } from "@/components/app/AppShell";
import { ErrorState, Field, Panel } from "@/components/app/ui";
import { MAX_PREVIEW_BYTES, type ParsedDataset, ParseError, detectFormat, parseDataset, suggestMapping } from "@/lib/dataset";
import { bytes, int } from "@/lib/format";
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

type Picked =
  | { kind: "none" }
  | { kind: "parsed"; fileName: string; size: number; format: "csv" | "jsonl"; parsed: ParsedDataset }
  | { kind: "parquet"; fileName: string; size: number }
  | { kind: "error"; fileName: string; message: string };

export function DatasetImportPage() {
  const { project, reloadProject } = useProject();
  const { refreshProjects } = useShell();
  const navigate = useNavigate();
  const fileInput = useRef<HTMLInputElement>(null);
  const [picked, setPicked] = useState<Picked>({ kind: "none" });
  const [name, setName] = useState("");
  const [mapping, setMapping] = useState<ColumnMapping>({ input: [], target: null, context: [] });
  const [dragging, setDragging] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const accept = (fileName: string, size: number, text: string | null, format: DatasetFormat) => {
    setSaveError(null);
    if (format === "parquet") return setPicked({ kind: "parquet", fileName, size });
    try {
      const parsed = parseDataset(text ?? "", format);
      setPicked({ kind: "parsed", fileName, size, format, parsed });
      setMapping(suggestMapping(parsed.columns));
      setName(fileName.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " "));
    } catch (err) {
      setPicked({ kind: "error", fileName, message: err instanceof ParseError ? err.message : `Could not read the file: ${errorMessage(err)}` });
    }
  };

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    const format = detectFormat(file.name);
    if (!format) return setPicked({ kind: "error", fileName: file.name, message: "Choose a .csv, .jsonl or .parquet file." });
    if (format !== "parquet" && file.size > MAX_PREVIEW_BYTES)
      return setPicked({
        kind: "error",
        fileName: file.name,
        message: `This file is ${bytes(file.size)}. Files over ${bytes(MAX_PREVIEW_BYTES)} need server-side ingestion, which is not available yet.`,
      });
    accept(file.name, file.size, format === "parquet" ? null : await file.text(), format);
  };

  const errors =
    picked.kind === "parsed"
      ? validateMapping(
          mapping,
          picked.parsed.columns.map((c) => c.name),
        )
      : [];

  const register = async () => {
    if (picked.kind !== "parsed") return;
    if (!name.trim()) return setSaveError("Name the dataset.");
    setSaving(true);
    setSaveError(null);
    try {
      const d = await api().createDataset(project.id, {
        name,
        format: picked.format,
        fileName: picked.fileName,
        sizeBytes: picked.size,
        rowCount: picked.parsed.rowCount,
        columns: picked.parsed.columns,
        preview: picked.parsed.preview,
        mapping,
      });
      reloadProject();
      refreshProjects();
      navigate(d.status === "ready" ? `/projects/${project.id}/experiments/new?dataset=${d.id}` : `/projects/${project.id}/datasets/${d.id}`);
    } catch (err) {
      setSaveError(errorMessage(err));
      setSaving(false);
    }
  };

  return (
    <div className="space-y-4">
      <Panel title="Add dataset" meta="CSV, JSONL or Parquet">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            onFile(e.dataTransfer.files[0]);
          }}
          className={cn(
            "flex flex-col items-center gap-2 rounded-xl border border-dashed px-6 py-8 text-center transition-colors",
            dragging ? "border-magenta-ink bg-magenta-wash" : "border-line-strong",
          )}
        >
          <FileArrowUp size={26} weight="duotone" className="text-magenta-ink" aria-hidden="true" />
          <p className="text-sm">
            Drop a file here or{" "}
            <button type="button" className="btn btn-text" onClick={() => fileInput.current?.click()}>
              choose one
            </button>
          </p>
          <p className="max-w-[60ch] text-xs text-ink-soft">
            The file is read in your browser to show its schema and a preview. Nothing is uploaded: server-side ingestion is not part of this build.
          </p>
          <input
            ref={fileInput}
            type="file"
            accept=".csv,.jsonl,.ndjson,.parquet"
            className="sr-only"
            aria-label="Dataset file"
            data-testid="dataset-file"
            onChange={(e) => onFile(e.target.files?.[0])}
          />
          <button
            type="button"
            className="btn btn-quiet btn-sm mt-1"
            onClick={() => accept("support_tickets_example.csv", EXAMPLE_CSV.length, EXAMPLE_CSV, "csv")}
          >
            Use an example CSV
          </button>
        </div>
      </Panel>

      {picked.kind === "error" && <ErrorState title={`Could not read ${picked.fileName}`} message={picked.message} />}

      {picked.kind === "parquet" && (
        <div role="status" className="flex items-start gap-3 rounded-[10px] border border-line-strong bg-wash px-4 py-3 text-sm">
          <Info size={18} weight="fill" className="mt-0.5 shrink-0 text-magenta-ink" aria-hidden="true" />
          <div>
            <p className="font-semibold">
              {picked.fileName} · {bytes(picked.size)}
            </p>
            <p className="mt-0.5 text-ink-soft">
              Parquet is a binary format read by Wynk's ingestion service, which is not available in this build, so it cannot be previewed or
              registered yet. Export it as CSV or JSONL to continue now.
            </p>
          </div>
        </div>
      )}

      {picked.kind === "parsed" && (
        <>
          <Panel
            title="Schema and column roles"
            meta={`${picked.fileName} · ${bytes(picked.size)} · ${int(picked.parsed.rowCount)} rows · ${picked.parsed.columns.length} columns`}
          >
            <MappingEditor columns={picked.parsed.columns} preview={picked.parsed.preview} mapping={mapping} onChange={setMapping} errors={errors} />
          </Panel>
          <Panel
            title="Preview"
            meta={`First ${picked.parsed.preview.length} of ${int(picked.parsed.rowCount)} rows. Types and counts are from these rows.`}
            bodyClassName="p-0"
          >
            <DatasetPreview columns={picked.parsed.columns} rows={picked.parsed.preview} mapping={mapping} />
          </Panel>
          <Panel>
            <div className="flex flex-wrap items-end justify-between gap-4">
              <Field label="Dataset name" className="w-full max-w-sm" error={saveError ?? undefined}>
                {(p) => <input {...p} className="input" value={name} onChange={(e) => setName(e.target.value)} maxLength={80} />}
              </Field>
              <div className="flex items-center gap-2">
                <Link to=".." relative="path" className="btn btn-quiet btn-sm">
                  Cancel
                </Link>
                <button type="button" className="btn btn-primary btn-sm" disabled={saving || errors.length > 0} onClick={register}>
                  {saving ? "Registering…" : "Register and configure"}
                </button>
              </div>
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}
