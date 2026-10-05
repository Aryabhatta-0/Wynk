import { Database, Lock, Sliders } from "@phosphor-icons/react";
import { type ReactNode, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { type Dataset, type ExperimentConfig, type ModelOption, type Objective, type TaskType, api, errorMessage } from "@/api";
import { EmptyState, ErrorState, Field, LoadingState, Panel } from "@/components/app/ui";
import { fieldsIn, labelsIn, suggestTaskType } from "@/lib/dataset";
import { OBJECTIVE_LABEL, TASK_LABEL, TASK_METRIC, int } from "@/lib/format";
import { useResource } from "@/lib/useResource";
import { cn } from "@/lib/utils";
import { type ConfigErrors, validateConfig } from "@/lib/validation";
import { useProject } from "./ProjectLayout";

/** Form state keeps raw strings so a half-typed number is never coerced under the cursor. */
interface Form {
  name: string;
  datasetId: string;
  taskType: TaskType;
  minQualityPct: string;
  maxCostPer1k: string;
  maxLatencySec: string;
  allowedModels: string[];
  objective: Objective;
  maxCandidates: string;
  maxGenerations: string;
  maxSpendUsd: string;
  maxDurationMin: string;
  optimization: string;
  validation: string;
  test: string;
}

const TASKS: { id: TaskType; description: string }[] = [
  { id: "classification", description: "Pick one label from a fixed set." },
  { id: "structured_extraction", description: "Fill a JSON object of named fields." },
  { id: "question_answering", description: "Write a short free-text answer." },
];

const OBJECTIVES: { id: Objective; description: string }[] = [
  { id: "quality", description: "Highest task metric among feasible workflows." },
  { id: "cost", description: "Cheapest feasible workflow." },
  { id: "latency", description: "Fastest feasible workflow (p95)." },
  { id: "balanced", description: "Trade quality against cost and latency." },
];

const optionalNumber = (s: string, scale = 1) => (s.trim() === "" ? null : Number(s) * scale);

export function toConfig(f: Form): ExperimentConfig {
  return {
    name: f.name,
    datasetId: f.datasetId,
    taskType: f.taskType,
    constraints: {
      minQuality: optionalNumber(f.minQualityPct, 0.01),
      maxCostPer1k: optionalNumber(f.maxCostPer1k),
      maxLatencyP95Ms: optionalNumber(f.maxLatencySec, 1000),
      allowedModels: f.allowedModels,
    },
    preferences: { objective: f.objective },
    budget: {
      maxCandidates: Number(f.maxCandidates),
      maxGenerations: Number(f.maxGenerations),
      maxSpendUsd: Number(f.maxSpendUsd),
      maxDurationMin: Number(f.maxDurationMin),
    },
    splits: { optimization: Number(f.optimization), validation: Number(f.validation), test: Number(f.test) },
  };
}

export function ConfigurePage() {
  const { project, reloadProject } = useProject();
  const [params] = useSearchParams();
  const data = useResource(() => Promise.all([api().listDatasets(project.id), api().listModels()]), [project.id]);

  if (data.state === "loading") return <LoadingState label="Loading datasets and models" />;
  if (data.state === "error") return <ErrorState title="Could not load what the form needs" message={data.error} onRetry={data.reload} />;

  const [datasets, models] = data.data;
  const ready = datasets.filter((d) => d.status === "ready");
  if (!ready.length) {
    return (
      <Panel>
        <EmptyState
          icon={Database}
          title="No dataset is ready"
          action={
            <Link to="../datasets/new" className="btn btn-primary btn-sm">
              Add dataset
            </Link>
          }
        >
          An experiment needs a dataset with at least one input column and a target column mapped.
        </EmptyState>
      </Panel>
    );
  }
  const initial = ready.find((d) => d.id === params.get("dataset")) ?? ready[0];
  return <ConfigureForm projectId={project.id} datasets={ready} models={models} initial={initial} onCreated={reloadProject} />;
}

function ConfigureForm({
  projectId,
  datasets,
  models,
  initial,
  onCreated,
}: {
  projectId: string;
  datasets: Dataset[];
  models: ModelOption[];
  initial: Dataset;
  onCreated: () => void;
}) {
  const navigate = useNavigate();
  const [form, setForm] = useState<Form>(() => ({
    name: `${initial.name}: optimization`,
    datasetId: initial.id,
    taskType: suggestTaskType(initial.columns, initial.mapping.target, initial.preview.length),
    minQualityPct: "",
    maxCostPer1k: "",
    maxLatencySec: "",
    allowedModels: models.slice(0, 2).map((m) => m.id),
    objective: "quality",
    maxCandidates: "64",
    maxGenerations: "8",
    maxSpendUsd: "25",
    maxDurationMin: "60",
    optimization: "60",
    validation: "20",
    test: "20",
  }));
  const [submitted, setSubmitted] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const set = <K extends keyof Form>(key: K, value: Form[K]) => setForm((f) => ({ ...f, [key]: value }));
  const dataset = datasets.find((d) => d.id === form.datasetId)!;
  const config = toConfig(form);
  const errors: ConfigErrors = validateConfig(config);
  const shown = submitted ? errors : {};
  const errorCount = Object.keys(errors).length;

  // a different dataset may suggest a different task type
  const chooseDataset = (id: string) => {
    const next = datasets.find((d) => d.id === id)!;
    setForm((f) => ({ ...f, datasetId: id, taskType: suggestTaskType(next.columns, next.mapping.target, next.preview.length) }));
  };

  const rows = dataset.rowCount ?? 0;
  const splitRows = (pct: string) => Math.round((rows * (Number(pct) || 0)) / 100);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    if (errorCount) return;
    setSaving(true);
    setSubmitError(null);
    try {
      const exp = await api().createExperiment(projectId, config);
      onCreated();
      navigate(`/projects/${projectId}/experiments/${exp.id}`);
    } catch (err) {
      setSubmitError(errorMessage(err));
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit} noValidate className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
      <div className="min-w-0 space-y-4">
        <Panel title="Task" meta="What the workflow must do with each row">
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Experiment name" error={shown.name}>
              {(p) => <input {...p} className="input" value={form.name} onChange={(e) => set("name", e.target.value)} maxLength={100} />}
            </Field>
            <Field
              label="Dataset"
              error={shown.datasetId}
              hint={`${dataset.rowCount === null ? "Unknown" : int(rows)} rows · inputs: ${dataset.mapping.input.join(", ")} · target: ${dataset.mapping.target}`}
            >
              {(p) => (
                <select {...p} className="input" value={form.datasetId} onChange={(e) => chooseDataset(e.target.value)}>
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name}
                    </option>
                  ))}
                </select>
              )}
            </Field>
          </div>
          <fieldset className="mt-4">
            <legend className="mb-1.5 text-[13px] font-medium">Task type</legend>
            <div className="grid gap-2 md:grid-cols-3">
              {TASKS.map((t) => (
                <Choice key={t.id} name="taskType" checked={form.taskType === t.id} onChange={() => set("taskType", t.id)} title={TASK_LABEL[t.id]}>
                  {t.description} Scored by {TASK_METRIC[t.id]}.
                </Choice>
              ))}
            </div>
          </fieldset>
          <TaskDetails dataset={dataset} taskType={form.taskType} />
        </Panel>

        <Panel
          title={
            <span className="inline-flex items-center gap-1.5">
              <Lock size={14} weight="bold" className="text-bad" aria-hidden="true" /> Hard constraints
            </span>
          }
          meta="Must hold. A workflow that breaks one is infeasible and can never become champion. Leave a limit empty for none."
        >
          <div className="grid gap-4 md:grid-cols-3">
            <Field label="Minimum quality (%)" hint={`Validation ${TASK_METRIC[form.taskType]}`} error={shown.minQuality}>
              {(p) => <NumberInput {...p} value={form.minQualityPct} onChange={(v) => set("minQualityPct", v)} placeholder="No limit" step="0.5" />}
            </Field>
            <Field label="Maximum cost ($ per 1k examples)" error={shown.maxCostPer1k}>
              {(p) => <NumberInput {...p} value={form.maxCostPer1k} onChange={(v) => set("maxCostPer1k", v)} placeholder="No limit" step="0.01" />}
            </Field>
            <Field label="Maximum p95 latency (s)" hint="Per example" error={shown.maxLatencyP95Ms}>
              {(p) => <NumberInput {...p} value={form.maxLatencySec} onChange={(v) => set("maxLatencySec", v)} placeholder="No limit" step="0.5" />}
            </Field>
          </div>
          <fieldset className="mt-4" aria-describedby={shown.allowedModels ? "models-error" : undefined}>
            <legend className="mb-1.5 text-[13px] font-medium">Allowed models</legend>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {models.map((m) => {
                const on = form.allowedModels.includes(m.id);
                return (
                  <label
                    key={m.id}
                    className={cn(
                      "flex cursor-pointer items-start gap-2 rounded-lg border px-3 py-2",
                      on ? "border-magenta-ink bg-magenta-wash/40" : "border-line",
                    )}
                  >
                    <input
                      type="checkbox"
                      className="mt-0.5 accent-[var(--color-magenta-ink)]"
                      checked={on}
                      onChange={() => set("allowedModels", on ? form.allowedModels.filter((x) => x !== m.id) : [...form.allowedModels, m.id])}
                    />
                    <span className="min-w-0">
                      <span className="block text-[13px] font-medium">{m.label}</span>
                      <span className="block truncate font-mono text-[11px] text-ink-soft">{m.id}</span>
                    </span>
                  </label>
                );
              })}
            </div>
            {shown.allowedModels && (
              <p id="models-error" className="mt-1 text-xs text-bad">
                {shown.allowedModels}
              </p>
            )}
          </fieldset>
        </Panel>

        <Panel
          title={
            <span className="inline-flex items-center gap-1.5">
              <Sliders size={14} weight="bold" className="text-magenta-ink" aria-hidden="true" /> Optimization preference
            </span>
          }
          meta="Steers the search among feasible workflows. It never overrides a hard constraint."
        >
          <fieldset>
            <legend className="sr-only">Objective</legend>
            <div className="grid gap-2 md:grid-cols-4">
              {OBJECTIVES.map((o) => (
                <Choice
                  key={o.id}
                  name="objective"
                  checked={form.objective === o.id}
                  onChange={() => set("objective", o.id)}
                  title={OBJECTIVE_LABEL[o.id]}
                >
                  {o.description}
                </Choice>
              ))}
            </div>
          </fieldset>
        </Panel>

        <div className="grid gap-4 2xl:grid-cols-2">
          <Panel title="Search budget" meta="The search stops at the first limit it reaches">
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4 2xl:grid-cols-2">
              <Field label="Candidates" error={shown.maxCandidates}>
                {(p) => <NumberInput {...p} value={form.maxCandidates} onChange={(v) => set("maxCandidates", v)} step="1" />}
              </Field>
              <Field label="Generations" error={shown.maxGenerations}>
                {(p) => <NumberInput {...p} value={form.maxGenerations} onChange={(v) => set("maxGenerations", v)} step="1" />}
              </Field>
              <Field label="Spend (USD)" error={shown.maxSpendUsd}>
                {(p) => <NumberInput {...p} value={form.maxSpendUsd} onChange={(v) => set("maxSpendUsd", v)} step="1" />}
              </Field>
              <Field label="Duration (minutes)" error={shown.maxDurationMin}>
                {(p) => <NumberInput {...p} value={form.maxDurationMin} onChange={(v) => set("maxDurationMin", v)} step="5" />}
              </Field>
            </div>
          </Panel>

          <Panel title="Data splits" meta="Kept apart for the whole experiment">
            <div className="grid grid-cols-3 gap-3">
              {(
                [
                  ["optimization", "Optimization", "Scores every candidate"],
                  ["validation", "Validation", "Picks the champion"],
                  ["test", "Held-out test", "Reported once, at the end"],
                ] as const
              ).map(([key, label, hint]) => (
                <Field key={key} label={`${label} (%)`} hint={`${hint} · ${int(splitRows(form[key]))} rows`}>
                  {(p) => <NumberInput {...p} aria-invalid={!!shown.splits} value={form[key]} onChange={(v) => set(key, v)} step="5" />}
                </Field>
              ))}
            </div>
            {shown.splits && <p className="mt-2 text-xs text-bad">{shown.splits}</p>}
          </Panel>
        </div>
      </div>

      <aside className="min-w-0">
        <div className="panel sticky top-5 space-y-3 p-4">
          <h2 className="font-sans text-sm font-semibold tracking-normal">Review</h2>
          <dl className="space-y-2 text-[13px]">
            <Row label="Dataset">{dataset.name}</Row>
            <Row label="Task">{TASK_LABEL[form.taskType]}</Row>
            <Row label="Objective">{OBJECTIVE_LABEL[form.objective]}</Row>
            <Row label="Constraints">
              {[
                config.constraints.minQuality !== null && `quality ≥ ${form.minQualityPct}%`,
                config.constraints.maxCostPer1k !== null && `cost ≤ $${form.maxCostPer1k}/1k`,
                config.constraints.maxLatencyP95Ms !== null && `p95 ≤ ${form.maxLatencySec} s`,
                `${form.allowedModels.length} model${form.allowedModels.length === 1 ? "" : "s"}`,
              ]
                .filter(Boolean)
                .join(" · ")}
            </Row>
            <Row label="Budget">
              {form.maxCandidates} candidates · {form.maxGenerations} generations · ${form.maxSpendUsd} · {form.maxDurationMin} min
            </Row>
            <Row label="Splits">
              {form.optimization} / {form.validation} / {form.test}
            </Row>
          </dl>
          {submitted && errorCount > 0 && (
            <p role="alert" className="text-xs text-bad">
              Fix {errorCount} field{errorCount === 1 ? "" : "s"} before starting.
            </p>
          )}
          {submitError && <ErrorState title="Could not start the experiment" message={submitError} />}
          <button type="submit" className="btn btn-primary w-full" disabled={saving}>
            {saving ? "Starting…" : "Start optimization"}
          </button>
          <p className="text-[11px] leading-snug text-ink-soft">
            The search evaluates candidates on the optimization split, re-measures the best on validation, and measures the champion once on held-out
            test.
          </p>
        </div>
      </aside>
    </form>
  );
}

function TaskDetails({ dataset, taskType }: { dataset: Dataset; taskType: TaskType }) {
  const target = dataset.mapping.target!;
  if (taskType === "classification") {
    const labels = labelsIn(dataset.preview, target);
    return (
      <Detail label={`Labels seen in the preview of ${target}`}>
        {labels.length ? labels.map((l) => <Tag key={l.label}>{l.label}</Tag>) : <span className="text-ink-soft">No labels in the preview.</span>}
      </Detail>
    );
  }
  if (taskType === "structured_extraction") {
    const fields = fieldsIn(dataset.preview, target);
    return (
      <Detail label={`Fields in ${target}`}>
        {fields.length ? (
          fields.map((f) => <Tag key={f}>{f}</Tag>)
        ) : (
          <span className="text-bad">The target column has no JSON objects in the preview, so there are no fields to extract.</span>
        )}
      </Detail>
    );
  }
  return (
    <Detail label="Answers">
      <span className="text-ink-soft">
        Free text in <span className="font-mono">{target}</span>
        {dataset.mapping.context.length ? `, with ${dataset.mapping.context.join(", ")} as context.` : ". No context column is mapped."}
      </span>
    </Detail>
  );
}

function Detail({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="mt-3 rounded-lg bg-wash px-3 py-2">
      <div className="kicker mb-1">{label}</div>
      <div className="flex flex-wrap gap-1.5 text-[13px]">{children}</div>
    </div>
  );
}

const Tag = ({ children }: { children: ReactNode }) => (
  <span className="rounded-md border border-line bg-white px-1.5 py-0.5 font-mono text-[12px]">{children}</span>
);

function Choice({
  name,
  checked,
  onChange,
  title,
  children,
}: {
  name: string;
  checked: boolean;
  onChange: () => void;
  title: string;
  children: ReactNode;
}) {
  return (
    <label
      className={cn(
        "flex cursor-pointer gap-2 rounded-lg border px-3 py-2.5",
        checked ? "border-magenta-ink bg-magenta-wash/40" : "border-line hover:border-line-strong",
      )}
    >
      <input type="radio" name={name} checked={checked} onChange={onChange} className="mt-1 accent-[var(--color-magenta-ink)]" />
      <span>
        <span className="block text-[13px] font-semibold">{title}</span>
        <span className="block text-xs leading-snug text-ink-soft">{children}</span>
      </span>
    </label>
  );
}

function NumberInput({
  value,
  onChange,
  ...rest
}: Omit<React.InputHTMLAttributes<HTMLInputElement>, "onChange" | "value"> & { value: string; onChange: (v: string) => void }) {
  return (
    <input {...rest} type="number" inputMode="decimal" min="0" className="input tnum" value={value} onChange={(e) => onChange(e.target.value)} />
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[88px_minmax(0,1fr)] gap-2">
      <dt className="text-ink-soft">{label}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </div>
  );
}
