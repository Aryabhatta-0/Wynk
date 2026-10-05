import { Cpu, Database, Lock, Scales, Sliders } from "@phosphor-icons/react";
import { type ReactNode, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import {
  type BalancedWeights,
  type ColumnType,
  type Dataset,
  type EvaluationConfig,
  type EvaluatorKind,
  type ExperimentConfig,
  type ModelOption,
  type Objective,
  type TaskType,
  api,
  errorMessage,
} from "@/api";
import { evaluatorsFor, splitSizes } from "@/api/contract/rules";
import { EmptyState, ErrorState, Field, LoadingState, Panel } from "@/components/app/ui";
import { labelsIn, suggestTaskType } from "@/lib/dataset";
import { EVALUATOR_LABEL, EVALUATOR_METRIC, EXAMPLES_PER_COST_UNIT, OBJECTIVE_LABEL, TASK_LABEL, int } from "@/lib/format";
import { balancedFormula } from "@/lib/objective";
import { useResource } from "@/lib/useResource";
import { cn } from "@/lib/utils";
import { type ConfigErrors, validateConfig } from "@/lib/validation";
import { useProject } from "./ProjectLayout";

/** Form state keeps raw strings so a half-typed number is never coerced under the cursor. */
interface Form {
  name: string;
  datasetId: string;
  taskType: TaskType;
  instructions: string;
  evaluator: EvaluatorKind;
  labels: string;
  caseSensitive: boolean;
  normalizeWhitespace: boolean;
  passThreshold: string;
  absoluteTolerance: string;
  relativeTolerance: string;
  minQualityPct: string;
  /** entered per 1,000 examples; the contract limit is per example */
  maxCostPer1k: string;
  maxMeanLatencyS: string;
  maxP95LatencyS: string;
  maxTokensPerExample: string;
  maxWorkflowSteps: string;
  models: string[];
  objective: Objective;
  wQuality: string;
  wCost: string;
  wLatency: string;
  wTokens: string;
  costScalePer1k: string;
  latencyScaleS: string;
  tokensScale: string;
  maxCandidates: string;
  maxGenerations: string;
  maxSpendUsd: string;
  maxDurationMin: string;
  validationPct: string;
  testPct: string;
  seed: string;
}

const TASKS: { id: TaskType; description: string }[] = [
  { id: "classification", description: "Pick one label from a fixed set. One text target." },
  { id: "structured_extraction", description: "Fill named fields. One target column per field." },
  { id: "question_answering", description: "Answer in short text or a number. One target." },
];

const OBJECTIVES: { id: Objective; description: string }[] = [
  { id: "quality", description: "Highest quality among feasible workflows." },
  { id: "cost", description: "Lowest mean cost per example, then quality. Needs a minimum quality." },
  { id: "latency", description: "Lowest mean latency, then quality. Needs a minimum quality." },
  { id: "balanced", description: "Weighted quality minus scaled cost, latency or tokens." },
];

const optionalNumber = (s: string, scale = 1) => (s.trim() === "" ? null : Number(s) * scale);
const targetTypesOf = (d: Dataset): ColumnType[] => d.mapping.target.map((t) => d.columns.find((c) => c.name === t)?.type ?? "string");
const defaultLabels = (d: Dataset) => (d.mapping.target.length === 1 ? labelsIn(d.preview, d.mapping.target[0]).map((l) => l.label) : []);

function evaluationOf(f: Form): EvaluationConfig {
  switch (f.evaluator) {
    case "classification_accuracy":
      return {
        evaluator: f.evaluator,
        labels: f.labels
          .split(",")
          .map((l) => l.trim())
          .filter(Boolean),
        caseSensitive: f.caseSensitive,
      };
    case "exact_match":
      return { evaluator: f.evaluator, caseSensitive: f.caseSensitive, normalizeWhitespace: f.normalizeWhitespace };
    case "token_f1":
      return { evaluator: f.evaluator, passThreshold: Number(f.passThreshold) };
    case "json_schema_validity":
      return { evaluator: f.evaluator };
    case "numeric_tolerance":
      return { evaluator: f.evaluator, absoluteTolerance: Number(f.absoluteTolerance), relativeTolerance: Number(f.relativeTolerance) };
  }
}

function balancedOf(f: Form): BalancedWeights {
  return {
    quality: Number(f.wQuality),
    cost: Number(f.wCost),
    latency: Number(f.wLatency),
    tokens: Number(f.wTokens),
    costScale: optionalNumber(f.costScalePer1k, 1 / EXAMPLES_PER_COST_UNIT),
    latencyScale: optionalNumber(f.latencyScaleS),
    tokensScale: optionalNumber(f.tokensScale),
  };
}

export function toConfig(f: Form): ExperimentConfig {
  return {
    name: f.name,
    datasetId: f.datasetId,
    taskType: f.taskType,
    instructions: f.instructions,
    evaluation: evaluationOf(f),
    constraints: {
      minQuality: optionalNumber(f.minQualityPct, 0.01),
      maxCostPerExample: optionalNumber(f.maxCostPer1k, 1 / EXAMPLES_PER_COST_UNIT),
      maxMeanLatencyS: optionalNumber(f.maxMeanLatencyS),
      maxP95LatencyS: optionalNumber(f.maxP95LatencyS),
      maxTokensPerExample: optionalNumber(f.maxTokensPerExample),
      maxWorkflowSteps: optionalNumber(f.maxWorkflowSteps),
    },
    preferences: { objective: f.objective, balanced: f.objective === "balanced" ? balancedOf(f) : null },
    models: f.models,
    budget: {
      maxCandidates: Number(f.maxCandidates),
      maxGenerations: Number(f.maxGenerations),
      maxSpendUsd: Number(f.maxSpendUsd),
      maxDurationMin: Number(f.maxDurationMin),
    },
    splits: { validationPct: Number(f.validationPct), testPct: Number(f.testPct), seed: Number(f.seed) },
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
          An experiment needs a dataset with an id column, at least one input column and at least one target column mapped.
        </EmptyState>
      </Panel>
    );
  }
  const initial = ready.find((d) => d.id === params.get("dataset")) ?? ready[0];
  return <ConfigureForm projectId={project.id} datasets={ready} models={models} initial={initial} onCreated={reloadProject} />;
}

/** Task type, evaluator and labels a dataset suggests; the person can change all of them. */
function suggestionsFor(d: Dataset): Pick<Form, "taskType" | "evaluator" | "labels"> {
  const taskType = suggestTaskType(d.columns, d.mapping.target, d.rowCount);
  return { taskType, evaluator: evaluatorsFor(taskType, targetTypesOf(d))[0] ?? "exact_match", labels: defaultLabels(d).join(", ") };
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
    ...suggestionsFor(initial),
    instructions: "",
    caseSensitive: false,
    normalizeWhitespace: true,
    passThreshold: "0.8",
    absoluteTolerance: "0",
    relativeTolerance: "0.01",
    minQualityPct: "",
    maxCostPer1k: "",
    maxMeanLatencyS: "",
    maxP95LatencyS: "",
    maxTokensPerExample: "",
    maxWorkflowSteps: "",
    models: models.slice(0, 2).map((m) => m.id),
    objective: "quality",
    wQuality: "70",
    wCost: "20",
    wLatency: "10",
    wTokens: "0",
    costScalePer1k: "0.5",
    latencyScaleS: "2",
    tokensScale: "",
    maxCandidates: "64",
    maxGenerations: "8",
    maxSpendUsd: "25",
    maxDurationMin: "60",
    validationPct: "20",
    testPct: "20",
    seed: "0",
  }));
  const [submitted, setSubmitted] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const set = <K extends keyof Form>(key: K, value: Form[K]) => setForm((f) => ({ ...f, [key]: value }));
  const dataset = datasets.find((d) => d.id === form.datasetId)!;
  const targetTypes = targetTypesOf(dataset);
  const allowedEvaluators = evaluatorsFor(form.taskType, targetTypes);
  const config = toConfig(form);
  const errors: ConfigErrors = validateConfig(config, dataset);
  const shown = submitted ? errors : {};
  const errorCount = Object.keys(errors).length;

  const chooseDataset = (id: string) => {
    const next = datasets.find((d) => d.id === id)!;
    setForm((f) => ({ ...f, datasetId: id, ...suggestionsFor(next) }));
  };
  // a different task type allows different evaluators; keep the choice when it is still allowed
  const chooseTask = (taskType: TaskType) =>
    setForm((f) => {
      const allowed = evaluatorsFor(taskType, targetTypes);
      return { ...f, taskType, evaluator: allowed.includes(f.evaluator) ? f.evaluator : (allowed[0] ?? f.evaluator) };
    });

  const optimizationPct = 100 - (Number(form.validationPct) || 0) - (Number(form.testPct) || 0);
  const sizes = splitSizes(dataset.rowCount, config.splits);

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

  const limitSummary = [
    config.constraints.minQuality !== null && `quality ≥ ${form.minQualityPct}%`,
    config.constraints.maxCostPerExample !== null && `cost ≤ $${form.maxCostPer1k}/1k`,
    config.constraints.maxMeanLatencyS !== null && `mean ≤ ${form.maxMeanLatencyS} s`,
    config.constraints.maxP95LatencyS !== null && `p95 ≤ ${form.maxP95LatencyS} s`,
    config.constraints.maxTokensPerExample !== null && `tokens ≤ ${form.maxTokensPerExample}`,
    config.constraints.maxWorkflowSteps !== null && `steps ≤ ${form.maxWorkflowSteps}`,
  ].filter(Boolean);

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
              hint={`${int(dataset.rowCount)} rows · inputs: ${dataset.mapping.input.join(", ")} · target: ${dataset.mapping.target.join(", ")}`}
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
          <fieldset className="mt-4" aria-describedby={shown.taskType ? "task-error" : undefined}>
            <legend className="mb-1.5 text-[13px] font-medium">Task type</legend>
            <div className="grid gap-2 md:grid-cols-3">
              {TASKS.map((t) => (
                <Choice key={t.id} name="taskType" checked={form.taskType === t.id} onChange={() => chooseTask(t.id)} title={TASK_LABEL[t.id]}>
                  {t.description}
                </Choice>
              ))}
            </div>
            {shown.taskType && (
              <p id="task-error" className="mt-1 text-xs text-bad">
                {shown.taskType}
              </p>
            )}
          </fieldset>
          <Field
            className="mt-4"
            label="Instructions"
            hint="Plain language, shown to every workflow. The target columns are never shown to it."
            error={shown.instructions}
          >
            {(p) => (
              <textarea
                {...p}
                className="input"
                rows={3}
                value={form.instructions}
                onChange={(e) => set("instructions", e.target.value)}
                placeholder="Read the ticket's subject and body and answer with the queue that should handle it."
              />
            )}
          </Field>
        </Panel>

        <Panel title="Evaluation" meta="How each output is scored. Quality is the mean score, 0 to 100%.">
          <fieldset aria-describedby={shown.evaluation ? "evaluation-error" : undefined}>
            <legend className="sr-only">Evaluator</legend>
            <div className="grid gap-2 md:grid-cols-3">
              {allowedEvaluators.map((k) => (
                <Choice key={k} name="evaluator" checked={form.evaluator === k} onChange={() => set("evaluator", k)} title={EVALUATOR_LABEL[k]}>
                  Quality = {EVALUATOR_METRIC[k]}.
                </Choice>
              ))}
            </div>
          </fieldset>
          <div className="mt-3 grid gap-4 md:grid-cols-3">
            {form.evaluator === "classification_accuracy" && (
              <Field
                className="md:col-span-2"
                label="Labels"
                hint={`Comma-separated. Seen in ${dataset.mapping.target[0]}: ${defaultLabels(dataset).join(", ") || "none"}`}
              >
                {(p) => <input {...p} className="input" value={form.labels} onChange={(e) => set("labels", e.target.value)} />}
              </Field>
            )}
            {(form.evaluator === "classification_accuracy" || form.evaluator === "exact_match") && (
              <Toggle label="Case sensitive" checked={form.caseSensitive} onChange={(v) => set("caseSensitive", v)} />
            )}
            {form.evaluator === "exact_match" && (
              <Toggle label="Collapse whitespace" checked={form.normalizeWhitespace} onChange={(v) => set("normalizeWhitespace", v)} />
            )}
            {form.evaluator === "token_f1" && (
              <Field label="Pass threshold" hint="An answer passes when its token F1 reaches this (0 to 1).">
                {(p) => <NumberInput {...p} value={form.passThreshold} onChange={(v) => set("passThreshold", v)} step="0.05" />}
              </Field>
            )}
            {form.evaluator === "numeric_tolerance" && (
              <>
                <Field label="Absolute tolerance">
                  {(p) => <NumberInput {...p} value={form.absoluteTolerance} onChange={(v) => set("absoluteTolerance", v)} step="0.01" />}
                </Field>
                <Field label="Relative tolerance" hint="Fraction of the target, e.g. 0.01 for 1%.">
                  {(p) => <NumberInput {...p} value={form.relativeTolerance} onChange={(v) => set("relativeTolerance", v)} step="0.01" />}
                </Field>
              </>
            )}
            {form.evaluator === "json_schema_validity" && (
              <p className="text-xs text-ink-soft md:col-span-3">
                Scores whether each output has every target field with the right type. No options.
              </p>
            )}
          </div>
          {shown.evaluation && (
            <p id="evaluation-error" className="mt-2 text-xs text-bad">
              {shown.evaluation}
            </p>
          )}
        </Panel>

        <Panel
          title={
            <span className="inline-flex items-center gap-1.5">
              <Lock size={14} weight="bold" className="text-bad" aria-hidden="true" /> Hard constraints
            </span>
          }
          meta="Must hold, measured per example. A workflow that breaks one, or lacks the measurement, is infeasible and can never become champion. Empty means no limit."
        >
          <div className="grid gap-4 md:grid-cols-3">
            <Field label="Minimum quality (%)" hint={`Mean ${EVALUATOR_METRIC[form.evaluator]}`} error={shown.minQuality}>
              {(p) => <NumberInput {...p} value={form.minQualityPct} onChange={(v) => set("minQualityPct", v)} placeholder="No limit" step="0.5" />}
            </Field>
            <Field label="Maximum cost ($ per 1k examples)" hint="Mean cost; checked per example" error={shown.maxCostPerExample}>
              {(p) => <NumberInput {...p} value={form.maxCostPer1k} onChange={(v) => set("maxCostPer1k", v)} placeholder="No limit" step="0.01" />}
            </Field>
            <Field label="Maximum mean latency (s)" error={shown.maxMeanLatencyS}>
              {(p) => (
                <NumberInput {...p} value={form.maxMeanLatencyS} onChange={(v) => set("maxMeanLatencyS", v)} placeholder="No limit" step="0.5" />
              )}
            </Field>
            <Field label="Maximum p95 latency (s)" error={shown.maxP95LatencyS}>
              {(p) => <NumberInput {...p} value={form.maxP95LatencyS} onChange={(v) => set("maxP95LatencyS", v)} placeholder="No limit" step="0.5" />}
            </Field>
            <Field label="Maximum tokens per example" hint="Worst single example" error={shown.maxTokensPerExample}>
              {(p) => (
                <NumberInput
                  {...p}
                  value={form.maxTokensPerExample}
                  onChange={(v) => set("maxTokensPerExample", v)}
                  placeholder="No limit"
                  step="100"
                />
              )}
            </Field>
            <Field label="Maximum workflow steps" error={shown.maxWorkflowSteps}>
              {(p) => (
                <NumberInput {...p} value={form.maxWorkflowSteps} onChange={(v) => set("maxWorkflowSteps", v)} placeholder="No limit" step="1" />
              )}
            </Field>
          </div>
        </Panel>

        <Panel
          title={
            <span className="inline-flex items-center gap-1.5">
              <Sliders size={14} weight="bold" className="text-magenta-ink" aria-hidden="true" /> Optimization preference
            </span>
          }
          meta="Ranks feasible workflows. It never overrides a hard constraint."
        >
          <fieldset aria-describedby={shown.objective ? "objective-error" : undefined}>
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
          {form.objective === "balanced" && (
            <div className="mt-4 space-y-3 rounded-lg bg-wash px-3 py-3">
              <div className="flex items-center gap-1.5 text-[13px] font-medium">
                <Scales size={14} aria-hidden="true" /> Weights (%, must add up to 100) and scales
              </div>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <Field label="Quality weight">
                  {(p) => <NumberInput {...p} value={form.wQuality} onChange={(v) => set("wQuality", v)} step="5" />}
                </Field>
                <Field label="Cost weight">{(p) => <NumberInput {...p} value={form.wCost} onChange={(v) => set("wCost", v)} step="5" />}</Field>
                <Field label="Latency weight">
                  {(p) => <NumberInput {...p} value={form.wLatency} onChange={(v) => set("wLatency", v)} step="5" />}
                </Field>
                <Field label="Tokens weight">{(p) => <NumberInput {...p} value={form.wTokens} onChange={(v) => set("wTokens", v)} step="5" />}</Field>
                <div className="hidden md:block" />
                <Field label="Cost scale ($ / 1k)" hint="Costs the full cost weight">
                  {(p) => <NumberInput {...p} value={form.costScalePer1k} onChange={(v) => set("costScalePer1k", v)} step="0.1" />}
                </Field>
                <Field label="Latency scale (s)" hint="Mean latency that costs its weight">
                  {(p) => <NumberInput {...p} value={form.latencyScaleS} onChange={(v) => set("latencyScaleS", v)} step="0.5" />}
                </Field>
                <Field label="Tokens scale" hint="Mean tokens that cost its weight">
                  {(p) => <NumberInput {...p} value={form.tokensScale} onChange={(v) => set("tokensScale", v)} step="100" />}
                </Field>
              </div>
              {!errors.objective && config.preferences.balanced && (
                <p className="font-mono text-[12px] text-ink-soft">utility = {balancedFormula(config.preferences.balanced)}</p>
              )}
            </div>
          )}
          {shown.objective && (
            <p id="objective-error" className="mt-2 text-xs text-bad">
              {shown.objective}
            </p>
          )}
        </Panel>

        <Panel
          title={
            <span className="inline-flex items-center gap-1.5">
              <Cpu size={14} weight="bold" className="text-ink-soft" aria-hidden="true" /> Models
            </span>
          }
          meta="The models the search may build workflows with. Part of the model configuration, not a constraint."
        >
          <fieldset aria-describedby={shown.models ? "models-error" : undefined}>
            <legend className="sr-only">Models the search may use</legend>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {models.map((m) => {
                const on = form.models.includes(m.id);
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
                      onChange={() => set("models", on ? form.models.filter((x) => x !== m.id) : [...form.models, m.id])}
                    />
                    <span className="min-w-0">
                      <span className="block text-[13px] font-medium">{m.label}</span>
                      <span className="block truncate font-mono text-[11px] text-ink-soft">{m.id}</span>
                    </span>
                  </label>
                );
              })}
            </div>
            {shown.models && (
              <p id="models-error" className="mt-1 text-xs text-bad">
                {shown.models}
              </p>
            )}
          </fieldset>
        </Panel>

        <div className="grid gap-4 2xl:grid-cols-2">
          <Panel title="Search budget" meta="Search settings: the search stops at the first limit it reaches">
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

          <Panel title="Data splits" meta={`Assigned by row id (${dataset.mapping.id}) with a seed; kept apart for the whole experiment`}>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <div>
                <div className="mb-1 text-[13px] font-medium">Optimization (%)</div>
                <div className="input flex items-center bg-wash tnum" aria-live="polite">
                  {optimizationPct}
                </div>
                <p className="mt-1 text-xs leading-snug text-ink-soft">Feeds the search · {int(sizes.optimization)} rows</p>
              </div>
              <Field label="Validation (%)" hint={`Selects the champion; never feeds the search · ${int(sizes.validation)} rows`}>
                {(p) => (
                  <NumberInput {...p} aria-invalid={!!shown.splits} value={form.validationPct} onChange={(v) => set("validationPct", v)} step="5" />
                )}
              </Field>
              <Field label="Held-out test (%)" hint={`Reported once, never used to choose · ${int(sizes.test)} rows`}>
                {(p) => <NumberInput {...p} aria-invalid={!!shown.splits} value={form.testPct} onChange={(v) => set("testPct", v)} step="5" />}
              </Field>
              <Field label="Split seed" hint="Same seed, same rows in each split">
                {(p) => <NumberInput {...p} aria-invalid={!!shown.splits} value={form.seed} onChange={(v) => set("seed", v)} step="1" />}
              </Field>
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
            <Row label="Evaluator">{EVALUATOR_LABEL[form.evaluator]}</Row>
            <Row label="Objective">{OBJECTIVE_LABEL[form.objective]}</Row>
            <Row label="Limits">{limitSummary.length ? limitSummary.join(" · ") : "none"}</Row>
            <Row label="Models">{form.models.length ? form.models.join(", ") : "none"}</Row>
            <Row label="Budget">
              {form.maxCandidates} candidates · {form.maxGenerations} generations · ${form.maxSpendUsd} · {form.maxDurationMin} min
            </Row>
            <Row label="Splits">
              {optimizationPct} / {form.validationPct} / {form.testPct} · seed {form.seed}
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
            Only optimization results feed the search. The best candidates are re-measured on validation, which picks the champion. Held-out test is
            measured once, afterwards, and only reported.
          </p>
        </div>
      </aside>
    </form>
  );
}

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

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center gap-2 self-end pb-2 text-[13px]">
      <input type="checkbox" className="accent-[var(--color-magenta-ink)]" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
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
