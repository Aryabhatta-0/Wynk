/*
  MOCK ONLY. An in-memory implementation of WynkApi for automated tests and demos. State lives for
  the page session and resets on reload; nothing is persisted or sent anywhere. Datasets follow the
  product API's lifecycle and error codes (upload -> register -> splits); inspection, hashes and
  split sizes are computed here, in the browser, only because there is no server in this mode.
  Experiments are simulated: no model is called and no optimization runs.
*/
import { validateConfig } from "@/lib/validation";
import { ApiError, type WynkApi } from "../client";
import { ContractMappingError, toTaskContract } from "../contract/mapping";
import { canBeId, mappingProblems, splitSizes } from "../contract/rules";
import { checkLimits, compareKeys, objectiveValue } from "../contract/semantics";
import type { TaskContractWire } from "../contract/wire";
import type {
  Candidate,
  Dataset,
  DatasetRow,
  DatasetSplitSet,
  DatasetUpload,
  DatasetVersion,
  Experiment,
  ExperimentConfig,
  ExperimentStatus,
  Measurement,
  MethodResult,
  Project,
  RegisterDataset,
  ResultsComparison,
} from "../types";
import { DATASETS, EXPERIMENTS, MODELS, PROJECTS } from "./fixtures";
import { inspect } from "./inspect";
import { hex } from "./random";
import { type Genome, type SimRun, measure, simulate, snapshot, stagesOf } from "./simulate";

export interface MockOptions {
  /** artificial response delay, ms */
  latencyMs?: number;
  /** how long a new run takes from start to champion, ms */
  runMs?: number;
  /** method names that fail, to exercise error states */
  failOn?: string[];
  /** clock, for tests */
  now?: () => number;
}

interface ExperimentRecord {
  id: string;
  projectId: string;
  config: ExperimentConfig;
  /** the TaskContract this configuration describes, built through the mapping layer */
  contract: TaskContractWire;
  createdAt: number;
  run: SimRun;
  runMs: number;
  queuedMs: number;
  /** fraction where the run was stopped (cancelled or failed) */
  stoppedAt: number | null;
  stoppedStatus: "cancelled" | "failed" | null;
  stoppedTime: number | null;
  failure: string | null;
}

interface UploadEntry {
  upload: DatasetUpload;
  /** every row of an uploaded file; null for seed datasets, which only have a preview */
  rows: DatasetRow[] | null;
}

const clone = <T>(v: T): T => structuredClone(v);

const RESET_NOTE = " Mock data resets when the page reloads, so anything created before a reload is gone.";
const SLUG = /^[a-z0-9][a-z0-9_.-]{0,127}$/;
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

async function sha256(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes as BufferSource);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export function createMockApi(options: MockOptions = {}): WynkApi {
  const latency = options.latencyMs ?? 180;
  const runMs = options.runMs ?? 16_000;
  const failOn = new Set(options.failOn ?? []);
  const now = options.now ?? Date.now;

  const projects: Project[] = clone(PROJECTS);
  const uploads = new Map<string, UploadEntry>();
  const versions: DatasetVersion[] = [];
  const splitSets: DatasetSplitSet[] = [];
  const experiments: ExperimentRecord[] = [];
  let seq = 0;
  const nextId = (prefix: string) => `${prefix}-${(now() + seq++).toString(36)}`;
  const iso = (t = now()) => new Date(t).toISOString();

  for (const seed of DATASETS) {
    const created = iso(now() - seed.createdDaysAgo * 86_400_000);
    const upload: DatasetUpload = {
      id: `u-${seed.id}`,
      projectId: seed.projectId,
      fileName: seed.fileName,
      format: seed.format,
      contentHash: hex(`mock-content:${seed.id}`, 64),
      sizeBytes: seed.sizeBytes,
      rowCount: seed.rowCount,
      columns: seed.columns,
      preview: seed.preview,
      parserVersion: "mock",
      createdAt: created,
    };
    uploads.set(upload.id, { upload, rows: null });
    versions.push(versionOf(upload, { datasetId: seed.id, name: seed.name, mapping: seed.mapping }, 1, created));
  }

  const versionsOf = (datasetId: string) => versions.filter((v) => v.datasetId === datasetId).sort((a, b) => a.version - b.version);
  const latest = (datasetId: string) => versionsOf(datasetId).at(-1);
  const rowsOf = (datasetId: string) => latest(datasetId)?.rowCount ?? 1000;

  for (const seed of EXPERIMENTS) {
    const created = now() - seed.createdDaysAgo * 86_400_000;
    experiments.push({
      id: seed.id,
      projectId: seed.projectId,
      config: seed.config,
      contract: toTaskContract(seed.id, seed.config, latest(seed.config.datasetId)!),
      createdAt: created,
      run: runFor(seed.id, seed.config, rowsOf(seed.config.datasetId)),
      runMs: 1,
      queuedMs: 0,
      stoppedAt: seed.stoppedAt < 1 ? seed.stoppedAt : null,
      stoppedStatus: seed.failure ? "failed" : null,
      stoppedTime: seed.stoppedAt < 1 ? created + 3_600_000 : null,
      failure: seed.failure ?? null,
    });
  }

  async function respond<T>(method: string, produce: () => T | Promise<T>): Promise<T> {
    if (latency > 0) await new Promise((r) => setTimeout(r, latency));
    if (failOn.has(method) || failOn.has("*")) throw new ApiError("mock_failure", `Mock failure injected for ${method}.`);
    return clone(await produce());
  }

  const datasetIdsOf = (projectId: string) => [...new Set(versions.filter((v) => v.projectId === projectId).map((v) => v.datasetId))];
  const project = (id: string) => {
    const p = projects.find((x) => x.id === id);
    if (!p) throw new ApiError("project_not_found", `no project ${id}.${RESET_NOTE}`);
    return {
      ...p,
      datasetCount: datasetIdsOf(id).length,
      experimentCount: experiments.filter((e) => e.projectId === id).length,
    };
  };
  const dataset = (id: string): Dataset => {
    const vs = versionsOf(id);
    if (!vs.length) throw new ApiError("dataset_not_found", `no dataset ${id}.${RESET_NOTE}`);
    const last = vs[vs.length - 1];
    return { id, projectId: last.projectId, name: last.name, latestVersion: last.version, versions: vs };
  };
  const version = (datasetId: string, n: number) => {
    dataset(datasetId);
    const v = versions.find((x) => x.datasetId === datasetId && x.version === n);
    if (!v) throw new ApiError("dataset_version_not_found", `dataset ${datasetId} has no version ${n}`);
    return v;
  };
  const uploadEntry = (id: string) => {
    const u = uploads.get(id);
    if (!u) throw new ApiError("upload_not_found", `no upload ${id}.${RESET_NOTE}`);
    return u;
  };
  const record = (id: string) => {
    const r = experiments.find((x) => x.id === id);
    if (!r) throw new ApiError("not_found", `This experiment does not exist.${RESET_NOTE}`);
    return r;
  };

  /** RegisterDataset checks, with the product API's error codes (ingestion/service.py). */
  function checkRegistration({ upload, rows }: UploadEntry, input: RegisterDataset) {
    const m = input.mapping;
    if (!SLUG.test(input.datasetId))
      throw new ApiError("invalid_request", `dataset_id: String should match pattern '${SLUG.source}'`, { details: { field: "dataset_id" } });
    if (!input.name.trim()) throw new ApiError("invalid_request", "name: String should have at least 1 character", { details: { field: "name" } });
    if (!m.input.length || !m.target.length)
      throw new ApiError("invalid_request", `${m.input.length ? "target_columns" : "input_columns"}: List should have at least 1 item`);
    const roles: [string, string[]][] = [
      ["input", m.input],
      ["target", m.target],
      ["context", m.context],
      ["id", m.id ? [m.id] : []],
    ];
    const seen = new Map<string, string>();
    for (const [role, cols] of roles)
      for (const name of cols) {
        const col = upload.columns.find((c) => c.name === name);
        if (!col) throw new ApiError("unknown_column", `${role} column '${name}' is not in the file`, { details: { column: name } });
        if (col.type === "json")
          throw new ApiError("json_column_role", `'${name}' holds nested JSON values and cannot take a schema role`, { details: { column: name } });
        if (seen.has(name))
          throw new ApiError("invalid_mapping", `column '${name}' is mapped as both ${seen.get(name)} and ${role}`, { details: { column: name } });
        seen.set(name, role);
      }
    const problems = mappingProblems(m, upload.columns);
    if (problems.length) throw new ApiError("invalid_mapping", problems[0]);
    if (m.id) {
      if (!canBeId(upload.columns.find((c) => c.name === m.id)!))
        throw new ApiError("invalid_id_column", `id column '${m.id}' must be a string or integer column with a value in every row`, {
          details: { column: m.id },
        });
      const ids = rows?.map((r) => String(r[m.id!])) ?? [];
      const dup = ids.find((v, i) => ids.indexOf(v) !== i);
      if (dup !== undefined)
        throw new ApiError("duplicate_row_id", `row id '${dup}' appears more than once in id column '${m.id}'`, {
          details: { column: m.id, row_id: dup },
        });
    }
    const owner = versions.find((v) => v.datasetId === input.datasetId)?.projectId;
    if (owner && owner !== upload.projectId)
      throw new ApiError("dataset_conflict", `dataset id '${input.datasetId}' belongs to another project`, { details: { dataset_id: input.datasetId } });
  }

  function fraction(r: ExperimentRecord): { f: number; status: ExperimentStatus } {
    if (r.stoppedStatus && r.stoppedAt !== null) return { f: r.stoppedAt, status: r.stoppedStatus };
    const t = now() - r.createdAt;
    if (t < r.queuedMs) return { f: 0, status: "queued" };
    const f = Math.min(1, (t - r.queuedMs) / r.runMs);
    return { f, status: f >= 1 ? "completed" : "running" };
  }

  function view(r: ExperimentRecord): Experiment {
    const { f, status } = fraction(r);
    const s = snapshot(r.run, f, r.id);
    const finishedAt = status === "completed" ? r.createdAt + r.queuedMs + r.runMs : r.stoppedTime;
    return {
      id: r.id,
      projectId: r.projectId,
      datasetId: r.config.datasetId,
      name: r.config.name,
      config: r.config,
      status,
      createdAt: new Date(r.createdAt).toISOString(),
      finishedAt: finishedAt ? new Date(finishedAt).toISOString() : null,
      // stands in for ExperimentIdentity.experiment_id: a hash of the contract the run optimizes
      identityHash: hex(JSON.stringify(r.contract), 64),
      progress: {
        phase: status === "completed" ? "done" : s.phase,
        generation: s.generation,
        evaluated: s.evaluatedCount,
        spendUsd: s.spendUsd,
        elapsedSec: s.elapsedSec,
        stopReason: s.phase === "searching" || status === "queued" ? null : r.run.stopReason,
      },
      baseline: status === "queued" ? null : r.run.baseline.optimization,
      curveBaseline: status === "queued" ? null : objectiveValue(r.run.baseline.optimization, r.config.preferences),
      bestId: s.bestId,
      championId: status === "completed" ? s.championId : null,
      curve: s.curve,
      generations: s.generations,
      edges: s.edges,
      candidates: s.candidates,
      failure: r.failure,
    };
  }

  function results(r: ExperimentRecord): ResultsComparison {
    const exp = view(r);
    const { constraints, taskType } = r.config;
    const run = r.run;
    const phase = exp.progress.phase;
    const validationReady = phase === "testing" || phase === "done";
    // reporting only: the held-out test split is measured once, after the champion is fixed
    const testReady = exp.status === "completed" && exp.championId !== null;
    const sizes = run.splitSizes;
    const seed = run.input.seed;
    const result = (m: Measurement) => ({ measurement: m, constraints: checkLimits(constraints, m) });

    const method = (id: MethodResult["method"], genome: Genome, evaluated: number, opt: Measurement): MethodResult => {
      const splits: MethodResult["splits"] = { optimization: result(opt) };
      if (validationReady) splits.validation = result(measure(genome, taskType, "validation", sizes.validation, seed));
      if (testReady) splits.test = result(measure(genome, taskType, "test", sizes.test, seed));
      return { method: id, stages: stagesOf(genome), model: genome.model, evaluated, splits };
    };

    const methods: MethodResult[] = [method("fixed_baseline", run.baselineGenome, 1, run.baseline.optimization)];
    const n = exp.progress.evaluated;
    const randomBest = run.random
      .slice(0, n)
      .filter((x) => x.feasible)
      .sort((a, b) => compareKeys(b.key, a.key))[0];
    if (randomBest) methods.push(method("random_search", randomBest.genome, n, randomBest.opt));
    const wynkId = exp.championId ?? exp.bestId;
    const wynk = exp.candidates.find((c) => c.id === wynkId);
    if (wynk) methods.push(method("wynk_aco", genomeOf(wynk), n, wynk.optimization));
    return { experimentId: r.id, splitSizes: sizes, methods, testLocked: !testReady };
  }

  return {
    mode: "mock",
    experiments: "simulated",

    listProjects: () => respond("listProjects", () => projects.map((p) => project(p.id))),
    getProject: (id) => respond("getProject", () => project(id)),
    createProject: (input) =>
      respond("createProject", () => {
        const name = input.name.trim();
        if (!name) throw new ApiError("invalid_request", "name: String should have at least 1 character");
        const p: Project = {
          id: nextId("p"),
          name,
          description: input.description.trim(),
          createdAt: iso(),
          datasetCount: 0,
          experimentCount: 0,
        };
        projects.unshift(p);
        return p;
      }),

    uploadDataset: (projectId, file) =>
      respond("uploadDataset", async () => {
        project(projectId);
        if (file.bytes.size > MAX_UPLOAD_BYTES)
          throw new ApiError("payload_too_large", `body is larger than ${MAX_UPLOAD_BYTES} bytes`, { details: { limit_bytes: MAX_UPLOAD_BYTES } });
        const bytes = new Uint8Array(await file.bytes.arrayBuffer());
        if (file.format !== "csv" && file.format !== "jsonl") throw new ApiError("unsupported_format", "format must be one of: csv, jsonl");
        const inspected = inspect(bytes, file.format);
        const contentHash = await sha256(bytes);
        const id = `u-${hex(`${projectId}:${file.format}:${contentHash}`, 32)}`;
        const existing = uploads.get(id);
        if (existing) return { value: existing.upload, created: false };
        const upload: DatasetUpload = {
          id,
          projectId,
          fileName: file.name,
          format: file.format,
          contentHash,
          sizeBytes: bytes.length,
          rowCount: inspected.rows.length,
          columns: inspected.columns,
          preview: inspected.preview,
          parserVersion: "mock",
          createdAt: iso(),
        };
        uploads.set(id, { upload, rows: inspected.rows });
        return { value: upload, created: true };
      }),
    getUpload: (uploadId) => respond("getUpload", () => uploadEntry(uploadId).upload),
    registerDataset: (uploadId, input) =>
      respond("registerDataset", () => {
        const entry = uploadEntry(uploadId);
        checkRegistration(entry, input);
        const same = versionsOf(input.datasetId).find((v) => v.uploadId === uploadId && JSON.stringify(v.mapping) === JSON.stringify(input.mapping));
        if (same) return { value: same, created: false };
        const v = versionOf(entry.upload, { ...input, name: input.name.trim() }, versionsOf(input.datasetId).length + 1, iso());
        versions.push(v);
        return { value: v, created: true };
      }),
    listDatasets: (projectId) =>
      respond("listDatasets", () => {
        project(projectId);
        return datasetIdsOf(projectId).map(dataset);
      }),
    getDataset: (id) => respond("getDataset", () => dataset(id)),
    getDatasetVersion: (id, n) => respond("getDatasetVersion", () => version(id, n)),
    createSplits: (datasetId, n, plan) =>
      respond("createSplits", () => {
        const v = version(datasetId, n);
        const vbps = Math.round(plan.validationPct * 100);
        const tbps = Math.round(plan.testPct * 100);
        if (!Number.isInteger(plan.seed) || vbps < 0 || tbps < 0)
          throw new ApiError("invalid_request", "seed must be an integer and fractions 0 or more");
        if (vbps + tbps >= 10_000) throw new ApiError("invalid_request", "Value error, validation + test fractions must leave rows for optimization");
        const splitsHash = hex(`mock-splits:${v.identityHash}:${plan.seed}:${vbps}:${tbps}`, 64);
        const existing = splitSets.find((x) => x.splitsHash === splitsHash);
        if (existing) return { value: existing, created: false };
        const set: DatasetSplitSet = {
          datasetId,
          version: n,
          splitsHash,
          datasetHash: v.identityHash,
          method: "seeded_hash/1",
          plan: { validationPct: vbps / 100, testPct: tbps / 100, seed: plan.seed },
          sizes: splitSizes(v.rowCount, plan),
          createdAt: iso(),
        };
        splitSets.push(set);
        return { value: set, created: true };
      }),
    listSplits: (datasetId, n) =>
      respond("listSplits", () => {
        version(datasetId, n);
        return splitSets.filter((x) => x.datasetId === datasetId && x.version === n);
      }),
    getSplits: (datasetId, n, splitsHash) =>
      respond("getSplits", () => {
        version(datasetId, n);
        const set = splitSets.find((x) => x.datasetId === datasetId && x.version === n && x.splitsHash === splitsHash);
        if (!set) throw new ApiError("splits_not_found", `no splits ${splitsHash} for ${datasetId}`);
        return set;
      }),

    listModels: () => respond("listModels", () => MODELS),

    listExperiments: (projectId) =>
      respond("listExperiments", () => {
        project(projectId);
        return experiments
          .filter((e) => e.projectId === projectId)
          .sort((a, b) => b.createdAt - a.createdAt)
          .map(view);
      }),
    getExperiment: (id) => respond("getExperiment", () => view(record(id))),
    createExperiment: (projectId, config) =>
      respond("createExperiment", () => {
        project(projectId);
        const d = latest(dataset(config.datasetId).id)!;
        if (d.projectId !== projectId) throw new ApiError("invalid_request", "The dataset belongs to another project.");
        const errors = Object.values(validateConfig(config, d));
        if (errors.length) throw new ApiError("invalid_request", errors[0]!);
        const id = nextId("e");
        let contract: TaskContractWire;
        try {
          contract = toTaskContract(id, config, d);
        } catch (err) {
          if (err instanceof ContractMappingError) throw new ApiError("invalid_request", err.message);
          throw err;
        }
        const r: ExperimentRecord = {
          id,
          projectId,
          config: clone(config),
          contract,
          createdAt: now(),
          run: runFor(id, config, d.rowCount),
          runMs,
          queuedMs: Math.min(900, runMs * 0.05),
          stoppedAt: null,
          stoppedStatus: null,
          stoppedTime: null,
          failure: null,
        };
        experiments.push(r);
        return view(r);
      }),
    cancelExperiment: (id) =>
      respond("cancelExperiment", () => {
        const r = record(id);
        const { f, status } = fraction(r);
        if (status !== "running" && status !== "queued") throw new ApiError("invalid_request", "Only a queued or running experiment can be cancelled.");
        r.stoppedAt = f;
        r.stoppedStatus = "cancelled";
        r.stoppedTime = now();
        return view(r);
      }),
    getResults: (experimentId) => respond("getResults", () => results(record(experimentId))),

    listWorkflows: (projectId) =>
      respond("listWorkflows", () => {
        project(projectId);
        return experiments
          .filter((r) => r.projectId === projectId)
          .flatMap((r) => {
            const exp = view(r);
            const ds = latest(r.config.datasetId);
            return exp.candidates
              .filter((c) => c.state !== "candidate")
              .map((candidate) => ({
                candidate,
                experimentId: r.id,
                experimentName: r.config.name,
                datasetName: ds?.name ?? r.config.datasetId,
                objective: r.config.preferences.objective,
              }));
          })
          .sort((a, b) => Number(b.candidate.state === "champion") - Number(a.candidate.state === "champion"));
      }),
  };
}

/** A registered version of an upload. Hashes are invented stand-ins for the server's. */
function versionOf(upload: DatasetUpload, input: RegisterDataset, n: number, createdAt: string): DatasetVersion {
  const identity = JSON.stringify({ datasetId: input.datasetId, contentHash: upload.contentHash, mapping: input.mapping, n });
  return {
    datasetId: input.datasetId,
    projectId: upload.projectId,
    uploadId: upload.id,
    version: n,
    name: input.name,
    format: upload.format,
    contentHash: upload.contentHash,
    identityHash: hex(`mock-identity:${identity}`, 64),
    rowCount: upload.rowCount,
    columns: upload.columns.map(({ name, type, nullable }) => ({ name, type, nullable })),
    mapping: input.mapping,
    rowIdSource: input.mapping.id ? "column" : "generated",
    rowIdScheme: input.mapping.id ? null : "row-content-sha256/1",
    rowIdsHash: hex(`mock-row-ids:${identity}`, 64),
    createdAt,
  };
}

function runFor(seed: string, config: ExperimentConfig, rowCount: number): SimRun {
  return simulate({
    seed,
    taskType: config.taskType,
    constraints: config.constraints,
    preferences: config.preferences,
    models: config.models,
    budget: config.budget,
    splits: config.splits,
    rowCount,
  });
}

const SLOT_ORDER = ["FILTER", "EXTRACT", "REASON", "VERIFY", "SYNTHESIZE"] as const;

function genomeOf(c: Candidate): Genome {
  return { choices: SLOT_ORDER.map((k) => c.stages.find((s) => s.kind === k)?.options[0] ?? null), model: c.model };
}
