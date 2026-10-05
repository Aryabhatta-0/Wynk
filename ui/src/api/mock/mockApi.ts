/*
  MOCK ONLY. An in-memory implementation of WynkApi. State lives for the page session and
  resets on reload; nothing is persisted or sent anywhere.
*/
import { validateConfig, validateMapping } from "@/lib/validation";
import { ApiError, type WynkApi } from "../client";
import type { Candidate, Dataset, Experiment, ExperimentConfig, ExperimentStatus, MethodResult, Project, ResultsComparison, SplitId } from "../types";
import { DATASETS, EXPERIMENTS, MODELS, PROJECTS } from "./fixtures";
import { type Genome, type SimRun, checkConstraints, measure, simulate, snapshot, stagesOf } from "./simulate";

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

const clone = <T>(v: T): T => structuredClone(v);

const RESET_NOTE = " Mock data resets when the page reloads, so anything created before a reload is gone.";

export function createMockApi(options: MockOptions = {}): WynkApi {
  const latency = options.latencyMs ?? 180;
  const runMs = options.runMs ?? 16_000;
  const failOn = new Set(options.failOn ?? []);
  const now = options.now ?? Date.now;

  const projects: Project[] = clone(PROJECTS);
  const datasets: Dataset[] = clone(DATASETS);
  const experiments: ExperimentRecord[] = [];
  let seq = 0;
  const nextId = (prefix: string) => `${prefix}-${(now() + seq++).toString(36)}`;

  const rowsOf = (datasetId: string) => datasets.find((d) => d.id === datasetId)?.rowCount ?? 1000;

  for (const seed of EXPERIMENTS) {
    const created = now() - seed.createdDaysAgo * 86_400_000;
    experiments.push({
      id: seed.id,
      projectId: seed.projectId,
      config: seed.config,
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

  async function respond<T>(method: string, produce: () => T): Promise<T> {
    if (latency > 0) await new Promise((r) => setTimeout(r, latency));
    if (failOn.has(method) || failOn.has("*")) throw new ApiError("unavailable", `Mock failure injected for ${method}.`);
    return clone(produce());
  }

  const project = (id: string) => {
    const p = projects.find((x) => x.id === id);
    if (!p) throw new ApiError("not_found", `This project does not exist.${RESET_NOTE}`);
    return {
      ...p,
      datasetCount: datasets.filter((d) => d.projectId === id).length,
      experimentCount: experiments.filter((e) => e.projectId === id).length,
    };
  };
  const dataset = (id: string) => {
    const d = datasets.find((x) => x.id === id);
    if (!d) throw new ApiError("not_found", `This dataset does not exist.${RESET_NOTE}`);
    return d;
  };
  const record = (id: string) => {
    const r = experiments.find((x) => x.id === id);
    if (!r) throw new ApiError("not_found", `This experiment does not exist.${RESET_NOTE}`);
    return r;
  };

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
      progress: {
        phase: status === "completed" ? "done" : s.phase,
        generation: s.generation,
        evaluated: s.evaluatedCount,
        spendUsd: s.spendUsd,
        elapsedSec: s.elapsedSec,
        stopReason: s.phase === "searching" || status === "queued" ? null : r.run.stopReason,
      },
      baseline: status === "queued" ? null : r.run.baseline.optimization,
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
    const testReady = exp.status === "completed" && exp.championId !== null;
    const sizes = run.splitSizes;
    const seed = run.input.seed;

    const method = (id: MethodResult["method"], genome: Genome, evaluated: number, opt: MethodResult["splits"]["optimization"]): MethodResult => {
      const splits: MethodResult["splits"] = { optimization: opt };
      if (validationReady) splits.validation = measure(genome, taskType, "validation", sizes.validation, seed);
      if (testReady) splits.test = measure(genome, taskType, "test", sizes.test, seed);
      const checkedOn = splits.test ?? splits.validation ?? opt!;
      return {
        method: id,
        stages: stagesOf(genome),
        model: genome.model,
        evaluated,
        splits,
        constraints: checkConstraints(checkedOn, genome.model, constraints),
      };
    };

    const methods: MethodResult[] = [method("fixed_baseline", run.baselineGenome, 1, run.baseline.optimization)];
    const n = exp.progress.evaluated;
    const randomBest = run.random
      .slice(0, n)
      .filter((x) => x.feasible)
      .sort((a, b) => b.score - a.score)[0];
    if (randomBest) methods.push(method("random_search", randomBest.genome, n, randomBest.opt));
    const wynkId = exp.championId ?? exp.bestId;
    const wynk = exp.candidates.find((c) => c.id === wynkId);
    if (wynk) methods.push(method("wynk_aco", genomeOf(wynk), n, wynk.optimization));
    return { experimentId: r.id, splitSizes: sizes as Record<SplitId, number>, methods, testLocked: !testReady };
  }

  return {
    mode: "mock",

    listProjects: () => respond("listProjects", () => projects.map((p) => project(p.id))),
    getProject: (id) => respond("getProject", () => project(id)),
    createProject: (input) =>
      respond("createProject", () => {
        const name = input.name.trim();
        if (!name) throw new ApiError("invalid", "Name the project.");
        if (projects.some((p) => p.name.toLowerCase() === name.toLowerCase()))
          throw new ApiError("invalid", "A project with this name already exists.");
        const p: Project = {
          id: nextId("p"),
          name,
          description: input.description.trim(),
          createdAt: new Date(now()).toISOString(),
          datasetCount: 0,
          experimentCount: 0,
        };
        projects.unshift(p);
        return p;
      }),

    listDatasets: (projectId) =>
      respond("listDatasets", () => {
        project(projectId);
        return datasets.filter((d) => d.projectId === projectId);
      }),
    getDataset: (id) => respond("getDataset", () => dataset(id)),
    createDataset: (projectId, draft) =>
      respond("createDataset", () => {
        project(projectId);
        if (!draft.name.trim()) throw new ApiError("invalid", "Name the dataset.");
        if (!draft.columns.length) throw new ApiError("invalid", "The file has no columns.");
        const errors = validateMapping(
          draft.mapping,
          draft.columns.map((c) => c.name),
        );
        const d: Dataset = {
          ...draft,
          name: draft.name.trim(),
          id: nextId("d"),
          projectId,
          createdAt: new Date(now()).toISOString(),
          status: errors.length ? "needs_mapping" : "ready",
        };
        datasets.unshift(d);
        return d;
      }),
    updateMapping: (id, mapping) =>
      respond("updateMapping", () => {
        const d = dataset(id);
        const errors = validateMapping(
          mapping,
          d.columns.map((c) => c.name),
        );
        d.mapping = mapping;
        d.status = errors.length ? "needs_mapping" : "ready";
        return d;
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
        const d = dataset(config.datasetId);
        if (d.projectId !== projectId) throw new ApiError("invalid", "The dataset belongs to another project.");
        if (d.status !== "ready") throw new ApiError("invalid", "Finish the dataset's column mapping first.");
        const errors = Object.values(validateConfig(config));
        if (errors.length) throw new ApiError("invalid", errors[0]!);
        const id = nextId("e");
        const r: ExperimentRecord = {
          id,
          projectId,
          config: clone(config),
          createdAt: now(),
          run: runFor(id, config, d.rowCount ?? 1000),
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
        if (status !== "running" && status !== "queued") throw new ApiError("invalid", "Only a queued or running experiment can be cancelled.");
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
            const ds = datasets.find((d) => d.id === r.config.datasetId);
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

function runFor(seed: string, config: ExperimentConfig, rowCount: number): SimRun {
  return simulate({
    seed,
    taskType: config.taskType,
    constraints: config.constraints,
    objective: config.preferences.objective,
    budget: config.budget,
    splits: config.splits,
    rowCount,
  });
}

const SLOT_ORDER = ["FILTER", "EXTRACT", "REASON", "VERIFY", "SYNTHESIZE"] as const;

function genomeOf(c: Candidate): Genome {
  return { choices: SLOT_ORDER.map((k) => c.stages.find((s) => s.kind === k)?.options[0] ?? null), model: c.model };
}
