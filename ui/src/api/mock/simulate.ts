/*
  MOCK ONLY. A deterministic stand-in for an optimization run, so the product screens can be
  built and tested before the backend exists. It is not Wynk's optimizer or evaluator and its
  numbers mean nothing: quality, cost and latency come from a made-up response surface plus
  seeded noise. The search is a small max-min ant system over the same stage vocabulary as
  `core/stages.py`, so the screens see realistic shapes (convergence, reinforced edges).
*/
import type {
  Candidate,
  ConstraintCheck,
  CurvePoint,
  EdgeSignal,
  ExperimentPhase,
  GenerationSummary,
  HardConstraints,
  Measurement,
  Objective,
  SearchBudget,
  SplitId,
  SplitPlan,
  StopReason,
  TaskType,
  WorkflowStage,
} from "../types";
import { hash32, hex, noise, rng } from "./random";

/* ------------------------------------------------------------- response surface */

interface Effect {
  q: number; // quality contribution
  tokens: number; // per example
  ms: number; // per example
  task?: Partial<Record<TaskType, number>>; // task-specific quality contribution
}

type Slot = { kind: WorkflowStage["kind"]; optional: boolean; options: Record<string, Effect> };

const SLOTS: Slot[] = [
  {
    kind: "FILTER",
    optional: true,
    options: {
      section_select: { q: 0.02, tokens: 150, ms: 300 },
      keyword_chunk: { q: 0.01, tokens: 80, ms: 150 },
    },
  },
  {
    kind: "EXTRACT",
    optional: false,
    options: {
      direct: { q: 0, tokens: 400, ms: 900 },
      schema_guided: { q: 0.05, tokens: 520, ms: 1100, task: { structured_extraction: 0.1 } },
      cot: { q: 0.04, tokens: 900, ms: 2200, task: { question_answering: 0.08 } },
    },
  },
  {
    kind: "REASON",
    optional: true,
    options: {
      single: { q: 0.03, tokens: 350, ms: 900, task: { question_answering: 0.05 } },
      decompose: { q: 0.03, tokens: 800, ms: 2100, task: { question_answering: 0.07, classification: -0.02 } },
    },
  },
  {
    kind: "VERIFY",
    optional: true,
    options: {
      schema_check: { q: 0.02, tokens: 120, ms: 250, task: { structured_extraction: 0.06 } },
      evidence_span: { q: 0.04, tokens: 300, ms: 700 },
      self_consistency: { q: 0.05, tokens: 1600, ms: 3200 },
    },
  },
  {
    kind: "SYNTHESIZE",
    optional: false,
    options: {
      direct: { q: 0, tokens: 200, ms: 500 },
      cite_evidence: { q: 0.01, tokens: 300, ms: 700, task: { question_answering: 0.03 } },
    },
  },
];

interface ModelProfile {
  q: number;
  usdPerMTok: number;
  speed: number; // latency multiplier
}

export const MODEL_PROFILES: Record<string, ModelProfile> = {
  "gemma-3-12b-it": { q: -0.05, usdPerMTok: 0.05, speed: 0.7 },
  "gemma-3-27b-it": { q: 0, usdPerMTok: 0.1, speed: 1 },
  "gemma-4-31b-it": { q: 0.03, usdPerMTok: 0.14, speed: 1.15 },
  "llama-3.3-70b-instruct": { q: 0.02, usdPerMTok: 0.25, speed: 1.3 },
};

const BASE: Record<TaskType, number> = {
  classification: 0.7,
  structured_extraction: 0.62,
  question_answering: 0.58,
};

/** A genome here: one option (or null when skipped) per slot, plus the model. */
export interface Genome {
  choices: (string | null)[];
  model: string;
}

export const genomeKey = (g: Genome) => `${g.choices.map((c) => c ?? "-").join("/")}@${g.model}`;

export function stagesOf(g: Genome): WorkflowStage[] {
  return g.choices.flatMap((c, i) => (c ? [{ kind: SLOTS[i].kind, options: [c] }] : []));
}

export const BASELINE_CHOICES: (string | null)[] = [null, "direct", null, null, "direct"];

const SPLIT_DRIFT: Record<SplitId, number> = { optimization: 0, validation: -0.012, test: -0.018 };

export function measure(g: Genome, task: TaskType, split: SplitId, n: number, seed: string): Measurement {
  let lift = 0;
  let tokens = 0;
  let ms = 0;
  g.choices.forEach((c, i) => {
    if (!c) return;
    const e = SLOTS[i].options[c];
    lift += e.q + (e.task?.[task] ?? 0);
    tokens += e.tokens;
    ms += e.ms;
  });
  const m = MODEL_PROFILES[g.model] ?? MODEL_PROFILES["gemma-3-27b-it"];
  const base = BASE[task] + m.q;
  const ceiling = 0.97;
  const key = `${seed}|${genomeKey(g)}|${split}`;
  const quality = base + (ceiling - base) * (1 - Math.exp(-Math.max(-0.2, lift) / 0.17));
  const q = clamp(quality + SPLIT_DRIFT[split] + noise(`${key}|q`) * 0.014, 0, 1);
  const costPer1k = ((tokens * 1000 * m.usdPerMTok) / 1e6) * (1 + noise(`${key}|c`) * 0.05);
  const latencyP95Ms = ms * m.speed * 1.35 * (1 + noise(`${key}|l`) * 0.08);
  return { quality: round(q, 4), costPer1k: round(costPer1k, 4), latencyP95Ms: Math.round(latencyP95Ms), n };
}

/* ------------------------------------------------------------- constraints and objective */

export function checkConstraints(m: Measurement, model: string, c: HardConstraints): ConstraintCheck[] {
  const checks: ConstraintCheck[] = [];
  if (c.minQuality !== null) checks.push({ key: "minQuality", ok: m.quality >= c.minQuality, observed: m.quality, limit: c.minQuality });
  if (c.maxCostPer1k !== null) checks.push({ key: "maxCostPer1k", ok: m.costPer1k <= c.maxCostPer1k, observed: m.costPer1k, limit: c.maxCostPer1k });
  if (c.maxLatencyP95Ms !== null)
    checks.push({ key: "maxLatencyP95Ms", ok: m.latencyP95Ms <= c.maxLatencyP95Ms, observed: m.latencyP95Ms, limit: c.maxLatencyP95Ms });
  checks.push({ key: "allowedModels", ok: c.allowedModels.includes(model), observed: model, limit: c.allowedModels.join(", ") });
  return checks;
}

/** The value plotted on the learning curve for an objective. */
export function objectiveMetric(m: Measurement, objective: Objective): number {
  if (objective === "cost") return m.costPer1k;
  if (objective === "latency") return m.latencyP95Ms;
  return m.quality;
}

export const lowerIsBetter = (objective: Objective) => objective === "cost" || objective === "latency";

/** Ranking score among feasible candidates; higher is better. Mock ranking only. */
export function objectiveScore(m: Measurement, objective: Objective, ref: Measurement): number {
  switch (objective) {
    case "quality":
      return m.quality - 1e-4 * (m.costPer1k / ref.costPer1k);
    case "cost":
      return -m.costPer1k / ref.costPer1k;
    case "latency":
      return -m.latencyP95Ms / ref.latencyP95Ms;
    case "balanced":
      return m.quality - 0.05 * Math.log2(m.costPer1k / ref.costPer1k) - 0.05 * Math.log2(m.latencyP95Ms / ref.latencyP95Ms);
  }
}

/* ------------------------------------------------------------- the run */

export interface SimInput {
  seed: string;
  taskType: TaskType;
  constraints: HardConstraints;
  objective: Objective;
  budget: SearchBudget;
  splits: SplitPlan;
  rowCount: number;
}

interface Evaluated {
  genome: Genome;
  generation: number;
  opt: Measurement;
  feasible: boolean;
  score: number;
}

export interface SimRun {
  input: SimInput;
  splitSizes: Record<SplitId, number>;
  baselineGenome: Genome;
  baseline: Record<SplitId, Measurement>;
  /** unique candidates in the order the search evaluated them */
  evaluated: Evaluated[];
  generations: GenerationSummary[];
  edgesByGeneration: EdgeSignal[][];
  /** cumulative spend and simulated seconds after each candidate */
  spend: number[];
  seconds: number[];
  random: { genome: Genome; opt: Measurement; feasible: boolean; score: number }[];
  validatedKeys: string[];
  championKey: string | null;
  stopReason: StopReason;
}

const ANTS_MIN = 4;
const RHO = 0.12;
const TAU_MIN = 0.15;
const TAU_MAX = 3;
const PARALLEL = 64;
const VALIDATE_TOP = 5;

export function simulate(input: SimInput): SimRun {
  const { seed, taskType, constraints, objective, budget } = input;
  const models = constraints.allowedModels.length ? constraints.allowedModels : ["gemma-3-27b-it"];
  const splitSizes = sizes(input.rowCount, input.splits);
  const nOpt = splitSizes.optimization;
  const random = rng(hash32(`${seed}|search`));

  const baselineGenome: Genome = { choices: BASELINE_CHOICES, model: models[0] };
  const baseline = {
    optimization: measure(baselineGenome, taskType, "optimization", nOpt, seed),
    validation: measure(baselineGenome, taskType, "validation", splitSizes.validation, seed),
    test: measure(baselineGenome, taskType, "test", splitSizes.test, seed),
  };
  const ref = baseline.optimization;
  const evalOne = (genome: Genome, generation: number): Evaluated => {
    const opt = measure(genome, taskType, "optimization", nOpt, seed);
    const feasible = checkConstraints(opt, genome.model, constraints).every((c) => c.ok);
    return { genome, generation, opt, feasible, score: objectiveScore(opt, objective, ref) };
  };

  // pheromone per slot option, per model, and per edge between consecutive chosen stages
  const tau = SLOTS.map((s) => {
    const t: Record<string, number> = Object.fromEntries(Object.keys(s.options).map((k) => [k, 1]));
    if (s.optional) t["-"] = 3; // cold start favours lean workflows; stages are added as they pay off
    return t;
  });
  const tauModel: Record<string, number> = Object.fromEntries(models.map((m) => [m, 1]));
  const tauEdge = new Map<string, number>();

  const ants = Math.max(ANTS_MIN, Math.ceil(budget.maxCandidates / Math.max(1, budget.maxGenerations)));
  const seen = new Map<string, Evaluated>();
  const evaluated: Evaluated[] = [];
  const generations: GenerationSummary[] = [];
  const edgesByGeneration: EdgeSignal[][] = [];
  const spend: number[] = [];
  const seconds: number[] = [];
  let spent = 0;
  let elapsed = 0;
  let best: Evaluated | null = null;
  let stopReason: StopReason = "generations";

  search: for (let gen = 1; gen <= budget.maxGenerations; gen++) {
    const proposals: Evaluated[] = [];
    let novel = 0;
    for (let a = 0; a < ants; a++) {
      const choices = tau.map((t) => {
        const pick = roulette(t, random);
        return pick === "-" ? null : pick;
      });
      const genome: Genome = { choices, model: roulette(tauModel, random) };
      const key = genomeKey(genome);
      let e = seen.get(key);
      if (!e) {
        if (evaluated.length >= budget.maxCandidates) {
          stopReason = "candidates";
          break search;
        }
        e = evalOne(genome, gen);
        const cost = (e.opt.costPer1k * nOpt) / 1000;
        const secs = ((e.opt.latencyP95Ms / 1000) * nOpt) / PARALLEL;
        if (spent + cost > budget.maxSpendUsd) {
          stopReason = "spend";
          break search;
        }
        if (elapsed + secs > budget.maxDurationMin * 60) {
          stopReason = "duration";
          break search;
        }
        spent += cost;
        elapsed += secs;
        seen.set(key, e);
        evaluated.push(e);
        spend.push(round(spent, 4));
        seconds.push(Math.round(elapsed));
        novel++;
      }
      proposals.push(e);
    }
    if (!proposals.length) break;

    // evaporate, then reinforce this generation's best and the best so far (max-min ant system)
    const genBest = pickBest(proposals);
    if (genBest && (!best || better(genBest, best))) best = genBest;
    for (const t of [...tau, tauModel]) for (const k of Object.keys(t)) t[k] *= 1 - RHO;
    for (const k of tauEdge.keys()) tauEdge.set(k, tauEdge.get(k)! * (1 - RHO));
    for (const winner of [genBest, best]) {
      if (!winner) continue;
      const amount = 0.3 + Math.max(0, winner.opt.quality - ref.quality) * 3;
      winner.genome.choices.forEach((c, i) => (tau[i][c ?? "-"] += amount));
      tauModel[winner.genome.model] += amount;
      edgesOf(winner.genome).forEach((k) => tauEdge.set(k, (tauEdge.get(k) ?? 0) + amount));
    }
    for (const t of [...tau, tauModel]) for (const k of Object.keys(t)) t[k] = clamp(t[k], TAU_MIN, TAU_MAX);

    const qualities = proposals.map((p) => p.opt.quality);
    generations.push({
      generation: gen,
      evaluated: novel,
      bestQuality: round(Math.max(...qualities), 4),
      meanQuality: round(qualities.reduce((s, q) => s + q, 0) / qualities.length, 4),
      agreement: round(agreementOf(proposals.map((p) => p.genome)), 3),
      novel,
    });
    edgesByGeneration.push(topEdges(tauEdge, edgesByGeneration.at(-2)));
  }

  // random search: same number of unique candidates, sampled uniformly
  const randomRng = rng(hash32(`${seed}|random`));
  const randomSeen = new Set<string>();
  const randomRuns: SimRun["random"] = [];
  for (let guard = 0; randomRuns.length < evaluated.length && guard < evaluated.length * 20; guard++) {
    const genome: Genome = {
      choices: SLOTS.map((s) => {
        const opts = [...Object.keys(s.options), ...(s.optional ? [null] : [])];
        return opts[Math.floor(randomRng() * opts.length)];
      }),
      model: models[Math.floor(randomRng() * models.length)],
    };
    const key = genomeKey(genome);
    if (randomSeen.has(key)) continue;
    randomSeen.add(key);
    const e = evalOne(genome, 0);
    randomRuns.push({ genome, opt: e.opt, feasible: e.feasible, score: e.score });
  }

  // validation: the top feasible candidates are re-measured; the champion must stay feasible there
  const ranked = evaluated.filter((e) => e.feasible).sort((a, b) => b.score - a.score);
  const validated = ranked.slice(0, VALIDATE_TOP);
  const championKey =
    validated
      .map((e) => {
        const val = measure(e.genome, taskType, "validation", splitSizes.validation, seed);
        const ok = checkConstraints(val, e.genome.model, constraints).every((c) => c.ok);
        return { e, ok, score: objectiveScore(val, objective, ref) };
      })
      .filter((v) => v.ok)
      .sort((a, b) => b.score - a.score)
      .map((v) => genomeKey(v.e.genome))[0] ?? null;

  return {
    input,
    splitSizes,
    baselineGenome,
    baseline,
    evaluated,
    generations,
    edgesByGeneration,
    spend,
    seconds,
    random: randomRuns,
    validatedKeys: validated.map((e) => genomeKey(e.genome)),
    championKey,
    stopReason,
  };
}

/* ------------------------------------------------------------- what the screen sees at a moment */

export interface Snapshot {
  phase: ExperimentPhase;
  generation: number;
  evaluatedCount: number;
  spendUsd: number;
  elapsedSec: number;
  candidates: Candidate[];
  bestId: string | null;
  championId: string | null;
  curve: CurvePoint[];
  generations: GenerationSummary[];
  edges: EdgeSignal[];
}

const SEARCH_END = 0.84;
const VALIDATE_END = 0.94;

/** The run as it looks a fraction `f` of the way through (0..1). */
export function snapshot(run: SimRun, f: number, idPrefix: string): Snapshot {
  const totalGens = run.generations.length;
  const phase: ExperimentPhase = f >= 1 ? "done" : f >= VALIDATE_END ? "testing" : f >= SEARCH_END ? "validating" : "searching";
  const shownGens = phase === "searching" ? Math.floor((f / SEARCH_END) * totalGens) : totalGens;
  const shown = run.evaluated.filter((e) => e.generation <= shownGens);
  const n = shown.length;
  const { objective, constraints, taskType } = run.input;
  const validatedShown = phase === "testing" || phase === "done";

  const candidates: Candidate[] = shown.map((e, i) => {
    const key = genomeKey(e.genome);
    const isValidated = validatedShown && run.validatedKeys.includes(key);
    const validation = isValidated ? measure(e.genome, taskType, "validation", run.splitSizes.validation, run.input.seed) : null;
    const state = phase === "done" && key === run.championKey ? "champion" : isValidated ? "validated" : "candidate";
    const checks = checkConstraints(validation ?? e.opt, e.genome.model, constraints);
    return {
      id: `${idPrefix}-c${i + 1}`,
      genomeHash: hex(key),
      stages: stagesOf(e.genome),
      model: e.genome.model,
      generation: e.generation,
      state,
      optimization: e.opt,
      validation,
      constraints: checks,
      feasible: checks.every((c) => c.ok),
    };
  });

  const bestIndex = shown.reduce((bi, e, i) => (e.feasible && (bi < 0 || e.score > shown[bi].score) ? i : bi), -1);
  const champIndex = phase === "done" && run.championKey ? shown.findIndex((e) => genomeKey(e.genome) === run.championKey) : -1;

  return {
    phase,
    generation: shownGens,
    evaluatedCount: n,
    spendUsd: n ? run.spend[n - 1] : 0,
    elapsedSec: n ? run.seconds[n - 1] : 0,
    candidates,
    bestId: bestIndex >= 0 ? candidates[bestIndex].id : null,
    championId: champIndex >= 0 ? candidates[champIndex].id : null,
    curve: curve(run, n, objective),
    generations: run.generations.slice(0, shownGens),
    edges: shownGens > 0 ? run.edgesByGeneration[shownGens - 1] : [],
  };
}

function curve(run: SimRun, n: number, objective: Objective): CurvePoint[] {
  const pts: CurvePoint[] = [];
  const step = Math.max(1, Math.ceil(n / 120));
  let wynk: Evaluated | null = null;
  let rand: SimRun["random"][number] | null = null;
  for (let i = 0; i < n; i++) {
    const w = run.evaluated[i];
    if (w.feasible && (!wynk || w.score > wynk.score)) wynk = w;
    const r = run.random[i];
    if (r && r.feasible && (!rand || r.score > rand.score)) rand = r;
    if (i % step === 0 || i === n - 1) {
      pts.push({
        evaluated: i + 1,
        wynk: wynk ? objectiveMetric(wynk.opt, objective) : null,
        random: rand ? objectiveMetric(rand.opt, objective) : null,
      });
    }
  }
  return pts;
}

/* ------------------------------------------------------------- helpers */

export function sizes(rows: number, s: SplitPlan): Record<SplitId, number> {
  const optimization = Math.round((rows * s.optimization) / 100);
  const validation = Math.round((rows * s.validation) / 100);
  return { optimization, validation, test: Math.max(0, rows - optimization - validation) };
}

/** Mean, over the stage and model decisions, of the share of ants that made the most common choice. */
function agreementOf(genomes: Genome[]): number {
  const decisions = [...SLOTS.map((_, i) => (g: Genome) => g.choices[i] ?? "-"), (g: Genome) => g.model];
  const shares = decisions.map((pick) => {
    const counts = new Map<string, number>();
    for (const g of genomes) counts.set(pick(g), (counts.get(pick(g)) ?? 0) + 1);
    return Math.max(...counts.values()) / genomes.length;
  });
  return shares.reduce((s, v) => s + v, 0) / shares.length;
}

function edgesOf(g: Genome): string[] {
  const nodes = g.choices.flatMap((c, i) => (c ? [`${SLOTS[i].kind} · ${c}`] : []));
  return nodes.slice(1).map((b, i) => `${nodes[i]}→${b}`);
}

function topEdges(tauEdge: Map<string, number>, earlier: EdgeSignal[] | undefined): EdgeSignal[] {
  const entries = [...tauEdge.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
  const max = entries[0]?.[1] ?? 1;
  return entries.map(([k, v]) => {
    const [from, to] = k.split("→");
    const strength = round(v / max, 3);
    const before = earlier?.find((e) => e.from === from && e.to === to)?.strength;
    const trend = before === undefined || strength - before > 0.05 ? "rising" : before - strength > 0.05 ? "falling" : "steady";
    return { from, to, strength, trend };
  });
}

function roulette(weights: Record<string, number>, random: () => number): string {
  const keys = Object.keys(weights);
  const total = keys.reduce((s, k) => s + weights[k], 0);
  let r = random() * total;
  for (const k of keys) {
    r -= weights[k];
    if (r <= 0) return k;
  }
  return keys[keys.length - 1];
}

function pickBest(es: Evaluated[]): Evaluated | null {
  return es.reduce<Evaluated | null>((b, e) => (!b || better(e, b) ? e : b), null);
}

/** Feasible beats infeasible; then the objective score. */
function better(a: Evaluated, b: Evaluated): boolean {
  if (a.feasible !== b.feasible) return a.feasible;
  return a.score > b.score;
}

function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}

function round(v: number, places: number) {
  const p = 10 ** places;
  return Math.round(v * p) / p;
}
