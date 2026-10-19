// The event stream one question produces, and the turn state the UI builds from it.
// A backend streams exactly these events; until one is wired, lib/demoEngine.ts emits them.

import type { StageKind } from "@/api/types";

export interface Stage {
  kind: StageKind; // the stage vocabulary of core/stages.py
  options: string[]; // e.g. ["fetch", "parallel-4"]
}

export type StageStatus = "idle" | "running" | "ok" | "failed" | "skipped";

// Where a GATHER stage reads its evidence (core/stages.py GatherSource). Each is shown distinctly:
// only websearch is live; the others read the frozen benchmark snapshot.
export type GatherSource = "fetch" | "api" | "jev" | "websearch";

export const GATHER_SOURCE_LABEL: Record<GatherSource, string> = {
  fetch: "frozen pages",
  api: "mock API",
  jev: "JEV browsing",
  websearch: "live web search",
};

/** Plain words for a GATHER source; an unknown source is shown as sent. */
export function gatherSourceLabel(source: string): string {
  return GATHER_SOURCE_LABEL[source as GatherSource] ?? source;
}

export function isLiveSource(source: string | undefined): boolean {
  return source === "websearch";
}

/** The source a GATHER stage reads (its first option), or undefined for any other stage. */
export function gatherSourceOf(stage: Stage): string | undefined {
  return stage.kind === "GATHER" ? stage.options[0] : undefined;
}

/** Which evidence sources the colony's ants read, in first-seen order, with the ants using each.
 *  A source the server reported for a stage wins over the one its plan names. */
export function evidenceSources(
  ants: (Pick<Ant, "id" | "stages"> & { sources?: (string | undefined)[] })[],
): { source: string; label: string; ants: number[] }[] {
  const out: { source: string; label: string; ants: number[] }[] = [];
  for (const a of ants) {
    for (const [i, s] of a.stages.entries()) {
      const source = a.sources?.[i] ?? gatherSourceOf(s);
      if (source === undefined) continue;
      const row = out.find((r) => r.source === source);
      if (!row) out.push({ source, label: gatherSourceLabel(source), ants: [a.id] });
      else if (!row.ants.includes(a.id)) row.ants.push(a.id);
    }
  }
  return out;
}

export interface Ant {
  id: number;
  role: "elite" | "explorer";
  // which recorded workflow an elite is: the routed template task's own champion, or the suite's
  elite?: "task" | "suite";
  elite_task?: string; // the template task id of a task elite
  stages: Stage[];
  trail: number; // mean pheromone on the ant's path, 0..1
}

export interface Quote {
  page: string;
  text: string;
}

export type RunEvent =
  | { type: "route"; source: { id: string; name: string }; matched: string; pages: string[] }
  | { type: "plan"; fields: { name: string; type: string }[] }
  | { type: "ants"; ants: Ant[] }
  // tool_calls: what the stage spent on tools (pages read, API calls, JEV browser actions, web
  // search requests); source: the evidence source of a GATHER stage
  | {
      type: "stage";
      ant: number;
      index: number;
      status: "running" | "ok" | "failed";
      tokens?: number;
      tool_calls?: number;
      seconds?: number;
      message?: string;
      source?: GatherSource;
    }
  | { type: "score"; ant: number; completed: boolean; evidence: boolean; tokens: number; score: number }
  | { type: "pick"; ant: number; reason: string }
  | { type: "answer"; delta: string }
  | { type: "done"; quotes: Quote[] }
  | { type: "refuse"; message: string }
  | { type: "error"; message: string };

export interface AntState extends Ant {
  status: StageStatus[];
  messages: (string | undefined)[];
  toolCalls: number[]; // per stage
  sources: (string | undefined)[]; // per stage: the GATHER source the server reported
  tokens: number;
  score?: { completed: boolean; evidence: boolean; score: number };
}

export interface Turn {
  id: number;
  question: string;
  source?: { id: string; name: string; matched: string; pages: string[] };
  fields?: { name: string; type: string }[];
  ants: AntState[];
  picked?: number;
  pickReason?: string;
  answer: string;
  quotes: Quote[];
  phase: "routing" | "planning" | "running" | "judging" | "answering" | "done" | "refused" | "error";
  note?: string; // refusal or error text
}

export function newTurn(id: number, question: string): Turn {
  return { id, question, ants: [], answer: "", quotes: [], phase: "routing" };
}

export function applyEvent(t: Turn, e: RunEvent): Turn {
  switch (e.type) {
    case "route":
      return { ...t, source: { ...e.source, matched: e.matched, pages: e.pages }, phase: "planning" };
    case "plan":
      return { ...t, fields: e.fields };
    case "ants":
      return {
        ...t,
        phase: "running",
        ants: e.ants.map((a) => ({
          ...a,
          status: a.stages.map(() => "idle"),
          messages: a.stages.map(() => undefined),
          toolCalls: a.stages.map(() => 0),
          sources: a.stages.map(gatherSourceOf),
          tokens: 0,
        })),
      };
    case "stage": {
      const ants = t.ants.map((a) => {
        if (a.id !== e.ant) return a;
        const status = [...a.status];
        const messages = [...a.messages];
        const toolCalls = [...a.toolCalls];
        const sources = [...a.sources];
        status[e.index] = e.status;
        messages[e.index] = e.message;
        toolCalls[e.index] += e.tool_calls ?? 0;
        if (e.source !== undefined) sources[e.index] = e.source;
        if (e.status === "failed") for (let i = e.index + 1; i < status.length; i++) status[i] = "skipped";
        return { ...a, status, messages, toolCalls, sources, tokens: a.tokens + (e.tokens ?? 0) };
      });
      return { ...t, ants };
    }
    case "score":
      return {
        ...t,
        phase: "judging",
        ants: t.ants.map((a) => (a.id === e.ant ? { ...a, tokens: e.tokens, score: { completed: e.completed, evidence: e.evidence, score: e.score } } : a)),
      };
    case "pick":
      return { ...t, picked: e.ant, pickReason: e.reason, phase: "answering" };
    case "answer":
      return { ...t, answer: t.answer + e.delta, phase: "answering" };
    case "done":
      return { ...t, quotes: e.quotes, phase: "done" };
    case "refuse":
      return { ...t, phase: "refused", note: e.message };
    case "error":
      return { ...t, phase: "error", note: e.message };
  }
}

/**
 * Completion of the workflow actually being followed: routing, planning, every stage of every
 * ant (finished, failed or skipped), judging, the answer. Nothing is estimated.
 */
export function progressOf(t: Turn): number {
  const stageUnits = t.ants.reduce((n, a) => n + a.stages.length, 0);
  const total = 2 + (stageUnits || 6) + 1 + 1;
  let done = 0;
  if (t.source || t.phase === "refused") done += 1;
  if (t.fields) done += 1;
  for (const a of t.ants) done += a.status.filter((s) => s === "ok" || s === "failed" || s === "skipped").length;
  if (t.picked !== undefined) done += 1;
  if (t.phase === "done") done += 1;
  if (t.phase === "refused" || t.phase === "error") return 100;
  return Math.round((done / total) * 100);
}

export const STAGE_LABEL: Record<Stage["kind"], string> = {
  GATHER: "Gather",
  FILTER: "Filter",
  EXTRACT: "Extract",
  REASON: "Reason",
  VERIFY: "Verify",
  SYNTHESIZE: "Synthesize",
  DIRECT: "Direct",
  CONFIDENCE_GATE: "Confidence gate",
};

/** What the run is doing right now, in a few words (the completion card's title). */
export function activityOf(t: Turn): string {
  switch (t.phase) {
    case "routing":
      return "Finding a source";
    case "planning":
      return "Planning the facts";
    case "running": {
      for (const a of t.ants) {
        const i = a.status.indexOf("running");
        if (i >= 0) return `Ant ${a.id} ${STAGE_LABEL[a.stages[i].kind].toLowerCase()}ing`.replace("eing", "ing");
      }
      return "Running workflows";
    }
    case "judging":
      return "Checking answers";
    case "answering":
      return "Writing the answer";
    case "done":
      return "Answer ready";
    case "refused":
      return "Not in the library";
    case "error":
      return "Run stopped";
  }
}
