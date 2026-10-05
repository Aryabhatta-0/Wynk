// The event stream one question produces, and the turn state the UI builds from it.
// A backend streams exactly these events; until one is wired, lib/demoEngine.ts emits them.

import type { StageKind } from "@/api/types";

export interface Stage {
  kind: StageKind; // the stage vocabulary of core/stages.py
  options: string[]; // e.g. ["fetch", "parallel-4"]
}

export type StageStatus = "idle" | "running" | "ok" | "failed" | "skipped";

export interface Ant {
  id: number;
  role: "elite" | "explorer";
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
  | { type: "stage"; ant: number; index: number; status: "running" | "ok" | "failed"; tokens?: number; seconds?: number; message?: string }
  | { type: "score"; ant: number; completed: boolean; evidence: boolean; tokens: number; score: number }
  | { type: "pick"; ant: number; reason: string }
  | { type: "answer"; delta: string }
  | { type: "done"; quotes: Quote[] }
  | { type: "refuse"; message: string }
  | { type: "error"; message: string };

export interface AntState extends Ant {
  status: StageStatus[];
  messages: (string | undefined)[];
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
        ants: e.ants.map((a) => ({ ...a, status: a.stages.map(() => "idle"), messages: a.stages.map(() => undefined), tokens: 0 })),
      };
    case "stage": {
      const ants = t.ants.map((a) => {
        if (a.id !== e.ant) return a;
        const status = [...a.status];
        const messages = [...a.messages];
        status[e.index] = e.status;
        messages[e.index] = e.message;
        if (e.status === "failed") for (let i = e.index + 1; i < status.length; i++) status[i] = "skipped";
        return { ...a, status, messages, tokens: a.tokens + (e.tokens ?? 0) };
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
