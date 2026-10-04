import type { Ant, Quote, RunEvent } from "./chat";
import { LIBRARY, type Source } from "./library";

/*
  Demo engine: emits the same events a wynk backend streams, with realistic timing, so the
  interface can be built and reviewed without a model. The workflows are real ones from the
  recorded experiments (the elite is the best ant colony workflow on fact sheets; explorer 2
  fails exactly as the jev gather source does in those runs). Answers quote the source
  verbatim. Swap `runQuery` for a fetch of the backend stream to go live.
*/

export const DEMO = true;

const FIELDS: { re: RegExp; name: string; type: string; line: RegExp }[] = [
  { re: /\b(ceo|chief executive|led by|boss|head)\b/i, name: "ceo", type: "string", line: /chief executive/i },
  { re: /\b(employees?|staff|headcount|people|workforce)\b/i, name: "employees", type: "integer", line: /employees/i },
  { re: /\b(revenue|sales|turnover|income)\b/i, name: "revenue", type: "number", line: /revenue/i },
  { re: /\b(founded|founding|established|started|year)\b/i, name: "founded", type: "integer", line: /founded/i },
  { re: /\b(hq|headquarter\w*|based|located|city)\b/i, name: "hq", type: "string", line: /headquartered/i },
  { re: /\b(products?|services?|sell|make|offer)\b/i, name: "products", type: "string_list", line: /products and services/i },
  { re: /\b(fiscal|financial year|fy)\b/i, name: "fy_end", type: "date", line: /fiscal year/i },
  { re: /\b(office|offices|branch)\b/i, name: "offices", type: "string", line: /office/i },
];

const ANTS: Ant[] = [
  {
    id: 1,
    role: "elite",
    trail: 0.86,
    stages: [
      { kind: "GATHER", options: ["fetch", "parallel-4"] },
      { kind: "EXTRACT", options: ["direct"] },
      { kind: "SYNTHESIZE", options: ["cite evidence"] },
      { kind: "VERIFY", options: ["evidence span", "retry-1"] },
    ],
  },
  {
    id: 2,
    role: "explorer",
    trail: 0.41,
    stages: [
      { kind: "GATHER", options: ["fetch", "sequential"] },
      { kind: "FILTER", options: ["keyword chunk"] },
      { kind: "EXTRACT", options: ["schema guided"] },
      { kind: "SYNTHESIZE", options: ["cite evidence"] },
    ],
  },
  {
    id: 3,
    role: "explorer",
    trail: 0.22,
    stages: [
      { kind: "GATHER", options: ["jev", "parallel-2"] },
      { kind: "EXTRACT", options: ["cot"] },
      { kind: "VERIFY", options: ["self consistency", "retry-1"] },
      { kind: "SYNTHESIZE", options: ["direct"] },
    ],
  },
];

function route(query: string): { source: Source; matched: string } | null {
  const q = query.toLowerCase();
  let best: { source: Source; matched: string; score: number } | null = null;
  for (const s of LIBRARY) {
    const words = s.name.toLowerCase().replace("&", " ").split(/\s+/).filter((w) => w.length > 2);
    const hits = words.filter((w) => q.includes(w));
    if (hits.length && (!best || hits.length > best.score)) best = { source: s, matched: hits.join(" "), score: hits.length };
  }
  return best;
}

function sentences(source: Source): { page: string; text: string }[] {
  return Object.entries(source.pages).flatMap(([page, body]) =>
    body
      .split("\n")
      .slice(1)
      .map((text) => ({ page, text: text.trim() }))
      .filter((s) => s.text),
  );
}

const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      clearTimeout(t);
      reject(new DOMException("stopped", "AbortError"));
    });
  });

const jitter = (base: number) => base * (0.75 + Math.random() * 0.5);

export async function runQuery(query: string, emit: (e: RunEvent) => void, signal: AbortSignal): Promise<void> {
  await sleep(jitter(650), signal);
  const hit = route(query);
  if (!hit) {
    emit({
      type: "refuse",
      message:
        "I could not match that to anything in my library, so I did not run a workflow. Ask about one of the companies below, for example who leads Quillon Labs.",
    });
    return;
  }
  const { source, matched } = hit;
  emit({ type: "route", source: { id: source.id, name: source.name }, matched, pages: Object.keys(source.pages) });

  await sleep(jitter(600), signal);
  let fields = FIELDS.filter((f) => f.re.test(query));
  if (!fields.length) fields = [FIELDS[0], FIELDS[3], FIELDS[4]]; // a general question: the headline facts
  emit({ type: "plan", fields: fields.map(({ name, type }) => ({ name, type })) });

  await sleep(jitter(500), signal);
  emit({ type: "ants", ants: ANTS });

  // the colony runs in parallel; each ant walks its own stages and reports what it spent
  const runAnt = async (ant: Ant): Promise<{ id: number; completed: boolean; tokens: number }> => {
    let spent = 0;
    for (let i = 0; i < ant.stages.length; i++) {
      const stage = ant.stages[i];
      emit({ type: "stage", ant: ant.id, index: i, status: "running" });
      await sleep(jitter(stage.kind === "EXTRACT" || stage.kind === "SYNTHESIZE" ? 1500 : 650), signal);
      if (ant.id === 3 && stage.kind === "GATHER") {
        emit({ type: "stage", ant: ant.id, index: i, status: "failed", message: "gather source 'jev' is not available (MVP)" });
        return { id: ant.id, completed: false, tokens: spent };
      }
      const tokens = Math.round(jitter(stage.kind === "EXTRACT" ? 380 + ant.id * 70 : stage.kind === "SYNTHESIZE" ? 170 + ant.id * 20 : 0));
      spent += tokens;
      emit({ type: "stage", ant: ant.id, index: i, status: "ok", tokens, seconds: Number(jitter(1.4).toFixed(1)) });
    }
    return { id: ant.id, completed: true, tokens: spent };
  };
  const runs = await Promise.all(ANTS.map(runAnt));

  // score and pick exactly as api/chat.py does: verified answers first, then fewer tokens
  const CAP = 2000;
  await sleep(jitter(500), signal);
  const scored = [];
  for (const r of runs) {
    const score = r.completed ? 1 + 0.1 * Math.max(0, 1 - r.tokens / CAP) : 0;
    scored.push({ ...r, score });
    emit({ type: "score", ant: r.id, completed: r.completed, evidence: r.completed, tokens: r.tokens, score: Number(score.toFixed(3)) });
    await sleep(220, signal);
  }
  const best = scored.reduce((a, b) => (b.score > a.score || (b.score === a.score && b.tokens < a.tokens) ? b : a));

  await sleep(jitter(450), signal);
  emit({ type: "pick", ant: best.id, reason: "Every quote verified on the page, at the lowest token cost." });

  const lines = sentences(source);
  const quotes: Quote[] = [];
  for (const f of fields) {
    const s = lines.find((l) => f.line.test(l.text));
    if (s && !quotes.some((q) => q.text === s.text)) quotes.push(s);
  }
  const answer = quotes.length
    ? `According to the ${source.name} fact sheet: ${quotes.map((q) => q.text).join(" ")}`
    : `The ${source.name} fact sheet does not say. I looked at ${Object.keys(source.pages).join(", ")} and found no line about that.`;

  await sleep(300, signal);
  for (const word of answer.split(/(?<=\s)/)) {
    emit({ type: "answer", delta: word });
    await sleep(28, signal);
  }
  emit({ type: "done", quotes });
}

export const SUGGESTIONS = [
  "Who is the CEO of Quillon Labs, and how many people work there?",
  "Where is Zephyr Dynamics headquartered, and when was it founded?",
  "What does Helio Pay sell?",
  "What was Brasswick Foods' latest revenue?",
];
