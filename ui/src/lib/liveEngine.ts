import type { RunEvent } from "./chat";

/** Runs a question on the wynk backend (`python -m api.chat`) and forwards its event stream. */
export async function runQuery(query: string, emit: (e: RunEvent) => void, signal: AbortSignal): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(res.status === 404 || res.status >= 500 ? "the wynk backend is not running (python -m api.chat)" : `HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut = buffer.indexOf("\n\n");
    while (cut >= 0) {
      const frame = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const data = frame
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trim())
        .join("");
      if (data) emit(JSON.parse(data) as RunEvent);
      cut = buffer.indexOf("\n\n");
    }
  }
}

export interface Health {
  ok: boolean;
  model: string | null;
  problem: string | null;
}

export async function health(): Promise<Health> {
  try {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error(String(r.status));
    return (await r.json()) as Health;
  } catch {
    return { ok: false, model: null, problem: "the wynk backend is not running (python -m api.chat)" };
  }
}
