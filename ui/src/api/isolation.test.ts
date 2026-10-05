import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { replay } from "@/test/productApi";

/*
  Live mode never loads mock code or fixtures; mock mode never touches the network.
*/

// Any attempt to load a mock module fails this file's tests.
vi.mock("./mock/mockApi", () => {
  throw new Error("live mode loaded the mock adapter");
});
vi.mock("./mock/fixtures", () => {
  throw new Error("live mode loaded mock fixtures");
});

// tests run from ui/ (npm test, CI); jsdom's import.meta.url is not a file URL
const SRC = resolve(process.cwd(), "src");

/** Runtime (non type-only) imports of a module, resolved to files under src/. */
function runtimeImports(file: string): string[] {
  const text = readFileSync(file, "utf-8");
  const specs = [...text.matchAll(/^import\s+(?!type\b)[^;]*?from\s+"([^"]+)";/gms), ...text.matchAll(/^export\s+(?!type\b)[^;]*?from\s+"([^"]+)";/gms)].map(
    (m) => m[1],
  );
  const dynamic = [...text.matchAll(/import\("([^"]+)"\)/g)].map((m) => `dynamic:${m[1]}`);
  return [...specs, ...dynamic]
    .filter((s) => s.startsWith(".") || s.startsWith("@/") || s.startsWith("dynamic:"))
    .map((s) => {
      const isDynamic = s.startsWith("dynamic:");
      const spec = isDynamic ? s.slice("dynamic:".length) : s;
      const base = spec.startsWith("@/") ? resolve(SRC, spec.slice(2)) : resolve(dirname(file), spec);
      const path = /\.\w+$/.test(base) ? base : `${base}.ts`;
      return isDynamic ? `dynamic:${path}` : path;
    });
}

function staticClosure(entry: string): Set<string> {
  const seen = new Set<string>();
  const stack = [entry];
  while (stack.length) {
    const file = stack.pop()!;
    if (seen.has(file)) continue;
    seen.add(file);
    for (const dep of runtimeImports(file)) if (!dep.startsWith("dynamic:")) stack.push(dep);
  }
  return seen;
}

describe("live mode is isolated from mock code", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("the live adapter and the contract layer import nothing from src/api/mock", () => {
    const closure = [...staticClosure(resolve(SRC, "api/live.ts"))];
    expect(closure.some((f) => f.includes(`${resolve(SRC, "api/mock")}`))).toBe(false);
    expect(closure.map((f) => f.slice(SRC.length + 1).replaceAll("\\", "/")).sort()).toEqual([
      "api/client.ts",
      "api/contract/decode.ts",
      "api/contract/mapping.ts",
      "api/contract/rules.ts",
      "api/live.ts",
    ]);
  });

  it("the adapter selector reaches the mock only through a lazy import", () => {
    const index = resolve(SRC, "api/index.ts");
    const deps = runtimeImports(index);
    const mock = (f: string) => f.includes(resolve(SRC, "api/mock"));
    expect(deps.filter((d) => !d.startsWith("dynamic:")).some(mock)).toBe(false);
    expect(deps.filter((d) => d.startsWith("dynamic:")).some(mock)).toBe(true);
  });

  it("by default the app talks to the product API and never loads the mock", async () => {
    const { api, setApi } = await import("./index");
    setApi(null);
    vi.stubEnv("VITE_WYNK_API", "");
    const r = replay("list_projects");
    vi.stubGlobal("fetch", r.fetch);
    window.history.replaceState(null, "", "/projects");
    expect(api().mode).toBe("live");
    expect(api().experiments).toBe("unavailable");
    const projects = await api().listProjects();
    expect(projects).toHaveLength(1);
    expect(r.calls.map((c) => c.url)).toEqual(["/api/v1/projects"]);
    setApi(null);
  });

  it("a live failure stays a failure: no fallback to mock data", async () => {
    const { api, setApi } = await import("./index");
    setApi(null);
    vi.stubEnv("VITE_WYNK_API", "");
    vi.stubGlobal("fetch", () => Promise.reject(new TypeError("Failed to fetch")));
    window.history.replaceState(null, "", "/projects");
    await expect(api().listProjects()).rejects.toMatchObject({ code: "network_error" });
    expect(api().mode).toBe("live");
    setApi(null);
  });
});
