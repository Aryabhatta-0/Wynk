/*
  Test helper: the product API's real exchanges (contract/fixtures/product-api.v1.json, generated
  from api/product.py by tests/test_product_api_ui_fixtures.py) served as a `fetch`, so the live
  adapter is exercised against exactly what the Python API sends and must send exactly what it
  received.
*/
import { expect } from "vitest";
import fixture from "@/api/contract/fixtures/product-api.v1.json";

export interface Exchange {
  request: { method: string; path: string; body?: { json: unknown } | { text: string } | { size: number } };
  response: { status: number; body: unknown };
}

export interface ProductApiFixture {
  error_status: Record<string, number>;
  exchanges: Record<string, Exchange>;
}

export const FIXTURE = fixture as unknown as ProductApiFixture;

export const exchange = (name: string): Exchange => {
  const e = FIXTURE.exchanges[name];
  if (!e) throw new Error(`no exchange ${name} in the product API fixture`);
  return e;
};

/** The response body of an exchange, typed loosely for assertions. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const body = (name: string): any => exchange(name).response.body;

export const respond = (e: Exchange) =>
  new Response(JSON.stringify(e.response.body), { status: e.response.status, headers: { "Content-Type": "application/json" } });

async function sentBody(init: RequestInit | undefined): Promise<{ json: unknown } | { text: string } | { size: number } | undefined> {
  const b = init?.body;
  if (b === undefined || b === null) return undefined;
  const headers = new Headers(init?.headers);
  if (headers.get("Content-Type") === "application/json") return { json: JSON.parse(String(b)) };
  const blob = b as Blob;
  return { text: await blob.text() };
}

/**
 * A fetch that expects the named exchanges in order: each request must match the recorded method,
 * path and body (a recorded `size` body only checks the byte count), and gets the recorded response.
 */
export function replay(...names: string[]) {
  const queue = [...names];
  const calls: { url: string; init?: RequestInit }[] = [];
  const fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    calls.push({ url, init });
    const name = queue.shift();
    if (!name) throw new Error(`unexpected request ${init?.method} ${url}`);
    const e = exchange(name);
    expect(`${init?.method ?? "GET"} ${url}`, `request for ${name}`).toBe(`${e.request.method} ${e.request.path}`);
    const recorded = e.request.body;
    if (recorded && "size" in recorded) expect((init!.body as Blob).size).toBe(recorded.size);
    else expect(await sentBody(init), `body for ${name}`).toEqual(recorded);
    return respond(e);
  };
  return { fetch: fetch as typeof globalThis.fetch, calls, remaining: queue };
}

/** A fetch that answers any recorded exchange by method + path (for screens that load in any order). */
export function router(names: string[]) {
  const byRoute = new Map(names.map((n) => [`${exchange(n).request.method} ${exchange(n).request.path}`, exchange(n)]));
  const calls: string[] = [];
  const fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const route = `${init?.method ?? "GET"} ${String(input)}`;
    calls.push(route);
    const e = byRoute.get(route);
    if (!e) return new Response("not routed in test", { status: 599 });
    return respond(e);
  };
  return { fetch: fetch as typeof globalThis.fetch, calls };
}
