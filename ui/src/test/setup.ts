import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";
import { setApi } from "@/api";

class NoopObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
// jsdom has no layout observers; components that measure themselves just skip measuring
globalThis.ResizeObserver ??= NoopObserver as unknown as typeof ResizeObserver;
globalThis.IntersectionObserver ??= NoopObserver as unknown as typeof IntersectionObserver;

afterEach(() => {
  cleanup();
  setApi(null);
});
