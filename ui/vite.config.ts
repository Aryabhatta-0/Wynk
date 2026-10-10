import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { type ProxyOptions, defineConfig } from "vite";

/*
  The API answers a refused upload (e.g. 413 payload_too_large) before reading its body, drains the
  rest for a few seconds and closes. A browser only shows a response once its upload has finished,
  so the dev proxy must let that upload finish:
  - it does not pass on Connection: close, which would cut the browser off mid-upload;
  - if the API's connection closes while the browser is still sending, the proxy reads and discards
    the rest itself (the pipe to the closed connection would otherwise stall the upload forever).
  Without this the browser reports a network error, or hangs, instead of showing the API's error.
*/
const finishRefusedUploads: ProxyOptions["configure"] = (proxy) => {
  proxy.on("proxyRes", (res) => {
    delete res.headers.connection;
  });
  proxy.on("proxyReq", (proxyReq, req) => {
    proxyReq.on("close", () => {
      if (!req.complete) {
        req.unpipe(proxyReq);
        req.resume();
      }
    });
  });
};

/*
  #32: every tenant endpoint needs an API key. The dev proxy adds `Authorization: Bearer
  $WYNK_API_KEY` server-side when that variable is set, so the key never reaches browser code,
  local storage or the bundle.
*/
const apiKey = process.env.WYNK_API_KEY;
const withApiKey: ProxyOptions["configure"] = (proxy, options) => {
  finishRefusedUploads?.(proxy, options);
  if (apiKey) proxy.on("proxyReq", (proxyReq) => proxyReq.setHeader("Authorization", `Bearer ${apiKey}`));
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  // the dev server forwards /api to the Wynk backend: `python -m api.chat` (chat + product API v1)
  // on 8787 by default, or any product API via WYNK_API_PROXY (the E2E suite runs `python -m api.product`)
  server: {
    port: 5173,
    proxy: {
      "/api": { target: process.env.WYNK_API_PROXY ?? "http://127.0.0.1:8787", changeOrigin: true, configure: withApiKey },
    },
  },
});
