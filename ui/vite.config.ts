import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  // `python -m api.chat` runs questions on Gemma; the dev server forwards /api to it
  server: { port: 5173, proxy: { "/api": { target: "http://127.0.0.1:8787", changeOrigin: true } } },
});
