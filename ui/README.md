# wynk UI

The chat interface: a landing page, the ghost intro, then a chat where every question runs an ant
colony of agent workflows and shows each step.

```bash
cd ui
npm install
npm run dev      # http://localhost:5173
npm run build    # typecheck + production bundle in dist/
```

## Where the data comes from

`src/lib/chat.ts` defines the event stream one question produces (`route`, `plan`, `ants`,
`stage`, `score`, `pick`, `answer`, `done`) and the reducer that turns it into what the screen
shows. The completion card's percentage is computed from those events: routing, planning, every
stage of every ant, judging, answering.

Until a backend streams these events, `src/lib/demoEngine.ts` emits them with realistic timing from
the benchmark fact sheets, and the header shows a "demo data" tag. To go live, replace `runQuery`
with a `fetch` of the backend's event stream and set `DEMO = false`.

## Design sources

- Completion card: `New folder/ui` (scan-line canvas), `src/components/SyncCard.tsx`
- Ghost logo and animation: `New folder/ghost`, `src/ghost/`
- Workflow visualizer: `workflow-visualizer.txt`, `src/components/WorkflowGraph.tsx`
- ACO cards: the BlobCard snippet, `src/components/ui/BlobCard.tsx` (with `FluidBlobs` and
  `GlowEffect` written here, in CSS)
