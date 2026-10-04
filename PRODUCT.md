# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Delegated to the agent (the user said "rest all are your choices"): Vite + React + TypeScript in
`ui/`, Tailwind v4, Motion for the intro, Phosphor icons. The chat consumes a stream of run events
(`ui/src/lib/chat.ts`); until a backend serves them, `ui/src/lib/demoEngine.ts` emits them and the
UI labels runs as demo data.

## Users

Inferred from the repository, not interviewed: the engineers building wynk, and the people they
show it to (reviewers, collaborators). They want to see *how* the optimizer reached its answer,
not just the final number.

## Product Purpose

wynk searches the space of agent workflows (ordered GATHER / FILTER / EXTRACT / REASON / VERIFY /
SYNTHESIZE stages, each with options) with ant colony optimization (MMAS), runs each candidate
end-to-end on a real model through Microsoft Agent Framework, and lets a deterministic evaluator
decide PASS / FAIL / INFEASIBLE. Success is finding a workflow that passes held-out validation
tasks cheaply, with fewer evaluations than random search.

## Positioning

The model never grades itself: every verdict comes from a deterministic evaluator that checks the
answer against ground truth and verifies every quoted evidence span against the frozen page bytes.
The UI's job is to make that visible run by run.

## Capabilities and Constraints

- Two optimizers today: `aco_mmas` and `random_search`; a run is (optimizer, seed), budget 100
  train evaluations, 2 trials per candidate.
- Task classes: A (fact sheets, `fetch`) and B (mock JSON API, `api`).
- Recorded experiments: Gemma 4 31B on A and B (5 seeds each, no stage traces), Gemma 3 27B on A
  (3 seeds, full stage traces).
- The UI is a chatbot: a question is routed to a library source, a small ant colony runs candidate
  workflows on it, results are checked without ground truth, the best ant's facts become the answer.

## Brand Commitments

Given by the user, binding:

- Name: "wynk" as the wordmark, "agent workflow optimizer" as the plain descriptor.
- Logo and mascot: the ghost from `New folder/ghost` (traced outline frames + its animation).
- Progress: the scan-line completion card from `New folder/ui` (plum card, magenta streaks with
  white tips, large white percentage, uppercase title).
- Colour: that card's magenta with white as the primary colours; white page base (user's choice).
- Live transparency: the beam-and-node workflow visualizer from `workflow-visualizer.txt`.
- ACO candidates shown as the user's BlobCard (fluid blobs + rotating glow).
- Flow: landing page, Start, the ghost crosses left to right, winks, and becomes the chat's logo.

## Evidence on Hand

Real, committed: `experiments/results/real-gemma4/{A,B}` and `experiments/results/real` (results,
per-evaluation logs, best genomes, final validation runs with evidence spans). The demo library is
the benchmark fact sheets verbatim. No users, testimonials or adoption figures exist; never invent
them.

## Product Principles

1. Show the work: every number on screen traces to a logged evaluation.
2. The evaluator decides, visibly: verdicts and evidence are first-class, not footnotes.
3. Honest comparisons: ACO is always shown next to random search on the same seed and budget.
4. Demo and live runs look the same, and the UI always says which one you are watching.
