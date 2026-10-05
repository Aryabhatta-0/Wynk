import { Flask, Info } from "@phosphor-icons/react";
import type { ReactNode } from "react";
import { api } from "@/api";
import { Panel } from "./ui";

const REPO = "https://github.com/Aryabhatta-0/Wynk";

/** The issues real experiment execution on uploaded datasets is blocked on. */
export const EXPERIMENT_BLOCKERS = [
  { n: 20, title: "Make TaskContract the authority in the optimization pipeline" },
  { n: 21, title: "Generalize the workflow grammar for uploaded datasets" },
  { n: 22, title: "Generic EvaluationSpec-driven evaluator execution" },
  { n: 23, title: "Optimize uploaded datasets: fixed vs random vs ACO (equal budgets)" },
];

/**
 * Experiment screens (configure, run, results, workflows). With the live API they show that the
 * experiment backend does not exist yet and make no request; with the mock they say, on every
 * screen, that the run is simulated.
 */
export function ExperimentGate({ children }: { children: ReactNode }) {
  if (api().experiments === "unavailable") return <ExperimentBackendRequired />;
  return (
    <div className="space-y-4">
      <SimulatedNotice />
      {children}
    </div>
  );
}

export function ExperimentBackendRequired() {
  return (
    <Panel>
      <div data-testid="experiment-backend-required" className="mx-auto flex max-w-[64ch] flex-col items-center px-4 py-8 text-center">
        <span className="grid size-11 place-items-center rounded-full border border-line text-magenta-ink" aria-hidden="true">
          <Flask size={20} weight="duotone" />
        </span>
        <h3 className="mt-3 font-sans text-base font-semibold tracking-normal">Experiments need the experiment backend, which is not built yet</h3>
        <p className="mt-1 text-sm leading-relaxed text-ink-soft">
          The Wynk API stores projects, datasets and splits. It does not run optimization experiments yet, so this screen has nothing real to
          show and sends no request. Running experiments on uploaded datasets is blocked on:
        </p>
        <ul className="mt-3 space-y-1 text-left text-sm">
          {EXPERIMENT_BLOCKERS.map((b) => (
            <li key={b.n}>
              <a href={`${REPO}/issues/${b.n}`} target="_blank" rel="noreferrer" className="font-medium text-magenta-ink hover:underline">
                #{b.n}
              </a>{" "}
              <span className="text-ink-soft">{b.title}</span>
            </li>
          ))}
        </ul>
        <p className="mt-4 text-xs text-ink-soft">
          To explore the flow with invented data, open the{" "}
          <a href="/projects?api=mock" className="font-medium text-magenta-ink hover:underline">
            mock demo
          </a>
          . It is labelled as simulated on every screen.
        </p>
      </div>
    </Panel>
  );
}

export function SimulatedNotice() {
  return (
    <div
      role="note"
      data-testid="simulated-notice"
      className="flex items-start gap-2.5 rounded-[10px] border border-dashed border-warn/50 bg-warn-wash px-4 py-2.5 text-xs text-warn"
    >
      <Info size={16} weight="fill" className="mt-px shrink-0" aria-hidden="true" />
      <p>
        <span className="font-semibold">Simulated experiment (mock data).</span> No model was called and no optimization ran: candidates, scores,
        costs, latencies and champions here are invented by the mock adapter. Real experiments are blocked on issues #20–#23.
      </p>
    </div>
  );
}
