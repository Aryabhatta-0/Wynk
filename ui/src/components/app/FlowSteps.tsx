import { Check } from "@phosphor-icons/react";
import { Fragment } from "react";
import { cn } from "@/lib/utils";

export const FLOW = ["Project", "Dataset", "Configure", "Optimize", "Compare", "Champion"] as const;
export type FlowStep = (typeof FLOW)[number];

/** The dataset flow against the product API; it ends on the dataset page. */
export const DATASET_FLOW = ["Upload", "Inspect", "Map columns", "Register", "Create splits"] as const;
export type DatasetStep = (typeof DATASET_FLOW)[number];

/** Where this screen sits in Project → Dataset → Configure → Optimize → Compare → Champion. */
export function FlowSteps({ current }: { current: FlowStep }) {
  return <Steps steps={FLOW} current={current} label="Workflow" />;
}

/** Where this screen sits in Upload → Inspect → Map columns → Register → Create splits; `done` marks the end. */
export function DatasetSteps({ current }: { current: DatasetStep | "done" }) {
  return <Steps steps={DATASET_FLOW} current={current} label="Dataset steps" />;
}

function Steps<S extends string>({ steps, current, label }: { steps: readonly S[]; current: S | "done"; label: string }) {
  const at = current === "done" ? steps.length : steps.indexOf(current);
  return (
    <ol className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-xs" aria-label={label}>
      {steps.map((step, i) => (
        <Fragment key={step}>
          {i > 0 && <li aria-hidden="true" className={cn("h-px w-4", i <= at ? "bg-magenta-ink" : "bg-line-strong")} />}
          <li
            aria-current={i === at ? "step" : undefined}
            className={cn(
              "inline-flex items-center gap-1 rounded-full px-2 py-0.5 whitespace-nowrap",
              i < at && "text-magenta-ink",
              i === at && "bg-magenta-ink font-semibold text-white",
              i > at && "text-ink-soft",
            )}
          >
            {i < at && <Check size={11} weight="bold" aria-hidden="true" />}
            {step}
          </li>
        </Fragment>
      ))}
    </ol>
  );
}
