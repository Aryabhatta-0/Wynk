import { Check } from "@phosphor-icons/react";
import { Fragment } from "react";
import { cn } from "@/lib/utils";

export const FLOW = ["Project", "Dataset", "Configure", "Optimize", "Compare", "Champion"] as const;
export type FlowStep = (typeof FLOW)[number];

/** Where this screen sits in Project → Dataset → Configure → Optimize → Compare → Champion. */
export function FlowSteps({ current }: { current: FlowStep }) {
  const at = FLOW.indexOf(current);
  return (
    <ol className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-xs" aria-label="Workflow">
      {FLOW.map((step, i) => (
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
