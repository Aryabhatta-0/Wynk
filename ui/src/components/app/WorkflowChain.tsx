import { CaretRight } from "@phosphor-icons/react";
import { Fragment } from "react";
import type { WorkflowStage } from "@/api";
import { STAGE_LABEL, optionLabel, workflowText } from "@/lib/format";
import { cn } from "@/lib/utils";

/** A workflow on one line, for tables: stage chips joined by carets. */
export function WorkflowChain({ stages, model, className }: { stages: WorkflowStage[]; model?: string; className?: string }) {
  return (
    <div
      className={cn("flex flex-wrap items-center gap-1", className)}
      aria-label={workflowText(stages) + (model ? ` on ${model}` : "")}
      role="group"
    >
      {stages.map((s, i) => (
        <Fragment key={`${i}-${s.kind}`}>
          {i > 0 && <CaretRight size={10} className="text-ink-soft" aria-hidden="true" />}
          <span
            className="inline-flex items-baseline gap-1 rounded-md border border-line bg-white px-1.5 py-0.5 text-xs leading-tight whitespace-nowrap"
            aria-hidden="true"
          >
            <span className="font-semibold text-ink">{STAGE_LABEL[s.kind]}</span>
            <span className="font-mono text-[11px] text-ink-soft">{s.options.map(optionLabel).join(", ")}</span>
          </span>
        </Fragment>
      ))}
      {model && (
        <span className="ml-1 rounded-md bg-wash px-1.5 py-0.5 font-mono text-[11px] whitespace-nowrap text-magenta-ink" aria-hidden="true">
          {model}
        </span>
      )}
    </div>
  );
}
