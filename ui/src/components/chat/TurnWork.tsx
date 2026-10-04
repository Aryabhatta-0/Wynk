import { CheckCircle, Circle, CircleNotch, Prohibit } from "@phosphor-icons/react";
import { useState } from "react";
import { activityOf, progressOf, type Turn } from "../../lib/chat";
import { cn } from "@/lib/utils";
import { SyncCard } from "../SyncCard";
import { WorkflowGraph } from "../WorkflowGraph";
import { AntCard } from "./AntCard";

const TYPE_LABEL: Record<string, string> = {
  string: "text",
  integer: "whole number",
  number: "number",
  date: "date",
  string_list: "list",
};

/** Everything wynk did for one question, as it happens. */
export function TurnWork({ turn }: { turn: Turn }) {
  const [focusedAnt, setFocusedAnt] = useState<number | null>(null);
  const runningAnt = turn.ants.find((a) => a.status.includes("running"))?.id;
  const shownId = focusedAnt ?? turn.picked ?? runningAnt ?? turn.ants[0]?.id;
  const shown = turn.ants.find((a) => a.id === shownId);

  const sourceLine = turn.source ? `${turn.source.name} fact sheet` : turn.phase === "refused" ? "No matching source" : "Reading your question";

  return (
    <div className="space-y-8">
      <SyncCard progress={progressOf(turn)} title={activityOf(turn)} subtitle={sourceLine} label="Progress through this workflow" />

      <section aria-labelledby={`steps-${turn.id}`}>
        <h3 id={`steps-${turn.id}`} className="text-lg font-semibold">
          What is happening
        </h3>
        <ol className="mt-4 space-y-3.5">
          <Step state={turn.source ? "done" : turn.phase === "refused" ? "stopped" : "active"} title="Find the source">
            {turn.source ? (
              <>
                Matched “{turn.source.matched}” to the {turn.source.name} fact sheet ({turn.source.pages.join(", ")}).
              </>
            ) : turn.phase === "refused" ? (
              turn.note
            ) : (
              "Matching your words against the library."
            )}
          </Step>
          {turn.phase !== "refused" && (
            <>
              <Step state={turn.fields ? "done" : turn.source ? "active" : "pending"} title="Decide what to look for">
                {turn.fields
                  ? `Looking for ${turn.fields.map((f) => `${f.name} (${TYPE_LABEL[f.type] ?? f.type})`).join(", ")}.`
                  : "Turning the question into named facts."}
              </Step>
              <Step state={turn.ants.length ? (turn.picked !== undefined || turn.phase === "judging" ? "done" : "active") : "pending"} title="Send the ant colony">
                {turn.ants.length
                  ? `${turn.ants.length} ants walk different workflows at once: 1 elite from the recorded search, ${turn.ants.length - 1} explorers sampled from the pheromone trail.`
                  : "Each ant follows a different workflow to the answer."}
              </Step>
              <Step state={turn.picked !== undefined ? "done" : turn.phase === "judging" ? "active" : "pending"} title="Check every result">
                Without ground truth: did the run finish, does every quote exist word for word on the page, how many tokens
                did it use.
              </Step>
              <Step state={turn.picked !== undefined ? "done" : "pending"} title="Keep the best ant">
                {turn.picked !== undefined ? `Ant ${turn.picked}: ${turn.pickReason} Its path gets more pheromone for next time.` : "The winner's path is reinforced."}
              </Step>
              <Step state={turn.phase === "done" ? "done" : turn.phase === "answering" ? "active" : "pending"} title="Write the answer">
                Gemma phrases the reply from the chosen ant's facts and quotes, nothing else.
              </Step>
            </>
          )}
        </ol>
      </section>

      {turn.ants.length > 0 && (
        <>
          <section aria-labelledby={`colony-${turn.id}`}>
            <div className="flex items-baseline justify-between gap-4">
              <h3 id={`colony-${turn.id}`} className="text-lg font-semibold">
                The colony
              </h3>
              <p className="text-sm text-ink-soft">Pick an ant to see its workflow</p>
            </div>
            <div className="mt-4 grid gap-4 sm:grid-cols-2 2xl:grid-cols-3">
              {turn.ants.map((a) => (
                <AntCard key={a.id} ant={a} picked={turn.picked} focused={a.id === shownId} onFocus={() => setFocusedAnt(a.id)} />
              ))}
            </div>
          </section>

          {shown && (
            <section aria-labelledby={`graph-${turn.id}`} className="rounded-[24px] border border-line bg-paper p-5 sm:p-6">
              <div className="mb-6 flex flex-wrap items-baseline justify-between gap-2">
                <h3 id={`graph-${turn.id}`} className="text-lg font-semibold">
                  Ant {shown.id}'s workflow
                </h3>
                <p className="tnum text-sm text-ink-soft">{shown.tokens > 0 ? `${shown.tokens} tokens so far` : "starting"}</p>
              </div>
              <WorkflowGraph
                stages={shown.stages}
                status={shown.status}
                answered={turn.phase === "done" && turn.picked === shown.id}
                failed={shown.status.includes("failed")}
              />
            </section>
          )}
        </>
      )}
    </div>
  );
}

function Step({ state, title, children }: { state: "done" | "active" | "pending" | "stopped"; title: string; children: React.ReactNode }) {
  return (
    <li className={cn("grid grid-cols-[1.25rem_1fr] gap-3", state === "pending" && "text-ink-soft")}>
      <span className="pt-0.5">
        {state === "done" ? (
          <CheckCircle size={18} weight="fill" className="text-magenta-ink" aria-label="done" />
        ) : state === "active" ? (
          <CircleNotch size={18} weight="bold" className="animate-spin text-magenta-ink motion-reduce:animate-none" aria-label="in progress" />
        ) : state === "stopped" ? (
          <Prohibit size={18} className="text-ink" aria-label="stopped" />
        ) : (
          <Circle size={18} className="text-line-strong" aria-label="not started" />
        )}
      </span>
      <div>
        <p className={cn("text-sm font-semibold", state !== "pending" && "text-ink")}>{title}</p>
        <p className="mt-0.5 text-sm leading-relaxed text-ink-soft">{children}</p>
      </div>
    </li>
  );
}
