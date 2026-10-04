import { CheckCircle, CircleNotch, Crown, MinusCircle, XCircle } from "@phosphor-icons/react";
import { STAGE_LABEL, type AntState } from "@/lib/chat";
import { cn } from "@/lib/utils";
import { BlobCard } from "../ui/BlobCard";
import { AntTrail } from "./AntTrail";

interface Props {
  ant: AntState;
  picked: number | undefined;
  focused: boolean;
  onFocus: () => void;
}

/** One ant of the colony: ACO at work in the header, its stages and score below. */
export function AntCard({ ant, picked, focused, onFocus }: Props) {
  const running = ant.status.includes("running");
  const failed = ant.status.includes("failed");
  const chosen = picked === ant.id;
  const lost = picked !== undefined && !chosen;

  return (
    <button
      type="button"
      onClick={onFocus}
      aria-pressed={focused}
      aria-label={`Ant ${ant.id}, show its workflow`}
      className={cn(
        "block w-full cursor-pointer rounded-[22px] text-left transition-[opacity,transform] duration-300 ease-out active:scale-[0.99]",
        lost && "opacity-55 saturate-50",
        focused && "outline-2 outline-offset-4 outline-magenta",
      )}
    >
      <BlobCard
        header={
          <div>
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="font-display text-2xl leading-none font-bold tracking-tight text-ink">Ant {ant.id}</p>
                <p className="mt-1.5 text-xs font-medium text-ink/80">{ant.role === "elite" ? "Elite: best recorded workflow" : "Explorer: sampled from the trail"}</p>
              </div>
              {chosen ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-white px-2.5 py-1 text-xs font-semibold text-magenta-ink shadow-sm">
                  <Crown size={12} weight="fill" aria-hidden="true" /> Chosen
                </span>
              ) : (
                <span className="tnum rounded-full bg-white/85 px-2.5 py-1 font-mono text-xs text-ink">trail {ant.trail.toFixed(2)}</span>
              )}
            </div>
            <div className="mt-4">
              <AntTrail antId={ant.id} stages={ant.stages} status={ant.status} trail={ant.trail} chosen={chosen} />
            </div>
          </div>
        }
      >
        <ol className="space-y-1 px-5 pt-1 pb-3">
          {ant.stages.map((s, i) => {
            const st = ant.status[i];
            return (
              <li key={i} className={cn("flex min-w-0 items-center gap-2 text-[13px]", st === "skipped" && "opacity-45")}>
                <StatusIcon status={st} />
                <span className="font-medium">{STAGE_LABEL[s.kind]}</span>
                <span className="truncate text-xs text-ink-soft">{s.options.join(", ")}</span>
              </li>
            );
          })}
        </ol>
        <div className="flex min-h-11 items-center justify-between gap-3 border-t border-line px-5 py-2.5 text-xs">
          {ant.score ? (
            <>
              <span className="min-w-0 truncate text-ink-soft">
                {!ant.score.completed
                  ? (ant.messages.find(Boolean) ?? "did not finish")
                  : ant.score.evidence
                    ? `quotes verified, ${ant.tokens} tokens`
                    : "finished, quotes not found"}
              </span>
              <span className="tnum font-mono text-sm font-semibold text-ink">{ant.score.score.toFixed(3)}</span>
            </>
          ) : failed ? (
            <span className="text-ink-soft">{ant.messages.find(Boolean)}</span>
          ) : (
            <span className="text-ink-soft">{running ? "walking its workflow" : "waiting"}</span>
          )}
        </div>
      </BlobCard>
    </button>
  );
}

function StatusIcon({ status }: { status: string }) {
  if (status === "running") return <CircleNotch size={16} weight="bold" className="animate-spin text-magenta-ink motion-reduce:animate-none" aria-label="running" />;
  if (status === "ok") return <CheckCircle size={16} weight="fill" className="text-magenta" aria-label="done" />;
  if (status === "failed") return <XCircle size={16} weight="fill" className="text-ink" aria-label="failed" />;
  if (status === "skipped") return <MinusCircle size={16} className="text-ink-soft" aria-label="not reached" />;
  return <span className="mx-[3px] size-2.5 rounded-full border border-line-strong" aria-label="waiting" />;
}
