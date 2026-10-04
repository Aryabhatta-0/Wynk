import { ArrowUp, CaretDown, CheckCircle, ListChecks, Quotes, Stop } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";
import { activityOf, type Turn } from "../../lib/chat";
import { cn } from "@/lib/utils";
import { GhostLogo } from "../GhostFigure";
import { TurnColony, TurnSteps } from "./TurnWork";

interface ThreadProps {
  turns: Turn[];
  selected: number | null;
  onSelect: (id: number) => void;
}

export function Thread({ turns, selected, onSelect }: ThreadProps) {
  const end = useRef<HTMLDivElement>(null);
  const last = turns.at(-1);
  // keep the newest message in view while it streams
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [turns.length, last?.answer.length, last?.phase]);

  return (
    <>
    <ol className="space-y-8" aria-label="Conversation">
      {turns.map((t) => (
        <li key={t.id} className="space-y-4">
          <div className="flex justify-end">
            <p className="max-w-[85%] rounded-[22px] rounded-br-md bg-magenta-wash px-4 py-3 text-[0.9375rem] leading-relaxed text-ink">{t.question}</p>
          </div>
          <AssistantMessage turn={t} selected={selected === t.id} onSelect={() => onSelect(t.id)} />
        </li>
      ))}
    </ol>
    <div ref={end} />
    </>
  );
}

function AssistantMessage({ turn, selected, onSelect }: { turn: Turn; selected: boolean; onSelect: () => void }) {
  const [openInline, setOpenInline] = useState(false);
  const [stepsOpen, setStepsOpen] = useState(true);
  const working = !["done", "refused", "error"].includes(turn.phase);
  const picked = turn.ants.find((a) => a.id === turn.picked);

  return (
    <div className="grid grid-cols-[2.25rem_1fr] gap-3">
      <span className="grid size-9 place-items-center rounded-full border border-line bg-white"><GhostLogo className="h-5 w-auto" /></span>
      <div className="min-w-0">
        {working && (
          <p className="flex items-center gap-2 pt-1.5 text-sm text-ink-soft" role="status">
            <span className="relative flex size-2">
              <span className="absolute inline-flex size-full animate-ping rounded-full bg-magenta opacity-60 motion-reduce:animate-none" />
              <span className="relative inline-flex size-2 rounded-full bg-magenta-ink" />
            </span>
            {activityOf(turn)}
          </p>
        )}

        {turn.phase !== "refused" && (
          <details
            open={stepsOpen}
            onToggle={(e) => setStepsOpen((e.currentTarget as HTMLDetailsElement).open)}
            className="group mt-2 mb-3 rounded-2xl border border-line"
          >
            <summary className="flex min-h-10 cursor-pointer list-none items-center gap-2 px-4 text-sm font-medium text-ink select-none [&::-webkit-details-marker]:hidden">
              <ListChecks size={16} className="text-magenta-ink" aria-hidden="true" />
              What wynk did
              <span className="text-ink-soft">{working ? "(live)" : ""}</span>
              <CaretDown size={14} className="ml-auto text-ink-soft transition-transform duration-200 group-open:rotate-180" aria-hidden="true" />
            </summary>
            <div className="border-t border-line px-4 py-4">
              <TurnSteps turn={turn} />
            </div>
          </details>
        )}

        {turn.answer && (
          <p className="pt-1 text-[0.9375rem] leading-relaxed text-ink">
            {turn.answer}
            {turn.phase === "answering" && <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-pulse bg-magenta-ink" aria-hidden="true" />}
          </p>
        )}

        {(turn.phase === "refused" || turn.phase === "error") && <p className="pt-1 text-[0.9375rem] leading-relaxed text-ink">{turn.note}</p>}

        {turn.quotes.length > 0 && (
          <ul className="mt-4 space-y-2" aria-label="Quotes the answer rests on">
            {turn.quotes.map((q) => (
              <li key={q.text} className="flex gap-3 rounded-2xl border border-line bg-white px-4 py-3 text-sm">
                <Quotes size={16} weight="fill" className="mt-0.5 shrink-0 text-magenta-ink" aria-hidden="true" />
                <span className="min-w-0">
                  <span className="text-ink">{q.text}</span>
                  <span className="mt-1 flex items-center gap-1.5 text-xs text-ink-soft">
                    <CheckCircle size={12} weight="fill" className="text-magenta-ink" aria-hidden="true" />
                    found word for word on <span className="font-mono">{q.page}</span>
                  </span>
                </span>
              </li>
            ))}
          </ul>
        )}

        {turn.phase === "done" && picked && (
          <p className="tnum mt-3 text-xs text-ink-soft">
            Answered with ant {picked.id}'s workflow, {picked.tokens} tokens, checked against the source.
          </p>
        )}

        {turn.phase !== "refused" && (
          <>
            {/* desktop: the work lives in the side panel; this selects which turn it shows */}
            <button
              type="button"
              onClick={onSelect}
              aria-pressed={selected}
              className={cn(
                "mt-3 hidden min-h-9 cursor-pointer items-center rounded-full border px-3 text-xs font-medium transition-colors duration-150 lg:inline-flex",
                selected ? "border-magenta bg-magenta text-ink" : "border-line-strong text-ink-soft hover:border-magenta hover:text-ink",
              )}
            >
              {selected ? "Colony shown on the right" : "Show this colony"}
            </button>
            {/* phones and tablets: the work opens inline */}
            <div className="lg:hidden">
              <button
                type="button"
                onClick={() => setOpenInline((o) => !o)}
                aria-expanded={openInline}
                className="mt-3 inline-flex min-h-9 cursor-pointer items-center gap-1.5 rounded-full border border-line-strong px-3 text-xs font-medium text-ink-soft"
              >
                {openInline ? "Hide the colony" : "Show the colony"}
                <CaretDown size={12} className={cn("transition-transform duration-200", openInline && "rotate-180")} aria-hidden="true" />
              </button>
              {openInline && (
                <div className="mt-4 rounded-[24px] border border-line p-4">
                  <TurnColony turn={turn} />
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

interface ComposerProps {
  busy: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  suggestions?: string[];
}

export function Composer({ busy, onSend, onStop, suggestions }: ComposerProps) {
  const [text, setText] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [text]);

  const send = (value = text) => {
    const v = value.trim();
    if (!v || busy) return;
    onSend(v);
    setText("");
  };

  return (
    <div>
      {suggestions && suggestions.length > 0 && (
        <ul className="mb-3 flex flex-wrap gap-2" aria-label="Example questions">
          {suggestions.map((s) => (
            <li key={s}>
              <button
                type="button"
                onClick={() => send(s)}
                disabled={busy}
                className="min-h-9 max-w-full cursor-pointer rounded-[18px] border border-line-strong bg-paper px-3.5 py-1.5 text-left text-sm text-ink transition-colors duration-150 hover:border-ink-soft disabled:opacity-50"
              >
                {s}
              </button>
            </li>
          ))}
        </ul>
      )}
      <form
        className="flex items-end gap-2 rounded-[26px] border border-line-strong bg-paper p-2 pl-4 shadow-[0_10px_30px_-18px_rgb(28_15_26/0.35)] focus-within:border-magenta-ink"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <label htmlFor="ask" className="sr-only">
          Ask wynk a question
        </label>
        <textarea
          id="ask"
          ref={area}
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Ask about a company in the library…"
          className="max-h-40 min-h-11 flex-1 resize-none bg-transparent py-2.5 text-[0.9375rem] leading-relaxed text-ink outline-none placeholder:text-ink-soft"
        />
        {busy ? (
          <button type="button" onClick={onStop} className="btn btn-quiet size-11 min-h-11 px-0" aria-label="Stop this run">
            <Stop size={18} weight="fill" />
          </button>
        ) : (
          <button type="submit" disabled={!text.trim()} className="btn btn-primary size-11 min-h-11 px-0" aria-label="Send">
            <ArrowUp size={18} weight="bold" />
          </button>
        )}
      </form>
    </div>
  );
}
