import { ArrowRight, GithubLogo, Plus } from "@phosphor-icons/react";
import { useAnimate } from "motion/react";
import { type RefObject, useEffect, useRef, useState } from "react";
import { Ghost } from "./components/Ghost";
import { GhostFigure, GhostTile } from "./components/GhostFigure";
import { SyncCard } from "./components/SyncCard";
import { WorkflowGraph } from "./components/WorkflowGraph";
import { Composer, Thread } from "./components/chat/Thread";
import { TurnWork } from "./components/chat/TurnWork";
import { applyEvent, newTurn, type Turn } from "./lib/chat";
import { cn } from "./lib/cn";
import { DEMO, SUGGESTIONS, runQuery } from "./lib/demoEngine";
import { LIBRARY } from "./lib/library";

const REPO = "https://github.com/Aryabhatta-0/Wynk";
const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

type Screen = "landing" | "intro" | "chat";

export default function App() {
  const [screen, setScreen] = useState<Screen>("landing");
  const [logoLanded, setLogoLanded] = useState(false);
  const logoSlot = useRef<HTMLSpanElement>(null);

  const start = () => {
    if (reduceMotion()) {
      setLogoLanded(true);
      setScreen("chat");
    } else {
      setScreen("intro");
    }
  };

  return (
    <>
      {screen === "landing" ? <Landing onStart={start} /> : <ChatView logoSlot={logoSlot} logoLanded={logoLanded} />}
      {screen === "intro" && (
        <Intro
          logoSlot={logoSlot}
          onDone={() => {
            setLogoLanded(true);
            setScreen("chat");
          }}
        />
      )}
    </>
  );
}

/* ------------------------------------------------------------------ landing */

function Landing({ onStart }: { onStart: () => void }) {
  return (
    <div className="flex min-h-dvh flex-col">
      <header className="mx-auto flex h-16 w-full max-w-[1280px] items-center justify-between px-4 sm:px-6 lg:px-8">
        <span className="flex items-center gap-2.5">
          <GhostTile className="size-8" />
          <span className="font-display text-[1.375rem] font-bold tracking-tight">wynk</span>
          <span className="hidden text-sm text-ink-soft sm:inline">agent workflow optimizer</span>
        </span>
        <a className="grid size-11 place-items-center rounded-full text-ink-soft hover:text-ink" href={REPO} aria-label="wynk on GitHub">
          <GithubLogo size={20} />
        </a>
      </header>

      <main className="mx-auto grid w-full max-w-[1280px] flex-1 items-center gap-10 px-4 pt-4 pb-10 sm:px-6 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)] lg:gap-14 lg:px-8">
        <div>
          <h1 className="max-w-[14ch] text-[2.75rem] leading-[1.02] font-extrabold sm:text-6xl lg:text-7xl">
            Ask a question. Watch the colony answer it.
          </h1>
          <p className="mt-6 max-w-[44ch] text-lg leading-relaxed text-ink-soft">
            wynk sends a colony of agent workflows after each question, checks every answer against its source, and shows
            you everything.
          </p>
          <button type="button" onClick={onStart} className="btn btn-primary mt-8 h-14 px-7 text-base">
            Start <ArrowRight size={18} weight="bold" aria-hidden="true" />
          </button>
        </div>
        <div className="relative aspect-[5/4] overflow-hidden rounded-[40px] bg-plum lg:aspect-square xl:aspect-[5/4]">
          <Ghost active className="absolute inset-0 block size-full" />
        </div>
      </main>
    </div>
  );
}

/* ------------------------------------------------------------------ intro */

/**
 * Start: a plum curtain, the ghost crosses left to right, winks, then flies into the logo
 * slot of the chat header as the curtain lifts. Runs once; skipped under reduced motion.
 */
function Intro({ logoSlot, onDone }: { logoSlot: RefObject<HTMLSpanElement | null>; onDone: () => void }) {
  const [scope, animate] = useAnimate<HTMLDivElement>();
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return; // StrictMode runs effects twice in development
    started.current = true;
    // never strand the visitor behind the curtain: finish anyway if frames stop (background tab)
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      onDone();
    };
    const fallback = window.setTimeout(finish, 6000);
    if (document.hidden) {
      finish();
      return;
    }
    const root = scope.current;
    const curtain = root.querySelector<HTMLElement>("[data-curtain]")!;
    const ghost = root.querySelector<HTMLElement>("[data-ghost]")!;
    const eye = ghost.querySelector<SVGPathElement>('[data-eye="right"]')!;

    (async () => {
      const w = window.innerWidth;
      const from = -w * 0.38;
      const to = w * 0.28;
      await animate(curtain, { opacity: [0, 1] }, { duration: 0.28, ease: "easeOut" });
      await animate(
        ghost,
        { x: [from, to], y: [0, -18, 0, -18, 0], rotate: [-5, 5, -5, 4, 0], opacity: [0, 1, 1, 1, 1] },
        { duration: 1.7, ease: [0.65, 0, 0.35, 1] },
      );
      await Promise.all([
        animate(eye, { scaleY: [1, 0.08, 1] }, { duration: 0.38, ease: "easeInOut" }),
        animate(ghost, { rotate: [0, -9, 0] }, { duration: 0.46, ease: "easeInOut" }),
      ]);
      await new Promise((r) => setTimeout(r, 140));

      const target = logoSlot.current?.getBoundingClientRect();
      const g = ghost.getBoundingClientRect();
      if (target && g.height) {
        const dx = target.left + target.width / 2 - (g.left + g.width / 2);
        const dy = target.top + target.height / 2 - (g.top + g.height / 2);
        animate(curtain, { opacity: 0 }, { duration: 0.65, ease: "easeOut", delay: 0.1 });
        await animate(ghost, { x: to + dx, y: dy, scale: target.height / g.height }, { duration: 0.8, ease: [0.32, 0.72, 0, 1] });
      }
      window.clearTimeout(fallback);
      finish();
    })();
  }, [animate, logoSlot, onDone, scope]);

  return (
    <div ref={scope} className="pointer-events-none fixed inset-0 z-50" aria-hidden="true">
      <div data-curtain className="absolute inset-0 bg-plum opacity-0" />
      <div className="absolute inset-0 grid place-items-center">
        <div data-ghost className="opacity-0">
          <GhostFigure className="h-[min(34vh,260px)] w-auto" />
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ chat */

function ChatView({ logoSlot, logoLanded }: { logoSlot: RefObject<HTMLSpanElement | null>; logoLanded: boolean }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const controller = useRef<AbortController | null>(null);
  const last = turns.at(-1);
  const busy = !!last && !["done", "refused", "error"].includes(last.phase);

  const ask = async (question: string) => {
    const id = Date.now();
    setTurns((ts) => [...ts, newTurn(id, question)]);
    setSelected(id);
    const ac = new AbortController();
    controller.current = ac;
    const update = (fn: (t: Turn) => Turn) => setTurns((ts) => ts.map((t) => (t.id === id ? fn(t) : t)));
    try {
      await runQuery(question, (e) => update((t) => applyEvent(t, e)), ac.signal);
    } catch (err) {
      const message =
        err instanceof DOMException && err.name === "AbortError"
          ? "Stopped. Nothing after this point ran."
          : `The run failed: ${err instanceof Error ? err.message : String(err)}`;
      update((t) => applyEvent(t, { type: "error", message }));
    }
  };
  const stop = () => controller.current?.abort();
  const newChat = () => {
    stop();
    setTurns([]);
    setSelected(null);
  };

  const shown = turns.find((t) => t.id === selected) ?? last;

  return (
    <div className="flex h-dvh flex-col">
      <header className="flex h-16 shrink-0 items-center justify-between border-b border-line px-4 sm:px-6">
        <span className="flex items-center gap-2.5">
          <span className="grid size-9 place-items-center rounded-[28%] bg-plum">
            <span ref={logoSlot} className={cn("block transition-opacity duration-150", logoLanded ? "opacity-100" : "opacity-0")}>
              <GhostFigure className="h-[22px] w-auto" />
            </span>
          </span>
          <span className="font-display text-[1.375rem] font-bold tracking-tight">wynk</span>
          {DEMO && (
            <span className="ml-1 rounded-full border border-line-strong px-2 py-0.5 text-xs text-ink-soft" title="Runs are simulated until the wynk backend is connected">
              demo data
            </span>
          )}
        </span>
        <button type="button" onClick={newChat} disabled={!turns.length} className="btn btn-quiet h-10 min-h-10 px-3.5 text-sm">
          <Plus size={16} aria-hidden="true" /> New chat
        </button>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        <section className="flex min-h-0 min-w-0 flex-col" aria-label="Chat">
          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-8 sm:px-6">
            <div className="mx-auto max-w-[44rem]">
              {turns.length ? <Thread turns={turns} selected={shown?.id ?? null} onSelect={setSelected} /> : <EmptyChat />}
            </div>
          </div>
          <div className="px-4 pt-2 pb-4 sm:px-6">
            <div className="mx-auto max-w-[44rem]">
              <Composer busy={busy} onSend={ask} onStop={stop} suggestions={turns.length ? undefined : SUGGESTIONS} />
              <p className="mt-2 text-center text-xs text-ink-soft">wynk answers only from its library, and shows every step it takes.</p>
            </div>
          </div>
        </section>

        <aside className="hidden min-h-0 overflow-y-auto border-l border-line bg-wash lg:block" aria-label="How wynk is answering">
          <div className="p-6 xl:p-8">{shown ? <TurnWork key={shown.id} turn={shown} /> : <IdleWork />}</div>
        </aside>
      </div>
    </div>
  );
}

function EmptyChat() {
  return (
    <div className="pt-6 sm:pt-12">
      <h2 className="max-w-[18ch] text-4xl font-bold sm:text-5xl">What would you like to know?</h2>
      <p className="mt-4 max-w-[52ch] leading-relaxed text-ink-soft">
        I answer from {LIBRARY.length} company fact sheets. Each answer comes with the workflow that produced it and the
        exact lines it rests on.
      </p>
      <ul className="mt-6 flex flex-wrap gap-2" aria-label="Companies in the library">
        {LIBRARY.map((s) => (
          <li key={s.id} className="rounded-full bg-wash px-3 py-1 text-sm text-ink">
            {s.name}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Before the first question: what the panel will show, at rest. */
function IdleWork() {
  return (
    <div className="space-y-8">
      <SyncCard progress={0} title="Ready" subtitle="Ask a question to start" label="Progress through the workflow" />
      <div>
        <h3 className="text-lg font-semibold">Your question's workflow appears here</h3>
        <p className="mt-1 max-w-[56ch] text-sm leading-relaxed text-ink-soft">
          Three ants each walk a different workflow at once. You will see every stage light up, which ant wins and why,
          and the quotes the answer rests on.
        </p>
      </div>
      <div className="rounded-[24px] border border-line bg-paper p-5 sm:p-6">
        <WorkflowGraph
          stages={[
            { kind: "GATHER", options: ["fetch", "parallel-4"] },
            { kind: "EXTRACT", options: ["direct"] },
            { kind: "SYNTHESIZE", options: ["cite evidence"] },
            { kind: "VERIFY", options: ["evidence span", "retry-1"] },
          ]}
          status={["idle", "idle", "idle", "idle"]}
          answered={false}
          failed={false}
        />
      </div>
    </div>
  );
}
