import { ArrowRight, CheckCircle, GithubLogo, Plus, Quotes, ShareNetwork } from "@phosphor-icons/react";
import { animate, motion } from "motion/react";
import { useRef, useState } from "react";
import { Ghost, type GhostHandle } from "./components/Ghost";
import { GhostLogo } from "./components/GhostFigure";
import { SyncCard } from "./components/SyncCard";
import { WorkflowGraph } from "./components/WorkflowGraph";
import { Composer, Thread } from "./components/chat/Thread";
import { TurnWork } from "./components/chat/TurnWork";
import { applyEvent, newTurn, type Turn } from "@/lib/chat";
import { cn } from "@/lib/utils";
import { DEMO, SUGGESTIONS, runQuery } from "./lib/demoEngine";
import { LIBRARY } from "./lib/library";

const REPO = "https://github.com/Aryabhatta-0/Wynk";
const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

// geometry of the ghost inside its canvas box (ghost-frames.js, first pose): the body is
// ~53% of the box tall and sits ~9% of the box left of centre
const BODY_H = 0.53;
const BODY_DX = -0.094;

type Screen = "landing" | "intro" | "chat";

export default function App() {
  const [screen, setScreen] = useState<Screen>("landing");
  const [chatMounted, setChatMounted] = useState(false);
  const [chatRevealed, setChatRevealed] = useState(false);
  const [logoLanded, setLogoLanded] = useState(false);
  const [ghostGone, setGhostGone] = useState(false);
  const [frozen, setFrozen] = useState(false);

  const landingRef = useRef<HTMLDivElement>(null);
  const ghostBox = useRef<HTMLDivElement>(null);
  const sparkle = useRef<SVGSVGElement>(null);
  const ghost = useRef<GhostHandle>(null);
  const logoSlot = useRef<HTMLSpanElement>(null);
  const running = useRef(false);

  const finishNow = () => {
    setChatMounted(true);
    setChatRevealed(true);
    setLogoLanded(true);
    setGhostGone(true);
    setScreen("chat");
  };

  const start = async () => {
    if (running.current) return;
    running.current = true;
    const land = landingRef.current;
    const box = ghostBox.current;
    if (reduceMotion() || !land || !box || document.hidden) return finishNow();

    // never strand the visitor if frames stop (a background tab): finish without the show
    const fallback = setTimeout(finishNow, 9000);

    setScreen("intro");
    setFrozen(true);
    const W = window.innerWidth;
    const S = box.getBoundingClientRect().width; // the ghost box, before any scaling
    const near = Math.min(300, W * 0.42) / S; // the ghost's size while it performs

    // 1. the copy steps back, the ghost comes forward and sharpens
    animate(land, { opacity: 0, filter: "blur(8px)", y: -16 }, { duration: 0.6, ease: [0.4, 0, 0.2, 1] });
    await animate(
      box,
      {
        opacity: [0.2, 1],
        scale: [1, near],
        filter: [
          "blur(3px) drop-shadow(0px 0px 0px rgba(239,117,211,0))",
          "blur(0px) drop-shadow(0px 22px 36px rgba(239,117,211,0.45))",
        ],
      },
      { duration: 0.8, ease: [0.22, 1, 0.36, 1] },
    );

    // 2. it swings to the left edge, then glides across the window
    const left = -W * 0.34;
    const x = W * 0.26;
    const y = 0;
    await animate(box, { x: left, y, rotate: -8 }, { duration: 0.6, ease: [0.65, 0, 0.35, 1] });
    await animate(
      box,
      { x: [left, x], y: [0, -34, 0, -34, 0], rotate: [-8, 7, -6, 5, 0] },
      { duration: 1.8, ease: [0.45, 0, 0.55, 1] },
    );

    // 3. it stops, faces you and winks
    setChatMounted(true);
    animate(box, { scale: [near, near * 1.06, near] }, { duration: 0.5, ease: "easeInOut" });
    await wait(120);
    ghost.current?.wink(480);
    if (sparkle.current) {
      animate(sparkle.current, { opacity: [0, 1, 0], scale: [0.4, 1.15, 0.6], rotate: [0, 40, 70] }, { duration: 0.7, ease: "easeOut" });
    }
    await wait(650);
    setChatRevealed(true);
    await wait(250);

    // 4. it flies up into the header and becomes the logo
    const slot = logoSlot.current?.getBoundingClientRect();
    const now = box.getBoundingClientRect();
    if (slot && now.width) {
      const end = slot.height / (BODY_H * S);
      const dx = slot.left + slot.width / 2 - BODY_DX * S * end - (now.left + now.width / 2);
      const dy = slot.top + slot.height / 2 - (now.top + now.height / 2);
      await animate(
        box,
        {
          x: [x, x + dx * 0.55, x + dx],
          y: [y, y + dy * 0.5 - 70, y + dy],
          scale: [near, near * 0.7, end],
          rotate: [0, -10, 0],
        },
        { duration: 0.95, ease: [0.32, 0.72, 0, 1] },
      );
    }
    setLogoLanded(true);
    await animate(box, { opacity: 0 }, { duration: 0.18 });
    clearTimeout(fallback);
    finishNow();
  };

  return (
    <>
      {/* the ghost: behind the landing copy, in front of everything during the intro */}
      {!ghostGone && (
        <div className={cn("pointer-events-none fixed inset-0 overflow-hidden", screen === "intro" ? "z-50" : "z-0")} aria-hidden="true">
          <div
            ref={ghostBox}
            className="absolute top-[56%] left-1/2 size-[min(82vh,760px)] -translate-x-1/2 -translate-y-1/2"
            style={{ opacity: 0.2, filter: "blur(3px) drop-shadow(0px 0px 0px rgba(239,117,211,0))" }}
          >
            <div className={cn("ghost-drift relative size-full", frozen && "is-frozen")}>
              <Ghost ref={ghost} active className="ghost-light block size-full" />
              <svg ref={sparkle} viewBox="0 0 24 24" className="absolute top-[30%] left-[52%] size-[9%] opacity-0" fill="#ef75d3">
                <path d="M12 0c.6 6.2 5.8 11.4 12 12-6.2.6-11.4 5.8-12 12-.6-6.2-5.8-11.4-12-12C6.2 11.4 11.4 6.2 12 0Z" />
              </svg>
            </div>
          </div>
        </div>
      )}

      {!chatRevealed && <Landing ref={landingRef} onStart={start} />}
      {chatMounted && <ChatView logoSlot={logoSlot} logoLanded={logoLanded} revealed={chatRevealed} />}
    </>
  );
}

/* ------------------------------------------------------------------ landing */

function Landing({ onStart, ref }: { onStart: () => void; ref: React.Ref<HTMLDivElement> }) {
  return (
    <div ref={ref} className="relative z-10 flex min-h-dvh flex-col">
      <div className="hero-ground pointer-events-none absolute inset-0 -z-10" aria-hidden="true" />
      <header className="mx-auto flex h-16 w-full max-w-[1200px] items-center justify-between px-4 sm:px-6 lg:px-8">
        <span className="flex items-center gap-2.5">
          <GhostLogo className="h-7 w-auto" />
          <span className="font-display text-[1.375rem] font-bold tracking-tight">wynk</span>
        </span>
        <a className="grid size-11 place-items-center rounded-full text-ink-soft hover:text-ink" href={REPO} aria-label="wynk on GitHub">
          <GithubLogo size={20} />
        </a>
      </header>

      <main className="mx-auto flex w-full max-w-[1200px] flex-1 flex-col items-center px-4 pt-[9vh] pb-12 text-center sm:px-6 lg:px-8">
        <h1 className="max-w-[15ch] text-[2.9rem] leading-[1.02] font-extrabold [text-shadow:0_0_40px_#fff,0_0_12px_#fff] sm:text-7xl lg:text-[5.5rem]">
          Ask a question. Watch the colony answer it.
        </h1>
        <p className="mt-6 max-w-[46ch] text-lg leading-relaxed text-ink-soft [text-shadow:0_0_18px_#fff] sm:text-xl">
          wynk sends a colony of agent workflows after each question, checks every answer against its source, and shows
          you all of it.
        </p>
        <button type="button" onClick={onStart} className="btn btn-primary mt-10 h-14 px-8 text-base">
          Start <ArrowRight size={18} weight="bold" aria-hidden="true" />
        </button>

        <ul className="mt-auto grid w-full max-w-3xl gap-4 pt-16 text-left text-sm sm:grid-cols-3">
          <Fact icon={<ShareNetwork size={18} weight="duotone" />} title="Three workflows race">
            An ant colony picks which agent steps to try for your question.
          </Fact>
          <Fact icon={<CheckCircle size={18} weight="duotone" />} title="Checked, not trusted">
            Every quote must exist word for word in the source.
          </Fact>
          <Fact icon={<Quotes size={18} weight="duotone" />} title="Answers you can trace">
            Each reply shows its workflow and the lines it rests on.
          </Fact>
        </ul>
      </main>
    </div>
  );
}

function Fact({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <li className="rounded-2xl border border-line bg-white/80 p-4 backdrop-blur-sm">
      <p className="flex items-center gap-2 font-semibold text-ink">
        <span className="text-magenta-ink">{icon}</span>
        {title}
      </p>
      <p className="mt-1.5 leading-relaxed text-ink-soft">{children}</p>
    </li>
  );
}

/* ------------------------------------------------------------------ chat */

const rise = (revealed: boolean, delay: number) => ({
  initial: { opacity: 0, y: 18 },
  animate: revealed ? { opacity: 1, y: 0 } : { opacity: 0, y: 18 },
  transition: { duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] as const },
});

function ChatView({
  logoSlot,
  logoLanded,
  revealed,
}: {
  logoSlot: React.RefObject<HTMLSpanElement | null>;
  logoLanded: boolean;
  revealed: boolean;
}) {
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
    <div className="fixed inset-0 z-20 flex flex-col bg-white">
      <header className="flex h-16 shrink-0 items-center justify-between border-b border-line px-4 sm:px-6">
        <span className="flex items-center gap-2.5">
          <motion.span
            ref={logoSlot}
            className="block"
            initial={{ opacity: 0 }}
            animate={logoLanded ? { opacity: 1, scale: [1.22, 0.94, 1] } : { opacity: 0 }}
            transition={{ duration: 0.45, ease: "easeOut" }}
          >
            <GhostLogo className="h-[30px] w-auto" />
          </motion.span>
          <motion.span className="flex items-center gap-2.5" {...rise(revealed, 0)}>
            <span className="font-display text-[1.375rem] font-bold tracking-tight">wynk</span>
            {DEMO && (
              <span
                className="rounded-full border border-line-strong px-2 py-0.5 text-xs text-ink-soft"
                title="Runs are simulated until the wynk backend is connected"
              >
                demo data
              </span>
            )}
          </motion.span>
        </span>
        <motion.div {...rise(revealed, 0.05)}>
          <button type="button" onClick={newChat} disabled={!turns.length} className="btn btn-quiet h-10 min-h-10 px-3.5 text-sm">
            <Plus size={16} aria-hidden="true" /> New chat
          </button>
        </motion.div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        <motion.section className="flex min-h-0 min-w-0 flex-col" aria-label="Chat" {...rise(revealed, 0.1)}>
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
        </motion.section>

        <motion.aside
          className="hidden min-h-0 overflow-y-auto border-l border-line lg:block"
          aria-label="How wynk is answering"
          {...rise(revealed, 0.2)}
        >
          <div className="p-6 xl:p-8">{shown ? <TurnWork key={shown.id} turn={shown} /> : <IdleWork />}</div>
        </motion.aside>
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
          <li key={s.id} className="rounded-full border border-line px-3 py-1 text-sm text-ink">
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
      <div className="rounded-[24px] border border-line p-5 sm:p-6">
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
