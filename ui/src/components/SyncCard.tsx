import { useEffect, useRef } from "react";

/*
  The scan-line completion card from the supplied design (New folder/ui), driven by real
  progress instead of its demo timeline. 65 magenta rows; while work is in flight their tips
  shimmer white, at 100% they settle into a clean rounded edge. The shown percentage eases
  toward the target so a jump (a new run starting at 0) travels instead of teleporting.
*/

const ROWS = 65;
const rows = Array.from({ length: ROWS }, (_, i) => ({
  phase: i * 2.3999632297,
  speed: 0.8 + fract(Math.sin(i * 78.233) * 43758.5453) * 1.5,
  brightness: 0.76 + fract(Math.sin(i * 17.719) * 9631.417) * 0.24,
}));

function fract(n: number) {
  return n - Math.floor(n);
}
function smoothstep(n: number) {
  return n * n * (3 - 2 * n);
}

interface Props {
  progress: number; // 0..100
  title: string;
  subtitle: string;
  label: string; // accessible name of the progress bar
}

export function SyncCard({ progress, title, subtitle, label }: Props) {
  const cardRef = useRef<HTMLElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const numberRef = useRef<HTMLSpanElement>(null);
  const target = useRef(progress);
  target.current = Math.max(0, Math.min(100, progress));

  useEffect(() => {
    const card = cardRef.current!;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)");
    let width = 0;
    let height = 0;
    let shown = target.current;
    let elapsed = 0;
    let last: number | null = null;
    let frame: number | null = null;
    let visible = true;
    let lastInt = -1;

    const draw = () => {
      const p = shown;
      const int = Math.round(p);
      if (int !== lastInt && numberRef.current) {
        numberRef.current.textContent = `${int}%`;
        lastInt = int;
      }
      ctx.clearRect(0, 0, width, height);
      if (p <= 0 || !width || !height) return;

      const rowH = height / ROWS;
      const lineH = rowH * 0.49;
      const energy = Math.min(1, Math.max(0, (100 - p) / 4));
      const reveal = Math.min(1, p / 4);
      const lead = smoothstep(Math.min(1, Math.max(0, (p - 65) / 20)));
      const front = width * (p / 100 + lead * 0.065 * energy);
      const fringe = width * 0.048 * energy * reveal;
      const t = elapsed * 4.8;

      rows.forEach((row, i) => {
        const wave =
          Math.sin(row.phase + t * row.speed) * 0.49 +
          Math.sin(i * 0.63 - t * 0.72) * 0.28 +
          Math.sin(i * 1.91 + t * 2.2) * 0.23;
        const tip = Math.max(0, Math.min(width, front + wave * fringe));
        if (tip <= 0) return;
        const y = i * rowH + rowH * 0.12;
        const tail = Math.min(tip, width * (0.11 + 0.05 * reveal));

        ctx.globalAlpha = row.brightness;
        const body = ctx.createLinearGradient(0, 0, Math.max(tip, 1), 0);
        body.addColorStop(0, "#361729");
        body.addColorStop(0.58, "#482039");
        body.addColorStop(1, "#853167");
        ctx.fillStyle = body;
        ctx.fillRect(0, y, tip, lineH);

        const edge = ctx.createLinearGradient(tip - tail, 0, tip, 0);
        edge.addColorStop(0, "rgba(159, 48, 120, 0)");
        edge.addColorStop(0.58, `rgba(201, 68, 158, ${0.6 + energy * 0.15})`);
        edge.addColorStop(0.86, "#ed73d3");
        edge.addColorStop(1, energy > 0.05 ? "#fff9ff" : "#dc69bc");
        ctx.fillStyle = edge;
        ctx.fillRect(Math.max(0, tip - tail), y, Math.min(tail, tip), lineH);

        if (energy > 0.05) {
          const tipW = Math.min(tip, width * 0.003);
          ctx.globalAlpha = energy * row.brightness;
          ctx.fillStyle = "#fffaff";
          ctx.shadowColor = "#f7a8e9";
          ctx.shadowBlur = width * 0.0018;
          ctx.fillRect(tip - tipW, y, tipW, lineH);
          ctx.shadowBlur = 0;
        }
      });
      ctx.globalAlpha = 1;
    };

    const tick = (now: number) => {
      frame = null;
      const dt = last === null ? 0 : Math.min((now - last) / 1000, 0.1);
      last = now;
      if (reduce.matches) {
        shown = target.current; // no travel, no shimmer
      } else {
        elapsed += dt;
        shown += (target.current - shown) * (1 - Math.exp(-dt * 5));
        if (Math.abs(target.current - shown) < 0.05) shown = target.current;
      }
      draw();
      schedule();
    };

    const schedule = () => {
      if (frame !== null || !visible || document.hidden) return;
      // a settled card with reduced motion has nothing left to animate
      if (reduce.matches && shown === target.current) return;
      frame = requestAnimationFrame(tick);
    };
    const stop = () => {
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
      last = null;
    };

    const resize = () => {
      const r = card.getBoundingClientRect();
      width = r.width;
      height = r.height;
      const ratio = Math.min(devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    };

    const ro = new ResizeObserver(resize);
    ro.observe(card);
    const io = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) schedule();
      else stop();
    });
    io.observe(card);
    const onVis = () => (document.hidden ? stop() : schedule());
    document.addEventListener("visibilitychange", onVis);
    const poke = setInterval(schedule, 250); // picks up new targets under reduced motion
    resize();
    schedule();
    return () => {
      stop();
      ro.disconnect();
      io.disconnect();
      clearInterval(poke);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, []);

  const value = Math.round(target.current);
  return (
    <section ref={cardRef} className="sync-card" aria-label={label}>
      <canvas ref={canvasRef} className="sync-lines" aria-hidden="true" />
      <div className="sync-content">
        <div className="min-w-0">
          <p className="sync-title">{title}</p>
          <p className="sync-subtitle">{subtitle}</p>
        </div>
        <div
          role="progressbar"
          aria-label={label}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={value}
          className="shrink-0 text-right"
        >
          <span ref={numberRef} className="sync-percent" aria-hidden="true">
            {value}%
          </span>
        </div>
      </div>
    </section>
  );
}
