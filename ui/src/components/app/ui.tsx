import { ArrowClockwise, CheckCircle, CircleDashed, Crown, type Icon, Prohibit, SealCheck, Spinner, Warning, XCircle } from "@phosphor-icons/react";
import { type ReactNode, useId } from "react";
import { ApiError, type ExperimentStatus, type WorkflowState, errorMessage } from "@/api";
import { STATUS_LABEL } from "@/lib/format";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------- page structure */

export function PageHeader({
  kicker,
  title,
  description,
  actions,
}: {
  kicker?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3 border-b border-line pb-4">
      <div className="min-w-0">
        {kicker && <div className="kicker mb-1">{kicker}</div>}
        <h1 className="text-2xl font-bold">{title}</h1>
        {description && <p className="mt-1 max-w-[72ch] text-sm leading-relaxed text-ink-soft">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Panel({
  title,
  meta,
  actions,
  children,
  className,
  bodyClassName,
}: {
  title?: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={cn("panel min-w-0", className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2.5">
          <div className="flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-0.5">
            {title && <h2 className="shrink-0 font-sans text-sm font-semibold tracking-normal whitespace-nowrap">{title}</h2>}
            {meta && <span className="min-w-0 text-xs text-ink-soft">{meta}</span>}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cn("p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

/* ------------------------------------------------------------- states */

export function LoadingState({ label = "Loading", rows = 4 }: { label?: string; rows?: number }) {
  return (
    <div role="status" aria-live="polite" className="space-y-2.5 py-2">
      <span className="sr-only">{label}…</span>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="skeleton h-9" style={{ opacity: 1 - i * 0.18 }} aria-hidden="true" />
      ))}
    </div>
  );
}

export function ErrorState({
  title = "Could not load this",
  message,
  code,
  details,
  onRetry,
}: {
  title?: string;
  message: string;
  /** stable error code from the API, shown so the real reason is never hidden */
  code?: string;
  details?: Record<string, string | number | null>;
  onRetry?: () => void;
}) {
  const facts = Object.entries(details ?? {}).filter(([, v]) => v !== null && v !== "");
  return (
    <div role="alert" data-error-code={code} className="flex items-start gap-3 rounded-[10px] border border-bad/30 bg-bad-wash px-4 py-3 text-sm">
      <Warning size={18} weight="fill" className="mt-0.5 shrink-0 text-bad" aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <p className="font-semibold text-ink">{title}</p>
        <p className="mt-0.5 break-words text-ink-soft">{message}</p>
        {(code || facts.length > 0) && (
          <p className="mt-1 font-mono text-[11px] text-ink-soft">
            {code && <span data-testid="error-code">{code}</span>}
            {facts.map(([k, v]) => (
              <span key={k}>
                {" · "}
                {k.replaceAll("_", " ")} {String(v)}
              </span>
            ))}
          </p>
        )}
      </div>
      {onRetry && (
        <button type="button" className="btn btn-quiet btn-sm shrink-0" onClick={onRetry}>
          <ArrowClockwise size={14} aria-hidden="true" /> Retry
        </button>
      )}
    </div>
  );
}

/** An adapter error with its heading, the server's own message, the stable code and details. */
export function ApiErrorState({ error, title, onRetry }: { error: unknown; title?: string; onRetry?: () => void }) {
  if (error instanceof ApiError)
    return <ErrorState title={title ?? error.title} message={error.message} code={error.code} details={error.details} onRetry={onRetry} />;
  return <ErrorState title={title ?? "Something went wrong"} message={errorMessage(error)} onRetry={onRetry} />;
}

export function EmptyState({ icon: IconCmp, title, children, action }: { icon: Icon; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center px-6 py-12 text-center">
      <span className="grid size-11 place-items-center rounded-full border border-line text-magenta-ink" aria-hidden="true">
        <IconCmp size={20} weight="duotone" />
      </span>
      <h3 className="mt-3 font-sans text-base font-semibold tracking-normal">{title}</h3>
      {children && <p className="mt-1 max-w-[52ch] text-sm leading-relaxed text-ink-soft">{children}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

/* ------------------------------------------------------------- badges */

const STATUS_STYLE: Record<ExperimentStatus, { icon: Icon; cls: string; spin?: boolean }> = {
  queued: { icon: CircleDashed, cls: "border-line-strong text-ink-soft" },
  running: { icon: Spinner, cls: "border-magenta/60 bg-magenta-wash text-magenta-ink", spin: true },
  completed: { icon: CheckCircle, cls: "border-good/30 bg-good-wash text-good" },
  failed: { icon: XCircle, cls: "border-bad/30 bg-bad-wash text-bad" },
  cancelled: { icon: Prohibit, cls: "border-line-strong text-ink-soft" },
};

export function StatusBadge({ status }: { status: ExperimentStatus }) {
  const s = STATUS_STYLE[status];
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium whitespace-nowrap", s.cls)}>
      <s.icon size={12} weight="bold" className={cn(s.spin && "animate-spin motion-reduce:animate-none")} aria-hidden="true" />
      {STATUS_LABEL[status]}
    </span>
  );
}

const STATE_STYLE: Record<WorkflowState, { icon: Icon; label: string; cls: string }> = {
  candidate: { icon: CircleDashed, label: "Candidate", cls: "border-line-strong text-ink-soft" },
  validated: { icon: SealCheck, label: "Validated", cls: "border-series-random/40 text-series-random" },
  champion: { icon: Crown, label: "Champion", cls: "border-magenta-ink bg-magenta-ink text-white" },
};

export function StateBadge({ state }: { state: WorkflowState }) {
  const s = STATE_STYLE[state];
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium whitespace-nowrap", s.cls)}>
      <s.icon size={12} weight="bold" aria-hidden="true" />
      {s.label}
    </span>
  );
}

/** Candidate → Validated → Champion, with what each step means. */
export function StateLegend() {
  return (
    <ol className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-soft" aria-label="Workflow states">
      <li className="flex items-center gap-1.5">
        <StateBadge state="candidate" /> measured on optimization split
      </li>
      <li aria-hidden="true">→</li>
      <li className="flex items-center gap-1.5">
        <StateBadge state="validated" /> re-measured on validation
      </li>
      <li aria-hidden="true">→</li>
      <li className="flex items-center gap-1.5">
        <StateBadge state="champion" /> best validated, all constraints met
      </li>
    </ol>
  );
}

export function Check({ ok, children }: { ok: boolean; children?: ReactNode }) {
  return (
    <span className={cn("inline-flex items-center gap-1 text-xs font-medium whitespace-nowrap", ok ? "text-good" : "text-bad")}>
      {ok ? <CheckCircle size={14} weight="fill" aria-hidden="true" /> : <XCircle size={14} weight="fill" aria-hidden="true" />}
      {children ?? (ok ? "Met" : "Broken")}
    </span>
  );
}

/** One labelled fact in a <dl>; `wide` spans the whole row (hashes). */
export function Fact({ label, children, wide, testId }: { label: string; children: ReactNode; wide?: boolean; testId?: string }) {
  return (
    <div className={cn(wide && "sm:col-span-2 xl:col-span-4")}>
      <dt className="text-xs text-ink-soft">{label}</dt>
      <dd className="font-medium" data-testid={testId}>
        {children}
      </dd>
    </div>
  );
}

/* ------------------------------------------------------------- forms */

export function Field({
  label,
  hint,
  error,
  children,
  className,
}: {
  label: ReactNode;
  hint?: ReactNode;
  error?: string;
  className?: string;
  children: (props: { id: string; "aria-invalid": boolean; "aria-describedby"?: string }) => ReactNode;
}) {
  const id = useId();
  const describedBy = [hint && `${id}-hint`, error && `${id}-error`].filter(Boolean).join(" ") || undefined;
  return (
    <div className={cn("min-w-0", className)}>
      <label htmlFor={id} className="mb-1 block text-[13px] font-medium text-ink">
        {label}
      </label>
      {children({ id, "aria-invalid": !!error, "aria-describedby": describedBy })}
      {hint && !error && (
        <p id={`${id}-hint`} className="mt-1 text-xs leading-snug text-ink-soft">
          {hint}
        </p>
      )}
      {error && (
        <p id={`${id}-error`} className="mt-1 text-xs leading-snug text-bad">
          {error}
        </p>
      )}
    </div>
  );
}

/* ------------------------------------------------------------- numbers */

export function Metric({
  label,
  value,
  detail,
  tone = "neutral",
  testId,
}: {
  label: ReactNode;
  value: ReactNode;
  detail?: ReactNode;
  tone?: "neutral" | "better" | "worse";
  testId?: string;
}) {
  return (
    <div className="min-w-0 px-4 py-3" data-testid={testId}>
      <div className="kicker">{label}</div>
      <div className="mt-1 font-display text-[1.625rem] leading-none font-bold tnum">{value}</div>
      {detail && (
        <div className={cn("mt-1.5 text-xs tnum", tone === "better" ? "text-good" : tone === "worse" ? "text-bad" : "text-ink-soft")}>{detail}</div>
      )}
    </div>
  );
}

export function Meter({ value, max, label }: { value: number; max: number; label: string }) {
  const p = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={max}
      aria-valuenow={value}
      className="h-1.5 overflow-hidden rounded-full bg-line"
    >
      <div className="h-full rounded-full bg-magenta-ink transition-[width] duration-500 ease-out" style={{ width: `${p}%` }} />
    </div>
  );
}
