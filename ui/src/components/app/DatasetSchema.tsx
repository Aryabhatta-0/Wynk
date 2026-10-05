import type { ColumnMapping, DatasetColumn, DatasetRow } from "@/api";
import { cellText } from "@/lib/dataset";
import { cn } from "@/lib/utils";

export type Role = "input" | "target" | "context" | "ignore";

export const roleOf = (m: ColumnMapping, name: string): Role =>
  m.target === name ? "target" : m.input.includes(name) ? "input" : m.context.includes(name) ? "context" : "ignore";

/** Assign one column a role, keeping a single target. */
export function withRole(m: ColumnMapping, name: string, role: Role): ColumnMapping {
  const input = m.input.filter((c) => c !== name);
  const context = m.context.filter((c) => c !== name);
  let target = m.target === name ? null : m.target;
  if (role === "input") input.push(name);
  if (role === "context") context.push(name);
  if (role === "target") target = name;
  return { input, target, context };
}

const ROLE_STYLE: Record<Role, string> = {
  input: "border-series-random/40 bg-[#eef4fb] text-series-random",
  target: "border-magenta-ink bg-magenta-wash text-magenta-ink",
  context: "border-line-strong bg-wash text-ink",
  ignore: "border-line text-ink-soft",
};

const ROLE_LABEL: Record<Role, string> = { input: "Input", target: "Target", context: "Context", ignore: "Ignored" };

export function RoleTag({ role }: { role: Role }) {
  return (
    <span className={cn("rounded border px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide", ROLE_STYLE[role])}>{ROLE_LABEL[role]}</span>
  );
}

/** Schema with a role per column: input, target (one), context (optional) or ignored. */
export function MappingEditor({
  columns,
  preview,
  mapping,
  onChange,
  errors,
  disabled,
}: {
  columns: DatasetColumn[];
  preview: DatasetRow[];
  mapping: ColumnMapping;
  onChange: (m: ColumnMapping) => void;
  errors: string[];
  disabled?: boolean;
}) {
  return (
    <div>
      <div className="overflow-x-auto">
        <table className="data-table min-w-[640px]" aria-label="Columns and roles">
          <thead>
            <tr>
              <th>Column</th>
              <th>Type</th>
              <th className="num">Missing</th>
              <th className="num">Distinct</th>
              <th>Example</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            {columns.map((c) => {
              const role = roleOf(mapping, c.name);
              const example = preview.map((r) => cellText(r[c.name])).find(Boolean) ?? "";
              return (
                <tr key={c.name}>
                  <td className="font-mono text-[12px] font-medium">{c.name}</td>
                  <td className="font-mono text-[12px] text-ink-soft">{c.type}</td>
                  <td className="num">{c.missing}</td>
                  <td className="num">{c.distinct}</td>
                  <td className="max-w-[280px] truncate text-ink-soft" title={example}>
                    {example}
                  </td>
                  <td>
                    <select
                      className="input h-8 w-32 py-0 text-[13px]"
                      aria-label={`Role of ${c.name}`}
                      value={role}
                      disabled={disabled}
                      onChange={(e) => onChange(withRole(mapping, c.name, e.target.value as Role))}
                    >
                      <option value="input">Input</option>
                      <option value="target">Target</option>
                      <option value="context">Context</option>
                      <option value="ignore">Ignore</option>
                    </select>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <dl className="mt-3 grid gap-x-6 gap-y-1 text-xs text-ink-soft sm:grid-cols-3">
        <div>
          <dt className="inline font-semibold text-ink">Input</dt> <dd className="inline">what the workflow reads. At least one.</dd>
        </div>
        <div>
          <dt className="inline font-semibold text-ink">Target</dt> <dd className="inline">the expected output it is scored against. Exactly one.</dd>
        </div>
        <div>
          <dt className="inline font-semibold text-ink">Context</dt> <dd className="inline">optional supporting text, not scored.</dd>
        </div>
      </dl>
      {errors.length > 0 && (
        <ul className="mt-3 space-y-1 text-xs text-bad" aria-label="Mapping problems">
          {errors.map((e) => (
            <li key={e}>• {e}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** The first rows, with each column's role in its header. */
export function DatasetPreview({ columns, rows, mapping }: { columns: DatasetColumn[]; rows: DatasetRow[]; mapping: ColumnMapping }) {
  return (
    <div className="max-h-[360px] overflow-auto">
      <table className="data-table" aria-label="Preview rows">
        <thead>
          <tr>
            <th className="num">#</th>
            {columns.map((c) => (
              <th key={c.name}>
                <div className="flex items-center gap-1.5 normal-case">
                  <span className="font-mono text-[11px] tracking-normal text-ink">{c.name}</span>
                  <RoleTag role={roleOf(mapping, c.name)} />
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td className="num text-ink-soft">{i + 1}</td>
              {columns.map((c) => {
                const v = cellText(r[c.name]);
                return (
                  <td key={c.name} className={cn("max-w-[320px] truncate", roleOf(mapping, c.name) === "ignore" && "text-ink-soft")} title={v}>
                    {v || <span className="text-ink-soft">—</span>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
