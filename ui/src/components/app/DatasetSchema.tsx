import type { ColumnMapping, DatasetColumn, DatasetRow } from "@/api";
import { canBeId, canTakeRole } from "@/api/contract/rules";
import { cellText } from "@/lib/dataset";
import { cn } from "@/lib/utils";

export type Role = "input" | "target" | "context" | "id" | "ignore";

export const roleOf = (m: ColumnMapping, name: string): Role =>
  m.id === name ? "id" : m.target.includes(name) ? "target" : m.input.includes(name) ? "input" : m.context.includes(name) ? "context" : "ignore";

/** Give one column one role (roles are disjoint; there is at most one id column). */
export function withRole(m: ColumnMapping, name: string, role: Role): ColumnMapping {
  const drop = (cols: string[]) => cols.filter((c) => c !== name);
  const next: ColumnMapping = { input: drop(m.input), target: drop(m.target), context: drop(m.context), id: m.id === name ? null : m.id };
  if (role === "input") next.input.push(name);
  if (role === "target") next.target.push(name);
  if (role === "context") next.context.push(name);
  if (role === "id") next.id = name;
  return next;
}

const ROLE_STYLE: Record<Role, string> = {
  input: "border-series-random/40 bg-[#eef4fb] text-series-random",
  target: "border-magenta-ink bg-magenta-wash text-magenta-ink",
  context: "border-line-strong bg-wash text-ink",
  id: "border-line-strong text-ink",
  ignore: "border-line text-ink-soft",
};

const ROLE_LABEL: Record<Role, string> = { input: "Input", target: "Target", context: "Context", id: "Row id", ignore: "Ignored" };

export function RoleTag({ role }: { role: Role }) {
  return (
    <span className={cn("rounded border px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide", ROLE_STYLE[role])}>{ROLE_LABEL[role]}</span>
  );
}

/** Schema with a role per column: input, target, context, row id or ignored (DatasetSpec roles). */
export function MappingEditor({
  columns,
  preview,
  mapping,
  onChange,
  errors,
  disabled,
}: {
  /** types and nullability as the server inferred them; null counts when the upload is known */
  columns: (DatasetColumn & { nullCount?: number })[];
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
              <th>Nullable</th>
              <th className="num">Missing</th>
              <th>Example</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            {columns.map((c) => {
              const role = roleOf(mapping, c.name);
              const example = preview.map((r) => cellText(r[c.name])).find(Boolean) ?? "";
              const usable = canTakeRole(c);
              return (
                <tr key={c.name}>
                  <td className="font-mono text-[12px] font-medium">{c.name}</td>
                  <td className="font-mono text-[12px] text-ink-soft">{c.type}</td>
                  <td className="text-[12px] text-ink-soft">{c.nullable ? "yes" : "no"}</td>
                  <td className="num">{c.nullCount ?? "—"}</td>
                  <td className="max-w-[280px] truncate text-ink-soft" title={example}>
                    {example}
                  </td>
                  <td>
                    <select
                      className="input h-8 w-32 py-0 text-[13px]"
                      aria-label={`Role of ${c.name}`}
                      value={role}
                      disabled={disabled}
                      title={usable ? undefined : c.type === "json" ? "Nested JSON cannot take a role" : "Column name must be an identifier"}
                      onChange={(e) => onChange(withRole(mapping, c.name, e.target.value as Role))}
                    >
                      <option value="input" disabled={!usable}>
                        Input
                      </option>
                      <option value="target" disabled={!usable}>
                        Target
                      </option>
                      <option value="context" disabled={!usable}>
                        Context
                      </option>
                      <option value="id" disabled={!usable || !canBeId(c)}>
                        Row id
                      </option>
                      <option value="ignore">Ignore</option>
                    </select>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <dl className="mt-3 grid gap-x-6 gap-y-1 text-xs text-ink-soft sm:grid-cols-2 xl:grid-cols-4">
        <div>
          <dt className="inline font-semibold text-ink">Input</dt> <dd className="inline">what the workflow reads. At least one.</dd>
        </div>
        <div>
          <dt className="inline font-semibold text-ink">Target</dt>{" "}
          <dd className="inline">
            the expected output it is scored against; never shown to the workflow. One for classification and QA, one per field for extraction.
          </dd>
        </div>
        <div>
          <dt className="inline font-semibold text-ink">Context</dt>{" "}
          <dd className="inline">optional supporting text the workflow may read; not scored.</dd>
        </div>
        <div>
          <dt className="inline font-semibold text-ink">Row id</dt> <dd className="inline">a stable id per row, used to split rows reproducibly.</dd>
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
