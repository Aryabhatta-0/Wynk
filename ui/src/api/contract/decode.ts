/*
  Runtime checks of product API responses against `wire.ts`, applied by the live adapter before
  anything is mapped. The Python models are strict (extra="forbid") and dump every field, so a
  response must carry exactly the expected keys: a renamed, added or dropped field fails here,
  loudly, instead of becoming `undefined` on a screen.
*/
import type {
  ColumnProfileWire,
  ColumnSpecWire,
  ColumnTypeWire,
  DatasetListWire,
  DatasetSpecWire,
  DatasetSplitWire,
  DatasetSplitsWire,
  DatasetVersionRecordWire,
  DatasetViewWire,
  ErrorResponseWire,
  ProjectListWire,
  ProjectRecordWire,
  SplitPlanWire,
  SplitsListWire,
  SplitsRecordWire,
  UploadRecordWire,
} from "./wire";

export class WireShapeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WireShapeError";
  }
}

type Rec = Record<string, unknown>;

function fail(path: string, expected: string, got: unknown): never {
  const shown = got === null ? "null" : Array.isArray(got) ? "array" : typeof got;
  throw new WireShapeError(`${path}: expected ${expected}, got ${shown}`);
}

function object(v: unknown, path: string, keys: readonly string[]): Rec {
  if (typeof v !== "object" || v === null || Array.isArray(v)) fail(path, "an object", v);
  const o = v as Rec;
  const missing = keys.filter((k) => !(k in o));
  const extra = Object.keys(o).filter((k) => !keys.includes(k));
  if (missing.length || extra.length)
    throw new WireShapeError(
      `${path}: fields differ from the contract${missing.length ? `; missing ${missing.join(", ")}` : ""}${extra.length ? `; unexpected ${extra.join(", ")}` : ""}`,
    );
  return o;
}

const str = (v: unknown, path: string): string => (typeof v === "string" ? v : fail(path, "a string", v));
const optStr = (v: unknown, path: string): string | null => (v === null ? null : str(v, path));
const bool = (v: unknown, path: string): boolean => (typeof v === "boolean" ? v : fail(path, "a boolean", v));
const int = (v: unknown, path: string): number => (Number.isInteger(v) ? (v as number) : fail(path, "an integer", v));
function arr<T>(v: unknown, path: string, item: (x: unknown, p: string) => T): T[] {
  if (!Array.isArray(v)) fail(path, "an array", v);
  return v.map((x, i) => item(x, `${path}[${i}]`));
}
function oneOf<T extends string>(v: unknown, path: string, values: readonly T[]): T {
  if (typeof v === "string" && (values as readonly string[]).includes(v)) return v as T;
  return fail(path, values.join(" | "), v);
}
const sha256 = (v: unknown, path: string): string => {
  const s = str(v, path);
  if (!/^[0-9a-f]{64}$/.test(s)) throw new WireShapeError(`${path}: expected a sha256 hex digest`);
  return s;
};

const COLUMN_TYPES: readonly ColumnTypeWire[] = ["string", "integer", "number", "boolean", "date", "string_list", "json"];
const FORMATS = ["csv", "jsonl", "wynk_snapshot"] as const;
const ROLES = ["optimization", "validation", "test"] as const;

export function project(v: unknown, path = "project"): ProjectRecordWire {
  const o = object(v, path, ["project_id", "name", "description", "created_at"]);
  return {
    project_id: str(o.project_id, `${path}.project_id`),
    name: str(o.name, `${path}.name`),
    description: str(o.description, `${path}.description`),
    created_at: str(o.created_at, `${path}.created_at`),
  };
}

export function projectList(v: unknown): ProjectListWire {
  const o = object(v, "body", ["projects"]);
  return { projects: arr(o.projects, "projects", project) };
}

function columnProfile(v: unknown, path: string): ColumnProfileWire {
  const o = object(v, path, ["name", "type", "nullable", "null_count"]);
  return {
    name: str(o.name, `${path}.name`),
    type: oneOf(o.type, `${path}.type`, COLUMN_TYPES),
    nullable: bool(o.nullable, `${path}.nullable`),
    null_count: int(o.null_count, `${path}.null_count`),
  };
}

export function upload(v: unknown, path = "upload"): UploadRecordWire {
  const o = object(v, path, [
    "upload_id",
    "project_id",
    "filename",
    "format",
    "content_hash",
    "size_bytes",
    "row_count",
    "columns",
    "preview",
    "parser_version",
    "created_at",
  ]);
  return {
    upload_id: str(o.upload_id, `${path}.upload_id`),
    project_id: str(o.project_id, `${path}.project_id`),
    filename: optStr(o.filename, `${path}.filename`),
    format: oneOf(o.format, `${path}.format`, FORMATS),
    content_hash: sha256(o.content_hash, `${path}.content_hash`),
    size_bytes: int(o.size_bytes, `${path}.size_bytes`),
    row_count: int(o.row_count, `${path}.row_count`),
    columns: arr(o.columns, `${path}.columns`, columnProfile),
    preview: arr(o.preview, `${path}.preview`, (x, p) => {
      if (typeof x !== "object" || x === null || Array.isArray(x)) fail(p, "an object", x);
      return x as Record<string, unknown>;
    }),
    parser_version: str(o.parser_version, `${path}.parser_version`),
    created_at: str(o.created_at, `${path}.created_at`),
  };
}

function columnSpec(v: unknown, path: string): ColumnSpecWire {
  const o = object(v, path, ["name", "type", "nullable"]);
  return { name: str(o.name, `${path}.name`), type: oneOf(o.type, `${path}.type`, COLUMN_TYPES), nullable: bool(o.nullable, `${path}.nullable`) };
}

function datasetSpec(v: unknown, path: string): DatasetSpecWire {
  const o = object(v, path, [
    "schema_version",
    "dataset_id",
    "dataset_version",
    "name",
    "content_hash",
    "format",
    "columns",
    "id_column",
    "input_columns",
    "target_columns",
    "context_columns",
    "row_count",
    "metadata",
  ]);
  const names = (x: unknown, p: string) => arr(x, p, str);
  if (typeof o.metadata !== "object" || o.metadata === null || Array.isArray(o.metadata)) fail(`${path}.metadata`, "an object", o.metadata);
  return {
    schema_version: oneOf(o.schema_version, `${path}.schema_version`, ["datasetspec/1"] as const),
    dataset_id: str(o.dataset_id, `${path}.dataset_id`),
    dataset_version: int(o.dataset_version, `${path}.dataset_version`),
    name: str(o.name, `${path}.name`),
    content_hash: sha256(o.content_hash, `${path}.content_hash`),
    format: oneOf(o.format, `${path}.format`, FORMATS),
    columns: arr(o.columns, `${path}.columns`, columnSpec),
    id_column: optStr(o.id_column, `${path}.id_column`),
    input_columns: names(o.input_columns, `${path}.input_columns`),
    target_columns: names(o.target_columns, `${path}.target_columns`),
    context_columns: names(o.context_columns, `${path}.context_columns`),
    row_count: int(o.row_count, `${path}.row_count`),
    metadata: o.metadata as DatasetSpecWire["metadata"],
  };
}

export function datasetVersion(v: unknown, path = "version"): DatasetVersionRecordWire {
  const o = object(v, path, ["project_id", "upload_id", "spec", "identity_hash", "row_id_source", "row_id_scheme", "row_ids_hash", "created_at"]);
  return {
    project_id: str(o.project_id, `${path}.project_id`),
    upload_id: str(o.upload_id, `${path}.upload_id`),
    spec: datasetSpec(o.spec, `${path}.spec`),
    identity_hash: sha256(o.identity_hash, `${path}.identity_hash`),
    row_id_source: oneOf(o.row_id_source, `${path}.row_id_source`, ["column", "generated"] as const),
    row_id_scheme: optStr(o.row_id_scheme, `${path}.row_id_scheme`),
    row_ids_hash: sha256(o.row_ids_hash, `${path}.row_ids_hash`),
    created_at: str(o.created_at, `${path}.created_at`),
  };
}

export function datasetView(v: unknown, path = "dataset"): DatasetViewWire {
  const o = object(v, path, ["dataset_id", "project_id", "name", "latest_version", "versions"]);
  return {
    dataset_id: str(o.dataset_id, `${path}.dataset_id`),
    project_id: str(o.project_id, `${path}.project_id`),
    name: str(o.name, `${path}.name`),
    latest_version: int(o.latest_version, `${path}.latest_version`),
    versions: arr(o.versions, `${path}.versions`, datasetVersion),
  };
}

export function datasetList(v: unknown): DatasetListWire {
  const o = object(v, "body", ["datasets"]);
  return { datasets: arr(o.datasets, "datasets", datasetView) };
}

function splitPlan(v: unknown, path: string): SplitPlanWire {
  const o = object(v, path, ["seed", "validation_bps", "test_bps"]);
  return { seed: int(o.seed, `${path}.seed`), validation_bps: int(o.validation_bps, `${path}.validation_bps`), test_bps: int(o.test_bps, `${path}.test_bps`) };
}

function split(v: unknown, path: string): DatasetSplitWire {
  const o = object(v, path, ["split_id", "role", "row_ids"]);
  return { split_id: str(o.split_id, `${path}.split_id`), role: oneOf(o.role, `${path}.role`, ROLES), row_ids: arr(o.row_ids, `${path}.row_ids`, str) };
}

function datasetSplits(v: unknown, path: string): DatasetSplitsWire {
  const o = object(v, path, ["schema_version", "dataset_hash", "method", "plan", "splits"]);
  return {
    schema_version: oneOf(o.schema_version, `${path}.schema_version`, ["datasetsplits/1"] as const),
    dataset_hash: sha256(o.dataset_hash, `${path}.dataset_hash`),
    method: oneOf(o.method, `${path}.method`, ["seeded_hash/1", "explicit"] as const),
    plan: o.plan === null ? null : splitPlan(o.plan, `${path}.plan`),
    splits: arr(o.splits, `${path}.splits`, split),
  };
}

export function splitsRecord(v: unknown, path = "splits"): SplitsRecordWire {
  const o = object(v, path, ["dataset_id", "dataset_version", "splits_hash", "splits", "sizes", "created_at"]);
  if (typeof o.sizes !== "object" || o.sizes === null || Array.isArray(o.sizes)) fail(`${path}.sizes`, "an object", o.sizes);
  const sizes: SplitsRecordWire["sizes"] = {};
  for (const [role, n] of Object.entries(o.sizes as Rec)) sizes[oneOf(role, `${path}.sizes key`, ROLES)] = int(n, `${path}.sizes.${role}`);
  return {
    dataset_id: str(o.dataset_id, `${path}.dataset_id`),
    dataset_version: int(o.dataset_version, `${path}.dataset_version`),
    splits_hash: sha256(o.splits_hash, `${path}.splits_hash`),
    splits: datasetSplits(o.splits, `${path}.splits`),
    sizes,
    created_at: str(o.created_at, `${path}.created_at`),
  };
}

export function splitsList(v: unknown): SplitsListWire {
  const o = object(v, "body", ["splits"]);
  return { splits: arr(o.splits, "splits", splitsRecord) };
}

/** The error envelope, or null when the body is not one (e.g. a proxy's own error page). */
export function errorResponse(v: unknown): ErrorResponseWire | null {
  try {
    const o = object(v, "body", ["error"]);
    const e = object(o.error, "error", ["code", "message", "details"]);
    if (typeof e.details !== "object" || e.details === null || Array.isArray(e.details)) return null;
    return { error: { code: str(e.code, "error.code"), message: str(e.message, "error.message"), details: e.details as ErrorResponseWire["error"]["details"] } };
  } catch {
    return null;
  }
}
