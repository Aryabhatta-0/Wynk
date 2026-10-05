"""Parse and inspect an uploaded dataset from its exact bytes. Deterministic, fail closed.

Only ``csv`` and ``jsonl`` are accepted. Nothing here trusts the client: the content hash, the row
count, the column types and nullability are all computed from the bytes, and any row that cannot
be read rejects the whole file (rows are never skipped).

Rules
-----
* Encoding: strict UTF-8 (a leading UTF-8 BOM is allowed and ignored by the parser; the hash still
  covers it). NUL characters are refused.
* CSV: comma-delimited, ``"``-quoted (RFC 4180), first record is the header. Every record must have
  exactly as many fields as the header; a blank line is a malformed record. An empty cell is null.
  Text after a closing quote is malformed; a quote inside an unquoted field is kept literally.
* JSONL: one JSON object per line (``\\n`` or ``\\r\\n``); a blank line is malformed, a single
  trailing newline is not. Duplicate keys, ``NaN`` / ``Infinity`` and non-finite numbers are
  refused. A key missing from a row is null in that row. Columns are ordered by first appearance.
* Column names must be unique identifiers (``core.dataset.COLUMN_NAME``), so every column can map
  onto a schema field.

Type inference (contract types only, see ``core.dataset.ColumnType``), over every non-null value:

    CSV    boolean (true/false, any case) > integer > number > date (YYYY-MM-DD) > string
    JSONL  boolean > integer > number (ints and floats mixed) > date (all strings ISO dates)
           > string > string_list (arrays of strings) ; any other mix -> json

A column with no non-null value is a nullable ``string``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from core.canonical import canonical_json, sha256_hex
from core.dataset import COLUMN_NAME, ColumnSpec, ColumnType, DatasetFormat

PARSER_VERSION = "ingest-parse/1"
ROW_ID_SCHEME = "row-content-sha256/1"  # generated row ids: see ``generated_row_ids``
MAX_ROW_ID_CHARS = 256

SUPPORTED_FORMATS = (DatasetFormat.CSV, DatasetFormat.JSONL)

_COLUMN_NAME = re.compile(COLUMN_NAME)
_INTEGER = re.compile(r"^(0|-?[1-9][0-9]*)$")  # no leading zeros, no "+", no "-0"
_NUMBER = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$")
_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_INT64 = 2**63


class IngestLimits(BaseModel):
    """Configurable bounds on what one upload may cost to inspect."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_upload_bytes: PositiveInt = 50 * 1024 * 1024
    max_columns: PositiveInt = 512
    preview_rows: NonNegativeInt = Field(default=20, le=1000)
    preview_cell_chars: PositiveInt = Field(default=200, le=10_000)


class IngestError(Exception):
    """A user-correctable problem with an upload or a mapping. ``code`` is stable."""

    def __init__(self, code: str, message: str, **details: str | int | None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class ColumnProfile(BaseModel):
    """One column as inspected: its contract type plus how many rows hold no value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=COLUMN_NAME)
    type: ColumnType
    nullable: bool
    null_count: NonNegativeInt

    def spec(self) -> ColumnSpec:
        return ColumnSpec(name=self.name, type=self.type, nullable=self.nullable)


@dataclass(frozen=True)
class ParsedDataset:
    """Every row of the file, as read. CSV values are ``str | None``; JSONL values are JSON."""

    format: DatasetFormat
    columns: tuple[ColumnProfile, ...]
    rows: tuple[dict[str, Any], ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def preview(self, limits: IngestLimits) -> tuple[dict[str, Any], ...]:
        """The first ``preview_rows`` rows, long cells truncated to ``preview_cell_chars``."""
        n = limits.preview_cell_chars

        def cell(v: Any) -> Any:
            if isinstance(v, str):
                return v if len(v) <= n else v[:n] + "…"
            if isinstance(v, (list, dict)):
                text = canonical_json(v)
                return v if len(text) <= n else text[:n] + "…"
            return v

        names = [c.name for c in self.columns]
        return tuple({k: cell(row[k]) for k in names} for row in self.rows[: limits.preview_rows])


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- parsing ------------------------------------------------------------------------------------
def parse_dataset(data: bytes, fmt: DatasetFormat | str, limits: IngestLimits) -> ParsedDataset:
    """Parse + inspect ``data``. Raises ``IngestError`` on anything malformed."""
    try:
        fmt = DatasetFormat(fmt)
    except ValueError:
        fmt = None  # type: ignore[assignment]
    if fmt not in SUPPORTED_FORMATS:
        raise IngestError("unsupported_format", "format must be one of: csv, jsonl")
    if len(data) > limits.max_upload_bytes:
        raise IngestError(
            "payload_too_large",
            f"upload is larger than {limits.max_upload_bytes} bytes",
            limit_bytes=limits.max_upload_bytes,
        )
    text = _decode(data)
    if fmt is DatasetFormat.CSV:
        names, rows = _parse_csv(text, limits)
        infer: Callable[[list[Any]], ColumnType] = _infer_csv
    else:
        names, rows = _parse_jsonl(text, limits)
        infer = _infer_json
    if not rows:
        raise IngestError("no_rows", "the file has no data rows")
    columns = []
    for name in names:
        values = [r[name] for r in rows]
        present = [v for v in values if v is not None]
        nulls = len(values) - len(present)
        columns.append(
            ColumnProfile(
                name=name,
                type=infer(present) if present else ColumnType.STRING,
                nullable=nulls > 0,
                null_count=nulls,
            )
        )
    return ParsedDataset(format=fmt, columns=tuple(columns), rows=tuple(rows))


def _decode(data: bytes) -> str:
    if not data:
        raise IngestError("empty_file", "the file is empty")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestError(
            "invalid_encoding", "the file is not valid UTF-8", byte_offset=exc.start
        ) from None
    if text.startswith("﻿"):
        text = text[1:]
    if "\x00" in text:
        raise IngestError("invalid_encoding", "the file contains NUL characters")
    if not text.strip():
        raise IngestError("empty_file", "the file holds no data")
    return text


def _check_columns(names: list[str], limits: IngestLimits) -> None:
    if len(names) > limits.max_columns:
        raise IngestError(
            "too_many_columns",
            f"the file has more than {limits.max_columns} columns",
            limit=limits.max_columns,
        )
    seen: set[str] = set()
    for name in names:
        if not _COLUMN_NAME.match(name):
            raise IngestError(
                "invalid_column_name",
                f"column name {name!r} must start with a letter or _ and use only letters, "
                "digits and _ (at most 64 characters)",
                column=name[:100],
            )
        if name in seen:
            raise IngestError("duplicate_column", f"column {name!r} appears twice", column=name)
        seen.add(name)


def _parse_csv(text: str, limits: IngestLimits) -> tuple[list[str], list[dict[str, Any]]]:
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader, None)
        if not header:
            raise IngestError("malformed_csv", "the first line must be a header", line=1)
        _check_columns(header, limits)
        rows: list[dict[str, Any]] = []
        for record in reader:
            if len(record) != len(header):
                raise IngestError(
                    "malformed_csv",
                    f"line {reader.line_num} has {len(record)} fields, the header has "
                    f"{len(header)}" + (" (blank line)" if not record else ""),
                    line=reader.line_num,
                )
            rows.append({k: (v if v != "" else None) for k, v in zip(header, record, strict=True)})
    except csv.Error as exc:
        raise IngestError(
            "malformed_csv", f"line {reader.line_num}: {exc}", line=reader.line_num
        ) from None
    return header, rows


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"number {text} is out of range")
    return value


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}")
        out[k] = v
    return out


def _parse_jsonl(text: str, limits: IngestLimits) -> tuple[list[str], list[dict[str, Any]]]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # one trailing newline terminates the last record
    names: list[str] = []
    known: set[str] = set()
    objects: list[dict[str, Any]] = []
    for i, raw in enumerate(lines, start=1):
        line = raw[:-1] if raw.endswith("\r") else raw
        if not line.strip():
            raise IngestError("malformed_jsonl", f"line {i} is blank", line=i)
        try:
            obj = json.loads(
                line,
                object_pairs_hook=_no_duplicate_keys,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
        except (ValueError, RecursionError) as exc:
            raise IngestError("malformed_jsonl", f"line {i}: {exc}"[:300], line=i) from None
        if not isinstance(obj, dict):
            raise IngestError("malformed_jsonl", f"line {i} is not a JSON object", line=i)
        for key in obj:
            if key not in known:
                known.add(key)
                names.append(key)
                if len(names) > limits.max_columns:
                    _check_columns(names, limits)
        objects.append(obj)
    _check_columns(names, limits)
    return names, [{k: obj.get(k) for k in names} for obj in objects]


# -- type inference -----------------------------------------------------------------------------
def _is_date(s: str) -> bool:
    if not _DATE.match(s):
        return False
    try:
        date.fromisoformat(s)
    except ValueError:
        return False
    return True


def _all(values: Iterable[Any], pred: Callable[[Any], bool]) -> bool:
    return all(pred(v) for v in values)


def _infer_csv(values: list[str]) -> ColumnType:
    if _all(values, lambda v: v.lower() in ("true", "false")):
        return ColumnType.BOOLEAN
    if _all(values, lambda v: bool(_INTEGER.match(v)) and abs(int(v)) < _INT64):
        return ColumnType.INTEGER
    if _all(values, lambda v: bool(_NUMBER.match(v)) and math.isfinite(float(v))):
        return ColumnType.NUMBER
    if _all(values, _is_date):
        return ColumnType.DATE
    return ColumnType.STRING


def _infer_json(values: list[Any]) -> ColumnType:
    if _all(values, lambda v: isinstance(v, bool)):
        return ColumnType.BOOLEAN
    if _all(values, lambda v: isinstance(v, int) and not isinstance(v, bool) and abs(v) < _INT64):
        return ColumnType.INTEGER
    if _all(
        values,
        lambda v: (
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and (isinstance(v, float) or abs(v) < _INT64)
        ),
    ):
        return ColumnType.NUMBER
    if _all(values, lambda v: isinstance(v, str)):
        return ColumnType.DATE if _all(values, _is_date) else ColumnType.STRING
    if _all(values, lambda v: isinstance(v, list) and _all(v, lambda x: isinstance(x, str))):
        return ColumnType.STRING_LIST
    return ColumnType.JSON


# -- row identity -------------------------------------------------------------------------------
def column_row_ids(parsed: ParsedDataset, id_column: str) -> tuple[str, ...]:
    """Row ids taken from a supplied id column. Every value must be present, non-empty, unique."""
    col = parsed.column(id_column)
    if col is None:
        raise IngestError("unknown_column", f"no column {id_column!r}", column=id_column)
    if col.nullable or col.type not in (ColumnType.STRING, ColumnType.INTEGER):
        raise IngestError(
            "invalid_id_column",
            f"id column {id_column!r} must be a string or integer column with a value in every "
            f"row (it is {'a nullable ' if col.nullable else ''}{col.type.value})",
            column=id_column,
        )
    ids: list[str] = []
    for i, row in enumerate(parsed.rows, start=1):
        value = row[id_column]
        rid = str(int(value)) if col.type is ColumnType.INTEGER else value
        if not rid or len(rid) > MAX_ROW_ID_CHARS:
            raise IngestError(
                "invalid_id_column",
                f"row {i} has an empty id or one longer than {MAX_ROW_ID_CHARS} characters",
                column=id_column,
                row=i,
            )
        ids.append(rid)
    _require_unique(ids, id_column)
    return tuple(ids)


def generated_row_ids(parsed: ParsedDataset) -> tuple[str, ...]:
    """Stable ids derived from row content: ``r-`` + 24 hex of sha256(canonical row values).

    The k-th (k >= 2) identical copy of a row, in file order, gets the suffix ``-k``. The ids are a
    pure function of the file bytes, so they are stable for an immutable dataset version, and (with
    the same columns) a row keeps its id when other rows are added, removed or reordered.
    """
    names = [c.name for c in parsed.columns]
    seen: dict[str, int] = {}
    ids: list[str] = []
    for row in parsed.rows:
        base = "r-" + sha256_hex("wynk-row/1:" + canonical_json({k: row[k] for k in names}))[:24]
        seen[base] = seen.get(base, 0) + 1
        ids.append(base if seen[base] == 1 else f"{base}-{seen[base]}")
    _require_unique(ids, None)  # a 96-bit prefix collision between different rows: fail closed
    return tuple(ids)


def _require_unique(ids: list[str], column: str | None) -> None:
    if len(set(ids)) == len(ids):
        return
    seen: set[str] = set()
    dup = ""
    for rid in ids:
        if rid in seen:
            dup = rid
            break
        seen.add(rid)
    raise IngestError(
        "duplicate_row_id",
        f"row id {dup!r} appears more than once"
        + (f" in id column {column!r}" if column else " (generated id collision)"),
        column=column,
        row_id=dup[:MAX_ROW_ID_CHARS],
    )


def row_ids_hash(row_ids: tuple[str, ...]) -> str:
    return sha256_hex(canonical_json(list(row_ids)))
