"""Dataset export formats and the CSV formula-injection rule (#33).

The rule is applied twice, with one definition (``is_formula_like`` / ``neutralize_cell``):

* **On CSV ingest** (``neutralize_rows``, called by ``DatasetService.upload`` after the
  secrets / PII policy and before serialization, hashing and storage): every formula-like cell
  is rewritten to its neutralized form, so the stored canonical bytes, the content hash, the
  row ids, the splits and every row a workflow sees hold the *neutralized* value. No copy of
  the formula-capable upload is kept. JSONL uploads are not rewritten (JSON is not evaluated by
  spreadsheets); their formula-like values are only counted.
* **On CSV export** (``safe_csv``): the same rule again, so a version stored before ingest
  neutralization existed (a pre-#33 blob) is still exported safely. A neutralized value is not
  formula-like, so the rule is idempotent: a version admitted by this code exports with zero
  further rewrites.

Export formats (``GET /api/v1/datasets/{id}/versions/{v}/export?format=...``):

``original``  the stored canonical bytes, byte for byte (``sha256`` == the version's
              ``content_hash``), labelled ``X-Wynk-Export-Safety: canonical``. For a CSV
              version these bytes are already neutralized. A legacy CSV version whose stored
              bytes still hold a formula-like cell is refused (``409 unsafe_original_export``):
              no API path re-emits formula-capable CSV content.
``csv``       the logical rows as CSV with every formula-like cell neutralized (rule below)
              (``X-Wynk-Export-Safety: spreadsheet-neutralized``).
``jsonl``     the logical rows as JSON lines (``serialize_rows``); JSON is not evaluated by
              spreadsheets, so values are not rewritten.

The neutralization rule (OWASP "CSV injection"), applied to every CSV cell ``v``:

    v is formula-like  iff  v.lstrip(" ") starts with one of
                            =  +  -  @  TAB  CR  (U+003D U+002B U+002D U+0040 U+0009 U+000D)
                            ＝ ＋ － ＠        (the full-width forms U+FF1D U+FF0B U+FF0D U+FF20)
                        and v is not a plain signed number (``-12``, ``+3.5``, ``-1e3``,
                        ``-.5``: ``_SIGNED_NUMBER``), which a spreadsheet reads as a number.
    a formula-like v is rewritten as  "'" + v   (a leading apostrophe: rendered as text)

Header cells are column names (identifiers), which can never be formula-like.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from core.canonical import canonical_json
from core.dataset import DatasetFormat
from ingestion.parse import serialize_rows

EXPORT_FORMATS = ("original", "csv", "jsonl")
_SIGNED_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "＝", "＋", "－", "＠")


def is_formula_like(text: str) -> bool:
    return text.lstrip(" ").startswith(FORMULA_PREFIXES) and _SIGNED_NUMBER.fullmatch(text) is None


def neutralize_cell(text: str) -> str:
    return "'" + text if is_formula_like(text) else text


def neutralize_rows(
    names: list[str], rows: tuple[dict[str, Any], ...]
) -> tuple[tuple[dict[str, Any], ...], int]:
    """CSV ingest: the rows with every formula-like cell neutralized, and how many were.
    ``rows`` itself (unchanged) when there is nothing to neutralize."""
    out: list[dict[str, Any]] = []
    neutralized = 0
    for row in rows:
        new = dict(row)
        for name in names:
            value = row.get(name)
            if isinstance(value, str) and is_formula_like(value):
                new[name] = "'" + value
                neutralized += 1
        out.append(new)
    return (tuple(out), neutralized) if neutralized else (rows, 0)


def _cell_text(value: Any) -> str:
    """A logical value as CSV text: JSON scalars / structures in their canonical JSON form."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return canonical_json(value)


def safe_csv(names: list[str], rows: Iterable[Mapping[str, Any]]) -> tuple[bytes, int]:
    """Spreadsheet-safe CSV (UTF-8, ``\\n``, minimal quoting) and how many cells were
    neutralized."""
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(names)
    neutralized = 0
    for row in rows:
        cells = []
        for name in names:
            text = _cell_text(row.get(name))
            safe = neutralize_cell(text)
            neutralized += safe is not text
            cells.append(safe)
        writer.writerow(cells)
    return out.getvalue().encode("utf-8"), neutralized


def jsonl(names: list[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    return serialize_rows(DatasetFormat.JSONL, names, rows)
