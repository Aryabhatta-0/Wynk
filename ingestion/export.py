"""Dataset export formats and the CSV formula-injection rule (#33).

Storage and experiments use the logical dataset exactly as admitted: a value such as
``=HYPERLINK(...)`` or ``@SUM(A1)`` is kept verbatim in the stored canonical bytes, in the content
hash and in every row a workflow sees. Only the *spreadsheet-facing* export rewrites it.

Export formats (``GET /api/v1/datasets/{id}/versions/{v}/export?format=...``):

``original``  the stored canonical bytes, byte for byte (``sha256`` == the version's
              ``content_hash``). For verification and reproduction; NOT spreadsheet-safe, so it is
              served as ``application/octet-stream`` with ``X-Wynk-Export-Safety: raw``.
``csv``       the logical rows as CSV with every formula-like cell neutralized (rule below).
              Not byte-identical to the stored bytes, and it says so
              (``X-Wynk-Export-Safety: spreadsheet-neutralized``).
``jsonl``     the logical rows as JSON lines (``serialize_rows``); JSON is not evaluated by
              spreadsheets, so values are not rewritten.

The neutralization rule (OWASP "CSV injection"), applied to every exported CSV cell ``v``:

    v is formula-like  iff  v.lstrip(" ") starts with one of
                            =  +  -  @  TAB  CR  (U+003D U+002B U+002D U+0040 U+0009 U+000D)
                            ＝ ＋ － ＠        (the full-width forms U+FF1D U+FF0B U+FF0D U+FF20)
                        and v is not a plain signed number (``-12``, ``+3.5``, ``-1e3``,
                        ``-.5``: ``_SIGNED_NUMBER``), which a spreadsheet reads as a number.
    a formula-like v is exported as  "'" + v   (a leading apostrophe: rendered as text)

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
