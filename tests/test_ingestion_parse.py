"""Upload parsing + inspection: formats, malformed input, limits, type/null inference, row ids."""

import hashlib
import json

import pytest

from core.dataset import ColumnType, DatasetFormat
from ingestion.parse import (
    IngestError,
    IngestLimits,
    column_row_ids,
    generated_row_ids,
    parse_dataset,
    sha256_bytes,
)

LIMITS = IngestLimits()


def parse_csv(text: str | bytes, limits: IngestLimits = LIMITS):
    data = text.encode("utf-8") if isinstance(text, str) else text
    return parse_dataset(data, "csv", limits)


def parse_jsonl(rows: list[dict] | str, limits: IngestLimits = LIMITS):
    text = rows if isinstance(rows, str) else "".join(json.dumps(r) + "\n" for r in rows)
    return parse_dataset(text.encode("utf-8"), "jsonl", limits)


def types(parsed) -> dict[str, tuple[str, bool]]:
    return {c.name: (c.type.value, c.nullable) for c in parsed.columns}


def code_of(fn, *args) -> str:
    with pytest.raises(IngestError) as info:
        fn(*args)
    return info.value.code


# -- formats ------------------------------------------------------------------------------------
def test_csv_parses_header_rows_and_quoting():
    p = parse_csv('id,text,label\n1,"hello, world",a\r\n2,"say ""hi""\nthere",b\n')
    assert p.format is DatasetFormat.CSV
    assert p.row_count == 2
    assert [c.name for c in p.columns] == ["id", "text", "label"]
    assert p.rows[0]["text"] == "hello, world"
    assert p.rows[1]["text"] == 'say "hi"\nthere'


def test_csv_quote_inside_an_unquoted_field_is_kept_literally():
    assert parse_csv('id,text\n1,a"b\n').rows[0]["text"] == 'a"b'


def test_csv_without_trailing_newline_and_with_bom():
    p = parse_csv(b"\xef\xbb\xbfid,text\n1,a\n2,b")
    assert [c.name for c in p.columns] == ["id", "text"]  # BOM is not part of the first name
    assert p.row_count == 2


def test_jsonl_parses_objects_and_orders_columns_by_first_appearance():
    p = parse_jsonl('{"b": 1, "a": "x"}\r\n{"a": "y", "c": true}\n')
    assert p.format is DatasetFormat.JSONL
    assert [c.name for c in p.columns] == ["b", "a", "c"]
    assert p.rows[1] == {"b": None, "a": "y", "c": True}  # a missing key is null


@pytest.mark.parametrize("fmt", ["parquet", "json", "wynk_snapshot", "CSV", ""])
def test_unsupported_formats_are_refused(fmt):
    assert code_of(parse_dataset, b"a\n1\n", fmt, LIMITS) == "unsupported_format"


# -- malformed input ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("data", "code"),
    [
        (b"", "empty_file"),
        (b"   \n\n", "empty_file"),
        (b"id,text\n", "no_rows"),
        (b"id,text\n1,a\n2\n", "malformed_csv"),  # too few fields
        (b"id,text\n1,a,extra\n", "malformed_csv"),  # too many fields
        (b"id,text\n1,a\n\n2,b\n", "malformed_csv"),  # a blank line is not silently skipped
        (b'id,text\n1,"unterminated\n', "malformed_csv"),
        (b'id,text\n1,"a"b\n', "malformed_csv"),  # text after a closing quote (strict)
        (b"id,text\n1,\xff\xfe\n", "invalid_encoding"),
        (b"id,text\n1,a\x00b\n", "invalid_encoding"),
        (b"\nid,text\n1,a\n", "malformed_csv"),  # header must be the first line
    ],
)
def test_malformed_csv_is_rejected(data, code):
    assert code_of(parse_csv, data) == code


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ('{"a": 1}\n[1, 2]\n', "malformed_jsonl"),  # not an object
        ('{"a": 1}\n{"a": \n', "malformed_jsonl"),  # truncated
        ('{"a": 1}\n\n{"a": 2}\n', "malformed_jsonl"),  # blank line
        ('{"a": 1}\n\n', "malformed_jsonl"),  # extra trailing blank line
        ('{"a": 1, "a": 2}\n', "malformed_jsonl"),  # duplicate key
        ('{"a": NaN}\n', "malformed_jsonl"),
        ('{"a": Infinity}\n', "malformed_jsonl"),
        ('{"a": 1e999}\n', "malformed_jsonl"),  # overflows to inf
        ('"just a string"\n', "malformed_jsonl"),
        ("\n", "empty_file"),
    ],
)
def test_malformed_jsonl_is_rejected(text, code):
    assert code_of(parse_jsonl, text) == code


def test_malformed_row_reports_its_line():
    with pytest.raises(IngestError) as info:
        parse_jsonl('{"a": 1}\n{"a": 2}\n{oops}\n')
    assert info.value.details["line"] == 3


# -- limits -------------------------------------------------------------------------------------
def test_upload_size_limit_is_enforced_exactly():
    data = b"id,text\n1,abcdef\n"
    assert parse_csv(data, IngestLimits(max_upload_bytes=len(data))).row_count == 1
    with pytest.raises(IngestError) as info:
        parse_csv(data, IngestLimits(max_upload_bytes=len(data) - 1))
    assert info.value.code == "payload_too_large"
    assert info.value.details["limit_bytes"] == len(data) - 1


def test_column_count_limit():
    header = ",".join(f"c{i}" for i in range(5))
    assert code_of(parse_csv, f"{header}\n1,2,3,4,5\n", IngestLimits(max_columns=4)) == (
        "too_many_columns"
    )
    rows = [{f"c{i}": i} for i in range(5)]
    assert code_of(parse_jsonl, rows, IngestLimits(max_columns=4)) == "too_many_columns"


def test_preview_is_bounded_in_rows_and_cell_size():
    long = "x" * 500
    text = "id,text\n" + "".join(f"{i},{long}\n" for i in range(50))
    limits = IngestLimits(preview_rows=5, preview_cell_chars=10)
    preview = parse_csv(text, limits).preview(limits)
    assert len(preview) == 5
    assert preview[0] == {"id": "0", "text": "x" * 10 + "…"}
    nested = parse_jsonl([{"j": {"k": long}}], limits).preview(limits)
    assert isinstance(nested[0]["j"], str) and len(nested[0]["j"]) == 11


# -- column names -------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("header", "code"),
    [
        ("id,text,id", "duplicate_column"),
        ("id,,label", "invalid_column_name"),  # empty
        ("id,my text,label", "invalid_column_name"),  # whitespace
        ("id,1st,label", "invalid_column_name"),  # leading digit
        ("id,a.b,label", "invalid_column_name"),
        ("id, text,label", "invalid_column_name"),  # names are not trimmed
        ("id," + "a" * 65 + ",label", "invalid_column_name"),
    ],
)
def test_invalid_and_duplicate_csv_column_names(header, code):
    assert code_of(parse_csv, f"{header}\n1,2,3\n") == code


def test_invalid_jsonl_key_is_refused():
    assert code_of(parse_jsonl, [{"ok": 1, "not ok": 2}]) == "invalid_column_name"


# -- type + null inference ----------------------------------------------------------------------
def test_csv_type_inference():
    p = parse_csv(
        "s,i,n,b,d,mixed,neg,lead0,blank,big\n"
        "a,1,1.5,true,2024-01-31,1,-3,007,,99999999999999999999\n"
        "b,2,2,FALSE,2024-02-29,x,0,1,,1\n"
        "c,30,3e2,False,1999-12-01,2.5,5,2,,2\n"
    )
    assert types(p) == {
        "s": ("string", False),
        "i": ("integer", False),
        "n": ("number", False),
        "b": ("boolean", False),
        "d": ("date", False),
        "mixed": ("string", False),
        "neg": ("integer", False),
        "lead0": ("string", False),  # "007" is not an integer literal: keep it text
        "blank": ("string", True),  # no value at all
        "big": ("number", False),  # beyond int64 -> number
    }


def test_csv_invalid_dates_and_partial_types_fall_back_to_string():
    p = parse_csv("d,n\n2024-02-30,1\n2024-01-01,1.0.0\n")
    assert types(p) == {"d": ("string", False), "n": ("string", False)}


def test_csv_nullable_columns_and_null_counts():
    p = parse_csv("id,score,note\n1,0.5,\n2,,x\n3,1,\n")
    assert types(p) == {
        "id": ("integer", False),
        "score": ("number", True),
        "note": ("string", True),
    }
    assert {c.name: c.null_count for c in p.columns} == {"id": 0, "score": 1, "note": 2}


def test_jsonl_type_inference():
    p = parse_jsonl(
        [
            {"s": "a", "i": 1, "n": 1, "b": True, "d": "2024-01-01", "l": ["x"], "j": {"k": 1}},
            {"s": "b", "i": 2, "n": 2.5, "b": False, "d": "2024-01-02", "l": [], "j": [1]},
            {"s": None, "i": 3, "n": 3, "b": True, "d": "2024-01-03", "l": ["y", "z"], "j": 1},
        ]
    )
    assert types(p) == {
        "s": ("string", True),
        "i": ("integer", False),
        "n": ("number", False),
        "b": ("boolean", False),
        "d": ("date", False),
        "l": ("string_list", False),
        "j": ("json", False),
    }


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, "1"], ColumnType.JSON),  # mixed scalar types are opaque, never coerced
        ([True, 1], ColumnType.JSON),  # a bool is not an integer
        ([["a", 1]], ColumnType.JSON),  # not a list of strings
        ([1.0, 2.0], ColumnType.NUMBER),
        (["2024-01-01", "soon"], ColumnType.STRING),
    ],
)
def test_jsonl_mixed_values(values, expected):
    p = parse_jsonl([{"v": v} for v in values])
    assert p.columns[0].type is expected


def test_inference_is_deterministic_and_order_independent_for_types():
    rows = [{"a": 1}, {"a": 2.5}, {"a": None}]
    first = parse_jsonl(rows).columns
    assert parse_jsonl(rows).columns == first
    assert parse_jsonl(list(reversed(rows))).columns == first


def test_content_hash_is_sha256_of_exact_bytes():
    data = b"\xef\xbb\xbfid,text\r\n1,a\r\n"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()
    assert sha256_bytes(data) != sha256_bytes(data.replace(b"\r\n", b"\n"))


# -- row identity -------------------------------------------------------------------------------
def test_column_row_ids_from_string_and_integer_columns():
    p = parse_csv("id,key,text\n10,a,x\n-2,b,y\n0,c,z\n")
    assert column_row_ids(p, "id") == ("10", "-2", "0")
    assert column_row_ids(p, "key") == ("a", "b", "c")
    j = parse_jsonl([{"id": 7, "t": "a"}, {"id": 8, "t": "b"}])
    assert column_row_ids(j, "id") == ("7", "8")


@pytest.mark.parametrize(
    ("text", "column", "code"),
    [
        ("id,text\n1,a\n1,b\n", "id", "duplicate_row_id"),
        ("id,text\na,x\nb,x\n", "text", "duplicate_row_id"),
        ("id,text\n1,a\n,b\n", "id", "invalid_id_column"),  # nullable
        ("id,text\n1.5,a\n2.5,b\n", "id", "invalid_id_column"),  # number ids
        ("id,text\ntrue,a\nfalse,b\n", "id", "invalid_id_column"),
        ("id,text\n1,a\n", "nope", "unknown_column"),
    ],
)
def test_invalid_id_columns(text, column, code):
    assert code_of(column_row_ids, parse_csv(text), column) == code


def test_empty_and_overlong_string_ids_are_refused():
    assert code_of(column_row_ids, parse_jsonl([{"id": ""}, {"id": "a"}]), "id") == (
        "invalid_id_column"
    )
    assert code_of(column_row_ids, parse_jsonl([{"id": "x" * 257}]), "id") == ("invalid_id_column")


def test_generated_row_ids_are_unique_stable_and_content_derived():
    text = "text,label\na,x\nb,y\na,x\na,x\n"
    ids = generated_row_ids(parse_csv(text))
    assert len(set(ids)) == 4
    assert ids[2] == ids[0] + "-2" and ids[3] == ids[0] + "-3"  # duplicate rows, file order
    assert all(r.startswith("r-") for r in ids)
    assert generated_row_ids(parse_csv(text)) == ids  # stable across parses
    # a row keeps its id when other rows are removed or reordered
    other = generated_row_ids(parse_csv("text,label\nb,y\na,x\n"))
    assert other == (ids[1], ids[0])


def test_generated_row_ids_depend_on_values_not_formatting():
    csv_ids = generated_row_ids(parse_csv("t\nhello\n"))
    assert generated_row_ids(parse_csv("t\r\nhello\r\n")) == csv_ids
    assert generated_row_ids(parse_csv("t\nhullo\n")) != csv_ids
