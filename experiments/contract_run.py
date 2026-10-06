"""Drive a search from a ``TaskContract`` alone: the contract, its dataset bytes and its splits.

    contract_suite(contract, splits, data) -> (ContractSuite, References)

* The dataset being executed is the contract's, byte for byte: ``data`` must hash to
  ``contract.dataset.content_hash`` and parse (``ingestion.parse``) to its format, columns and
  row count. Row ids come from the contract's ``id_column`` (or the ingestion row-id scheme).
* Each row is split at this boundary: input + context columns become the ``ExampleInput`` that
  search and execution see; target columns become ``References``, held only by the evaluator.
* The splits must be made for this dataset (``ContractSuite`` checks the dataset identity) and
  assign every row; their roles then decide feedback / selection / reporting.

Everything else - prompt text, admission, caps, evaluator, objective, hard limits - is read from
the contract by ``core.run_contract`` and ``evaluation.contract_eval``. There is no other input.
"""

from __future__ import annotations

from typing import Any

from core.dataset import ColumnType, DatasetFormat, DatasetSplits
from core.run_contract import ContractSuite, ExampleInput, ExecutionTask
from core.task_contract import ContractError, TaskContract
from evaluation.contract_eval import References
from ingestion.parse import (
    IngestLimits,
    column_row_ids,
    generated_row_ids,
    parse_dataset,
    sha256_bytes,
)

_CSV_CELL = {
    ColumnType.STRING: str,
    ColumnType.DATE: str,
    ColumnType.INTEGER: int,
    ColumnType.NUMBER: float,
    ColumnType.BOOLEAN: lambda v: v.lower() == "true",
}


def load_rows(
    contract: TaskContract, data: bytes, limits: IngestLimits | None = None
) -> list[tuple[str, dict[str, Any]]]:
    """``(row id, typed row)`` for every row of the contract's dataset, in file order."""
    ds = contract.dataset
    if ds.format is DatasetFormat.WYNK_SNAPSHOT:
        raise ContractError("wynk_snapshot datasets enter through benchmarks.legacy_adapter")
    if sha256_bytes(data) != ds.content_hash:
        raise ContractError(f"bytes do not match dataset {ds.dataset_id} v{ds.dataset_version}")
    parsed = parse_dataset(data, ds.format, limits or IngestLimits())
    if parsed.row_count != ds.row_count:
        raise ContractError(f"dataset has {parsed.row_count} rows, contract says {ds.row_count}")
    missing = {c.name for c in ds.columns} - {c.name for c in parsed.columns}
    if missing:
        raise ContractError(f"dataset is missing columns {sorted(missing)}")
    row_ids = column_row_ids(parsed, ds.id_column) if ds.id_column else generated_row_ids(parsed)

    def cell(column: str, value: Any) -> Any:
        if value is None or ds.format is not DatasetFormat.CSV:
            return value
        convert = _CSV_CELL.get(ds.column(column).type)
        if convert is None:
            raise ContractError(f"column {column!r} cannot be read from csv")
        return convert(value)

    return [
        (rid, {c.name: cell(c.name, row[c.name]) for c in ds.columns})
        for rid, row in zip(row_ids, parsed.rows, strict=True)
    ]


def contract_suite(
    contract: TaskContract,
    splits: DatasetSplits,
    data: bytes,
    *,
    limits: IngestLimits | None = None,
) -> tuple[ContractSuite, References]:
    """The search unit for ``contract`` and the evaluator-side expected values of every row."""
    ds = contract.dataset
    inputs = (*ds.input_columns, *ds.context_columns)
    tasks: list[ExecutionTask] = []
    expected: dict[str, dict[str, Any]] = {}
    for rid, row in load_rows(contract, data, limits):
        values = {name: row[name] for name in inputs if row[name] is not None}
        tasks.append(
            ExecutionTask(contract=contract, example=ExampleInput(row_id=rid, values=values))
        )
        expected[rid] = {name: row[name] for name in ds.target_columns}
    suite = ContractSuite(name=contract.task_id, tasks=tuple(tasks), splits=splits)
    return suite, References(expected)
