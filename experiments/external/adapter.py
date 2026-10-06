"""One reusable adapter: external dataset -> sanitized TaskContract -> frozen manifest -> runner.

    official file ──(pinned SHA-256, row count)──> official rows
      ──(stratified hash sampling: ids + a dataset-native stratum only)──> selected ids
      ──(sanitize: declared columns only, gold annotations dropped)──> Wynk JSONL bytes
      ──> DatasetSpec + TaskContract (from the frozen protocol) ──> seeded DatasetSplits
      ──> manifest (provenance, preprocessing hash, selected ids, dataset/split/contract hashes)

    protocol-<v>.json (canonical hash pinned in protocols.lock.json) + manifest
      ──> ExperimentPlan ──> experiments.optimization_experiment (fixed / random / ACO)

A benchmark (``Benchmark``) defines ONLY:

    source / provenance     ``source``: repository, exact revision, file, SHA-256, row count
    sanitizer / schema      ``parse``, ``sanitize``, ``columns``, ``gold_fields``, ``preprocessing``
    evaluator               ``task_type`` + the protocol's ``contract.evaluation``
    sampling                ``stratum`` (a dataset-native field) and the sampling rule's name
    fixed-baseline rule     the protocol's ``fixed_baseline_rule`` (``FIXED_RULES``)

Everything else - hashing, sampling, contract and split construction, the manifest, protocol
locking, the plan, leakage gates, the runner, budgets, artifacts - is shared and benchmark-blind.
Nothing here can depend on a model outcome: the only inputs are the official bytes, the frozen
protocol and the benchmark definition.

Leakage boundary. ``sanitize`` must return EXACTLY ``columns.all`` (checked for every row). The
target columns go to the evaluator only (``experiments.contract_run``); execution sees the input
and context columns and nothing else (``check_isolation``). Raw fields listed in ``gold_fields``
(answers, rationales, decompositions, support annotations, target-revealing metadata) never
appear in a sanitized row.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from core.canonical import canonical_hash, sha256_hex
from core.constraints import ConstraintLimits
from core.dataset import (
    ColumnSpec,
    ColumnType,
    DatasetFormat,
    DatasetSpec,
    DatasetSplits,
    SplitPlan,
    seeded_splits,
)
from core.evaluation_spec import EvaluationSpec
from core.experiment import ModelConfiguration
from core.run_contract import ContractSuite
from core.task_contract import TaskContract, TaskType
from core.task_spec import AnswerField, AnswerSchema, FieldType
from experiments.budget_ledger import ExperimentBudget
from experiments.optimization_experiment import ExperimentPlan, Strategy
from ingestion.parse import sha256_bytes


class SourceMismatch(ValueError):
    """The bytes are not the pinned official file (or the rebuilt subset is not the frozen one)."""


class LeakageError(ValueError):
    """A sanitized row or an execution task carries something execution must never see."""


@dataclass(frozen=True)
class Columns:
    """The sanitized row layout. Every column is a string column."""

    id: str
    inputs: tuple[str, ...]
    context: tuple[str, ...]
    targets: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        return (self.id, *self.inputs, *self.context, *self.targets)

    @property
    def execution(self) -> frozenset[str]:
        """What an ``ExecutionTask`` may hold: inputs + context, never the id or targets."""
        return frozenset((*self.inputs, *self.context))


class Benchmark(ABC):
    """The benchmark-specific part of an adapter. Subclasses set the class attributes and
    implement ``parse`` / ``sanitize`` / ``stratum``; they never touch the runner."""

    key: ClassVar[str]  # short slug, e.g. "mmlu-pro"
    title: ClassVar[str]
    task_family: ClassVar[str]
    metric: ClassVar[str]  # what the evaluator's quality means, for reports
    source: ClassVar[Mapping[str, Any]]  # must pin "file_sha256" and "rows"
    preprocessing: ClassVar[Mapping[str, Any]]
    sampling_rule: ClassVar[str]
    manifest_schema: ClassVar[str]
    per_stratum_key: ClassVar[str]  # the protocol's sampling key for the per-stratum count
    columns: ClassVar[Columns]
    task_type: ClassVar[TaskType]
    gold_fields: ClassVar[tuple[str, ...]]  # raw fields that must never reach execution
    dataset_name: ClassVar[str]
    frozen_dir: ClassVar[Path]  # protocol-<v>.json, protocols.lock.json, manifest.json
    results_dir: ClassVar[Path]  # results/<key>/protocol-<v>

    @abstractmethod
    def parse(self, data: bytes) -> list[dict[str, Any]]:
        """The official rows, from the (already hash-checked) official bytes."""

    @abstractmethod
    def row_id(self, row: Mapping[str, Any]) -> str:
        """The official row id, as a string."""

    @abstractmethod
    def stratum(self, row: Mapping[str, Any]) -> str:
        """The dataset-native stratum of a row. Must never read an answer or a gold field."""

    @abstractmethod
    def sanitize(self, row: Mapping[str, Any]) -> dict[str, str]:
        """The only view of a row that enters Wynk: exactly ``columns.all``."""

    def strata(self, sampling: Mapping[str, Any]) -> tuple[str, ...] | None:
        """The strata to sample, in order; ``None`` = every stratum present in the source."""
        return None

    def sampling_manifest(self, sampling: Mapping[str, Any], strata: Sequence[str]) -> dict:
        """How the sample was drawn, as recorded in the manifest."""
        return {
            "rule": self.sampling_rule,
            "seed": sampling["seed"],
            self.per_stratum_key: sampling[self.per_stratum_key],
            "strata": list(strata),
            "selection_inputs": "official ids + the dataset-native stratum only",
        }

    @property
    def preprocessing_hash(self) -> str:
        return canonical_hash(self.preprocessing)


# -- source -------------------------------------------------------------------------------------
def load_source(bench: Benchmark, path: Path) -> list[dict[str, Any]]:
    """The official rows, after checking the file's SHA-256 and row count against ``source``."""
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if digest != bench.source["file_sha256"]:
        raise SourceMismatch(
            f"{path.name} has sha256 {digest}, expected {bench.source['file_sha256']}"
        )
    rows = bench.parse(data)
    if len(rows) != bench.source["rows"]:
        raise SourceMismatch(f"{len(rows)} rows, expected {bench.source['rows']}")
    ids = [bench.row_id(r) for r in rows]
    if len(set(ids)) != len(ids):
        raise SourceMismatch("official row ids are not unique")
    return rows


# -- sampling -----------------------------------------------------------------------------------
def sample_key(rule: str, seed: int, row_id: str) -> str:
    return sha256_hex(f"{rule}:{seed}:{row_id}")


def stratified_sample(
    keyed: Iterable[tuple[str, str]],
    *,
    rule: str,
    seed: int,
    per_stratum: int,
    strata: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Equal-count, hash-ordered sample per stratum from ``(row id, stratum)`` pairs.

    Within each stratum ids are ordered by ``sha256("<rule>:<seed>:<id>")`` and the first
    ``per_stratum`` are kept (then sorted). Uses ids and strata only - never answers/outcomes.
    """
    by: dict[str, list[str]] = {}
    for rid, stratum in keyed:
        by.setdefault(stratum, []).append(rid)
    wanted = list(strata) if strata is not None else sorted(by)
    out: dict[str, list[str]] = {}
    for s in wanted:
        group = by.get(s, [])
        if len(group) < per_stratum:
            raise ValueError(f"only {len(group)} rows in stratum {s!r}, need {per_stratum}")
        ranked = sorted(group, key=lambda r: (sample_key(rule, seed, r), r))
        out[s] = sorted(ranked[:per_stratum])
    return out


# -- sanitized subset ---------------------------------------------------------------------------
def check_sanitized(bench: Benchmark, row: Mapping[str, Any]) -> None:
    """A sanitized row is exactly the declared columns, all strings, no gold field."""
    keys = set(row)
    if keys != set(bench.columns.all):
        raise LeakageError(f"sanitized row has columns {sorted(keys)}, not {bench.columns.all}")
    gold = keys & set(bench.gold_fields)
    if gold:
        raise LeakageError(f"sanitized row keeps gold fields {sorted(gold)}")
    if not all(isinstance(v, str) for v in row.values()):
        raise LeakageError("sanitized values must be strings")


def dataset_bytes(bench: Benchmark, rows: Sequence[Mapping[str, Any]], ids: Iterable[str]) -> bytes:
    """The sanitized subset as Wynk JSONL (deterministic bytes, rows sorted by id)."""
    by_id = {bench.row_id(r): r for r in rows}
    lines = []
    for rid in sorted(set(ids)):
        clean = bench.sanitize(by_id[rid])
        check_sanitized(bench, clean)
        lines.append(json.dumps(clean, sort_keys=True, ensure_ascii=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _schema(names: Iterable[str]) -> AnswerSchema:
    return AnswerSchema(fields=tuple(AnswerField(name=n, type=FieldType.STRING) for n in names))


def build_contract(
    bench: Benchmark, data: bytes, protocol: Mapping[str, Any], *, dataset_version: int = 1
) -> TaskContract:
    """The frozen TaskContract for the subset bytes, from the frozen protocol."""
    c, cols = protocol["contract"], bench.columns
    dataset = DatasetSpec(
        dataset_id=c["dataset_id"],
        dataset_version=dataset_version,
        name=bench.dataset_name,
        content_hash=sha256_bytes(data),
        format=DatasetFormat.JSONL,
        columns=tuple(ColumnSpec(name=n, type=ColumnType.STRING) for n in cols.all),
        id_column=cols.id,
        input_columns=cols.inputs,
        context_columns=cols.context,
        target_columns=cols.targets,
        row_count=sum(1 for line in data.splitlines() if line.strip()),
    )
    return TaskContract(
        task_id=c["task_id"],
        contract_version=c["contract_version"],
        task_type=bench.task_type,
        instructions=c["instructions"],
        input_schema=_schema((*cols.inputs, *cols.context)),
        output_schema=_schema(cols.targets),
        dataset=dataset,
        evaluation=EvaluationSpec(**c["evaluation"]),
        constraints=ConstraintLimits(**c["constraints"]),
    )


def manifest(
    bench: Benchmark,
    selected: Mapping[str, Sequence[str]],
    *,
    protocol: Mapping[str, Any],
    contract: TaskContract,
    splits: DatasetSplits,
) -> dict[str, Any]:
    """Everything needed to rebuild and verify the frozen subset, plus its own hash."""
    body = {
        "schema": bench.manifest_schema,
        "source": dict(bench.source),
        "preprocessing": {**bench.preprocessing, "hash": bench.preprocessing_hash},
        "sampling": bench.sampling_manifest(protocol["sampling"], list(selected)),
        "selected_ids": {str(s): list(ids) for s, ids in selected.items()},
        "dataset": {
            "rows": contract.dataset.row_count,
            "content_hash": contract.dataset.content_hash,
            "identity_hash": contract.dataset.identity_hash,
        },
        "splits": {
            "plan": splits.plan.model_dump(mode="json") if splits.plan else None,
            "identity_hash": splits.identity_hash,
            "rows": {s.role.value: list(s.row_ids) for s in splits.splits},
        },
        "contract_hash": contract.contract_hash,
    }
    return {**body, "manifest_hash": canonical_hash(body)}


@dataclass(frozen=True)
class Prepared:
    contract: TaskContract
    splits: DatasetSplits
    data: bytes
    manifest: dict[str, Any]
    rows: list[dict[str, Any]]  # the official rows (evaluator-side cross-checks only)


def prepare(bench: Benchmark, source: Path, protocol: Mapping[str, Any]) -> Prepared:
    """Official file + frozen protocol -> contract, splits, subset bytes and manifest."""
    return prepare_rows(bench, load_source(bench, source), protocol)


def prepare_rows(
    bench: Benchmark, rows: list[dict[str, Any]], protocol: Mapping[str, Any]
) -> Prepared:
    sampling = protocol["sampling"]
    selected = stratified_sample(
        ((bench.row_id(r), bench.stratum(r)) for r in rows),
        rule=bench.sampling_rule,
        seed=sampling["seed"],
        per_stratum=sampling[bench.per_stratum_key],
        strata=bench.strata(sampling),
    )
    ids = sorted(rid for group in selected.values() for rid in group)
    data = dataset_bytes(bench, rows, ids)
    contract = build_contract(bench, data, protocol)
    splits = seeded_splits(
        contract.dataset.identity_hash, tuple(ids), SplitPlan(**protocol["split_plan"])
    )
    frozen = manifest(bench, selected, protocol=protocol, contract=contract, splits=splits)
    return Prepared(contract, splits, data, frozen, rows)


def check_manifest(fresh: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    """The rebuilt manifest must equal the committed one, hash for hash."""
    if dict(fresh) != dict(frozen):
        diff = sorted(k for k in set(fresh) | set(frozen) if fresh.get(k) != frozen.get(k))
        raise SourceMismatch(f"rebuilt manifest differs from the frozen one in {diff}")


def check_isolation(bench: Benchmark, suite: ContractSuite) -> None:
    """Mechanical leakage gate: every execution task holds only input + context values."""
    allowed = bench.columns.execution
    for task in suite.tasks:
        extra = set(task.example.values) - allowed
        if extra:
            raise LeakageError(f"execution task {task.id} holds {sorted(extra)}")


# -- frozen protocols ---------------------------------------------------------------------------
def protocol_path(bench: Benchmark, version: str) -> Path:
    return bench.frozen_dir / f"protocol-{version}.json"


def lock_path(bench: Benchmark) -> Path:
    return bench.frozen_dir / "protocols.lock.json"


def manifest_path(bench: Benchmark) -> Path:
    return bench.frozen_dir / "manifest.json"


def protocol_versions(bench: Benchmark) -> list[str]:
    lock = json.loads(lock_path(bench).read_text(encoding="utf-8"))
    return sorted(lock["protocols"])


def load_protocol(bench: Benchmark, version: str) -> dict[str, Any]:
    """The frozen protocol, refused unless its canonical hash equals its lock entry."""
    protocol = json.loads(protocol_path(bench, version).read_text(encoding="utf-8"))
    lock = json.loads(lock_path(bench).read_text(encoding="utf-8"))
    if lock["protocols"][version]["canonical_hash"] != canonical_hash(protocol):
        raise SourceMismatch(f"{bench.key} protocol {version} does not match its frozen hash")
    return protocol


def frozen_manifest(bench: Benchmark) -> dict[str, Any]:
    return json.loads(manifest_path(bench).read_text(encoding="utf-8"))


def plan_from(protocol: Mapping[str, Any], manifest_hash: str) -> ExperimentPlan:
    """The ExperimentPlan a frozen protocol describes (shared by every benchmark)."""
    # MuSiQue protocol-v1 predates ``protocol_version`` / ``fixed_baseline_rule``: its plan (and
    # hence its experiment identity) is exactly what it was when it ran.
    versioned = "protocol_version" in protocol
    return ExperimentPlan(
        fixed_rule=protocol.get("fixed_baseline_rule"),
        protocol_id=canonical_hash(protocol) if versioned else None,
        model=ModelConfiguration(**protocol["model"]),
        expected_model_hash=protocol["expected_model_hash"],
        expected_prompt_version=protocol["expected_prompt_version"],
        dataset_manifest_hash=manifest_hash,
        budget=ExperimentBudget(**protocol["budget"]),
        seeds=tuple(protocol["seeds"]),
        strategies=tuple(Strategy(s) for s in protocol["strategies"]),
        trials=protocol["trials"],
        batch_size=protocol["batch_size"],
        lcb_z=protocol["lcb_z"],
    )
