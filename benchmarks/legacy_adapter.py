"""Legacy adapter: THE one boundary between the frozen A/B benchmark and the generic pipeline.

    TaskSpec (benchmarks/tasks.json)  ->  TaskContract (DatasetSpec, EvaluationSpec,
                                          ObjectiveSpec, ConstraintLimits, WorkflowSpec)
                                          + ExampleInput  =  ExecutionTask
    splits.json + heldout/splits.json ->  DatasetSplits (optimization / validation / test)
    one task class                    ->  ContractSuite (what one search runs on)
    ground truth                      ->  references (row id -> expected values), evaluator only

Past this module, search, execution and evaluation read only contracts: nothing downstream
branches on a task class or reads ``RuntimeTask`` / ``TaskSpec`` authority. Benchmark-only
concepts stay here: the task class selects a suite and survives only as non-authoritative
dataset metadata and the suite's display name; snapshots are the ``wynk_snapshot`` dataset
format; the per-field matchers + evidence check are the explicitly named ``legacy_field_match``
evaluator (executed by ``evaluation.contract_eval.ContractEvaluator``).

Mapping decisions:
  * Each legacy task asks for its OWN answer fields, so each becomes its own contract over a
    one-row dataset (input ``question``, context ``snapshot_id``, one target column per answer
    field). A class is a suite of contracts that share caps, sources and objective.
  * ``Caps`` become per-example ``ConstraintLimits`` with the same numbers (``from_caps``);
    ``allowed_sources`` / ``interaction_required`` become the contract's ``WorkflowSpec``, whose
    ``stages`` is the frozen six-stage vocabulary (``LEGACY_STAGE_KINDS``, ``grammar/1``). This is
    the only place the legacy grammar is selected: the benchmark's search space is unchanged, and
    generic contracts never fall back to it.
  * The objective is ``maximize_quality``: the legacy shaped fitness ranks by verdict and field
    matches first; its small budget-headroom bonus among PASSes has no ObjectiveSpec equivalent
    yet and remains owned by ``evaluation/fitness.py``.
  * Expected values never enter a contract; ``content_hash`` covers them through
    ``TaskSpec.content_hash`` so a changed truth still changes dataset identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.loader import BENCH_DIR, HELDOUT_DIR, load_splits, load_task_specs
from benchmarks.snapshot_store import SnapshotStore
from core.canonical import canonical_hash
from core.constraints import ConstraintLimits
from core.dataset import (
    ColumnSpec,
    ColumnType,
    DatasetFormat,
    DatasetSpec,
    DatasetSplit,
    DatasetSplits,
    SplitMethod,
    SplitRole,
)
from core.evaluation_spec import EvaluationSpec, EvaluatorKind
from core.objective import ObjectiveSpec
from core.run_contract import ContractSuite, ExampleInput, ExecutionTask, suite_dataset_hash
from core.stages import LEGACY_STAGE_KINDS
from core.task_contract import TaskContract, TaskType, WorkflowSpec
from core.task_spec import AnswerField, AnswerSchema, FieldType, RuntimeTask, TaskClass, TaskSpec

LEGACY_CONTRACT_VERSION = 1
LEGACY_INSTRUCTIONS = (
    "Answer the question using only the pages gathered from the task's snapshot. Return exactly "
    "the requested answer fields, each with evidence spans pointing into the gathered pages."
)
INPUT_SCHEMA = AnswerSchema(
    fields=(
        AnswerField(name="question", type=FieldType.STRING),
        AnswerField(name="snapshot_id", type=FieldType.STRING),
    )
)


def _dataset_spec(rt: RuntimeTask, prefix: str, content_hash: str, name: str) -> DatasetSpec:
    targets = tuple(
        ColumnSpec(name=f.name, type=ColumnType(f.type.value), nullable=not f.required)
        for f in rt.answer_schema.fields
    )
    return DatasetSpec(
        dataset_id=f"{prefix}-{rt.id.lower()}",
        dataset_version=1,
        name=name,
        content_hash=content_hash,
        format=DatasetFormat.WYNK_SNAPSHOT,
        columns=(
            ColumnSpec(name="task_id", type=ColumnType.STRING),
            ColumnSpec(name="question", type=ColumnType.STRING),
            ColumnSpec(name="snapshot_id", type=ColumnType.STRING),
            *targets,
        ),
        id_column="task_id",
        input_columns=("question",),
        context_columns=("snapshot_id",),
        target_columns=tuple(c.name for c in targets),
        row_count=1,
        metadata={"task_class": rt.task_class.value},  # legacy label, not authority
    )


def _contract(rt: RuntimeTask, dataset: DatasetSpec, evaluation: EvaluationSpec) -> TaskContract:
    return TaskContract(
        task_id=dataset.dataset_id,
        contract_version=LEGACY_CONTRACT_VERSION,
        task_type=TaskType.STRUCTURED_EXTRACTION,
        instructions=LEGACY_INSTRUCTIONS,
        input_schema=INPUT_SCHEMA,
        output_schema=rt.answer_schema,
        dataset=dataset,
        evaluation=evaluation,
        objective=ObjectiveSpec(),
        constraints=ConstraintLimits.from_caps(rt.caps),
        workflow=WorkflowSpec(
            allowed_sources=rt.allowed_sources,
            interaction_required=rt.interaction_required,
            stages=LEGACY_STAGE_KINDS,
        ),
    )


def _example(rt: RuntimeTask) -> ExampleInput:
    return ExampleInput(
        row_id=rt.id, values={"question": rt.question, "snapshot_id": rt.snapshot_id}
    )


# -- benchmark tasks ------------------------------------------------------------------------------
def legacy_dataset_spec(spec: TaskSpec, store: SnapshotStore) -> DatasetSpec:
    rt = spec.runtime
    content = canonical_hash(
        {"task": spec.content_hash, "snapshot": store.snapshot_hash(rt.snapshot_id)}
    )
    return _dataset_spec(rt, "wynk-benchmark", content, f"Wynk benchmark task {rt.id}")


def legacy_task_contract(spec: TaskSpec, store: SnapshotStore) -> TaskContract:
    return _contract(
        spec.runtime,
        legacy_dataset_spec(spec, store),
        EvaluationSpec(
            evaluator=EvaluatorKind.LEGACY_FIELD_MATCH,
            config={
                "matchers": {k: m.model_dump(mode="json") for k, m in spec.matchers.items()},
                "require_evidence": True,
            },
        ),
    )


def legacy_execution_task(spec: TaskSpec, store: SnapshotStore) -> ExecutionTask:
    return ExecutionTask(contract=legacy_task_contract(spec, store), example=_example(spec.runtime))


def legacy_contracts(bench_dir: Path = BENCH_DIR) -> dict[str, TaskContract]:
    """Every task of a frozen benchmark directory as a contract, keyed by legacy task id."""
    store = SnapshotStore(bench_dir / "snapshots")
    return {
        tid: legacy_task_contract(s, store) for tid, s in sorted(load_task_specs(bench_dir).items())
    }


def legacy_execution_tasks(bench_dir: Path = BENCH_DIR) -> dict[str, ExecutionTask]:
    store = SnapshotStore(bench_dir / "snapshots")
    return {
        tid: legacy_execution_task(s, store)
        for tid, s in sorted(load_task_specs(bench_dir).items())
    }


def legacy_references(*bench_dirs: Path) -> dict[str, dict[str, Any]]:
    """Expected answer values by task id - for ``ContractEvaluator`` only, never for search."""
    out: dict[str, dict[str, Any]] = {}
    for bench_dir in bench_dirs or (BENCH_DIR,):
        for tid, spec in load_task_specs(bench_dir).items():
            out[tid] = dict(spec.ground_truth.values)
    return out


# -- splits and suites ----------------------------------------------------------------------------
def _splits(
    tasks: Mapping[str, ExecutionTask],
    train: Sequence[str],
    val: Sequence[str],
    test: Sequence[str],
) -> DatasetSplits:
    return DatasetSplits(
        dataset_hash=suite_dataset_hash(t.contract for t in tasks.values()),
        method=SplitMethod.EXPLICIT,
        splits=tuple(
            DatasetSplit(split_id=split_id, role=role, row_ids=tuple(rows))
            for split_id, role, rows in (
                ("train", SplitRole.OPTIMIZATION, train),
                ("validation", SplitRole.VALIDATION, val),
                ("test", SplitRole.TEST, test),
            )
            if rows
        ),
    )


def _split_ids(
    bench_dir: Path, heldout_dir: Path, task_class: TaskClass | None
) -> tuple[dict[str, ExecutionTask], list[str], list[str], list[str]]:
    main, heldout = load_splits(bench_dir), load_splits(heldout_dir)
    if set(main) != {"train", "validation"} or set(heldout) != {"test"}:
        raise ValueError("unexpected legacy split names")
    main_specs, held_specs = load_task_specs(bench_dir), load_task_specs(heldout_dir)
    main_store = SnapshotStore(bench_dir / "snapshots")
    held_store = SnapshotStore(heldout_dir / "snapshots")

    def keep(ids: Sequence[str], specs: Mapping[str, TaskSpec]) -> list[str]:
        return [i for i in ids if task_class is None or specs[i].runtime.task_class is task_class]

    train, val = keep(main["train"], main_specs), keep(main["validation"], main_specs)
    test = keep(heldout["test"], held_specs)
    tasks = {i: legacy_execution_task(main_specs[i], main_store) for i in (*train, *val)}
    tasks |= {i: legacy_execution_task(held_specs[i], held_store) for i in test}
    return tasks, train, val, test


def legacy_splits(bench_dir: Path = BENCH_DIR, heldout_dir: Path = HELDOUT_DIR) -> DatasetSplits:
    """The benchmark's existing split files as one split contract over legacy task ids.

    ``train`` -> optimization, ``validation`` -> validation, the held-out set -> final test.
    """
    tasks, train, val, test = _split_ids(bench_dir, heldout_dir, None)
    return _splits(tasks, train, val, test)


def legacy_suite(
    task_class: TaskClass | str,
    bench_dir: Path = BENCH_DIR,
    heldout_dir: Path = HELDOUT_DIR,
) -> ContractSuite:
    """One task class as a ``ContractSuite``: its train / validation tasks from ``bench_dir`` and
    its held-out test tasks from ``heldout_dir``, in split-file order."""
    cls = TaskClass(task_class)
    tasks, train, val, test = _split_ids(bench_dir, heldout_dir, cls)
    return ContractSuite(
        name=cls.value,
        tasks=tuple(tasks[i] for i in (*train, *val, *test)),
        splits=_splits(tasks, train, val, test),
    )


# -- ad-hoc questions over the benchmark library (api.chat) ---------------------------------------
def legacy_adhoc_task(rt: RuntimeTask, store: SnapshotStore) -> ExecutionTask:
    """A question asked over a benchmark snapshot, with no expected answer: its contract can
    only check the answer's shape (``json_schema_validity``)."""
    content = canonical_hash(
        {"task": rt.model_dump(mode="json"), "snapshot": store.snapshot_hash(rt.snapshot_id)}
    )
    dataset = _dataset_spec(rt, "wynk-adhoc", content, f"Question over snapshot {rt.snapshot_id}")
    contract = _contract(rt, dataset, EvaluationSpec(evaluator=EvaluatorKind.JSON_SCHEMA_VALIDITY))
    return ExecutionTask(contract=contract, example=_example(rt))
