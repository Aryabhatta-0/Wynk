"""Legacy adapter: the frozen A/B benchmark expressed as generic task contracts.

    TaskSpec (benchmarks/tasks.json)  ->  DatasetSpec + TaskContract
                                          (EvaluationSpec, ObjectiveSpec, ConstraintLimits)
    splits.json + heldout/splits.json ->  DatasetSplits (optimization / validation / test)

This is the migration seam, not a rewrite: the benchmark, its golden hashes and the experiment
harness are untouched and still run on ``RuntimeTask`` / ``TaskSpec``. Benchmark-only concepts
stay here: the task class is kept as non-authoritative dataset metadata, snapshots are the
``wynk_snapshot`` dataset format, and the per-field matchers + evidence check are the explicitly
named ``legacy_field_match`` evaluator (executed by ``evaluation.gate.DeterministicEvaluator``).

Mapping decisions:
  * Each legacy task asks for its OWN answer fields, so each becomes its own contract over a
    one-row dataset (input ``question``, context ``snapshot_id``, one target column per answer
    field). A class is a suite of contracts that share caps and sources.
  * ``Caps`` become per-example ``ConstraintLimits`` with the same numbers (``from_caps``).
  * The objective is ``maximize_quality``: the legacy shaped fitness ranks by verdict and field
    matches first; its small budget-headroom bonus among PASSes has no ObjectiveSpec equivalent
    yet and remains owned by ``evaluation/fitness.py``.
  * Expected values never enter a contract; ``content_hash`` covers them through
    ``TaskSpec.content_hash`` so a changed truth still changes dataset identity.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.loader import BENCH_DIR, HELDOUT_DIR, benchmark_hash, load_splits, load_task_specs
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
from core.task_contract import TaskContract, TaskType
from core.task_spec import AnswerField, AnswerSchema, FieldType, TaskSpec

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


def legacy_dataset_spec(spec: TaskSpec, store: SnapshotStore) -> DatasetSpec:
    rt = spec.runtime
    targets = tuple(
        ColumnSpec(name=f.name, type=ColumnType(f.type.value), nullable=not f.required)
        for f in rt.answer_schema.fields
    )
    return DatasetSpec(
        dataset_id=f"wynk-benchmark-{rt.id.lower()}",
        dataset_version=1,
        name=f"Wynk benchmark task {rt.id}",
        content_hash=canonical_hash(
            {"task": spec.content_hash, "snapshot": store.snapshot_hash(rt.snapshot_id)}
        ),
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


def legacy_task_contract(spec: TaskSpec, store: SnapshotStore) -> TaskContract:
    rt = spec.runtime
    return TaskContract(
        task_id=f"wynk-benchmark-{rt.id.lower()}",
        contract_version=LEGACY_CONTRACT_VERSION,
        task_type=TaskType.STRUCTURED_EXTRACTION,
        instructions=LEGACY_INSTRUCTIONS,
        input_schema=INPUT_SCHEMA,
        output_schema=rt.answer_schema,
        dataset=legacy_dataset_spec(spec, store),
        evaluation=EvaluationSpec(
            evaluator=EvaluatorKind.LEGACY_FIELD_MATCH,
            config={
                "matchers": {k: m.model_dump(mode="json") for k, m in spec.matchers.items()},
                "require_evidence": True,
            },
        ),
        objective=ObjectiveSpec(),
        constraints=ConstraintLimits.from_caps(rt.caps),
    )


def legacy_contracts(bench_dir: Path = BENCH_DIR) -> dict[str, TaskContract]:
    """Every task of a frozen benchmark directory as a contract, keyed by legacy task id."""
    store = SnapshotStore(bench_dir / "snapshots")
    return {
        tid: legacy_task_contract(s, store) for tid, s in sorted(load_task_specs(bench_dir).items())
    }


def legacy_splits(bench_dir: Path = BENCH_DIR, heldout_dir: Path = HELDOUT_DIR) -> DatasetSplits:
    """The benchmark's existing split files as one split contract over legacy task ids.

    ``train`` -> optimization, ``validation`` -> validation, the held-out set -> final test.
    """
    main, heldout = load_splits(bench_dir), load_splits(heldout_dir)
    if set(main) != {"train", "validation"} or set(heldout) != {"test"}:
        raise ValueError("unexpected legacy split names")
    return DatasetSplits(
        dataset_hash=canonical_hash(
            {"benchmark": benchmark_hash(bench_dir), "heldout": benchmark_hash(heldout_dir)}
        ),
        method=SplitMethod.EXPLICIT,
        splits=(
            DatasetSplit(split_id="train", role=SplitRole.OPTIMIZATION, row_ids=main["train"]),
            DatasetSplit(
                split_id="validation", role=SplitRole.VALIDATION, row_ids=main["validation"]
            ),
            DatasetSplit(split_id="test", role=SplitRole.TEST, row_ids=heldout["test"]),
        ),
    )
