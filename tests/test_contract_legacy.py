"""Legacy adapter: every frozen benchmark task as a contract, and the split/leakage boundary
proven end-to-end through the existing experiment harness."""

import pytest

from benchmarks.heldout import HELDOUT_DIR
from benchmarks.legacy_adapter import legacy_contracts, legacy_splits
from benchmarks.loader import BENCH_DIR, load_splits, load_task_specs, runtime_tasks
from core.dataset import DatasetFormat, SplitAccessError, SplitRole
from core.evaluation_spec import EvaluatorKind
from core.task_contract import TaskType
from core.task_spec import GroundTruth, TaskClass, TaskSpec
from evaluation.gate import EVALUATOR_VERSION
from experiments.learning_curves import ExperimentConfig, run_search
from experiments.synthetic import synthetic_evaluate
from optimizers.aco_mmas import MMASACO

DIRS = {"main": BENCH_DIR, "heldout": HELDOUT_DIR}


@pytest.mark.parametrize("bench", sorted(DIRS))
def test_every_benchmark_task_converts_faithfully(bench):
    specs = load_task_specs(DIRS[bench])
    contracts = legacy_contracts(DIRS[bench])
    assert set(contracts) == set(specs) and len(contracts) == 16
    for tid, spec in specs.items():
        c, rt = contracts[tid], spec.runtime
        assert c.task_type is TaskType.STRUCTURED_EXTRACTION
        assert c.output_schema == rt.answer_schema
        assert c.constraints.to_caps() == rt.caps  # same hard caps, same numbers
        assert c.evaluation.evaluator is EvaluatorKind.LEGACY_FIELD_MATCH
        assert c.evaluation.evaluator_version == EVALUATOR_VERSION
        assert c.evaluation.config["matchers"] == {
            k: m.model_dump(mode="json") for k, m in spec.matchers.items()
        }
        assert c.dataset.format is DatasetFormat.WYNK_SNAPSHOT
        assert c.dataset.target_columns == tuple(f.name for f in rt.answer_schema.fields)
        # the task class survives only as non-authoritative metadata
        assert c.dataset.metadata == {"task_class": rt.task_class.value}
        assert "task_class" not in str(c.authoritative())


def test_conversion_is_deterministic_and_identities_are_unique():
    a, b = legacy_contracts(), legacy_contracts()
    assert {k: c.contract_hash for k, c in a.items()} == {k: c.contract_hash for k, c in b.items()}
    assert len({c.contract_hash for c in a.values()}) == len(a)
    assert len({c.dataset.identity_hash for c in a.values()}) == len(a)


def _truth_strings(value):
    items = value if isinstance(value, list) else [value]
    return [str(x) for x in items if isinstance(x, str) and len(x) >= 4]


def test_contracts_never_carry_expected_answer_values():
    for tid, spec in load_task_specs().items():
        text = legacy_contracts()[tid].canonical_json()
        for value in spec.ground_truth.values.values():
            for s in _truth_strings(value):
                assert s not in text, (tid, s)


def test_dataset_identity_still_changes_when_the_expected_answer_changes():
    from benchmarks.legacy_adapter import legacy_dataset_spec
    from benchmarks.snapshot_store import SnapshotStore

    spec = load_task_specs()["A-001"]
    changed = TaskSpec(
        runtime=spec.runtime,
        ground_truth=GroundTruth(values={**spec.ground_truth.values, "hq": "Elsewhere"}),
        matchers=spec.matchers,
    )
    store = SnapshotStore()
    assert (
        legacy_dataset_spec(spec, store).identity_hash
        != legacy_dataset_spec(changed, store).identity_hash
    )


def test_legacy_splits_mirror_the_existing_split_files():
    s = legacy_splits()
    main, heldout = load_splits(BENCH_DIR), load_splits(HELDOUT_DIR)
    assert s.split(SplitRole.OPTIMIZATION).row_ids == tuple(sorted(main["train"]))
    assert s.split(SplitRole.VALIDATION).row_ids == tuple(sorted(main["validation"]))
    assert s.split(SplitRole.TEST).row_ids == tuple(sorted(heldout["test"]))
    assert s.split(SplitRole.TEST).is_final
    view = s.optimizer_view()
    assert not (set(view.optimization_row_ids) | set(view.validation_row_ids)) & set(
        heldout["test"]
    )


# -- final-test results cannot become optimizer feedback ---------------------------------------


class GatedACO(MMASACO):
    """MMAS ACO whose feedback passes through the split contract's gate before it is used."""

    def __init__(self, splits):
        super().__init__()
        self.splits = splits
        self.observed: set[str] = set()

    def observe(self, results):
        ids = {r.execution.task_id for r in results}
        self.splits.check_feedback(ids)  # raises before any state changes
        self.observed |= ids
        super().observe(results)


@pytest.mark.parametrize("cls", [TaskClass.A, TaskClass.B])
def test_harness_feeds_the_optimizer_only_optimization_rows(cls):
    splits = legacy_splits()
    view = splits.optimizer_view()
    train = [t for t in runtime_tasks("train", cls) if t.id in view.optimization_row_ids]
    val = [t for t in runtime_tasks("validation", cls) if t.id in view.validation_row_ids]
    assert train and val
    evaluated: set[str] = set()

    def spy(genome, task, trial, seed):
        evaluated.add(task.id)
        return synthetic_evaluate(genome, task, trial, seed)

    opt = GatedACO(splits)
    result = run_search(opt, spy, train, val, ExperimentConfig(budget=40, trials=1), seed=0)
    assert result["workflow_evaluations"] > 0
    # validation runs were MEASURED (selection) but never observed by the optimizer ...
    assert {t.id for t in val} <= evaluated
    assert opt.observed == {t.id for t in train}
    # ... and no final-test row was ever executed or observed.
    test_rows = set(splits.split(SplitRole.TEST).row_ids)
    assert not (evaluated | opt.observed) & test_rows


def test_a_final_test_result_is_refused_as_feedback_and_leaves_state_untouched():
    splits = legacy_splits()
    test_task = runtime_tasks("test", TaskClass.A, HELDOUT_DIR)[0]
    assert splits.role_of(test_task.id) is SplitRole.TEST
    opt = GatedACO(splits)
    context_task = runtime_tasks("train", TaskClass.A)[0]
    from core.constraints import ConstraintChecker
    from optimizers.base import SearchContext

    genome = opt.propose(1, SearchContext(task=context_task, checker=ConstraintChecker(), seed=0))[
        0
    ]
    run = synthetic_evaluate(genome, test_task, 0, 0)
    before = (opt.epoch, opt.pheromone_snapshot())
    with pytest.raises(SplitAccessError):
        opt.observe([run])
    assert (opt.epoch, opt.pheromone_snapshot()) == before
    assert not opt.observed
