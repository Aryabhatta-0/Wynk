"""Architectural invariants: who may see / do what, enforced mechanically."""

import ast
import inspect
from collections.abc import Sequence
from pathlib import Path

import pytest

from core.constraints import ConstraintChecker
from core.genome import Genome
from core.results import EvaluatedRun, Evaluation, ExecutionResult, Verdict
from core.task_spec import GroundTruth, RuntimeTask, TaskSpec
from optimizers.base import Optimizer, SearchContext, ensure_admissible
from tests.conftest import SECRET_ANSWER, make_caps, make_runtime_task, make_task_spec

ROOT = Path(__file__).resolve().parent.parent

# Packages that must be ground-truth-free. core/ modules other than task_spec.py are included.
RUNTIME_SIDE = ["runtime", "compiler", "optimizers", "router"]
CORE_RUNTIME_SAFE = [
    p for p in (ROOT / "core").glob("*.py") if p.name not in ("task_spec.py", "__init__.py")
]
FORBIDDEN_NAMES = {"TaskSpec", "GroundTruth"}
FORBIDDEN_MODULES = ("evaluation", "benchmarks", "store")


def _py_files(pkg: str) -> list[Path]:
    return sorted((ROOT / pkg).rglob("*.py"))


def _imports(path: Path) -> list[tuple[str, list[str]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            out.append((node.module or "", [a.name for a in node.names]))
        elif isinstance(node, ast.Import):
            out += [(a.name, []) for a in node.names]
    return out


RUNTIME_FILES = [f for pkg in RUNTIME_SIDE for f in _py_files(pkg)] + CORE_RUNTIME_SAFE


@pytest.mark.parametrize("path", RUNTIME_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_runtime_side_code_never_imports_ground_truth_or_the_evaluator(path):
    for module, names in _imports(path):
        assert not (set(names) & FORBIDDEN_NAMES), f"{path.name} imports {names} from {module}"
        assert module.split(".")[0] not in FORBIDDEN_MODULES, f"{path.name} imports {module}"
    assert "ground_truth" not in path.read_text(encoding="utf-8").lower()


def test_maf_is_imported_only_from_compiler_and_runtime():
    offenders = []
    for pkg_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()):
        if pkg_dir.name in ("compiler", "runtime", "tests"):
            continue
        for f in pkg_dir.rglob("*.py"):
            if any(m.split(".")[0] == "agent_framework" for m, _ in _imports(f)):
                offenders.append(str(f.relative_to(ROOT)))
    assert offenders == []


def test_runtime_task_carries_no_ground_truth_and_taskspec_is_not_a_runtime_task():
    assert not {"ground_truth", "matchers"} & set(RuntimeTask.model_fields)
    spec = make_task_spec()
    assert not isinstance(spec, RuntimeTask)
    view = spec.runtime_view()
    assert isinstance(view, RuntimeTask)
    assert SECRET_ANSWER not in view.model_dump_json()
    assert SECRET_ANSWER not in repr(view)


def test_ground_truth_is_redacted_in_repr_and_str():
    spec = make_task_spec()
    for text in (repr(spec), str(spec), repr(spec.ground_truth), str(spec.ground_truth)):
        assert SECRET_ANSWER not in text
    assert isinstance(spec.ground_truth, GroundTruth)


def test_execution_results_cannot_carry_a_verdict_or_fitness():
    assert not {"verdict", "fitness"} & set(ExecutionResult.model_fields)
    assert {"verdict", "fitness"} <= set(Evaluation.model_fields)
    assert set(EvaluatedRun.model_fields) == {"execution", "evaluation"}


def test_verdict_vocabulary_is_closed():
    assert {v.value for v in Verdict} == {"PASS", "FAIL", "INFEASIBLE"}


def test_optimizer_contract_consumes_evaluated_runs_and_a_ground_truth_free_context():
    hints = inspect.get_annotations(Optimizer.observe, eval_str=True)
    assert hints["results"] == Sequence[EvaluatedRun]
    ctx_hints = inspect.get_annotations(SearchContext, eval_str=True)
    assert ctx_hints["task"] is RuntimeTask
    assert TaskSpec not in ctx_hints.values()
    assert set(inspect.signature(Optimizer.propose).parameters) == {"self", "k", "context"}


class _GreedyOptimizer(Optimizer):
    """Smallest possible optimizer: legality comes ONLY from the shared checker."""

    name = "greedy-test"

    def __init__(self):
        self.seen: list[EvaluatedRun] = []

    def propose(self, k, context):
        g = Genome()
        while not context.checker.grammar.can_terminate(g):
            g = g.extend(context.checker.admissible_successors(g, context.task)[0])
        return [g]

    def observe(self, results):
        self.seen.extend(results)


def test_optimizer_can_propose_without_duplicating_grammar_or_constraint_rules():
    ctx = SearchContext(task=make_runtime_task(), checker=ConstraintChecker(), seed=0)
    opt = _GreedyOptimizer()
    proposals = opt.propose(1, ctx)
    ensure_admissible(proposals, ctx)
    assert len(proposals) == 1


def test_ensure_admissible_rejects_proposals_that_break_the_shared_rules():
    from core.stages import GatherMode, GatherSource
    from tests.conftest import extract, gather, synth

    ctx = SearchContext(task=make_runtime_task(), checker=ConstraintChecker(), seed=0)
    bad = Genome.of(gather(GatherSource.JEV, GatherMode.PARALLEL_4), extract(), synth())
    from optimizers.base import InadmissibleProposal

    with pytest.raises(InadmissibleProposal):
        ensure_admissible([bad], ctx)


def test_stub_optimizers_exist_but_are_explicitly_unimplemented():
    from optimizers.aco_mmas import MMASACO

    ctx = SearchContext(
        task=make_runtime_task(caps=make_caps()), checker=ConstraintChecker(), seed=0
    )
    with pytest.raises(NotImplementedError):
        MMASACO().propose(1, ctx)
