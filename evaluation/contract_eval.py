"""Contract-driven evaluation: judge each run under ITS task's ``TaskContract``.

The contract decides how an output is judged - evaluator kind, pinned implementation version
and config (``EvaluationSpec``), which per-run caps make a run INFEASIBLE (``ConstraintLimits``),
which output fields exist. The evaluator holds the one thing a contract deliberately lacks: the
expected (target) values, keyed by row id (``References``). They never leave this boundary.

    legacy_field_match   DeterministicEvaluator.evaluate_fields, with matchers and the evidence
                         policy from the contract's config, evidence checked against the task's
                         snapshot source
    any other kind       evaluation.metrics.score, looked up by (kind, pinned version)

``check_task`` fails closed BEFORE execution: missing expected values, an unimplemented
evaluator or a version mismatch is an error, never a silent PASS or a default.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from core.evaluation_spec import EvaluatorKind, LegacyFieldMatchConfig
from core.results import (
    EvaluatedRun,
    Evaluation,
    ExecutionResult,
    FailureKind,
    Verdict,
    usage_exceeds,
)
from core.run_contract import ExecutionTask, SnapshotSource
from core.task_contract import ContractError
from evaluation.evidence import EvidenceVerifier
from evaluation.fitness import FitnessFunction
from evaluation.gate import EVALUATOR_VERSION, DeterministicEvaluator
from evaluation.metrics import get_metric, score


class References:
    """Expected target values by row id. Evaluator side only; never printed."""

    def __init__(self, values: Mapping[str, Mapping[str, Any]]) -> None:
        self._values = {row_id: dict(v) for row_id, v in values.items()}

    def __contains__(self, row_id: object) -> bool:
        return row_id in self._values

    def __len__(self) -> int:
        return len(self._values)

    def expected(self, row_id: str) -> dict[str, Any]:
        try:
            return self._values[row_id]
        except KeyError:
            raise ContractError(f"no expected values for row {row_id!r}") from None

    def __repr__(self) -> str:  # never leak values into logs/tracebacks
        return f"References(<{len(self._values)} rows redacted>)"

    __str__ = __repr__


class ContractEvaluator:
    def __init__(
        self,
        references: References | Mapping[str, Mapping[str, Any]],
        *,
        verifier: EvidenceVerifier | None = None,
        fitness: FitnessFunction | None = None,
    ) -> None:
        self.references = (
            references if isinstance(references, References) else References(references)
        )
        self._fields = DeterministicEvaluator(verifier, fitness=fitness)
        self.fitness_fn = self._fields.fitness_fn

    # -- preflight ------------------------------------------------------------------------------
    def check_task(self, task: ExecutionTask) -> None:
        """Raise unless ``task`` can be evaluated exactly as its contract says."""
        expected = self.references.expected(task.id)
        if set(expected) != task.answer_schema.field_names:
            raise ContractError(f"expected values of row {task.id} do not match the output schema")
        spec = task.contract.evaluation
        if spec.evaluator is EvaluatorKind.LEGACY_FIELD_MATCH:
            if spec.evaluator_version != EVALUATOR_VERSION:
                raise ContractError(
                    f"legacy_field_match is implemented at {EVALUATOR_VERSION!r}, "
                    f"contract pins {spec.evaluator_version!r}"
                )
            if not isinstance(task.source, SnapshotSource):
                raise ContractError("legacy_field_match needs a snapshot data source")
        else:
            get_metric(spec)  # EvaluatorUnavailable on an unknown kind or version mismatch

    def check_tasks(self, tasks: Iterable[ExecutionTask]) -> None:
        for task in tasks:
            self.check_task(task)

    # -- judging --------------------------------------------------------------------------------
    def evaluate(self, task: ExecutionTask, result: ExecutionResult) -> Evaluation:
        self.check_task(task)
        if result.key.task_id != task.id or result.key.contract_hash != task.contract_hash:
            raise ContractError(f"result {result.run_id[:12]} was not produced for this task")
        expected = self.references.expected(task.id)
        spec = task.contract.evaluation
        if spec.evaluator is EvaluatorKind.LEGACY_FIELD_MATCH:
            cfg = spec.typed_config()
            assert isinstance(cfg, LegacyFieldMatchConfig) and isinstance(
                task.source, SnapshotSource
            )
            return self._fields.evaluate_fields(
                schema=task.answer_schema,
                expected=expected,
                matchers=cfg.matchers,
                snapshot_id=task.source.snapshot_id,
                caps=task.caps,
                result=result,
                require_evidence=cfg.require_evidence,
            )
        return self._metric(task, expected, result)

    def evaluate_run(self, task: ExecutionTask, result: ExecutionResult) -> EvaluatedRun:
        return EvaluatedRun(execution=result, evaluation=self.evaluate(task, result))

    def _metric(
        self, task: ExecutionTask, expected: Mapping[str, Any], result: ExecutionResult
    ) -> Evaluation:
        caps, usage, spec = task.caps, result.budget_usage, task.contract.evaluation
        fitness_version = self.fitness_fn.version
        breached = bool(usage_exceeds(usage, caps)) or (
            result.failure is not None and result.failure.kind is FailureKind.BUDGET_EXCEEDED
        )
        if breached:
            return Evaluation(
                verdict=Verdict.INFEASIBLE,
                fitness=self.fitness_fn.score_fitness(Verdict.INFEASIBLE, 0.0, usage, caps),
                evaluator_version=f"{spec.evaluator_version}+{fitness_version}",
            )
        answer = result.answer
        predicted = answer.values if result.failure is None and answer is not None else None
        measured = score(spec, task.answer_schema, expected, predicted)
        verdict = Verdict.PASS if measured.passed else Verdict.FAIL
        return Evaluation(
            verdict=verdict,
            fitness=self.fitness_fn.score_fitness(verdict, measured.score, usage, caps),
            evaluator_version=f"{measured.evaluator_version}+{fitness_version}",
        )
