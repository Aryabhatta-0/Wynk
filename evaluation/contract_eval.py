"""Contract-driven evaluation: judge each run under ITS task's ``TaskContract``.

The contract decides how an output is judged - evaluator kind, pinned implementation version
and config (``EvaluationSpec``), which per-run caps make a run INFEASIBLE (``ConstraintLimits``),
which output fields exist. The evaluator holds the one thing a contract deliberately lacks: the
expected (target) values, keyed by row id (``References``). They never leave this boundary.

One run is judged in three separate steps, each owned by a different authority:

    feasible?   per-run caps from contract.constraints     -> INFEASIBLE, quality never measured
    quality     evaluation.dispatch, from contract.evaluation (kind, pinned version, config)
    fitness     evaluation.fitness, from (verdict, quality, usage, caps)

Every generic evaluator kind goes through the one dispatcher. ``legacy_field_match`` (the frozen
benchmark's matchers + snapshot evidence check) is not generic: it runs only when this evaluator
was built at the benchmark compatibility boundary (``benchmarks.legacy_adapter.legacy_evaluator``
passes ``legacy_verifier``); otherwise it is refused like any unsupported kind.

``check_task`` fails closed BEFORE execution and ``evaluate`` fails closed after it: missing
expected values, an unknown or unsupported evaluator, a version mismatch or an invalid config
raise ``EvaluationFailed`` / ``ContractError`` - never a PASS, a fitness or a default.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from core.evaluation_spec import EvaluatorKind, LegacyFieldMatchConfig
from core.results import (
    EvaluatedRun,
    Evaluation,
    EvaluatorFailure,
    ExecutionResult,
    FailureKind,
    Verdict,
    usage_exceeds,
)
from core.run_contract import ExecutionTask, SnapshotSource
from core.task_contract import ContractError
from evaluation.dispatch import EvaluationFailed, ResolvedEvaluator, resolve
from evaluation.evidence import EvidenceVerifier
from evaluation.fitness import FitnessFunction, ShapedFitness
from evaluation.gate import EVALUATOR_VERSION, DeterministicEvaluator


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
        fitness: FitnessFunction | None = None,
        legacy_verifier: EvidenceVerifier | None = None,
    ) -> None:
        """``legacy_verifier`` enables ``legacy_field_match`` (benchmark compatibility only)."""
        self.references = (
            references if isinstance(references, References) else References(references)
        )
        self.fitness_fn = fitness or ShapedFitness()
        self._legacy = (
            DeterministicEvaluator(legacy_verifier, fitness=self.fitness_fn)
            if legacy_verifier is not None
            else None
        )

    # -- preflight ------------------------------------------------------------------------------
    def check_task(self, task: ExecutionTask) -> ResolvedEvaluator | None:
        """Raise unless ``task`` can be evaluated exactly as its contract says. Returns the
        resolved generic evaluator (``None`` for the legacy kind)."""
        expected = self.references.expected(task.id)
        if set(expected) != task.answer_schema.field_names:
            raise ContractError(f"expected values of row {task.id} do not match the output schema")
        spec = task.contract.evaluation
        if spec.evaluator is not EvaluatorKind.LEGACY_FIELD_MATCH:
            resolved = resolve(spec)
            # probe with no prediction: surfaces an unusable target before any model call
            probe = resolved.evaluate(task.answer_schema, expected, None)
            if not probe.ok:
                raise EvaluationFailed(probe)
            return resolved
        kind, version = spec.evaluator.value, spec.evaluator_version
        if self._legacy is None:
            raise EvaluationFailed.of(
                kind,
                version,
                EvaluatorFailure.UNSUPPORTED,
                "legacy_field_match runs only behind the benchmark compatibility boundary "
                "(benchmarks.legacy_adapter.legacy_evaluator)",
                spec.identity_hash,
            )
        if version != EVALUATOR_VERSION:
            raise EvaluationFailed.of(
                kind,
                version,
                EvaluatorFailure.VERSION_MISMATCH,
                f"spec pins {version!r}, implementation is {EVALUATOR_VERSION!r}",
                spec.identity_hash,
            )
        if not isinstance(task.source, SnapshotSource):
            raise EvaluationFailed.of(
                kind,
                version,
                EvaluatorFailure.UNSUPPORTED,
                "legacy_field_match needs a snapshot data source",
                spec.identity_hash,
            )
        return None

    def check_tasks(self, tasks: Iterable[ExecutionTask]) -> None:
        for task in tasks:
            self.check_task(task)

    # -- judging --------------------------------------------------------------------------------
    def evaluate(self, task: ExecutionTask, result: ExecutionResult) -> Evaluation:
        resolved = self.check_task(task)
        if result.key.task_id != task.id or result.key.contract_hash != task.contract_hash:
            raise ContractError(f"result {result.run_id[:12]} was not produced for this task")
        expected = self.references.expected(task.id)
        if resolved is None:
            return self._evaluate_legacy(task, expected, result)

        caps, usage = task.caps, result.budget_usage
        fitness_version = self.fitness_fn.version
        evaluator_version = f"{resolved.version}+{fitness_version}"
        breached = bool(usage_exceeds(usage, caps)) or (
            result.failure is not None and result.failure.kind is FailureKind.BUDGET_EXCEEDED
        )
        if breached:  # a hard constraint, decided before (and instead of) measuring quality
            return Evaluation(
                verdict=Verdict.INFEASIBLE,
                fitness=self.fitness_fn.score_fitness(Verdict.INFEASIBLE, 0.0, usage, caps),
                evaluator_version=evaluator_version,
                evaluator=resolved.record(),
            )
        answer = result.answer
        predicted = answer.values if result.failure is None and answer is not None else None
        record = resolved.evaluate(task.answer_schema, expected, predicted)
        if not record.ok:
            raise EvaluationFailed(record)
        assert record.quality is not None
        verdict = Verdict.PASS if record.passed else Verdict.FAIL
        return Evaluation(
            verdict=verdict,
            fitness=self.fitness_fn.score_fitness(verdict, record.quality, usage, caps),
            evaluator_version=evaluator_version,
            evaluator=record,
        )

    def evaluate_run(self, task: ExecutionTask, result: ExecutionResult) -> EvaluatedRun:
        return EvaluatedRun(execution=result, evaluation=self.evaluate(task, result))

    def _evaluate_legacy(
        self, task: ExecutionTask, expected: Mapping[str, Any], result: ExecutionResult
    ) -> Evaluation:
        assert self._legacy is not None and isinstance(task.source, SnapshotSource)
        cfg = task.contract.evaluation.typed_config()
        assert isinstance(cfg, LegacyFieldMatchConfig)
        return self._legacy.evaluate_fields(
            schema=task.answer_schema,
            expected=expected,
            matchers=cfg.matchers,
            snapshot_id=task.source.snapshot_id,
            caps=task.caps,
            result=result,
            require_evidence=cfg.require_evidence,
        )
