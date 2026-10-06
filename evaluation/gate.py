"""Evaluator boundary.

The offline evaluator is the ONLY authority for PASS / FAIL / INFEASIBLE and search fitness.
It is deterministic and never calls an LLM. It is the only place (with benchmarks/ and
store/) that may hold a ``TaskSpec`` and therefore ground truth.

``DeterministicEvaluator.evaluate_fields`` is the ``legacy_field_match`` implementation - the frozen
benchmark's per-field matchers + evidence check. It is NOT a generic evaluator: in the
optimization pipeline it is reached only through a ``ContractEvaluator`` built by
``benchmarks.legacy_adapter.legacy_evaluator`` (the benchmark compatibility boundary), which
supplies matchers, evidence policy and caps from the task's ``TaskContract``. Generic kinds run in
``evaluation.dispatch``. ``evaluate(TaskSpec, ...)`` remains for code that scores the frozen
benchmark directly (the OSS-baseline comparison).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from core.evaluation_spec import EvaluationSpec, EvaluatorKind
from core.evidence import EvidenceSpan
from core.results import (
    BudgetUsage,
    EvaluatedRun,
    Evaluation,
    EvaluatorRecord,
    ExecutionResult,
    FailureKind,
    FieldResult,
    Verdict,
    usage_exceeds,
)
from core.task_spec import AnswerSchema, Caps, MatcherConfig, TaskSpec
from evaluation.evidence import EvidenceVerifier, SnapshotEvidenceVerifier
from evaluation.fitness import FitnessFunction, ShapedFitness
from evaluation.matchers import DefaultMatcher, Matcher
from evaluation.schema import validate_answer

EVALUATOR_VERSION = "evaluator/mvp-2"


class Evaluator(Protocol):
    version: str

    def evaluate(self, task: TaskSpec, result: ExecutionResult) -> Evaluation:
        """Pure function of (task, result): same inputs -> same Evaluation.

        Rules fixed by the architecture contract:
          * a ``BUDGET_EXCEEDED`` failure (or usage above caps) => INFEASIBLE;
          * otherwise PASS iff every field matches ground truth under its matcher.
        """
        ...


class DeterministicEvaluator:
    """Order of decisions: INFEASIBLE (caps) -> no answer -> schema -> field match -> evidence.

    PASS iff the answer is schema-valid, every field matches its ground truth under its
    matcher, and (when ``require_evidence``) every field cites >= 1 span and all cited spans
    verify against the snapshot. Anything else that completed is FAIL.
    """

    version = EVALUATOR_VERSION

    def __init__(
        self,
        verifier: EvidenceVerifier | None = None,
        matcher: Matcher | None = None,
        fitness: FitnessFunction | None = None,
        *,
        require_evidence: bool = True,
    ) -> None:
        self.verifier = verifier or SnapshotEvidenceVerifier()
        self.matcher = matcher or DefaultMatcher()
        self.fitness_fn = fitness or ShapedFitness()
        self.require_evidence = require_evidence

    def evaluate(self, task: TaskSpec, result: ExecutionResult) -> Evaluation:
        return self.evaluate_fields(
            schema=task.runtime.answer_schema,
            expected=task.ground_truth.values,
            matchers=task.matchers,
            snapshot_id=task.runtime.snapshot_id,
            caps=task.caps,
            result=result,
        )

    def evaluate_fields(
        self,
        *,
        schema: AnswerSchema,
        expected: Mapping[str, Any],
        matchers: Mapping[str, MatcherConfig],
        snapshot_id: str,
        caps: Caps,
        result: ExecutionResult,
        require_evidence: bool | None = None,
    ) -> Evaluation:
        """Judge ``result`` with every input explicit (no task object, so the caller decides
        where matchers, caps and the evidence policy come from)."""
        require_evidence = self.require_evidence if require_evidence is None else require_evidence
        spec_hash = legacy_spec_hash(schema, matchers, require_evidence)
        usage = result.budget_usage
        breached = bool(usage_exceeds(usage, caps)) or (
            result.failure is not None and result.failure.kind is FailureKind.BUDGET_EXCEEDED
        )
        if breached:
            return self._build(Verdict.INFEASIBLE, (), usage, caps, spec_hash)

        names = [f.name for f in schema.fields]
        answer = result.answer
        if result.failure is not None or answer is None:
            return self._build(
                Verdict.FAIL,
                tuple(FieldResult(field=n, matched=False) for n in names),
                usage,
                caps,
                spec_hash,
            )

        problems = validate_answer(schema, answer.values)
        spans_by_field: dict[str, list[EvidenceSpan]] = {}
        for fe in answer.evidence:
            spans_by_field.setdefault(fe.field, []).extend(fe.spans)

        results = []
        for name in names:
            matched = (
                name in answer.values
                and name not in problems
                and self.matcher.matches(expected[name], answer.values[name], matchers[name])
            )
            results.append(
                FieldResult(
                    field=name,
                    matched=matched,
                    evidence_valid=self._evidence_valid(spans_by_field.get(name, []), snapshot_id),
                )
            )
        passed = (
            not problems
            and all(r.matched for r in results)
            and (not require_evidence or all(r.evidence_valid for r in results))
        )
        verdict = Verdict.PASS if passed else Verdict.FAIL
        return self._build(verdict, tuple(results), usage, caps, spec_hash)

    def evaluate_run(self, task: TaskSpec, result: ExecutionResult) -> EvaluatedRun:
        return EvaluatedRun(execution=result, evaluation=self.evaluate(task, result))

    def _evidence_valid(self, spans: list[EvidenceSpan], snapshot_id: str) -> bool:
        return bool(spans) and all(self.verifier.is_valid(s, snapshot_id) for s in spans)

    def _build(
        self,
        verdict: Verdict,
        field_results: tuple[FieldResult, ...],
        usage: BudgetUsage,
        caps: Caps,
        spec_hash: str,
    ) -> Evaluation:
        measured = verdict is not Verdict.INFEASIBLE
        quality = (
            sum(r.matched for r in field_results) / len(field_results) if field_results else 0.0
        )
        return Evaluation(
            verdict=verdict,
            fitness=self.fitness_fn.fitness(verdict, field_results, usage, caps),
            evaluator_version=f"{self.version}+{self.fitness_fn.version}",
            field_results=field_results,
            evaluator=EvaluatorRecord(
                kind=EvaluatorKind.LEGACY_FIELD_MATCH.value,
                version=self.version,
                spec_hash=spec_hash,
                quality=quality if measured else None,
                passed=(verdict is Verdict.PASS) if measured else None,
            ),
        )


def legacy_spec_hash(
    schema: AnswerSchema, matchers: Mapping[str, MatcherConfig], require_evidence: bool
) -> str:
    """Identity of the ``legacy_field_match`` spec these matchers + evidence policy amount to
    (equal to the adapter-built contract's ``evaluation.identity_hash``)."""
    return EvaluationSpec(
        evaluator=EvaluatorKind.LEGACY_FIELD_MATCH,
        config={
            "matchers": {f.name: matchers[f.name].model_dump(mode="json") for f in schema.fields},
            "require_evidence": require_evidence,
        },
    ).identity_hash
