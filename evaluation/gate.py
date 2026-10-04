"""Evaluator boundary.

The offline evaluator is the ONLY authority for PASS / FAIL / INFEASIBLE and search fitness.
It is deterministic and never calls an LLM. It is the only place (with benchmarks/ and
store/) that may hold a ``TaskSpec`` and therefore ground truth.
"""

from __future__ import annotations

from typing import Protocol

from core.evidence import EvidenceSpan
from core.results import (
    EvaluatedRun,
    Evaluation,
    ExecutionResult,
    FailureKind,
    FieldResult,
    Verdict,
    usage_exceeds,
)
from core.task_spec import TaskSpec
from evaluation.evidence import EvidenceVerifier, SnapshotEvidenceVerifier
from evaluation.fitness import FitnessFunction, ShapedFitness
from evaluation.matchers import DefaultMatcher, Matcher
from evaluation.schema import validate_answer

EVALUATOR_VERSION = "evaluator/mvp-1"


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
        usage = result.budget_usage
        breached = bool(usage_exceeds(usage, task.caps)) or (
            result.failure is not None and result.failure.kind is FailureKind.BUDGET_EXCEEDED
        )
        if breached:
            return self._build(Verdict.INFEASIBLE, (), usage)

        schema = task.runtime.answer_schema
        names = [f.name for f in schema.fields]
        answer = result.answer
        if answer is None:
            return self._build(
                Verdict.FAIL, tuple(FieldResult(field=n, matched=False) for n in names), usage
            )

        problems = validate_answer(schema, answer.values)
        spans_by_field: dict[str, list[EvidenceSpan]] = {}
        for fe in answer.evidence:
            spans_by_field.setdefault(fe.field, []).extend(fe.spans)

        results = []
        for name in names:
            matched = name not in problems and self.matcher.matches(
                task.ground_truth.values[name], answer.values[name], task.matchers[name]
            )
            results.append(
                FieldResult(
                    field=name,
                    matched=matched,
                    evidence_valid=self._evidence_valid(
                        spans_by_field.get(name, []), task.runtime.snapshot_id
                    ),
                )
            )
        passed = (
            not problems
            and all(r.matched for r in results)
            and (not self.require_evidence or all(r.evidence_valid for r in results))
        )
        return self._build(Verdict.PASS if passed else Verdict.FAIL, tuple(results), usage)

    def evaluate_run(self, task: TaskSpec, result: ExecutionResult) -> EvaluatedRun:
        return EvaluatedRun(execution=result, evaluation=self.evaluate(task, result))

    def _evidence_valid(self, spans: list[EvidenceSpan], snapshot_id: str) -> bool:
        return bool(spans) and all(self.verifier.is_valid(s, snapshot_id) for s in spans)

    def _build(self, verdict: Verdict, field_results: tuple[FieldResult, ...], usage) -> Evaluation:
        return Evaluation(
            verdict=verdict,
            fitness=self.fitness_fn.fitness(verdict, field_results, usage),
            evaluator_version=f"{self.version}+{self.fitness_fn.version}",
            field_results=field_results,
        )
