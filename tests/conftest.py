"""Shared test builders. Kept tiny and explicit so tests read like the architecture."""

from __future__ import annotations

import pytest

from core.genome import Genome
from core.stages import (
    ExtractMethod,
    ExtractStage,
    FailureStrategy,
    FilterMethod,
    FilterStage,
    GatherMode,
    GatherSource,
    GatherStage,
    ReasonMethod,
    ReasonStage,
    SynthesizeMethod,
    SynthesizeStage,
    VerifyMethod,
    VerifyStage,
)
from core.task_spec import (
    AnswerField,
    AnswerSchema,
    Caps,
    FieldType,
    GroundTruth,
    MatcherConfig,
    MatcherKind,
    RuntimeTask,
    TaskClass,
    TaskSpec,
)

SECRET_ANSWER = "TOP-SECRET-GROUND-TRUTH-42"


def gather(source=GatherSource.FETCH, mode=GatherMode.SEQUENTIAL) -> GatherStage:
    return GatherStage(source=source, mode=mode)


def flt(method=FilterMethod.KEYWORD_CHUNK) -> FilterStage:
    return FilterStage(method=method)


def extract(method=ExtractMethod.DIRECT) -> ExtractStage:
    return ExtractStage(method=method)


def reason(method=ReasonMethod.SINGLE) -> ReasonStage:
    return ReasonStage(method=method)


def verify(method=VerifyMethod.SCHEMA_CHECK, on_failure=FailureStrategy.RETRY_1) -> VerifyStage:
    return VerifyStage(method=method, on_failure=on_failure)


def synth(method=SynthesizeMethod.DIRECT) -> SynthesizeStage:
    return SynthesizeStage(method=method)


def minimal_genome() -> Genome:
    return Genome.of(gather(), extract(), synth())


def make_caps(**overrides) -> Caps:
    base = {"tokens": 20_000, "wall_time_s": 120.0, "tool_calls": 20, "retries": 2}
    return Caps(**{**base, **overrides})


def make_runtime_task(**overrides) -> RuntimeTask:
    base = {
        "id": "task-001",
        "task_class": TaskClass.A,
        "question": "What is the capital of France?",
        "answer_schema": AnswerSchema(fields=(AnswerField(name="capital", type=FieldType.STRING),)),
        "caps": make_caps(),
        "allowed_sources": (GatherSource.FETCH, GatherSource.API, GatherSource.JEV),
        "snapshot_id": "snap-001",
    }
    return RuntimeTask(**{**base, **overrides})


def make_task_spec(runtime: RuntimeTask | None = None) -> TaskSpec:
    return TaskSpec(
        runtime=runtime or make_runtime_task(),
        ground_truth=GroundTruth(values={"capital": SECRET_ANSWER}),
        matchers={"capital": MatcherConfig(kind=MatcherKind.NORMALIZED_TEXT)},
    )


@pytest.fixture
def task() -> RuntimeTask:
    return make_runtime_task()


@pytest.fixture
def task_spec() -> TaskSpec:
    return make_task_spec()
