import json

import pytest
from pydantic import ValidationError

from core.stages import GatherSource
from core.task_spec import (
    AnswerField,
    AnswerSchema,
    FieldType,
    GroundTruth,
    MatcherConfig,
    MatcherKind,
    TaskSpec,
)
from tests.conftest import make_caps, make_runtime_task, make_task_spec


@pytest.mark.parametrize(
    "bad",
    [
        {"tokens": 0},
        {"tokens": -5},
        {"wall_time_s": 0},
        {"wall_time_s": -1.0},
        {"tool_calls": 0},
        {"retries": -1},
    ],
)
def test_invalid_caps_rejected(bad):
    with pytest.raises(ValidationError):
        make_caps(**bad)


def test_zero_retries_is_a_valid_cap():
    assert make_caps(retries=0).retries == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"allowed_sources": ()},  # nothing to gather from
        {"allowed_sources": (GatherSource.API, GatherSource.API)},  # duplicate source
        {"allowed_sources": (GatherSource.FETCH,), "interaction_required": True},  # needs jev
        {"allowed_sources": ("carrier-pigeon",)},  # unknown source
        {"id": ""},
        {"question": ""},
        {"snapshot_id": ""},
    ],
)
def test_invalid_task_configuration_rejected(overrides):
    with pytest.raises(ValidationError):
        make_runtime_task(**overrides)


def test_interaction_required_with_jev_is_valid():
    t = make_runtime_task(allowed_sources=(GatherSource.JEV,), interaction_required=True)
    assert t.interaction_required


def test_answer_schema_needs_unique_nonempty_fields():
    with pytest.raises(ValidationError):
        AnswerSchema(fields=())
    f = AnswerField(name="x", type=FieldType.STRING)
    with pytest.raises(ValidationError):
        AnswerSchema(fields=(f, f))


def test_ground_truth_and_matchers_must_cover_exactly_the_schema_fields():
    rt = make_runtime_task()
    m = {"capital": MatcherConfig(kind=MatcherKind.EXACT)}
    with pytest.raises(ValidationError):
        TaskSpec(
            runtime=rt, ground_truth=GroundTruth(values={"capital": "x", "extra": 1}), matchers=m
        )
    with pytest.raises(ValidationError):
        TaskSpec(runtime=rt, ground_truth=GroundTruth(values={"capital": "x"}), matchers={})


def test_tolerance_only_for_numeric_matchers():
    with pytest.raises(ValidationError):
        MatcherConfig(kind=MatcherKind.EXACT, abs_tol=0.1)
    assert MatcherConfig(kind=MatcherKind.NUMERIC_TOLERANCE, rel_tol=0.01).rel_tol == 0.01


def test_serialization_is_stable_and_roundtrips():
    spec = make_task_spec()
    again = TaskSpec.model_validate_json(spec.model_dump_json())
    assert again == spec
    assert again.canonical_json() == spec.canonical_json()
    assert again.content_hash == spec.content_hash

    # key order of incoming JSON must not matter
    raw = json.loads(spec.model_dump_json())
    reordered = json.loads(json.dumps(raw, sort_keys=True))
    assert TaskSpec.model_validate(reordered).content_hash == spec.content_hash


def test_content_hash_changes_when_ground_truth_changes():
    a = make_task_spec()
    b = TaskSpec(
        runtime=a.runtime,
        ground_truth=GroundTruth(values={"capital": "different"}),
        matchers=a.matchers,
    )
    assert a.content_hash != b.content_hash
