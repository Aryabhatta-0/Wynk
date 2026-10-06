"""EvaluationSpec (configuration) and its evaluator implementations, run through the dispatcher.

Every metric test goes through ``evaluation.dispatch.evaluate_prediction`` - the same path real
candidate evaluation takes - so it also checks what is recorded. Fail-closed behaviour, the
legacy boundary, leakage and an end-to-end run live in ``test_evaluation_dispatch.py``.
"""

import math

import pytest
from pydantic import ValidationError

from core.evaluation_spec import EVALUATOR_VERSIONS, EvaluationSpec, EvaluatorKind
from core.results import EvaluatorFailure
from core.task_spec import FieldType
from evaluation import metrics
from evaluation.dispatch import evaluate_prediction
from evaluation.gate import EVALUATOR_VERSION
from evaluation.metrics import token_f1
from tests.contract_helpers import schema

VALID_CONFIGS = {
    EvaluatorKind.EXACT_MATCH: {"case_sensitive": False},
    EvaluatorKind.CLASSIFICATION_ACCURACY: {"labels": ["spam", "ham"]},
    EvaluatorKind.TOKEN_F1: {"pass_threshold": 0.5},
    EvaluatorKind.JSON_SCHEMA_VALIDITY: {},
    EvaluatorKind.NUMERIC_TOLERANCE: {"absolute_tolerance": 0.01},
    EvaluatorKind.LEGACY_FIELD_MATCH: {"matchers": {"hq": {"kind": "normalized_text"}}},
}


@pytest.mark.parametrize("kind", list(EvaluatorKind))
def test_every_supported_evaluator_accepts_a_valid_config_and_pins_its_version(kind):
    spec = EvaluationSpec(evaluator=kind, config=VALID_CONFIGS[kind])
    assert spec.evaluator_version == EVALUATOR_VERSIONS[kind]
    assert EvaluationSpec.model_validate_json(spec.model_dump_json()) == spec


def test_config_is_normalized_so_equivalent_specs_share_one_identity():
    implicit = EvaluationSpec(evaluator="numeric_tolerance", config={"absolute_tolerance": 0.01})
    explicit = EvaluationSpec(
        evaluator="numeric_tolerance",
        evaluator_version="numeric_tolerance/1",
        config={"relative_tolerance": 0.0, "absolute_tolerance": 0.01},
    )
    assert implicit == explicit and implicit.identity_hash == explicit.identity_hash
    assert implicit.config == {"absolute_tolerance": 0.01, "relative_tolerance": 0.0}
    other = EvaluationSpec(evaluator="numeric_tolerance", config={"absolute_tolerance": 0.02})
    assert other.identity_hash != implicit.identity_hash


@pytest.mark.parametrize(
    ("kind", "config"),
    [
        ("exact_match", {"case_insensitive": True}),  # unknown option
        ("classification_accuracy", {}),  # labels required
        ("classification_accuracy", {"labels": ["only-one"]}),
        ("classification_accuracy", {"labels": ["a", "a"]}),
        ("classification_accuracy", {"labels": ["a", " "]}),
        ("token_f1", {"pass_threshold": 0.0}),
        ("token_f1", {"pass_threshold": 1.5}),
        ("json_schema_validity", {"strict": True}),
        ("numeric_tolerance", {"absolute_tolerance": -0.1}),
        ("numeric_tolerance", {"absolute_tolerance": math.nan}),
        ("numeric_tolerance", {"relative_tolerance": math.inf}),
        ("legacy_field_match", {}),
        ("legacy_field_match", {"matchers": {"x": {"kind": "exact", "abs_tol": 0.1}}}),
        (
            "legacy_field_match",
            {"matchers": {"x": {"kind": "numeric_tolerance", "abs_tol": math.inf}}},
        ),
    ],
)
def test_invalid_configs_fail_closed(kind, config):
    with pytest.raises(ValidationError):
        EvaluationSpec(evaluator=kind, config=config)


def test_unknown_evaluator_fails_closed():
    for kind in ("llm_judge", "self_grade", "bleu", ""):
        with pytest.raises(ValidationError):
            EvaluationSpec(evaluator=kind)


def test_missing_evaluator_fails_closed():
    with pytest.raises(ValidationError):
        EvaluationSpec(config={"absolute_tolerance": 0.1})


def test_pinned_version_mismatch_fails_closed():
    with pytest.raises(ValidationError, match="implemented at"):
        EvaluationSpec(evaluator="exact_match", evaluator_version="exact_match/0")


def test_implementation_registry_agrees_with_the_pinned_versions():
    for kind, metric in metrics.METRICS.items():
        assert EVALUATOR_VERSIONS[kind] == metric.version
    # every generic kind has exactly one implementation; the legacy kind has none here
    assert set(metrics.METRICS) == set(EvaluatorKind) - {EvaluatorKind.LEGACY_FIELD_MATCH}
    # the legacy kind names the existing deterministic evaluator, not a new one
    assert EVALUATOR_VERSIONS[EvaluatorKind.LEGACY_FIELD_MATCH] == EVALUATOR_VERSION


# -- metric correctness (through the dispatcher) ------------------------------------------------

ANSWER = schema(("answer", FieldType.STRING))
LABEL = schema(("label", FieldType.STRING))
VALUE = schema(("value", FieldType.NUMBER))


def run(kind, config, expected, predicted, out=ANSWER):
    record = evaluate_prediction(
        EvaluationSpec(evaluator=kind, config=config), out, expected, predicted
    )
    assert record.ok, record.detail
    return record


def passes(kind, config, expected, predicted, out=ANSWER) -> bool:
    return run(kind, config, expected, predicted, out).passed


def failure(kind, config, expected, predicted, out=ANSWER) -> EvaluatorFailure | None:
    spec = EvaluationSpec(evaluator=kind, config=config)
    return evaluate_prediction(spec, out, expected, predicted).failure


# exact_match
def test_exact_match_is_case_whitespace_and_type_strict_by_default():
    assert passes("exact_match", {}, {"answer": "Paris"}, {"answer": "Paris"})
    assert not passes("exact_match", {}, {"answer": "Paris"}, {"answer": "paris"})
    assert not passes("exact_match", {}, {"answer": "Paris"}, {"answer": "Paris "})
    assert not passes("exact_match", {}, {"answer": "New York"}, {"answer": "New  York"})


def test_exact_match_normalization_options():
    folded = {"case_sensitive": False}
    assert passes("exact_match", folded, {"answer": "Straße"}, {"answer": "STRASSE"})  # casefold
    assert not passes("exact_match", folded, {"answer": "a b"}, {"answer": "A  B"})
    relaxed = {"case_sensitive": False, "normalize_whitespace": True}
    assert passes("exact_match", relaxed, {"answer": "New  York"}, {"answer": "\tnew\nyork "})
    # normalization never deletes characters: punctuation still counts
    assert not passes("exact_match", relaxed, {"answer": "New York"}, {"answer": "New-York"})


def test_exact_match_boundaries():
    assert passes("exact_match", {}, {"answer": ""}, {"answer": ""})  # empty == empty
    assert not passes("exact_match", {}, {"answer": ""}, {"answer": " "})
    count = schema(("count", FieldType.INTEGER))
    assert passes("exact_match", {}, {"count": 0}, {"count": 0}, count)
    assert not passes("exact_match", {}, {"count": 1}, {"count": True}, count)  # bool is not int
    num = schema(("x", FieldType.NUMBER))
    assert not passes("exact_match", {}, {"x": 1}, {"x": 1.0}, num)  # type-strict: int != float


def test_exact_match_scores_the_fraction_of_matching_fields():
    out = schema(("city", FieldType.STRING), ("year", FieldType.INTEGER))
    want = {"city": "Lisbon", "year": 1987}
    r = run("exact_match", {}, want, {"city": "Lisbon", "year": 1988}, out)
    assert (r.quality, r.passed) == (0.5, False)
    r = run("exact_match", {}, want, dict(want), out)
    assert (r.quality, r.passed) == (1.0, True)


# classification_accuracy
CLASSES = {"labels": ["billing", "bugs", "sales"]}


def test_classification_correct_and_incorrect_labels():
    r = run("classification_accuracy", CLASSES, {"label": "bugs"}, {"label": "bugs"}, LABEL)
    assert (r.quality, r.passed) == (1.0, True)
    r = run("classification_accuracy", CLASSES, {"label": "bugs"}, {"label": "billing"}, LABEL)
    assert (r.quality, r.passed) == (0.0, False)


def test_classification_prediction_outside_the_label_set_is_wrong():
    for got in ("Bugs", "bugs ", "a bug", ""):
        assert not passes(
            "classification_accuracy", CLASSES, {"label": "bugs"}, {"label": got}, LABEL
        )
    folded = {**CLASSES, "case_sensitive": False}
    assert passes("classification_accuracy", folded, {"label": "bugs"}, {"label": "BUGS"}, LABEL)


def test_classification_target_outside_the_label_set_fails_closed():
    assert (
        failure("classification_accuracy", CLASSES, {"label": "hr"}, {"label": "hr"}, LABEL)
        is EvaluatorFailure.INVALID_TARGET
    )


# token_f1
def test_token_f1_values():
    assert token_f1("the Eiffel Tower", "Eiffel tower!") == 1.0  # articles/punct/case ignored
    assert token_f1("cat sat", "cat") == pytest.approx(2 / 3)
    assert token_f1("cat sat", "dog") == 0.0
    assert token_f1("x y", "x z") == 0.5


def test_token_f1_empty_partial_and_full():
    # empty: both normalize to no tokens -> perfect; only one empty -> zero
    assert run("token_f1", {}, {"answer": ""}, {"answer": ""}).quality == 1.0
    assert run("token_f1", {}, {"answer": "the"}, {"answer": "a"}).quality == 1.0
    assert run("token_f1", {}, {"answer": "Paris"}, {"answer": ""}).quality == 0.0
    assert run("token_f1", {}, {"answer": ""}, {"answer": "Paris"}).quality == 0.0
    # partial: scored, but default threshold 1.0 demands the full token bag
    partial = run("token_f1", {}, {"answer": "the cat sat"}, {"answer": "cat"})
    assert partial.quality == pytest.approx(2 / 3) and not partial.passed
    # full
    full = run("token_f1", {}, {"answer": "The cat sat."}, {"answer": "cat sat"})
    assert (full.quality, full.passed) == (1.0, True)


def test_token_f1_threshold_is_inclusive():
    case = {"answer": "x y"}, {"answer": "x z"}  # F1 exactly 0.5
    assert passes("token_f1", {"pass_threshold": 0.5}, *case)
    assert not passes("token_f1", {"pass_threshold": 0.5000001}, *case)


# json_schema_validity
RECORD = schema(("name", FieldType.STRING), ("count", FieldType.INTEGER))


def test_json_schema_validity_judges_shape_not_values():
    ok = run(
        "json_schema_validity", {}, {"name": "x", "count": 1}, {"name": "y", "count": 9}, RECORD
    )
    assert (ok.quality, ok.passed) == (1.0, True)
    # it needs no target at all
    assert run("json_schema_validity", {}, None, {"name": "y", "count": 9}, RECORD).passed


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "y"},  # missing required field
        {"name": "y", "count": 1.5},  # float for integer
        {"name": "y", "count": True},  # bool for integer
        {"name": 3, "count": 1},  # int for string
        {"name": "y", "count": 1, "extra": 0},  # unknown field
        {"name": None, "count": 1},  # null for required field
        {},
    ],
)
def test_json_schema_validity_rejects_invalid_structures(bad):
    r = run("json_schema_validity", {}, None, bad, RECORD)
    assert (r.quality, r.passed) == (0.0, False)


def test_json_schema_validity_typed_fields():
    out = schema(("when", FieldType.DATE), ("tags", FieldType.STRING_LIST))
    assert passes("json_schema_validity", {}, None, {"when": "2024-02-29", "tags": []}, out)
    assert not passes("json_schema_validity", {}, None, {"when": "2023-02-29", "tags": []}, out)
    assert not passes("json_schema_validity", {}, None, {"when": "2024-01-01", "tags": [1]}, out)


# numeric_tolerance
def test_numeric_tolerance_absolute_boundary_is_inclusive():
    tol = {"absolute_tolerance": 0.25}  # exactly representable, so the boundary is exact
    assert passes("numeric_tolerance", tol, {"value": 1.0}, {"value": 1.25}, VALUE)
    assert passes("numeric_tolerance", tol, {"value": 1.0}, {"value": 0.75}, VALUE)
    assert not passes("numeric_tolerance", tol, {"value": 1.0}, {"value": 1.2500001}, VALUE)
    assert not passes("numeric_tolerance", tol, {"value": 1.0}, {"value": 0.7499999}, VALUE)


def test_numeric_tolerance_relative_boundary_and_sign():
    tol = {"relative_tolerance": 0.25}
    assert passes("numeric_tolerance", tol, {"value": 8}, {"value": 10}, VALUE)
    assert not passes("numeric_tolerance", tol, {"value": 8}, {"value": 10.001}, VALUE)
    assert passes("numeric_tolerance", tol, {"value": -8}, {"value": -6}, VALUE)  # |target|
    # the larger of the two bounds applies
    both = {"absolute_tolerance": 3.0, "relative_tolerance": 0.25}
    assert passes("numeric_tolerance", both, {"value": 8}, {"value": 11}, VALUE)
    # relative tolerance around zero allows nothing
    assert not passes("numeric_tolerance", tol, {"value": 0}, {"value": 1e-12}, VALUE)


def test_numeric_tolerance_zero_tolerance_and_types():
    assert passes("numeric_tolerance", {}, {"value": 1}, {"value": 1.0}, VALUE)  # numeric equality
    assert not passes("numeric_tolerance", {}, {"value": 1.0}, {"value": 1.0000001}, VALUE)
    for got in (True, "1.0", None, math.nan, math.inf):
        assert not passes("numeric_tolerance", {}, {"value": 1.0}, {"value": got}, VALUE)


def test_numeric_tolerance_scores_each_field():
    out = schema(("a", FieldType.NUMBER), ("b", FieldType.NUMBER))
    r = run(
        "numeric_tolerance", {"absolute_tolerance": 0.5}, {"a": 1, "b": 2}, {"a": 1.4, "b": 3}, out
    )
    assert (r.quality, r.passed) == (0.5, False)


def test_numeric_tolerance_non_finite_target_fails_closed():
    assert (
        failure("numeric_tolerance", {}, {"value": math.inf}, {"value": 1.0}, VALUE)
        is EvaluatorFailure.INVALID_TARGET
    )
