"""EvaluationSpec (configuration) and evaluation.metrics (implementation)."""

import math

import pytest
from pydantic import ValidationError

from core.evaluation_spec import EVALUATOR_VERSIONS, EvaluationSpec, EvaluatorKind
from core.task_spec import FieldType
from evaluation import metrics
from evaluation.gate import EVALUATOR_VERSION
from evaluation.metrics import EvaluatorUnavailable, get_metric, score, token_f1
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
    for kind, (version, _fn) in metrics.METRICS.items():
        assert EVALUATOR_VERSIONS[kind] == version
    assert set(metrics.METRICS) == set(EvaluatorKind) - {EvaluatorKind.LEGACY_FIELD_MATCH}
    # the legacy kind names the existing deterministic evaluator, not a new one
    assert EVALUATOR_VERSIONS[EvaluatorKind.LEGACY_FIELD_MATCH] == EVALUATOR_VERSION


def test_no_silent_substitution_when_the_implementation_version_moves(monkeypatch):
    spec = EvaluationSpec(evaluator="exact_match")
    _old, fn = metrics.METRICS[EvaluatorKind.EXACT_MATCH]
    monkeypatch.setitem(metrics.METRICS, EvaluatorKind.EXACT_MATCH, ("exact_match/2", fn))
    with pytest.raises(EvaluatorUnavailable):
        get_metric(spec)


def test_legacy_kind_is_not_a_per_example_metric():
    spec = EvaluationSpec(
        evaluator="legacy_field_match", config=VALID_CONFIGS["legacy_field_match"]
    )
    with pytest.raises(EvaluatorUnavailable, match="DeterministicEvaluator"):
        get_metric(spec)


# -- metric correctness -------------------------------------------------------------------------

ANSWER = schema(("answer", FieldType.STRING))


def _score(kind, config, expected, predicted, out=ANSWER):
    return score(EvaluationSpec(evaluator=kind, config=config), out, expected, predicted)


def test_exact_match_is_case_and_type_strict_by_default():
    assert _score("exact_match", {}, {"answer": "Paris"}, {"answer": "Paris"}).passed
    assert not _score("exact_match", {}, {"answer": "Paris"}, {"answer": "paris"}).passed
    relaxed = {"case_sensitive": False, "normalize_whitespace": True}
    assert _score("exact_match", relaxed, {"answer": "New  York"}, {"answer": " new york "}).passed


def test_exact_match_scores_the_fraction_of_matching_fields():
    out = schema(("city", FieldType.STRING), ("year", FieldType.INTEGER))
    r = _score(
        "exact_match", {}, {"city": "Lisbon", "year": 1987}, {"city": "Lisbon", "year": 1988}, out
    )
    assert (r.score, r.passed) == (0.5, False)
    r = _score(
        "exact_match", {}, {"city": "Lisbon", "year": 1987}, {"city": "Lisbon", "year": 1987}, out
    )
    assert (r.score, r.passed) == (1.0, True)


def test_classification_accuracy():
    cfg = {"labels": ["billing", "bugs"]}
    out = schema(("label", FieldType.STRING))
    assert _score("classification_accuracy", cfg, {"label": "bugs"}, {"label": "bugs"}, out).passed
    assert not _score(
        "classification_accuracy", cfg, {"label": "bugs"}, {"label": "billing"}, out
    ).passed
    # a prediction outside the label set is wrong even if it "looks" right
    assert not _score(
        "classification_accuracy", cfg, {"label": "bugs"}, {"label": "Bugs"}, out
    ).passed
    folded = {"labels": ["billing", "bugs"], "case_sensitive": False}
    assert _score(
        "classification_accuracy", folded, {"label": "bugs"}, {"label": "BUGS"}, out
    ).passed
    with pytest.raises(ValueError, match="not one of the configured labels"):
        _score("classification_accuracy", cfg, {"label": "sales"}, {"label": "sales"}, out)


def test_token_f1_values_and_threshold():
    assert token_f1("the Eiffel Tower", "Eiffel tower!") == 1.0  # articles/punct/case ignored
    assert token_f1("cat sat", "cat") == pytest.approx(2 / 3)
    assert token_f1("cat sat", "dog") == 0.0
    assert token_f1("the", "a") == 1.0  # both normalize to empty
    partial = {"answer": "the cat sat"}, {"answer": "cat"}
    assert not _score("token_f1", {}, *partial).passed  # default threshold 1.0 = exact tokens
    r = _score("token_f1", {"pass_threshold": 0.6}, *partial)
    assert r.passed and r.score == pytest.approx(2 / 3)


def test_json_schema_validity():
    out = schema(("name", FieldType.STRING), ("count", FieldType.INTEGER))
    ok = _score(
        "json_schema_validity", {}, {"name": "x", "count": 1}, {"name": "y", "count": 9}, out
    )
    assert ok.passed and ok.score == 1.0  # validity only: values are not compared
    for bad in ({"name": "y"}, {"name": "y", "count": 1.5}, {"name": "y", "count": 1, "x": 0}):
        assert not _score("json_schema_validity", {}, {"name": "x", "count": 1}, bad, out).passed


def test_numeric_tolerance():
    out = schema(("value", FieldType.NUMBER))
    absolute = {"absolute_tolerance": 0.01}
    assert _score("numeric_tolerance", absolute, {"value": 1.0}, {"value": 1.005}, out).passed
    assert not _score("numeric_tolerance", absolute, {"value": 1.0}, {"value": 1.02}, out).passed
    relative = {"relative_tolerance": 0.1}
    assert _score("numeric_tolerance", relative, {"value": 100}, {"value": 109}, out).passed
    assert not _score("numeric_tolerance", relative, {"value": 100}, {"value": 111}, out).passed
    assert not _score("numeric_tolerance", absolute, {"value": 1.0}, {"value": True}, out).passed
    assert not _score("numeric_tolerance", absolute, {"value": 1.0}, {"value": "1.0"}, out).passed


def test_missing_or_schema_invalid_prediction_scores_zero():
    for predicted in (None, {}, {"answer": 3}, {"answer": "x", "extra": "y"}):
        r = _score("exact_match", {}, {"answer": "x"}, predicted)
        assert (r.score, r.passed) == (0.0, False)


def test_expected_values_must_cover_exactly_the_output_fields():
    with pytest.raises(ValueError, match="cover exactly"):
        _score("exact_match", {}, {"answer": "x", "other": 1}, {"answer": "x"})
