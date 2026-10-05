"""ObjectiveSpec, measured hard limits, and the rule that hard constraints outrank fitness."""

import json
import math
from itertools import product

import pytest
from pydantic import ValidationError

from core.constraints import ConstraintLimits, check_limits
from core.objective import (
    CandidateMeasurements,
    Metric,
    MissingMetric,
    ObjectiveMode,
    ObjectiveSpec,
)
from core.task_spec import Caps
from core.violations import ViolationCode
from tests.contract_helpers import classification_contract

BALANCED = {
    "mode": "balanced",
    "weights": {"quality": 0.7, "cost": 0.2, "latency": 0.1},
    "scales": {"cost": 0.01, "latency": 2.0},
}


def m(**values) -> CandidateMeasurements:
    return CandidateMeasurements(**values)


# -- ObjectiveSpec ------------------------------------------------------------------------------


def test_default_objective_is_quality_first():
    o = ObjectiveSpec()
    assert o.mode is ObjectiveMode.MAXIMIZE_QUALITY
    assert o.required_metrics() == (Metric.QUALITY,)


def test_valid_balanced_objective_and_its_utility():
    o = ObjectiveSpec(**BALANCED)
    assert o.required_metrics() == (Metric.COST, Metric.LATENCY, Metric.QUALITY)
    key = o.rank_key(m(quality=0.9, mean_cost_per_example=0.005, mean_latency_s=1.0))
    # 0.7 * 0.9 - 0.2 * (0.005 / 0.01) - 0.1 * (1.0 / 2.0)
    assert key == pytest.approx((0.48,))


def test_balanced_serialization_is_deterministic_across_dict_order():
    a = ObjectiveSpec(**BALANCED)
    b = ObjectiveSpec(
        mode="balanced",
        weights={"latency": 0.1, "cost": 0.2, "quality": 0.7},
        scales={"latency": 2.0, "cost": 0.01},
    )
    assert a == b and a.identity_hash == b.identity_hash
    again = ObjectiveSpec.model_validate(json.loads(a.model_dump_json()))
    assert again.identity_hash == a.identity_hash


@pytest.mark.parametrize(
    "spec",
    [
        {"mode": "maximize_quality", "weights": {"quality": 1.0}},  # weights only for balanced
        {"mode": "minimize_cost", "scales": {"cost": 1.0}},
        {"mode": "balanced"},  # balanced needs explicit weights
        {**BALANCED, "weights": {"quality": 0.7, "cost": 0.2}},  # sums to 0.9
        {**BALANCED, "weights": {"quality": 0.7, "cost": 0.2, "latency": 0.2}},  # sums to 1.1
        {**BALANCED, "weights": {"quality": math.nan, "cost": 0.2, "latency": 0.1}},
        {**BALANCED, "weights": {"quality": math.inf, "cost": 0.2, "latency": 0.1}},
        {**BALANCED, "weights": {"quality": 1.2, "cost": -0.1, "latency": -0.1}},
        {"mode": "balanced", "weights": {"cost": 1.0}, "scales": {"cost": 1.0}},  # no quality
        {**BALANCED, "scales": {"cost": 0.01}},  # latency is penalized but has no scale
        {**BALANCED, "scales": {"cost": 0.01, "latency": 2.0, "tokens": 100.0}},  # extra scale
        {**BALANCED, "scales": {"cost": 0.0, "latency": 2.0}},
        {**BALANCED, "scales": {"cost": math.nan, "latency": 2.0}},
        {**BALANCED, "weights": {"quality": 0.7, "fun": 0.3}},  # unknown metric
        {"mode": "pareto"},  # not implemented (yet)
        {"mode": "maximize_quality", "surprise": 1},
    ],
)
def test_invalid_objectives_fail_closed(spec):
    with pytest.raises(ValidationError):
        ObjectiveSpec(**spec)


def test_zero_weight_metrics_are_allowed_and_not_penalized():
    o = ObjectiveSpec(
        mode="balanced", weights={"quality": 0.9, "cost": 0.1, "tokens": 0.0}, scales={"cost": 1.0}
    )
    assert Metric.TOKENS not in o.required_metrics()


@pytest.mark.parametrize(
    ("spec", "measured"),
    [
        ({}, m(mean_cost_per_example=0.1)),  # quality missing
        ({"mode": "minimize_cost"}, m(quality=0.9)),  # cost missing
        ({"mode": "minimize_latency"}, m(quality=0.9, mean_cost_per_example=0.1)),
        (BALANCED, m(quality=0.9, mean_cost_per_example=0.1)),  # latency missing
    ],
)
def test_missing_metrics_are_never_silently_accepted(spec, measured):
    with pytest.raises(MissingMetric):
        ObjectiveSpec(**spec).rank_key(measured)


def test_single_objective_modes_order_candidates_as_documented():
    cheap = m(quality=0.8, mean_cost_per_example=0.001, mean_latency_s=5.0)
    good = m(quality=0.95, mean_cost_per_example=0.01, mean_latency_s=1.0)
    q, c, lat = (ObjectiveSpec(mode=x) for x in ObjectiveMode if x is not ObjectiveMode.BALANCED)
    assert q.rank_key(good) > q.rank_key(cheap)
    assert c.rank_key(cheap) > c.rank_key(good)
    assert lat.rank_key(good) > lat.rank_key(cheap)


@pytest.mark.parametrize(
    "values",
    [
        {"quality": math.nan},
        {"quality": 1.5},
        {"quality": -0.1},
        {"mean_cost_per_example": math.inf},
        {"mean_latency_s": -1.0},
        {"max_tokens_per_example": -1},
        {"max_wall_time_s_per_example": math.nan},
        {"accuracy": 0.9},
    ],
)
def test_measurements_reject_non_finite_negative_or_unknown_values(values):
    with pytest.raises(ValidationError):
        CandidateMeasurements(**values)


# -- measured hard limits -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "limits",
    [
        {"minimum_quality": 1.5},
        {"minimum_quality": math.nan},
        {"maximum_cost_per_example": -0.01},
        {"maximum_cost_per_example": math.inf},
        {"maximum_mean_latency_s": 0},
        {"maximum_p95_latency_s": -1},
        {"maximum_tokens_per_example": 0},
        {"maximum_workflow_steps": 0},
        {"maximum_model_calls": -1},
        {"maximum_retries": -1},
        {"maximum_wall_time_s": math.inf},
        {"maximum_latency": 3.0},  # unknown limit
    ],
)
def test_invalid_limits_fail_closed(limits):
    with pytest.raises(ValidationError):
        ConstraintLimits(**limits)


def test_caps_round_trip_through_limits():
    caps = Caps(tokens=9000, wall_time_s=180.0, tool_calls=12, retries=0)
    limits = ConstraintLimits.from_caps(caps)
    assert limits.to_caps() == caps
    with pytest.raises(ValueError, match="maximum_retries"):
        ConstraintLimits(
            maximum_tokens_per_example=1, maximum_wall_time_s=1.0, maximum_tool_calls=1
        ).to_caps()


LIMITS = ConstraintLimits(
    minimum_quality=0.8,
    maximum_cost_per_example=0.01,
    maximum_p95_latency_s=3.0,
    maximum_tokens_per_example=4000,
    maximum_workflow_steps=6,
)
WITHIN = m(
    quality=0.85,
    mean_cost_per_example=0.004,
    p95_latency_s=2.0,
    max_tokens_per_example=3000,
    workflow_steps=4,
)


def test_limits_hold_and_equality_is_allowed():
    assert check_limits(LIMITS, WITHIN) == ()
    at_limit = WITHIN.model_copy(update={"quality": 0.8, "max_tokens_per_example": 4000})
    assert check_limits(LIMITS, at_limit) == ()


@pytest.mark.parametrize(
    "update",
    [
        {"quality": 0.79},
        {"mean_cost_per_example": 0.011},
        {"p95_latency_s": 3.01},
        {"max_tokens_per_example": 4001},
        {"workflow_steps": 7},
    ],
)
def test_each_breached_limit_is_a_violation(update):
    violations = check_limits(LIMITS, WITHIN.model_copy(update=update))
    assert [v.code for v in violations] == [ViolationCode.LIMIT_VIOLATED]


def test_an_active_limit_without_its_measurement_fails_closed():
    violations = check_limits(LIMITS, WITHIN.model_copy(update={"p95_latency_s": None}))
    assert [v.code for v in violations] == [ViolationCode.METRIC_MISSING]
    # an inactive limit does not need its measurement
    assert check_limits(ConstraintLimits(), m()) == ()


# -- hard constraints outrank fitness -----------------------------------------------------------


def _contract(**objective):
    return classification_contract(objective=ObjectiveSpec(**objective), constraints=LIMITS)


def test_infeasible_candidate_never_beats_a_feasible_one_whatever_its_score():
    contract = _contract(**BALANCED)
    feasible = contract.rank(
        WITHIN.model_copy(
            update={"quality": 0.8, "mean_cost_per_example": 0.0099, "mean_latency_s": 2.0}
        )
    )
    # Perfect quality, zero latency, barely over the cost limit: its weighted score is higher.
    over = {"quality": 1.0, "mean_cost_per_example": 0.0101, "mean_latency_s": 0.0}
    assert ObjectiveSpec(**BALANCED).rank_key(WITHIN.model_copy(update=over)) > ObjectiveSpec(
        **BALANCED
    ).rank_key(
        WITHIN.model_copy(
            update={"quality": 0.8, "mean_cost_per_example": 0.0099, "mean_latency_s": 2.0}
        )
    )
    infeasible = contract.rank(WITHIN.model_copy(update=over))
    assert feasible.feasible and not infeasible.feasible
    assert infeasible.objective_key == ()  # objective is not even computed
    assert feasible.sort_key > infeasible.sort_key
    ranked = sorted([infeasible, feasible], key=lambda r: r.sort_key, reverse=True)
    assert ranked[0] is feasible


@pytest.mark.parametrize("mode", list(ObjectiveMode))
def test_feasibility_dominates_for_every_objective_mode(mode):
    spec = BALANCED if mode is ObjectiveMode.BALANCED else {"mode": mode.value}
    contract = _contract(**spec)
    qualities, costs = (0.0, 0.5, 0.79, 0.8, 0.9, 1.0), (0.0, 0.005, 0.01, 0.02, 1.0)
    ranks = []
    for q, c in product(qualities, costs):
        measured = WITHIN.model_copy(
            update={"quality": q, "mean_cost_per_example": c, "mean_latency_s": 1.0}
        )
        ranks.append(contract.rank(measured))
    worst_feasible = min(r.sort_key for r in ranks if r.feasible)
    best_infeasible = max(r.sort_key for r in ranks if not r.feasible)
    assert worst_feasible > best_infeasible


def test_missing_measurement_makes_a_candidate_infeasible_not_free():
    rank = _contract().rank(WITHIN.model_copy(update={"mean_cost_per_example": None}))
    assert not rank.feasible
    assert rank.violations[0].code is ViolationCode.METRIC_MISSING


def test_feasible_candidate_missing_an_objective_metric_raises():
    contract = classification_contract(
        objective=ObjectiveSpec(**BALANCED), constraints=ConstraintLimits()
    )
    with pytest.raises(MissingMetric):
        contract.rank(m(quality=0.9, mean_cost_per_example=0.001))  # no latency
