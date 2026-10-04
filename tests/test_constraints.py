import random

import pytest

from core.constraints import ConstraintChecker, ConstraintConfig
from core.genome import Genome
from core.stages import (
    FailureStrategy,
    GatherMode,
    GatherSource,
    VerifyMethod,
)
from core.violations import ViolationCode as V
from tests.conftest import (
    extract,
    gather,
    make_caps,
    make_runtime_task,
    minimal_genome,
    reason,
    synth,
    verify,
)

C = ConstraintChecker()
JEV = GatherSource.JEV


def codes(genome, task=None, complete=True) -> set[V]:
    return {v.code for v in C.check(genome, task, complete=complete)}


def test_valid_workflow_has_no_violations(task):
    assert C.check(minimal_genome(), task) == ()
    assert C.is_valid(minimal_genome())  # task-free check also works


def test_jev_cannot_use_parallel_4():
    bad = Genome.of(gather(JEV, GatherMode.PARALLEL_4), extract(), synth())
    assert codes(bad) == {V.JEV_PARALLEL_4}
    for mode in (GatherMode.SEQUENTIAL, GatherMode.PARALLEL_2):
        assert codes(Genome.of(gather(JEV, mode), extract(), synth())) == set()
    # parallel-4 is fine for the other sources
    assert (
        codes(Genome.of(gather(GatherSource.FETCH, GatherMode.PARALLEL_4), extract(), synth()))
        == set()
    )


def test_more_than_two_active_verifiers_rejected():
    two = Genome.of(gather(), extract(), verify(), reason(), verify(), synth())
    three = Genome.of(gather(), extract(), verify(), reason(), verify(), synth(), verify())
    assert codes(two) == set()
    assert codes(three) == {V.TOO_MANY_VERIFIERS}


def test_self_consistency_at_most_once():
    sc = verify(VerifyMethod.SELF_CONSISTENCY)
    once = Genome.of(gather(), extract(), sc, synth())
    twice = Genome.of(gather(), extract(), sc, reason(), sc, synth())
    assert codes(once) == set()
    assert codes(twice) == {V.SELF_CONSISTENCY_REPEATED}


def test_jev_cannot_regather():
    regather = verify(on_failure=FailureStrategy.REGATHER)
    assert codes(Genome.of(gather(JEV), extract(), regather, synth())) == {V.JEV_REGATHER}
    assert codes(Genome.of(gather(GatherSource.API), extract(), regather, synth())) == set()
    assert (
        codes(
            Genome.of(gather(JEV), extract(), verify(on_failure=FailureStrategy.RETRY_2), synth())
        )
        == set()
    )


def test_limits_are_configurable_not_hardcoded():
    strict = ConstraintChecker(config=ConstraintConfig(max_active_verifiers=1))
    two = Genome.of(gather(), extract(), verify(), reason(), verify(), synth())
    assert {v.code for v in strict.check(two)} == {V.TOO_MANY_VERIFIERS}


def test_required_stages_typing_and_answer_terminal_reported_by_constraint_layer():
    assert V.MISSING_REQUIRED_STAGE in codes(Genome())
    assert V.INVALID_TRANSITION in codes(Genome.of(gather(), synth()))
    assert V.NO_ANSWER_TERMINAL in codes(Genome.of(gather(), extract()))


def test_prefix_checking_catches_violations_early_and_accepts_good_partials(task):
    assert codes(Genome.of(gather(JEV, GatherMode.PARALLEL_4)), task, complete=False) == {
        V.JEV_PARALLEL_4
    }
    assert codes(Genome.of(gather()), task, complete=False) == set()


def test_task_source_rules():
    api_only = make_runtime_task(allowed_sources=(GatherSource.API,))
    assert codes(minimal_genome(), api_only) == {V.SOURCE_NOT_ALLOWED}  # fetch not allowed
    interactive = make_runtime_task(interaction_required=True)
    assert codes(minimal_genome(), interactive) == {V.INTERACTION_REQUIRES_JEV}
    assert codes(Genome.of(gather(JEV), extract(), synth()), interactive) == set()


def test_budget_infeasibility_is_decided_before_execution():
    # complete workflow, tokens: direct extract (1500) + direct synth (800) = 2300 > 2000
    tight = make_runtime_task(caps=make_caps(tokens=2000))
    assert codes(minimal_genome(), tight) == {V.BUDGET_INFEASIBLE}
    roomy = make_runtime_task(caps=make_caps(tokens=2300))
    assert codes(minimal_genome(), roomy) == set()


@pytest.mark.parametrize(
    "cap",
    [{"tokens": 1000}, {"tool_calls": 1}, {"wall_time_s": 5.0}],
)
def test_partial_workflow_that_provably_cannot_finish_is_rejected(cap):
    task = make_runtime_task(caps=make_caps(**cap))
    # even the cheapest completion of an EMPTY prefix exceeds this cap, so nothing is admissible
    assert V.BUDGET_INFEASIBLE in codes(Genome(), task, complete=False)
    assert C.admissible_successors(Genome(), task) == ()


def test_retries_never_make_a_workflow_provably_infeasible():
    task = make_runtime_task(caps=make_caps(retries=0))
    g = Genome.of(gather(), extract(), verify(on_failure=FailureStrategy.RETRY_2), synth())
    assert codes(g, task) == set()  # retries are only a risk, not a proof


def test_admissible_successors_filter_by_constraints_and_budget(task):
    after_jev = Genome.of(gather(JEV), extract())
    kinds_cfg = {
        (s.kind, getattr(s, "on_failure", None)) for s in C.admissible_successors(after_jev, task)
    }
    assert ("VERIFY", FailureStrategy.REGATHER) not in kinds_cfg
    assert ("VERIFY", FailureStrategy.RETRY_1) in kinds_cfg

    gathers = C.admissible_successors(Genome(), task)
    assert not any(g.source == JEV and g.mode == GatherMode.PARALLEL_4 for g in gathers)
    assert len(gathers) == 8

    # tool_calls cap 2: only the api source (2 calls) is still feasible
    capped = make_runtime_task(caps=make_caps(tool_calls=2))
    assert {g.source for g in C.admissible_successors(Genome(), capped)} == {GatherSource.API}


def test_random_walks_over_admissible_successors_always_yield_valid_genomes(task):
    rng = random.Random(0)
    seen = set()
    for _ in range(150):
        g = Genome()
        while True:
            options = C.admissible_successors(g, task)
            can_stop = C.grammar.can_terminate(g)
            if not options or (can_stop and rng.random() < 0.3):
                break
            g = g.extend(rng.choice(options))
        assert C.check(g, task) == (), g.canonical_json()
        seen.add(g.genome_hash)
    assert len(seen) > 20
