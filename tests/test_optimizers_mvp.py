"""Random search + MMAS ACO against the shared grammar/constraints and a SYNTHETIC objective.

``synthetic_evaluate`` is a made-up deterministic objective (see experiments/synthetic.py);
nothing here is a benchmark result.
"""

import statistics

import pytest

from core.constraints import ConstraintChecker
from core.genome import Genome
from core.stages import GatherMode, GatherSource
from experiments.synthetic import synthetic_evaluate, synthetic_score
from optimizers.aco_mmas import MMASACO, ACOConfig
from optimizers.base import SearchContext, ensure_admissible
from optimizers.construct import END, START, node_key, path_edges
from optimizers.random_search import RandomSearch
from optimizers.scoring import ScoreBoard, lcb
from tests.conftest import extract, gather, make_caps, make_runtime_task, synth

CHECKER = ConstraintChecker()


def ctx(task=None, seed=0, round_=0):
    return SearchContext(task=task or make_runtime_task(), checker=CHECKER, seed=seed, round=round_)


def evaluated(genome, task=None, trial=0, seed=0):
    return synthetic_evaluate(genome, task or make_runtime_task(), trial, seed)


def tight_task():
    """Caps that provably exclude expensive stages: cot extract / self-consistency / jev."""
    return make_runtime_task(
        caps=make_caps(tokens=3000, wall_time_s=12.0, tool_calls=6),
        allowed_sources=(GatherSource.FETCH, GatherSource.API),
    )


# -- validity ---------------------------------------------------------------------------------
@pytest.mark.parametrize("make", [RandomSearch, MMASACO])
@pytest.mark.parametrize("task_factory", [make_runtime_task, tight_task])
def test_proposals_are_complete_and_admissible(make, task_factory):
    task = task_factory()
    context = ctx(task, seed=3)
    proposals = make().propose(30, context)
    assert len(proposals) == 30
    ensure_admissible(proposals, context)
    for g in proposals:
        assert CHECKER.is_valid(g, task, complete=True)


def test_hard_constraints_prune_candidates_in_both_optimizers():
    task = tight_task()
    for opt in (RandomSearch(), MMASACO()):
        for g in opt.propose(40, ctx(task, seed=1)):
            sources = {s.source for s in g.stages if s.kind == "GATHER"}
            assert GatherSource.JEV not in sources  # not allowed for this task
            assert not any(
                s.kind == "EXTRACT" and s.method.value == "cot" for s in g.stages
            )  # provably over the token cap


# -- seeds ------------------------------------------------------------------------------------
@pytest.mark.parametrize("make", [RandomSearch, MMASACO])
def test_same_seed_reproduces_proposals_and_different_seed_differs(make):
    a = [g.genome_hash for g in make().propose(10, ctx(seed=5))]
    b = [g.genome_hash for g in make().propose(10, ctx(seed=5))]
    c = [g.genome_hash for g in make().propose(10, ctx(seed=6))]
    assert a == b
    assert a != c


def test_aco_with_identical_history_and_seed_reproduces_later_proposals():
    def run():
        opt = MMASACO()
        out = []
        for r in range(4):
            gs = opt.propose(3, ctx(seed=2, round_=r))
            opt.observe([evaluated(g, trial=t) for g in gs for t in range(2)])
            out.append([g.genome_hash for g in gs])
        return out, opt.pheromone_snapshot()

    assert run() == run()


# -- pheromone mechanics ------------------------------------------------------------------------
BEST = Genome.of(gather(), extract(), synth())


def one_update(opt, genome=BEST):
    opt._genomes[genome.genome_hash] = genome
    opt.observe([evaluated(genome, trial=t) for t in range(2)])


def test_edge_pheromone_updates_deterministically():
    cfg = ACOConfig(rho=0.2, tau_max=1.0, tau_min=0.05)
    a, b = MMASACO(cfg), MMASACO(cfg)
    one_update(a)
    one_update(b)
    assert a.pheromone_snapshot() == b.pheromone_snapshot()
    edges = path_edges(BEST)
    assert edges[0][0] == START and edges[-1][1] == END
    assert len(edges) == len(BEST.stages) + 1
    # all candidates start at tau_max; the only candidate seen has quality 1, so deposit =
    # rho * tau_max, clamped: reinforced edges stay at tau_max...
    for e in edges:
        assert a.pheromone(e) == pytest.approx(1.0)
    # ...while every untouched edge has evaporated by one step
    other = (START, node_key(gather(GatherSource.JEV, GatherMode.SEQUENTIAL)))
    assert a.pheromone(other) == pytest.approx(0.8)


def test_evaporation_is_lazy_but_exact_and_bounded_below_by_tau_min():
    cfg = ACOConfig(rho=0.5, tau_max=1.0, tau_min=0.1)
    opt = MMASACO(cfg)
    stray = ("x", "y")
    for _ in range(1, 6):
        one_update(opt)
    assert opt.pheromone(stray) == pytest.approx(0.1)  # 1 * 0.5**5 < tau_min -> clamped
    opt2 = MMASACO(cfg)
    one_update(opt2)
    assert opt2.pheromone(stray) == pytest.approx(0.5)  # one step of (1 - rho)


def test_pheromone_stays_within_min_max_bounds():
    cfg = ACOConfig(rho=0.3, tau_max=2.0, tau_min=0.2)
    opt = MMASACO(cfg)
    for r in range(12):
        gs = opt.propose(3, ctx(seed=0, round_=r))
        opt.observe([evaluated(g, trial=t) for g in gs for t in range(2)])
    values = list(opt.pheromone_snapshot().values())
    assert values
    assert all(cfg.tau_min <= v <= cfg.tau_max for v in values)


def test_best_candidate_edges_are_reinforced_over_a_worse_candidates_edges():
    opt = MMASACO(ACOConfig(rho=0.3, global_best_period=100))
    good = Genome.of(gather(GatherSource.API, GatherMode.PARALLEL_2), extract(), synth())
    bad = Genome.of(gather(GatherSource.JEV, GatherMode.SEQUENTIAL), extract(), synth())
    assert synthetic_score(good) > synthetic_score(bad)
    for g in (good, bad):
        opt._genomes[g.genome_hash] = g
    for _ in range(4):
        opt.observe([evaluated(g, trial=t) for g in (good, bad) for t in range(2)])
    good_first = opt.pheromone(path_edges(good)[0])
    bad_first = opt.pheromone(path_edges(bad)[0])
    assert good_first > bad_first


def test_aco_ignores_results_for_genomes_it_never_proposed():
    opt = MMASACO()
    opt.observe([evaluated(BEST)])
    assert opt.epoch == 0 and opt.pheromone_snapshot() == {}


def test_aco_config_validation():
    with pytest.raises(ValueError):
        ACOConfig(tau_min=2.0, tau_max=1.0)


# -- noise handling ----------------------------------------------------------------------------
def test_lcb_is_below_mean_and_degrades_gracefully_for_one_sample():
    assert lcb([1.0]) == 1.0
    vals = [0.5, 0.7, 0.9, 0.7]
    assert lcb(vals) < statistics.fmean(vals)
    assert lcb(vals, z=0.0) == pytest.approx(statistics.fmean(vals))


def test_scoreboard_picks_highest_lcb_with_deterministic_ties():
    board = ScoreBoard()
    good = Genome.of(gather(GatherSource.API), extract(), synth())
    bad = Genome.of(gather(GatherSource.JEV), extract(), synth())
    board.add([evaluated(g, trial=t) for g in (bad, good) for t in range(2)])
    assert board.best().genome_hash == good.genome_hash
    assert board.best(among=[bad.genome_hash]).genome_hash == bad.genome_hash


# -- learning (SYNTHETIC objective) -------------------------------------------------------------
def _best_true_score(opt, n_rounds, seed):
    best = -1e9
    for r in range(n_rounds):
        gs = opt.propose(2, ctx(seed=seed, round_=r))
        opt.observe([evaluated(g, trial=t) for g in gs for t in range(2)])
        best = max(best, *(synthetic_score(g) for g in gs))
    return best


def test_aco_beats_random_on_the_synthetic_objective_at_equal_proposal_budget():
    seeds = range(6)
    aco = statistics.fmean(_best_true_score(MMASACO(), 60, s) for s in seeds)
    rnd = statistics.fmean(_best_true_score(RandomSearch(), 60, s) for s in seeds)
    assert aco > rnd
