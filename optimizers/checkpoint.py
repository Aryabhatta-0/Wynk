"""Optimizer checkpoints: an optimizer's complete state as plain JSON, and back.

``checkpoint_state(optimizer)`` captures everything an optimizer needs so that
``restore(state)`` proposes and updates exactly like the original would from that point on. It
lives here, outside the optimizers, so the search implementations stay byte for byte the ones
the frozen protocols ran (``tests/test_benchmark_adapter.py`` pins their sources). It only reads
and sets their state; it never changes how they search.

Per optimizer (``state["optimizer"]`` / ``state["version"]`` name the exact implementation):

    aco_mmas            config; ``epoch`` (evaporation steps - it also fixes the iteration-best /
                        global-best phase); the lazily evaporated edges ``(value, epoch set at)``;
                        every proposed genome; the score board (every observation, hence every
                        LCB and the normalisation range); the propose-call index that seeds the
                        next ant. Recorded for inspection and re-checked on restore: current
                        ``pheromones``, ``global_best`` and the ``lcb_range`` a deposit is
                        normalised against.
    random_search[_distinct]
                        the propose-call index (each proposal's rng is ``derive_rng(name, seed,
                        round, calls)``: the complete rng-equivalent state); the distinct variant
                        adds the proposed set and whether the space is exhausted. Proposal ORDER is
                        the experiment's candidate log.
    fixed baselines     whether the one workflow was proposed (it is re-derived from the contract).

Only these exact classes are checkpointed; any other optimizer has no checkpoint (``None``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from core.genome import Genome
from optimizers.aco_mmas import MMASACO, ACOConfig
from optimizers.base import Optimizer
from optimizers.fixed_baseline import FIXED_RULES, FixedBaseline
from optimizers.random_search import DistinctRandomSearch, RandomSearch
from optimizers.scoring import ScoreBoard

STATE_VERSION = "optimizer-checkpoint/1"
_RANDOM = (RandomSearch, DistinctRandomSearch)
_KNOWN: dict[str, type[Optimizer]] = {
    MMASACO.name: MMASACO,
    RandomSearch.name: RandomSearch,
    DistinctRandomSearch.name: DistinctRandomSearch,
    **{cls.name: cls for cls in FIXED_RULES.values()},
}


class OptimizerStateError(ValueError):
    """A checkpoint does not belong to this optimizer, or does not reproduce its own state."""


def _version(optimizer: Optimizer | type[Optimizer]) -> str:
    return str(getattr(optimizer, "version", optimizer.name))


# -- score board --------------------------------------------------------------------------------
def board_state(board: ScoreBoard) -> dict[str, Any]:
    """Every observation per genome, in observation order."""
    return {
        "z": board.z,
        "fitness": {h: list(v) for h, v in sorted(board._fitness.items())},
        "passes": {h: list(v) for h, v in sorted(board._passes.items())},
    }


def restore_board(state: Mapping[str, Any]) -> ScoreBoard:
    board = ScoreBoard(float(state["z"]))
    if set(state["fitness"]) != set(state["passes"]):
        raise OptimizerStateError("score board: fitness and passes cover different genomes")
    for h in sorted(state["fitness"]):
        fitness, passes = state["fitness"][h], state["passes"][h]
        if not fitness or len(fitness) != len(passes):
            raise OptimizerStateError(f"score board: inconsistent observations for {h}")
        board._fitness[h] = [float(f) for f in fitness]
        board._passes[h] = [bool(p) for p in passes]
    return board


# -- optimizers ---------------------------------------------------------------------------------
def checkpoint_state(optimizer: Optimizer) -> dict[str, Any] | None:
    """The optimizer's complete state (plain JSON), or ``None`` for an unknown optimizer."""
    cls = type(optimizer)
    if _KNOWN.get(optimizer.name) is not cls:
        return None
    head = {"state_version": STATE_VERSION, "optimizer": cls.name, "version": _version(cls)}
    if isinstance(optimizer, MMASACO):
        board = optimizer._board
        best = board.best()
        lcbs = [board.score(h).lcb for h in board.hashes()]
        return head | {
            "config": asdict(optimizer.config),
            "epoch": optimizer.epoch,
            "calls": optimizer._calls,
            "edges": [[a, b, v, at] for (a, b), (v, at) in sorted(optimizer._edges.items())],
            "genomes": {h: g.canonical() for h, g in sorted(optimizer._genomes.items())},
            "board": board_state(board),
            "pheromones": [[a, b, t] for (a, b), t in optimizer.pheromone_snapshot().items()],
            "global_best": best.genome_hash if best is not None else None,
            "lcb_range": [min(lcbs), max(lcbs)] if lcbs else None,
        }
    if isinstance(optimizer, _RANDOM):
        out = head | {"calls": optimizer._calls, "n_observed": optimizer.n_observed}
        if isinstance(optimizer, DistinctRandomSearch):
            out |= {"seen": sorted(optimizer._seen), "exhausted": optimizer.exhausted}
        return out
    assert isinstance(optimizer, FixedBaseline)
    return head | {"proposed": optimizer._proposed, "n_observed": optimizer.n_observed}


def restore(state: Mapping[str, Any]) -> Optimizer:
    """The optimizer ``state`` was taken from, in exactly that state."""
    cls = _KNOWN.get(str(state.get("optimizer")))
    if (
        cls is None
        or state.get("version") != _version(cls)
        or state.get("state_version") != STATE_VERSION
    ):
        raise OptimizerStateError(
            f"no checkpointed optimizer {state.get('optimizer')!r} {state.get('version')!r}"
        )
    if cls is MMASACO:
        opt: Optimizer = MMASACO(ACOConfig(**state["config"]))
        assert isinstance(opt, MMASACO)
        opt.epoch = int(state["epoch"])
        opt._calls = int(state["calls"])
        opt._edges = {(a, b): (float(v), int(at)) for a, b, v, at in state["edges"]}
        opt._genomes = {h: Genome.from_canonical(g) for h, g in state["genomes"].items()}
        if any(g.genome_hash != h for h, g in opt._genomes.items()):
            raise OptimizerStateError("ACO checkpoint: a genome does not match its hash")
        opt._board = restore_board(state["board"])
    elif issubclass(cls, RandomSearch):
        opt = cls()
        assert isinstance(opt, RandomSearch)
        opt._calls = int(state["calls"])
        opt.n_observed = int(state["n_observed"])
        if isinstance(opt, DistinctRandomSearch):
            opt._seen = set(state["seen"])
            opt.exhausted = bool(state["exhausted"])
    else:
        opt = cls()
        assert isinstance(opt, FixedBaseline)
        opt._proposed = bool(state["proposed"])
        opt.n_observed = int(state["n_observed"])
    if checkpoint_state(opt) != dict(state):
        raise OptimizerStateError(f"{cls.name} checkpoint does not reproduce its own state")
    return opt
