"""MAX-MIN Ant System over the typed workflow-construction graph (MVP).

* An ant builds a genome stage by stage; the legal next stages come from the shared checker
  (grammar successors + hard-constraint pruning), never from this module.
* Pheromone lives on EDGES (transitions): ``(previous node, next node)`` where a node is a
  configured stage, plus the pseudo-nodes ``START`` and ``END``.
* Selection probability of a legal successor is proportional to ``tau(edge) ** alpha``.
* MMAS update per ``observe`` call: evaporate every edge (``tau <- max(tau_min, (1-rho) * tau)``),
  then deposit on the edges of ONE candidate - the iteration-best, or the best-so-far every
  ``global_best_period`` updates - and clamp to ``[tau_min, tau_max]``.
* Candidate quality is the lower-confidence-bound score from ``optimizers.scoring`` (noise-aware),
  normalised against the best/worst LCB over ALL genomes seen so far - not just this batch.
  Per-batch normalisation is degenerate here: one ant is reinforced per epoch, so the batch min
  and max both come from that same ant and quality would collapse to 1.0 every time. Measured
  over 24 seeds x 60 rounds, per-batch scoring scored *worse* (0.9375 vs 0.9950 mean best-so-far).
* ``rho = 0.30``. One deposit per epoch against whole-field evaporation is thin, and 0.15 was
  measured to be well off the optimum: sweep over {0.15, 0.20, ..., 0.45} gave 0.9838 / 0.9858 /
  0.9846 / 1.0000 / 0.9738 / 0.9758 / 0.9529 mean best-so-far, with 0.30 also the lowest spread
  and the fastest to first PASS. Raise it past ~0.35 and the colony over-exploits and loses.
* A batch is deduplicated by genome hash: a repeat would burn a real-LLM evaluation and add a
  *non-independent* sample to that genome's score, shrinking sd/sqrt(n) spuriously.
* Evaporation is applied lazily (closed form per edge from its last-touched epoch), which is
  exactly equivalent to evaporating every edge each epoch, so edges never seen are handled too.
* All randomness derives from ``context.seed`` (+ round + call index): same seed, same proposals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from core.genome import Genome
from core.results import EvaluatedRun
from optimizers.base import Optimizer, SearchContext, ensure_admissible
from optimizers.construct import construct_genome, derive_rng, path_edges
from optimizers.scoring import DEFAULT_Z, ScoreBoard

Edge = tuple[str, str]
ATTEMPTS_PER_ANT = 20
QUALITY_FLOOR = 0.1  # even the worst reinforced candidate deposits a little


@dataclass(frozen=True)
class ACOConfig:
    alpha: float = 1.0
    rho: float = 0.30  # evaporation rate; measured optimum over {0.15..0.45}, see class docstring
    tau_max: float = 1.0
    tau_min: float = 0.05
    global_best_period: int = 5  # every n-th update reinforces best-so-far instead of iter-best
    lcb_z: float = DEFAULT_Z

    def __post_init__(self) -> None:
        if not (0.0 < self.rho <= 1.0 and 0.0 < self.tau_min < self.tau_max and self.alpha >= 0):
            raise ValueError("require 0 < rho <= 1, 0 < tau_min < tau_max, alpha >= 0")


class MMASACO(Optimizer):
    name = "aco_mmas"

    def __init__(self, config: ACOConfig | None = None) -> None:
        self.config = config or ACOConfig()
        self.epoch = 0  # number of pheromone updates (evaporation steps) so far
        self._edges: dict[Edge, tuple[float, int]] = {}  # edge -> (value, epoch it was set at)
        self._genomes: dict[str, Genome] = {}  # genomes this optimizer proposed, by hash
        self._board = ScoreBoard(self.config.lcb_z)
        self._calls = 0

    # -- pheromone ------------------------------------------------------------
    def pheromone(self, edge: Edge) -> float:
        """Current (lazily evaporated) pheromone of an edge; unseen edges start at tau_max."""
        cfg = self.config
        value, at = self._edges.get(edge, (cfg.tau_max, 0))
        return max(cfg.tau_min, value * (1.0 - cfg.rho) ** (self.epoch - at))

    def pheromone_snapshot(self) -> dict[Edge, float]:
        return {e: self.pheromone(e) for e in sorted(self._edges)}

    # -- Optimizer ------------------------------------------------------------
    def propose(self, k: int, context: SearchContext) -> list[Genome]:
        rng = derive_rng(self.name, context.seed, context.round, self._calls)
        self._calls += 1
        alpha = self.config.alpha

        def weights(prev: str, options: Sequence[str]) -> list[float]:
            return [self.pheromone((prev, o)) ** alpha for o in options]

        out: list[Genome] = []
        seen: set[str] = set()
        for _ in range(k * ATTEMPTS_PER_ANT):
            if len(out) == k:
                break
            g = construct_genome(context, rng, weights)
            # Dedupe within the batch: a repeat would waste a real-LLM evaluation and would add a
            # *non-independent* second sample to that genome's ScoreBoard, shrinking sd/sqrt(n)
            # and making it look better-measured than two independent runs would.
            if g is not None and g.genome_hash not in seen:
                seen.add(g.genome_hash)
                out.append(g)
        ensure_admissible(out, context)
        for g in out:
            self._genomes[g.genome_hash] = g
        return out

    def observe(self, results: Sequence[EvaluatedRun]) -> None:
        known = [r for r in results if r.execution.genome_hash in self._genomes]
        if not known:
            return
        self._board.add(known)
        self.epoch += 1  # evaporation of every edge happens implicitly via the epoch gap

        in_batch = sorted({r.execution.genome_hash for r in known})
        use_global = self.epoch % self.config.global_best_period == 0
        chosen = self._board.best(None if use_global else in_batch)
        if chosen is None:
            return
        lcbs = [self._board.score(h).lcb for h in self._board.hashes()]
        lo, hi = min(lcbs), max(lcbs)
        quality = 1.0 if hi == lo else max(QUALITY_FLOOR, (chosen.lcb - lo) / (hi - lo))
        # Fixed point on a repeatedly reinforced edge is quality * tau_max (<= tau_max).
        delta = quality * self.config.rho * self.config.tau_max
        for edge in path_edges(self._genomes[chosen.genome_hash]):
            new = min(self.config.tau_max, self.pheromone(edge) + delta)
            self._edges[edge] = (max(self.config.tau_min, new), self.epoch)
