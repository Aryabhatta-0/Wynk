"""Noisy-objective scoring: mean and a simple lower confidence bound per genome.

MVP noise handling (no racing): every genome is run on several tasks x trials; its score is the
mean fitness and ``lcb = mean - z * sd / sqrt(n)`` (n = number of runs; n < 2 -> lcb = mean).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from core.results import EvaluatedRun, FailureKind, Verdict

DEFAULT_Z = 1.0


@dataclass(frozen=True)
class CandidateScore:
    genome_hash: str
    n: int
    mean: float
    lcb: float
    pass_rate: float


def lcb(values: Sequence[float], z: float = DEFAULT_Z) -> float:
    if len(values) < 2:
        return float(values[0]) if values else -math.inf
    return statistics.fmean(values) - z * statistics.stdev(values) / math.sqrt(len(values))


class ScoreBoard:
    """Accumulates observations per genome hash (deterministic; insertion-order independent)."""

    def __init__(self, z: float = DEFAULT_Z) -> None:
        self.z = z
        self._fitness: dict[str, list[float]] = {}
        self._passes: dict[str, list[bool]] = {}

    def add(self, results: Sequence[EvaluatedRun]) -> None:
        for r in results:
            if r.execution.failure and r.execution.failure.kind is FailureKind.MODEL_ERROR:
                continue
            h = r.execution.genome_hash
            self._fitness.setdefault(h, []).append(r.evaluation.fitness)
            self._passes.setdefault(h, []).append(r.evaluation.verdict is Verdict.PASS)

    def score(self, genome_hash: str) -> CandidateScore:
        f, p = self._fitness[genome_hash], self._passes[genome_hash]
        return CandidateScore(
            genome_hash=genome_hash,
            n=len(f),
            mean=statistics.fmean(f),
            lcb=lcb(f, self.z),
            pass_rate=sum(p) / len(p),
        )

    def hashes(self) -> list[str]:
        return sorted(self._fitness)

    def best(self, among: Sequence[str] | None = None) -> CandidateScore | None:
        """Highest lcb; ties broken by mean, then by hash (fully deterministic)."""
        pool = [h for h in (among if among is not None else self._fitness) if h in self._fitness]
        if not pool:
            return None
        scores = [self.score(h) for h in sorted(set(pool))]
        return max(scores, key=lambda s: (s.lcb, s.mean, s.genome_hash))
