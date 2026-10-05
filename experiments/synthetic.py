"""SYNTHETIC deterministic fake objective - for tests and harness demos ONLY.

This is NOT a benchmark result. It mimics "an evaluator behind a runtime" with a made-up
additive-plus-edge-interaction score so optimizers can be verified without Track A. Every
artifact built from it is labelled ``synthetic`` (``evaluator_version`` starts with
``synthetic-fake``), and the experiment writer refuses to emit it without that label.
"""

from __future__ import annotations

import math

from core.canonical import canonical_hash
from core.genome import Genome
from core.results import (
    BudgetUsage,
    EvaluatedRun,
    Evaluation,
    ExecutionResult,
    RunKey,
    RunVersions,
    Verdict,
)
from core.run_contract import ExecutionTask

SYNTHETIC_VERSION = "synthetic-fake/1"
PASS_THRESHOLD = 0.8

_NODE = {
    "GATHER:fetch": 0.10,
    "GATHER:api": 0.20,
    "GATHER:jev": 0.0,
    "FILTER:keyword_chunk": 0.04,
    "FILTER:section_select": 0.06,
    "EXTRACT:direct": 0.05,
    "EXTRACT:schema_guided": 0.20,
    "EXTRACT:cot": 0.12,
    "REASON:single": -0.04,
    "REASON:decompose": -0.02,
    "VERIFY:schema_check": 0.03,
    "VERIFY:evidence_span": 0.05,
    "VERIFY:self_consistency": 0.08,
    "SYNTHESIZE:direct": 0.05,
    "SYNTHESIZE:cite_evidence": 0.15,
}
_MODE = {"sequential": 0.0, "parallel-2": 0.05, "parallel-4": 0.03}
_EDGE = {  # bonuses for consecutive stages: the structure that ACO's edge pheromone can learn
    ("EXTRACT:schema_guided", "SYNTHESIZE:cite_evidence"): 0.15,
    ("SYNTHESIZE:cite_evidence", "VERIFY:evidence_span"): 0.15,
    ("GATHER:api", "EXTRACT:schema_guided"): 0.10,
}
_REGATHER = -0.01


def _key(stage) -> str:
    option = stage.source if stage.kind == "GATHER" else stage.method
    return f"{stage.kind}:{option.value}"


def synthetic_score(genome: Genome) -> float:
    """Noise-free quality of a genome under the fake objective."""
    keys = [_key(s) for s in genome.stages]
    # fsum, not sum: float sum() rounding changed in 3.12, which would break pinned hashes on 3.11
    score = math.fsum(_NODE[k] for k in keys)
    score += math.fsum(_MODE[s.mode.value] for s in genome.stages if s.kind == "GATHER")
    score += math.fsum(
        _REGATHER for s in genome.stages if s.kind == "VERIFY" and s.on_failure.value == "regather"
    )
    score += math.fsum(_EDGE.get(pair, 0.0) for pair in zip(keys, keys[1:], strict=False))
    return score - 0.02 * max(0, len(keys) - 3)


def _unit(*parts) -> float:
    """Deterministic value in [-1, 1) from the hash of ``parts``."""
    return int(canonical_hash(list(parts))[:8], 16) / 2**31 - 1.0


def synthetic_evaluate(genome: Genome, task: ExecutionTask, trial: int, seed: int) -> EvaluatedRun:
    """``EvaluateFn``-shaped fake: same (genome, task, trial, seed) -> same result."""
    key = RunKey(
        genome_hash=genome.genome_hash,
        task_id=task.id,
        contract_hash=task.contract_hash,
        trial=trial,
        seed=seed,
        versions=RunVersions(
            model_hash="synthetic",
            prompt_template_version="synthetic",
            benchmark_hash="synthetic",
            compiler_version="synthetic",
            grammar_version="synthetic",
        ),
    )
    task_offset = 0.03 * _unit("task", task.id)
    noise = 0.05 * _unit("noise", key.run_id)
    fitness = synthetic_score(genome) + task_offset + noise
    verdict = Verdict.PASS if fitness >= PASS_THRESHOLD else Verdict.FAIL
    execution = ExecutionResult(key=key, budget_usage=BudgetUsage(tokens=1000 * len(genome)))
    return EvaluatedRun(
        execution=execution,
        evaluation=Evaluation(
            verdict=verdict, fitness=fitness, evaluator_version=SYNTHETIC_VERSION
        ),
    )
