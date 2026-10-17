"""Per-task champions (#10): the validated genome each validation task's contract ranks best.

An additional SELECTION layer over a finished search, next to - never instead of - the suite
champion (one genome for every row) and the optimizer (which never sees any of this):

    input    the FINAL validation runs of every evaluated genome (MODEL_ERROR retries are
             superseded), as the search measured them. Optimization rows, optimizer scores and
             test rows are never read: a run on any other split is refused (``SplitAccessError``
             semantics, fail closed).
    grouping by validation ``task_id`` (one dataset row), then by genome.
    ranking  the suite's ranking authority (``ContractSuite.rank`` = ``rank_measured`` under the
             shared search policy: hard limits first, objective second) over that genome's runs
             on that ONE task. Every task of a suite shares that policy, so this is also the
             task's own ``TaskContract`` ranking.
    order    (1) feasible over infeasible, (2) the contract rank key, (3) mean evaluator fitness,
             (4) the genome hash (lexicographically greatest) - a total order on distinct
             genomes, so neither execution order nor concurrency can decide a winner.

A task whose best candidate is infeasible still gets a record (``feasible: false``): diagnostic
evidence only, never a usable elite (``usable``). JEV is an evidence source like fetch/api: a JEV
genome's runs carry the pinned ``evidence_source`` the runtime reported (#52) and the record keeps
it (``evidence_sources``); nothing here asks JEV, or anything else, which genome is best.

``select`` is pure: the same stored runs always give the same mapping, so ``rederive`` can check
a stored mapping against the artifact's own workflow-run entries.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.canonical import canonical_json
from core.dataset import SplitRole
from core.results import EvaluatedRun, Verdict
from core.run_contract import RunMeasure, rank_measured
from core.task_contract import CandidateRank, TaskContract

SCHEMA = "wynk-task-champions/1"


def run_score(run: EvaluatedRun) -> float:
    """The evaluator's measured quality (INFEASIBLE: 0); fitness when there is no record."""
    record = run.evaluation.evaluator
    if record is None:
        return run.evaluation.fitness
    return record.quality if record.quality is not None else 0.0


@dataclass(frozen=True)
class TaskRun:
    """One final validation run of one genome on one task, as selection reads it."""

    genome_hash: str
    task_id: str
    trial: int
    measure: RunMeasure
    fitness: float
    score: float
    run_id: str | None = None  # RunKey.run_id (known when built from the run itself)
    contract_hash: str | None = None  # RunKey.contract_hash (known when built from the run)
    path: str | None = None  # trace path inside the strategy run (``candidate.<n>.runs.<j>``)
    run_versions: str | None = None  # canonical JSON of the RunVersions the run reported
    evidence_source: str | None = None  # RunVersions.evidence_source (a JEV run's provider)

    @classmethod
    def of(cls, run: EvaluatedRun, *, path: str | None = None) -> TaskRun:
        key = run.execution.key
        return cls(
            genome_hash=key.genome_hash,
            task_id=key.task_id,
            trial=key.trial,
            measure=RunMeasure.of(run),
            fitness=run.evaluation.fitness,
            score=run_score(run),
            run_id=key.run_id,
            contract_hash=key.contract_hash,
            path=path,
            run_versions=canonical_json(key.versions.model_dump(mode="json")),
            evidence_source=key.versions.evidence_source,
        )

    @classmethod
    def from_entry(
        cls,
        genome_hash: str,
        entry: Mapping[str, Any],
        *,
        path: str | None = None,
        run_versions: Mapping[str, Any] | None = None,
    ) -> TaskRun:
        """A stored workflow-run entry of an experiment artifact (``candidates[].runs[]``)."""
        if entry["split"] != SplitRole.VALIDATION.value:
            raise ValueError(f"row {entry['row_id']} is not a validation run")
        if entry["verdict"] is None:
            raise ValueError(f"row {entry['row_id']} has no verdict (a superseded attempt)")
        return cls(
            genome_hash=genome_hash,
            task_id=entry["row_id"],
            trial=entry["trial"],
            measure=RunMeasure(
                verdict=Verdict(entry["verdict"]),
                tokens=entry["tokens"],
                model_calls=entry["model_calls"],
                tool_calls=entry["tool_calls"],
                retries=entry["retries"],
                wall_time_s=entry["wall_time_s"],
            ),
            fitness=entry["fitness"],
            score=entry["score"],
            path=path,
            run_versions=canonical_json(run_versions) if run_versions is not None else None,
            evidence_source=(run_versions or {}).get("evidence_source"),
        )


def select(
    contract: TaskContract,
    genomes: Mapping[str, Mapping[str, Any]],
    runs: Iterable[TaskRun],
    *,
    validation_tasks: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """One record per validation task (sorted by task id): its best candidate under
    ``contract``'s ranking. ``genomes`` maps every genome hash to its ``Genome.canonical()``;
    ``validation_tasks`` (when given) are the only task ids a run may be on."""
    allowed = set(validation_tasks) if validation_tasks is not None else None
    grouped: dict[str, dict[str, list[TaskRun]]] = {}
    for r in runs:
        if allowed is not None and r.task_id not in allowed:
            raise ValueError(f"task {r.task_id} is not a validation task: not used for selection")
        grouped.setdefault(r.task_id, {}).setdefault(r.genome_hash, []).append(r)
    out = []
    for task_id in sorted(grouped):
        options = []
        for h, rs in grouped[task_id].items():
            rs = sorted(rs, key=lambda r: r.trial)
            if len({r.trial for r in rs}) != len(rs):
                raise ValueError(f"genome {h} has two final runs of one trial on task {task_id}")
            steps = len(genomes[h]["stages"])
            rank = rank_measured(contract, steps, [r.measure for r in rs])
            options.append((rank, statistics.fmean(r.fitness for r in rs), h, rs))
        rank, _, h, rs = max(options, key=lambda o: (o[0].sort_key, o[1], o[2]))
        out.append(_record(task_id, h, genomes, rs, rank=rank, compared=len(options)))
    return out


def _record(
    task_id: str,
    h: str,
    genomes: Mapping[str, Mapping[str, Any]],
    rs: Sequence[TaskRun],
    *,
    rank: CandidateRank | None = None,
    compared: int | None = None,
) -> dict[str, Any]:
    rs = sorted(rs, key=lambda r: r.trial)
    return {
        "task_id": task_id,
        "genome_hash": h,
        "genome": dict(genomes[h]),
        "feasible": rank.feasible if rank else None,
        "rank_key": list(rank.sort_key) if rank else None,
        "violations": [v.message for v in rank.violations] if rank else None,
        "validation": {
            "runs": len(rs),
            "pass_rate": sum(r.measure.verdict is Verdict.PASS for r in rs) / len(rs),
            "fitness_mean": statistics.fmean(r.fitness for r in rs),
            "score_mean": statistics.fmean(r.score for r in rs),
        },
        "compared_genomes": compared,
        "evidence_sources": sorted({r.evidence_source for r in rs if r.evidence_source}),
        "run_versions": [
            json.loads(v) for v in sorted({r.run_versions for r in rs if r.run_versions})
        ],
        "contract_hashes": sorted({r.contract_hash for r in rs if r.contract_hash}),
        "run_ids": [r.run_id for r in rs if r.run_id is not None],
        "run_paths": [r.path for r in rs if r.path is not None],
    }


def usable(record: Mapping[str, Any]) -> bool:
    """May product routing use this task champion as an elite? Only a feasible one."""
    return record.get("feasible") is True


def artifact_task_runs(run: Mapping[str, Any]) -> list[TaskRun]:
    """The final validation runs of one strategy-run record of an experiment artifact, with
    their trace paths. A strategy run reports one runtime identity (``run_versions``), so every
    run's RunVersions - and evidence source - is that identity."""
    versions = run["run_versions"]
    if run["candidates"] and len(versions) != 1:
        raise ValueError("a strategy run with validation runs reports exactly one RunVersions")
    out = []
    for cand in run["candidates"]:
        last: dict[tuple[str, int], int] = {}  # (row, trial) -> its final attempt's entry index
        for j, e in enumerate(cand["runs"]):
            if e["split"] == SplitRole.VALIDATION.value:
                last[(e["row_id"], e["trial"])] = j
        for j in sorted(last.values()):
            out.append(
                TaskRun.from_entry(
                    cand["genome_hash"],
                    cand["runs"][j],
                    path=f"candidate.{cand['evaluation']}.runs.{j}",
                    run_versions=versions[0],
                )
            )
    return out


_REDERIVED = (
    "task_id",
    "genome_hash",
    "genome",
    "feasible",
    "rank_key",
    "violations",
    "validation",
    "compared_genomes",
    "evidence_sources",
    "run_versions",
    "run_paths",
)
_EVIDENCE = ("genome", "validation", "evidence_sources", "run_versions", "run_paths")


def rederive(run: Mapping[str, Any], contract: TaskContract | None) -> list[str]:
    """Where a strategy run's stored ``task_champions`` disagree with the mapping its own
    validation-run entries give (``[]``: they agree). Ranking needs the ``contract``; without
    it (an artifact whose provenance does not carry one) only the rank-free parts are checked:
    the tasks covered, and each winner's genome, statistics and run references."""
    stored = run.get("task_champions")
    if stored is None:
        return []  # a strategy run recorded before per-task selection existed
    name = f"{run['strategy']}/{run['seed']}"
    genomes = {c["genome_hash"]: c["genome"] for c in run["candidates"]}
    runs = artifact_task_runs(run)
    problems = []
    if contract is not None:
        derived = {t["task_id"]: t for t in select(contract, genomes, runs)}
        keys = _REDERIVED
    else:
        derived = {}
        for t in stored:
            mine = [
                r for r in runs if (r.task_id, r.genome_hash) == (t["task_id"], t["genome_hash"])
            ]
            if mine and t["genome_hash"] in genomes:
                derived[t["task_id"]] = _record(t["task_id"], t["genome_hash"], genomes, mine)
        keys = _EVIDENCE
    if [t["task_id"] for t in stored] != sorted({r.task_id for r in runs}):
        problems.append(f"{name} task_champions cover other tasks than its validation runs")
    for got in stored:
        want = derived.get(got["task_id"])
        bad = [k for k in keys if want is None or got.get(k) != want.get(k)]
        if bad:
            problems.append(f"{name} task {got['task_id']}: {', '.join(bad)} do not re-derive")
    return problems


__all__ = [
    "SCHEMA",
    "TaskRun",
    "artifact_task_runs",
    "rederive",
    "run_score",
    "select",
    "usable",
]
