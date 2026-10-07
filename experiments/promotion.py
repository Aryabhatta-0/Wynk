"""Champion promotion: a COMPLETED experiment -> ONE validation-selected challenger -> held-out
gate -> promote or reject.

    ChampionPromotions.promote(job_id)
      1. eligibility  only a COMPLETED durable experiment job (#24) with its assembled #23
                      artifact; cancelled / failed / interrupted / running jobs promote nothing
      2. challenger   ``select_challenger``: VALIDATION evidence only. Each (strategy, seed) run's
                      validation-selected champion is a candidate; its rank is recomputed from
                      the job's stored validation runs with ``TaskContract.rank`` (hard limits
                      first - infeasible candidates are REJECTED - objective second) and must
                      equal the artifact's. Exactly one feasible candidate is VALIDATED; it may
                      come from fixed, random or ACO.
      3. incumbent    the lineage's current champion, if any. Its pinned identity must equal the
                      challenger's (``compatibility_identity``) or promotion fails closed.
      4. open test    the experiment's test split is opened DURABLY (``SQLiteChampionStore.open``)
                      with the selection and the incumbent pinned, BEFORE any test target is
                      bound. One look per experiment: a second promote of the same job resumes
                      or returns this promotion; it never picks another candidate.
      5. held-out     the challenger - and the incumbent, on the exact same rows, trials and run
                      seeds - through the write-ahead attempt log (#24's no-double-spend rule)
      6. gate         ``gate``: the challenger must satisfy every hard constraint on held-out,
                      and must not rank below the incumbent under ``CandidateRank.sort_key``
                      (``TaskContract.rank``: maximize -> ``>=``, minimize -> ``<=``, ties as the
                      contract ranks them). No incumbent: the constraints alone decide.
      7. decide       an immutable ``PromotionDecision``; PROMOTED is compare-and-promote against
                      the incumbent version pinned in step 4 (a newer champion is never
                      overwritten: the decision becomes REJECTED / ``incumbent_superseded``).

Candidate lifecycle: ``CANDIDATE -> VALIDATED -> CHAMPION``, or ``REJECTED`` (infeasible on
validation, or rejected by the held-out gate). Test data never reaches an optimizer, candidate
generation, validation selection or tie-breaking: those all finished inside the experiment, or
in step 2, before step 4 recorded that the test split was opened. ``verify`` re-derives every
decision from stored evidence (validation runs in the job store, held-out runs here).

Promotion never runs a search and never changes an experiment: it is a separate, durable step
after one (no second experiment runner, no change to optimizers or evaluators).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from core.canonical import canonical_hash
from core.dataset import SplitRole, SplitUse, require_use
from core.genome import Genome
from core.results import EvaluatedRun, FailureKind, Verdict
from core.run_contract import ExecutionTask, candidate_measurements, rank_candidate
from core.task_contract import TaskContract
from experiments.budget_ledger import MeasuredUsage
from experiments.jobs import (
    AmbiguousAttempt,
    ExperimentJobDefinition,
    HeldoutBinding,
    HeldoutRuntime,
    JobNotFound,
    Lease,
    RuntimeMismatch,
    _Heartbeat,
    job_identity,
)
from experiments.optimization_experiment import (
    MODEL_ATTEMPTS,
    ExperimentError,
    ModelUnavailable,
    run_score,
)
from store.champions import (
    TERMINAL,
    ChampionRow,
    NewChampion,
    PromotionClaim,
    PromotionReason,
    PromotionRow,
    PromotionState,
    SQLiteChampionStore,
    StaleIncumbent,
)
from store.datasets import Conflict
from store.jobs import AttemptRow, AttemptState, JobState, LeaseLost, NewAttempt, SQLiteJobStore

DECISION_SCHEMA = "wynk-promotion-decision/1"
CHAMPION_SCHEMA = "wynk-champion/1"
SELECTION_RULE = "validation-rank/1"
HELDOUT_PROTOCOL = "heldout-gate/1"
HELDOUT_TRIAL_OFFSET = 20_000  # held-out runs never share a trial with feedback or validation
DEFAULT_LEASE_S = 60.0
LINEAGE = r"[a-z0-9][a-z0-9_.-]{0,255}"  # like a dataset id (never a path), but longer

COMPARATOR = (
    "TaskContract.rank -> CandidateRank.sort_key, higher is better: every feasible key outranks "
    "every infeasible one; maximize objectives compare (quality,), minimize objectives "
    "(-metric, quality), balanced (utility,)"
)
SELECTION_DESCRIPTION = (
    "validation runs only: candidates that violate a hard constraint are rejected; feasible "
    "candidates are ranked by TaskContract.rank (CandidateRank.sort_key); ties: higher mean "
    "validation score, then earlier first evaluation, then plan strategy order, then plan seed "
    "order"
)


# -- vocabulary ---------------------------------------------------------------------------------
class CandidateStatus(StrEnum):
    CANDIDATE = "CANDIDATE"  # feasible on validation, outranked by the challenger
    VALIDATED = "VALIDATED"  # THE challenger: the one candidate allowed to reach the test split
    CHAMPION = "CHAMPION"
    REJECTED = "REJECTED"


class Decision(StrEnum):
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


class ReasonCode(StrEnum):
    """Machine-readable reasons. Stable strings: part of the decision record."""

    # validation selection (per candidate)
    SELECTED_ON_VALIDATION = "selected_on_validation"
    OUTRANKED_ON_VALIDATION = "outranked_on_validation"
    VALIDATION_INFEASIBLE = "validation_infeasible"
    # decision
    NO_FEASIBLE_CHALLENGER = "no_feasible_challenger"
    HELDOUT_CONSTRAINTS_PASSED = "heldout_constraints_passed"
    HELDOUT_CONSTRAINT_VIOLATION = "heldout_constraint_violation"
    INITIAL_CHAMPION = "initial_champion"
    NO_REGRESSION = "no_regression"
    EQUAL_TO_INCUMBENT = "equal_to_incumbent"
    REGRESSION = "regression"
    INCUMBENT_HELDOUT_INFEASIBLE = "incumbent_heldout_infeasible"
    CHALLENGER_IS_INCUMBENT = "challenger_is_incumbent"
    INCUMBENT_SUPERSEDED = "incumbent_superseded"


# -- errors -------------------------------------------------------------------------------------
class PromotionError(Exception):
    code = "promotion_error"


class PromotionNotFound(PromotionError):
    code = "promotion_not_found"


class ChampionNotFound(PromotionError):
    code = "champion_not_found"


class NotPromotable(PromotionError):
    """The experiment cannot enter promotion (not COMPLETED, already promoted elsewhere, ...)."""

    code = "experiment_not_promotable"


class IncompatibleIncumbent(PromotionError):
    """The lineage's champion was produced under a different identity: never compared."""

    code = "incompatible_incumbent"


class PromotionInProgress(PromotionError):
    code = "promotion_in_progress"


class PromotionAmbiguous(PromotionError):
    """A held-out call's outcome is unknown and nothing proves it: nothing is re-sent."""

    code = "promotion_ambiguous_attempt"


class PromotionFailed(PromotionError):
    code = "promotion_failed"


class EvidenceMismatch(PromotionError):
    """Stored evidence does not reproduce what was recorded (fail closed)."""

    code = "promotion_evidence_mismatch"


class HeldoutUnavailable(PromotionError):
    code = "heldout_backend_unavailable"


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def default_lineage(contract: TaskContract) -> str:
    """One lineage per dataset + task unless the caller names another: a changed dataset
    version, contract, model or evaluator then fails closed instead of starting silently."""
    return f"{contract.dataset.dataset_id}.{contract.task_id}"


def promotion_id_for(job_id: str) -> str:
    return "p-" + canonical_hash(["promotion", job_id])[:24]


# -- identity -----------------------------------------------------------------------------------
def compatibility_identity(
    definition: ExperimentJobDefinition, artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Everything two champions must share to be compared on held-out data: dataset version and
    splits (the same test rows, never another champion's training rows), TaskContract (objective,
    constraints, evaluation, workflow vocabulary), grammar, evaluator, exact model + model_hash,
    prompt version and runtime versions, and the held-out protocol. The plan (budget, seeds,
    strategies) is NOT part of it: champions of different experiments are compared."""
    p = artifact["identity"]["problem"]
    contract = definition.contract
    out = {
        "dataset_id": p["dataset_id"],
        "dataset_version": p["dataset_version"],
        "dataset_hash": p["dataset_hash"],
        "dataset_content_hash": p["dataset_content_hash"],
        "splits_hash": p["splits_hash"],
        "test_row_ids": list(definition.splits.rows_for(SplitRole.TEST, SplitUse.PROMOTION_GATE)),
        "task_id": p["task_id"],
        "contract_version": p["contract_version"],
        "task_contract_hash": p["task_contract_hash"],
        "objective": contract.objective.model_dump(mode="json"),
        "objective_hash": p["objective_hash"],
        "constraints": contract.constraints.model_dump(mode="json"),
        "constraints_hash": p["constraints_hash"],
        "evaluation_hash": p["evaluation_hash"],
        "evaluator_version": p["evaluator_run_version"],
        "grammar_version": p["grammar_version"],
        "model": p["model"],
        "model_config_hash": p["model_config_hash"],
        "model_hash": p["model_hash"],
        "prompt_template_version": p["prompt_template_version"],
        "run_versions": artifact["run_versions"],
        "synthetic": artifact["synthetic"],
        "heldout_protocol": HELDOUT_PROTOCOL,
    }
    return out | {"compat_hash": canonical_hash(out)}


def _incompatibilities(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    return sorted(k for k in set(a) | set(b) if k != "compat_hash" and a.get(k) != b.get(k))


# -- 2. validation-only challenger selection ----------------------------------------------------
def final_runs(attempts: Sequence[AttemptRow], genome_hash: str, rows: set[str]):
    """The final stored run of every (row, trial, seed) of ``genome_hash`` on ``rows`` (the
    last attempt: a MODEL_ERROR is retried), with its attempt id, ordered by (row, trial)."""
    final: dict[tuple[str, int, int], AttemptRow] = {}
    for a in attempts:
        if a.genome_hash != genome_hash or a.task_id not in rows:
            continue
        if a.state is not AttemptState.COMPLETED or a.result_json is None:
            raise EvidenceMismatch(f"stored run {a.attempt_id} has no result")
        key = (a.task_id, a.trial, a.run_seed)
        if key not in final or a.attempt > final[key].attempt:
            final[key] = a
    return [
        (final[k].attempt_id, EvaluatedRun.model_validate_json(final[k].result_json or ""))
        for k in sorted(final)
    ]


def _summary(runs: Sequence[EvaluatedRun]) -> dict[str, Any]:
    return {
        "runs": len(runs),
        "score_mean": statistics.fmean(run_score(r) for r in runs),
        "pass_rate": sum(r.evaluation.verdict is Verdict.PASS for r in runs) / len(runs),
        "fitness_mean": statistics.fmean(r.evaluation.fitness for r in runs),
    }


def _rank(contract: TaskContract, genome: Genome, runs: Sequence[EvaluatedRun]) -> dict[str, Any]:
    rank = rank_candidate(contract, genome, runs)  # TaskContract.rank: hard limits first
    return {
        "feasible": rank.feasible,
        "rank_key": list(rank.sort_key),
        "violations": [v.message for v in rank.violations],
        "violation_codes": [v.code.value for v in rank.violations],
    }


AttemptsFn = Callable[[str, int], Sequence[AttemptRow]]


def select_challenger(
    definition: ExperimentJobDefinition, artifact: Mapping[str, Any], attempts: AttemptsFn
) -> dict[str, Any]:
    """Exactly one challenger (or none), from VALIDATION evidence only.

    Candidates are the validation-selected champions of every (strategy, seed) run of the
    artifact - each run's best under the same ``TaskContract.rank``, so the best of them is the
    best of every candidate the experiment evaluated. Each one's validation rank is recomputed
    from the job's stored validation runs (``attempts(strategy, seed)``) and must equal the
    artifact's (fail closed). Nothing here can see a test row: the job never ran one, and the
    held-out runtime is bound only after this selection is pinned."""
    contract, plan = definition.contract, definition.plan
    val_rows = set(definition.splits.rows_for(SplitRole.VALIDATION, SplitUse.SELECTION))
    strategy_order = {s.value: i for i, s in enumerate(plan.strategies)}
    seed_order = {s: i for i, s in enumerate(plan.seeds)}
    candidates: list[dict[str, Any]] = []
    for run in artifact["runs"]:
        champ = run["champion"]
        if champ is None:
            continue  # the run evaluated nothing (e.g. its budget fit no candidate)
        genome = Genome.from_canonical(champ["genome"])
        if genome.genome_hash != champ["genome_hash"]:
            raise EvidenceMismatch(f"run {run['run_id']}: champion genome does not hash to itself")
        stored = final_runs(attempts(run["strategy"], run["seed"]), genome.genome_hash, val_rows)
        runs = [r for _, r in stored]
        if not runs or len(runs) != champ["validation"]["runs"]:
            raise EvidenceMismatch(f"run {run['run_id']}: stored validation runs are incomplete")
        rank = _rank(contract, genome, runs)
        validation = _summary(runs)
        if (rank["feasible"], rank["rank_key"], validation["score_mean"]) != (
            champ["feasible"],
            champ["rank_key"],
            champ["validation"]["score_mean"],
        ):
            raise EvidenceMismatch(
                f"run {run['run_id']}: stored validation runs do not reproduce its champion"
            )
        candidates.append(
            {
                "candidate_id": f"{run['strategy']}/{run['seed']}/{genome.genome_hash}",
                "strategy": run["strategy"],
                "seed": run["seed"],
                "run_id": run["run_id"],
                "optimizer": run["optimizer"],
                "optimizer_version": run["optimizer_version"],
                "genome_hash": genome.genome_hash,
                "genome": genome.canonical(),
                "first_evaluation": champ["first_evaluation"],
                "validation": validation | {"attempt_ids": [a for a, _ in stored]},
                "rank": rank,
            }
        )

    def key(c: Mapping[str, Any]) -> tuple:
        return (
            tuple(c["rank"]["rank_key"]),
            c["validation"]["score_mean"],
            -c["first_evaluation"],
            -strategy_order[c["strategy"]],
            -seed_order[c["seed"]],
        )

    feasible = sorted((c for c in candidates if c["rank"]["feasible"]), key=key, reverse=True)
    challenger = feasible[0]["candidate_id"] if feasible else None
    for c in candidates:
        if not c["rank"]["feasible"]:
            c["status"], c["reason_codes"] = (
                CandidateStatus.REJECTED,
                [ReasonCode.VALIDATION_INFEASIBLE],
            )
        elif c["candidate_id"] == challenger:
            c["status"], c["reason_codes"] = (
                CandidateStatus.VALIDATED,
                [ReasonCode.SELECTED_ON_VALIDATION],
            )
        else:
            c["status"], c["reason_codes"] = (
                CandidateStatus.CANDIDATE,
                [ReasonCode.OUTRANKED_ON_VALIDATION],
            )
    return json.loads(
        _json(
            {
                "rule": SELECTION_RULE,
                "description": SELECTION_DESCRIPTION,
                "split": SplitRole.VALIDATION.value,
                "validation_row_ids": sorted(val_rows),
                "objective": contract.objective.model_dump(mode="json"),
                "constraints": contract.constraints.model_dump(mode="json"),
                "candidates": candidates,
                "ranking": [c["candidate_id"] for c in feasible],
                "challenger": challenger,
            }
        )
    )


def challenger_of(selection: Mapping[str, Any]) -> dict[str, Any] | None:
    return next(
        (c for c in selection["candidates"] if c["candidate_id"] == selection["challenger"]), None
    )


# -- 5./6. held-out evidence and the gate -------------------------------------------------------
def heldout_seed(task_id: str, trial: int) -> int:
    """The execution seed of one held-out (row, trial): the SAME for challenger and incumbent."""
    return int(canonical_hash([HELDOUT_PROTOCOL, task_id, trial])[:8], 16)


def heldout_layout(tasks: Sequence[ExecutionTask], trials: int):
    return [
        (t, HELDOUT_TRIAL_OFFSET + i, heldout_seed(t.id, HELDOUT_TRIAL_OFFSET + i))
        for t in tasks
        for i in range(trials)
    ]


def heldout_protocol(definition: ExperimentJobDefinition) -> dict[str, Any]:
    rows = list(definition.splits.rows_for(SplitRole.TEST, SplitUse.PROMOTION_GATE))
    trials = definition.plan.trials
    return {
        "protocol": HELDOUT_PROTOCOL,
        "split": SplitRole.TEST.value,
        "row_ids": rows,
        "trials": [HELDOUT_TRIAL_OFFSET + i for i in range(trials)],
        "seeds": "heldout_seed(row, trial): identical for challenger and incumbent",
        "model_attempts": MODEL_ATTEMPTS,
    }


def heldout_evidence(
    contract: TaskContract, genome: Genome, stored: Sequence[tuple[str, EvaluatedRun]]
) -> dict[str, Any]:
    runs = [r for _, r in stored]
    return json.loads(
        _json(
            {
                "genome_hash": genome.genome_hash,
                "row_ids": sorted({r.execution.task_id for r in runs}),
                **_summary(runs),
                "measurements": candidate_measurements(genome, runs).model_dump(mode="json"),
                **_rank(contract, genome, runs),
                "attempt_ids": [a for a, _ in stored],
            }
        )
    )


def gate(
    challenger: Mapping[str, Any], incumbent: Mapping[str, Any] | None, same_genome: bool = False
) -> tuple[Decision, list[ReasonCode], dict[str, Any]]:
    """The binary held-out gate from held-out evidence alone. Hard constraints first; then the
    contract's own ranking (``CandidateRank.sort_key``) - the challenger must not rank below the
    incumbent. Equal is no regression. No incumbent: the constraints alone decide."""
    comparison: dict[str, Any] = {
        "comparator": COMPARATOR,
        "challenger_key": challenger["rank_key"],
        "incumbent_key": incumbent["rank_key"] if incumbent is not None else None,
        "relation": None,
    }
    if not challenger["feasible"]:
        return Decision.REJECTED, [ReasonCode.HELDOUT_CONSTRAINT_VIOLATION], comparison
    reasons = [ReasonCode.HELDOUT_CONSTRAINTS_PASSED]
    if incumbent is None:
        return Decision.PROMOTED, reasons + [ReasonCode.INITIAL_CHAMPION], comparison
    if same_genome:
        reasons.append(ReasonCode.CHALLENGER_IS_INCUMBENT)
    if not incumbent["feasible"]:
        reasons.append(ReasonCode.INCUMBENT_HELDOUT_INFEASIBLE)
    mine, theirs = tuple(challenger["rank_key"]), tuple(incumbent["rank_key"])
    if mine < theirs:
        comparison["relation"] = "worse"
        return Decision.REJECTED, reasons + [ReasonCode.REGRESSION], comparison
    if mine == theirs:
        comparison["relation"] = "equal"
        return (
            Decision.PROMOTED,
            reasons + [ReasonCode.NO_REGRESSION, ReasonCode.EQUAL_TO_INCUMBENT],
            comparison,
        )
    comparison["relation"] = "better"
    return Decision.PROMOTED, reasons + [ReasonCode.NO_REGRESSION], comparison


def heldout_record(
    definition: ExperimentJobDefinition,
    pinned: Mapping[str, Any],
    attempts: Sequence[tuple[str, AttemptRow]],
) -> dict[str, Any]:
    """Held-out evidence of the challenger (and incumbent) from the stored held-out runs."""
    contract = definition.contract
    rows = set(definition.splits.rows_for(SplitRole.TEST, SplitUse.PROMOTION_GATE))
    challenger = challenger_of(pinned["selection"])
    assert challenger is not None
    c_genome = Genome.from_canonical(challenger["genome"])
    by_subject = {s: [a for subj, a in attempts if subj == s] for s in ("challenger", "incumbent")}
    out: dict[str, Any] = {
        "protocol": heldout_protocol(definition),
        "challenger": heldout_evidence(
            contract,
            c_genome,
            final_runs(by_subject["challenger"], c_genome.genome_hash, rows),
        ),
        "incumbent": None,
    }
    inc = pinned["incumbent"]
    if inc is not None:
        i_genome = Genome.from_canonical(inc["genome"])
        same = i_genome.genome_hash == c_genome.genome_hash
        source = "challenger" if same else "incumbent"  # one workflow is evaluated once
        out["incumbent"] = heldout_evidence(
            contract, i_genome, final_runs(by_subject[source], i_genome.genome_hash, rows)
        )
    expected = len(rows) * definition.plan.trials
    for subject in ("challenger", "incumbent"):
        ev = out[subject]
        if ev is not None and ev["runs"] != expected:
            raise EvidenceMismatch(f"{subject}: {ev['runs']} held-out runs, expected {expected}")
    return out


def decision_record(
    *,
    promotion: PromotionRow,
    definition: ExperimentJobDefinition,
    identity: Mapping[str, Any],
    pinned: Mapping[str, Any],
    heldout: Mapping[str, Any] | None,
    superseded: bool = False,
    champion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The immutable PromotionDecision: everything needed to reproduce it."""
    selection = pinned["selection"]
    challenger = challenger_of(selection)
    incumbent = pinned["incumbent"]
    if challenger is None:
        decision, reasons, comparison = Decision.REJECTED, [ReasonCode.NO_FEASIBLE_CHALLENGER], None
        gate_out = {"decision": decision, "reason_codes": reasons}
    else:
        assert heldout is not None
        same = incumbent is not None and incumbent["genome_hash"] == challenger["genome_hash"]
        decision, reasons, comparison = gate(heldout["challenger"], heldout["incumbent"], same)
        gate_out = {"decision": decision, "reason_codes": list(reasons)}
        if superseded:
            decision, reasons = Decision.REJECTED, [*reasons, ReasonCode.INCUMBENT_SUPERSEDED]
    status = CandidateStatus.CHAMPION if decision is Decision.PROMOTED else CandidateStatus.REJECTED
    record = {
        "schema": DECISION_SCHEMA,
        "promotion_id": promotion.promotion_id,
        "lineage_id": promotion.lineage_id,
        "experiment": {
            "job_id": promotion.job_id,
            "experiment_id": identity["experiment_id"],
            "definition_hash": identity["definition_hash"],
            "problem_id": identity["problem_id"],
            "synthetic": identity["synthetic"],
        },
        "decision": decision,
        "reason_codes": reasons,
        "gate": gate_out,
        "challenger": None
        if challenger is None
        else {
            k: challenger[k]
            for k in (
                "candidate_id",
                "strategy",
                "seed",
                "run_id",
                "optimizer",
                "optimizer_version",
                "genome_hash",
                "genome",
            )
        }
        | {"status": status},
        "incumbent": incumbent,
        "selection": selection,
        "test_opened": promotion.test_opened_at is not None,
        "heldout": heldout,
        "constraints": None
        if heldout is None
        else {
            s: None
            if heldout[s] is None
            else {k: heldout[s][k] for k in ("feasible", "violations", "violation_codes")}
            for s in ("challenger", "incumbent")
        },
        "comparison": comparison,
        "identities": pinned["compatibility"]
        | {"selection_rule": SELECTION_RULE, "decision_schema": DECISION_SCHEMA},
        "champion": champion,
    }
    record = json.loads(_json(record))
    return record | {"decision_hash": canonical_hash(record)}


def champion_record(
    *,
    lineage_id: str,
    champion_id: str,
    version: int,
    promotion: PromotionRow,
    identity: Mapping[str, Any],
    pinned: Mapping[str, Any],
    heldout: Mapping[str, Any],
) -> dict[str, Any]:
    challenger = challenger_of(pinned["selection"])
    assert challenger is not None
    incumbent = pinned["incumbent"]
    record = {
        "schema": CHAMPION_SCHEMA,
        "champion_id": champion_id,
        "lineage_id": lineage_id,
        "version": version,
        "genome_hash": challenger["genome_hash"],
        "genome": challenger["genome"],
        "compatibility": pinned["compatibility"],  # dataset, contract, grammar, evaluator, model
        "provenance": {
            "promotion_id": promotion.promotion_id,
            "job_id": promotion.job_id,
            "experiment_id": identity["experiment_id"],
            "definition_hash": identity["definition_hash"],
            "run_id": challenger["run_id"],
            "strategy": challenger["strategy"],
            "seed": challenger["seed"],
            "optimizer": challenger["optimizer"],
            "optimizer_version": challenger["optimizer_version"],
        },
        "validation": {"rank": challenger["rank"], **challenger["validation"]},
        "heldout": heldout["challenger"],
        "previous_champion_id": incumbent["champion_id"] if incumbent is not None else None,
    }
    return json.loads(_json(record))


# -- views --------------------------------------------------------------------------------------
class _View(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HeldoutAttempts(_View):
    completed: int
    in_flight: int
    errored: int
    reused_by_proof: int


class PromotionView(_View):
    promotion_id: str
    job_id: str
    lineage_id: str
    state: PromotionState
    reason: str | None
    detail: str | None
    created_at: str
    updated_at: str
    test_opened_at: str | None
    incumbent_id: str | None
    incumbent_version: int
    challenger: dict[str, Any] | None
    decision: Decision | None
    record: dict[str, Any] | None  # the PromotionDecision, once DECIDED
    heldout_attempts: HeldoutAttempts


class ChampionView(_View):
    champion_id: str
    lineage_id: str
    version: int
    promotion_id: str
    created_at: str
    record: dict[str, Any]


class ChampionHistory(_View):
    lineage_id: str
    champions: list[ChampionView]


class PromotionList(_View):
    promotions: list[PromotionView]


def champion_view(row: ChampionRow) -> ChampionView:
    return ChampionView(
        champion_id=row.champion_id,
        lineage_id=row.lineage_id,
        version=row.version,
        promotion_id=row.promotion_id,
        created_at=row.created_at,
        record=json.loads(row.record_json),
    )


# -- the service --------------------------------------------------------------------------------
@dataclass
class ChampionPromotions:
    """promote / inspect / verify. ``runtime`` is ``None`` in a process that cannot execute
    held-out runs (champions and decisions stay inspectable)."""

    store: SQLiteChampionStore
    jobs: SQLiteJobStore
    runtime: HeldoutRuntime | None = None
    clock: Callable[[], float] = time.time
    owner: str = field(default_factory=lambda: f"promoter-{os.getpid()}-{secrets.token_hex(4)}")
    lease_s: float = DEFAULT_LEASE_S
    heartbeat: bool = True

    # -- inspection -------------------------------------------------------------------------
    def get(self, promotion_id: str) -> PromotionView:
        row = self.store.get(promotion_id)
        if row is None:
            raise PromotionNotFound(f"no promotion {promotion_id}")
        attempts = [a for _, a in self.store.attempts(promotion_id)]
        return PromotionView(
            promotion_id=row.promotion_id,
            job_id=row.job_id,
            lineage_id=row.lineage_id,
            state=row.state,
            reason=row.reason,
            detail=row.detail,
            created_at=row.created_at,
            updated_at=row.updated_at,
            test_opened_at=row.test_opened_at,
            incumbent_id=row.incumbent_id,
            incumbent_version=row.incumbent_version,
            challenger=challenger_of(json.loads(row.selection_json)["selection"]),
            decision=Decision(row.decision) if row.decision else None,
            record=json.loads(row.decision_json) if row.decision_json else None,
            heldout_attempts=HeldoutAttempts(
                completed=sum(a.state is AttemptState.COMPLETED for a in attempts),
                in_flight=sum(a.state is AttemptState.STARTED for a in attempts),
                errored=sum(a.state is AttemptState.ERRORED for a in attempts),
                reused_by_proof=sum(a.resolution == "replay_proof" for a in attempts),
            ),
        )

    def for_job(self, job_id: str) -> PromotionView:
        row = self.store.for_job(job_id)
        if row is None:
            raise PromotionNotFound(f"experiment {job_id} has not entered promotion")
        return self.get(row.promotion_id)

    def list(self, lineage_id: str | None = None) -> PromotionList:
        return PromotionList(
            promotions=[self.get(r.promotion_id) for r in self.store.promotions(lineage_id)]
        )

    def current(self, lineage_id: str) -> ChampionView:
        row = self.store.current(lineage_id)
        if row is None:
            raise ChampionNotFound(f"lineage {lineage_id} has no champion")
        return champion_view(row)

    def history(self, lineage_id: str) -> ChampionHistory:
        rows = self.store.history(lineage_id)
        if not rows:
            raise ChampionNotFound(f"lineage {lineage_id} has no champion")
        return ChampionHistory(lineage_id=lineage_id, champions=[champion_view(r) for r in rows])

    def recover(self) -> list[tuple[str, PromotionReason]]:
        """Process start: every held-out evaluation whose process is gone becomes INTERRUPTED
        (ambiguous when a call was in flight - never resumed without proof)."""
        return self.store.interrupt_stale(self.clock())

    # -- promotion --------------------------------------------------------------------------
    def promote(self, job_id: str, lineage: str | None = None) -> PromotionView:
        """Enter a COMPLETED experiment into promotion, or continue / return its promotion. An
        experiment is promoted (or rejected) once: its test split is opened at most once."""
        if lineage is not None and not _valid_lineage(lineage):
            raise NotPromotable(f"lineage {lineage!r} is not a valid lineage id")
        existing = self.store.for_job(job_id)
        if existing is not None:
            return self._existing(existing, lineage)
        job = self.jobs.get_job(job_id)
        if job is None:
            raise JobNotFound(f"no experiment job {job_id}")
        if job.state is not JobState.COMPLETED or job.artifact_json is None:
            raise NotPromotable(
                f"experiment {job_id} is {job.state.value}; only a COMPLETED experiment may "
                "enter promotion"
            )
        definition, identity, artifact = _experiment(self.jobs, job_id)
        lineage_id = lineage or default_lineage(definition.contract)
        if not _valid_lineage(lineage_id):
            raise NotPromotable(f"default lineage {lineage_id!r} is too long; name a lineage")
        selection = select_challenger(definition, artifact, self._attempts(job_id))
        compat = compatibility_identity(definition, artifact)
        promotion_id = promotion_id_for(job_id)
        if selection["challenger"] is None:  # nothing feasible: the test split stays closed
            return self._reject_without_test(
                promotion_id, job_id, lineage_id, definition, identity, selection, compat
            )
        self._check_runtime(definition, identity)  # before the test split is opened
        while True:
            incumbent = self.store.current(lineage_id)
            pinned = {
                "selection": selection,
                "compatibility": compat,
                "incumbent": self._pin_incumbent(incumbent, compat),
            }
            try:
                claim = self.store.open(
                    promotion_id,
                    job_id,
                    lineage_id,
                    _json(pinned),
                    incumbent.champion_id if incumbent else None,
                    incumbent.version if incumbent else 0,
                    self.owner,
                    self.clock(),
                    self.lease_s,
                )
            except StaleIncumbent:
                continue  # the lineage moved before anything was opened: pin the new incumbent
            except Conflict:  # another process entered this experiment first
                row = self.store.for_job(job_id)
                assert row is not None
                return self._existing(row, lineage)
            return self._evaluate(claim)

    def _existing(self, row: PromotionRow, lineage: str | None) -> PromotionView:
        if lineage is not None and lineage != row.lineage_id:
            raise NotPromotable(
                f"experiment {row.job_id} already entered promotion in lineage {row.lineage_id}; "
                "its test split is never opened again"
            )
        if row.state in TERMINAL:
            return self.get(row.promotion_id)
        return self._continue(row)

    def _continue(self, row: PromotionRow) -> PromotionView:
        """Resume an unfinished held-out evaluation: the same challenger and incumbent on the
        same rows. Ambiguous calls are proven (replay proof) or the resume is refused."""
        if row.state is PromotionState.OPEN and (row.lease_until or 0.0) >= self.clock():
            raise PromotionInProgress(f"promotion {row.promotion_id} is being evaluated")
        claim = self.store.claim(row.promotion_id, self.owner, self.clock(), self.lease_s)
        if claim is None:
            row = self.store.get(row.promotion_id) or row
            if row.state in TERMINAL:
                return self.get(row.promotion_id)
            if row.reason == PromotionReason.AMBIGUOUS_ATTEMPT:
                self._prove(row)
                claim = self.store.claim(row.promotion_id, self.owner, self.clock(), self.lease_s)
            if claim is None:
                raise PromotionInProgress(f"promotion {row.promotion_id} is not claimable")
        return self._evaluate(claim)

    def _reject_without_test(
        self, promotion_id, job_id, lineage_id, definition, identity, selection, compat
    ) -> PromotionView:
        pinned = {"selection": selection, "compatibility": compat, "incumbent": None}
        stub = PromotionRow(
            promotion_id=promotion_id,
            job_id=job_id,
            lineage_id=lineage_id,
            state=PromotionState.DECIDED,
            reason=None,
            detail=None,
            created_at="",
            updated_at="",
            test_opened_at=None,
            incumbent_id=None,
            incumbent_version=0,
            selection_json=_json(pinned),
            lease_owner=None,
            lease_until=None,
            fence=0,
            decision=None,
            decision_json=None,
            decided_at=None,
        )
        record = decision_record(
            promotion=stub, definition=definition, identity=identity, pinned=pinned, heldout=None
        )
        try:
            self.store.record_without_test(
                promotion_id, job_id, lineage_id, _json(pinned), record["decision"], _json(record)
            )
        except Conflict:
            pass  # recorded concurrently: the same deterministic decision
        return self.get(promotion_id)

    def _pin_incumbent(
        self, incumbent: ChampionRow | None, compat: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if incumbent is None:
            return None
        record = json.loads(incumbent.record_json)
        mismatched = _incompatibilities(record["compatibility"], compat)
        if mismatched or incumbent.compat_hash != compat["compat_hash"]:
            raise IncompatibleIncumbent(
                f"lineage {incumbent.lineage_id}'s champion {incumbent.champion_id} differs in "
                f"{', '.join(mismatched) or 'identity'}; it is never compared with this "
                "challenger (name a new lineage to start one)"
            )
        return {
            "champion_id": incumbent.champion_id,
            "version": incumbent.version,
            "genome_hash": record["genome_hash"],
            "genome": record["genome"],
            "provenance": record["provenance"],
            "compat_hash": incumbent.compat_hash,
        }

    def _runtime(self) -> HeldoutRuntime:
        if self.runtime is None or not hasattr(self.runtime, "bind_heldout"):
            raise HeldoutUnavailable("this process has no held-out evaluation backend")
        return self.runtime

    def _check_runtime(self, definition: ExperimentJobDefinition, identity: Mapping[str, Any]):
        """The runtime must reproduce the experiment's identity (dataset, contract, evaluator,
        model, pricing, optimizer versions) - checked before the test split is opened."""
        runtime = self._runtime()
        if runtime.synthetic and not definition.synthetic:
            raise HeldoutUnavailable("a synthetic runtime cannot judge a real experiment")
        try:
            if job_identity(definition, runtime.bind(definition)) != identity:
                raise RuntimeMismatch("the runtime does not reproduce the experiment's identity")
        except (ValueError, RuntimeMismatch) as exc:
            raise HeldoutUnavailable(str(exc)) from exc

    def _attempts(self, job_id: str) -> AttemptsFn:
        return lambda strategy, seed: self.jobs.attempts(job_id, strategy, seed)

    def _evaluate(self, claim: PromotionClaim) -> PromotionView:
        # the #24 heartbeat, renewing this promotion's fenced lease instead of a unit's
        lease = Lease(self.store, claim, self.lease_s, self.clock)  # type: ignore[arg-type]
        if not self.heartbeat:
            return self._evaluate_owned(claim)
        with _Heartbeat(lease, self.lease_s / 3):
            return self._evaluate_owned(claim)

    def _evaluate_owned(self, claim: PromotionClaim) -> PromotionView:
        row = self.store.get(claim.promotion_id)
        assert row is not None
        definition, identity, artifact = _experiment(self.jobs, row.job_id)
        pinned = json.loads(row.selection_json)
        try:
            try:
                self._check_runtime(definition, identity)
                binding = self._runtime().bind_heldout(definition)
                _check_binding(definition, identity, binding)
            except Exception as exc:
                self.store.interrupt(claim, PromotionReason.RUNTIME_UNAVAILABLE, str(exc))
                raise HeldoutUnavailable(f"held-out runtime unavailable: {exc}") from exc
            self._run_heldout(claim, definition, identity, artifact, pinned, binding)
        except LeaseLost:
            raise PromotionInProgress(f"another process took over {claim.promotion_id}") from None
        except AmbiguousAttempt as exc:
            self._close(claim, self.store.interrupt, PromotionReason.AMBIGUOUS_ATTEMPT, exc)
            raise PromotionAmbiguous(str(exc)) from None
        except ModelUnavailable as exc:
            self._close(claim, self.store.fail, PromotionReason.MODEL_UNAVAILABLE, exc)
            raise PromotionFailed(str(exc)) from None
        except PromotionError:
            raise
        except Exception as exc:
            self._close(claim, self.store.fail, PromotionReason.HELDOUT_ERROR, exc)
            raise PromotionFailed(f"{type(exc).__name__}: {exc}") from None
        heldout = heldout_record(definition, pinned, self.store.attempts(claim.promotion_id))
        self._decide(claim, row, definition, identity, pinned, heldout)
        return self.get(claim.promotion_id)

    @staticmethod
    def _close(claim, close, reason: PromotionReason, exc: BaseException) -> None:
        try:
            close(claim, reason, f"{type(exc).__name__}: {exc}")
        except LeaseLost:
            pass

    def _run_heldout(
        self,
        claim: PromotionClaim,
        definition: ExperimentJobDefinition,
        identity: Mapping[str, Any],
        artifact: Mapping[str, Any],
        pinned: Mapping[str, Any],
        binding: HeldoutBinding,
    ) -> None:
        """Every held-out run of the challenger (then the incumbent) through the write-ahead
        log: a stored result is reused (no model call), a new call is written STARTED first and
        COMPLETED atomically after; a call that started without a stored result is ambiguous
        and is never re-sent."""
        stored = {a.attempt_id: a for _, a in self.store.attempts(claim.promotion_id)}
        consumed: set[str] = set()
        challenger = challenger_of(pinned["selection"])
        assert challenger is not None
        subjects = [("challenger", Genome.from_canonical(challenger["genome"]))]
        inc = pinned["incumbent"]
        if inc is not None and inc["genome_hash"] != challenger["genome_hash"]:
            subjects.append(("incumbent", Genome.from_canonical(inc["genome"])))
        layout = heldout_layout(binding.tasks, definition.plan.trials)
        versions = artifact["run_versions"]
        for subject, genome in subjects:
            for task, trial, seed in layout:
                n = 0
                while True:
                    attempt_id = canonical_hash(
                        [claim.promotion_id, subject, genome.genome_hash, task.id, trial, seed, n]
                    )
                    run = self._attempt(
                        claim, subject, genome, task, trial, seed, n, attempt_id, stored, binding
                    )
                    consumed.add(attempt_id)
                    _check_run(run, definition, identity, versions, genome, task, trial, seed)
                    failure = run.execution.failure
                    if failure is None or failure.kind is not FailureKind.MODEL_ERROR:
                        break
                    n += 1
                    if n >= MODEL_ATTEMPTS:
                        raise ModelUnavailable(
                            f"model unavailable on held-out {task.id} after {n} attempt(s)"
                        )
        if set(stored) - consumed:
            raise ExperimentError("stored held-out runs were never reached by the re-execution")

    def _attempt(
        self, claim, subject, genome, task, trial, seed, n, attempt_id, stored, binding
    ) -> EvaluatedRun:
        row = stored.get(attempt_id)
        if row is not None:
            if row.state is AttemptState.STARTED:
                raise AmbiguousAttempt(f"held-out attempt {attempt_id} ({task.id}) has no result")
            if row.state is not AttemptState.COMPLETED:
                raise ExperimentError(f"held-out attempt {attempt_id} errored: {row.error}")
            run = EvaluatedRun.model_validate_json(row.result_json or "")
            if _usage(run) != json.loads(row.entry_json or "null"):
                raise ExperimentError(f"stored held-out run {attempt_id} disagrees with its usage")
            return run
        self.store.start_attempt(
            claim,
            subject,
            NewAttempt(
                attempt_id=attempt_id,
                genome_hash=genome.genome_hash,
                genome_json=genome.canonical_json(),
                task_id=task.id,
                trial=trial,
                run_seed=seed,
                attempt=n,
            ),
            self.clock(),
            self.lease_s,
        )
        try:
            run = binding.evaluate(genome, task, trial, seed)
        except Exception as exc:
            self.store.error_attempt(claim, attempt_id, f"{type(exc).__name__}: {exc}")
            raise
        timing = getattr(binding.evaluate, "timing", None)
        spent = timing(genome, task, trial, seed) if timing is not None else None
        self.store.complete_attempt(
            claim,
            attempt_id,
            run.model_dump_json(),
            _json(_usage(run)),
            _json(list(spent)) if spent is not None else None,
        )
        return run

    def _decide(
        self,
        claim: PromotionClaim,
        row: PromotionRow,
        definition: ExperimentJobDefinition,
        identity: Mapping[str, Any],
        pinned: Mapping[str, Any],
        heldout: Mapping[str, Any],
    ) -> None:
        record = decision_record(
            promotion=row, definition=definition, identity=identity, pinned=pinned, heldout=heldout
        )
        if record["decision"] == Decision.REJECTED:
            self.store.decide(claim, Decision.REJECTED.value, _json(record))
            return
        version = row.incumbent_version + 1
        champion_id = "c-" + canonical_hash([row.lineage_id, row.promotion_id, version])[:24]
        champion = {"champion_id": champion_id, "version": version}
        record = decision_record(
            promotion=row,
            definition=definition,
            identity=identity,
            pinned=pinned,
            heldout=heldout,
            champion=champion,
        )
        new = NewChampion(
            champion_id=champion_id,
            compat_hash=pinned["compatibility"]["compat_hash"],
            record_json=_json(
                champion_record(
                    lineage_id=row.lineage_id,
                    champion_id=champion_id,
                    version=version,
                    promotion=row,
                    identity=identity,
                    pinned=pinned,
                    heldout=heldout,
                )
            ),
        )
        try:
            self.store.decide(claim, Decision.PROMOTED.value, _json(record), new)
        except StaleIncumbent:  # a newer champion exists: never overwrite it
            record = decision_record(
                promotion=row,
                definition=definition,
                identity=identity,
                pinned=pinned,
                heldout=heldout,
                superseded=True,
            )
            self.store.decide(claim, Decision.REJECTED.value, _json(record))

    def _prove(self, row: PromotionRow) -> None:
        """Resolve ambiguous held-out calls WITHOUT a model call (the runtime's replay proof),
        or refuse: nothing is ever re-sent to resolve one."""
        open_ = [
            a for _, a in self.store.attempts(row.promotion_id) if a.state is AttemptState.STARTED
        ]
        definition, identity, _ = _experiment(self.jobs, row.job_id)
        proof = None
        if self.runtime is not None and hasattr(self.runtime, "bind_heldout"):
            self._check_runtime(definition, identity)
            proof = self.runtime.bind_heldout(definition).replay_proof
        if proof is None:
            raise PromotionAmbiguous(
                f"{len(open_)} held-out model call(s) started without a stored result; their "
                "outcome is unknown and no replay proof is available (nothing is re-sent)"
            )
        for a in open_:
            run = proof(a)
            key = run.execution.key if run is not None else None
            if (
                run is None
                or key is None
                or (
                    key.genome_hash,
                    key.task_id,
                    key.trial,
                    key.seed,
                )
                != (a.genome_hash, a.task_id, a.trial, a.run_seed)
            ):
                raise PromotionAmbiguous(
                    f"held-out attempt {a.attempt_id} ({a.task_id}) cannot be proven"
                )
            self.store.resolve_attempt(
                a.attempt_id, run.model_dump_json(), _json(_usage(run)), "replay_proof"
            )

    # -- reproducibility --------------------------------------------------------------------
    def verify(self, promotion_id: str) -> PromotionView:
        """Re-derive a decision from stored evidence alone - the validation runs in the job
        store, the held-out runs here, the contract - and fail closed unless it reproduces the
        recorded decision (and champion) exactly."""
        row = self.store.get(promotion_id)
        if row is None:
            raise PromotionNotFound(f"no promotion {promotion_id}")
        if row.state is not PromotionState.DECIDED or row.decision_json is None:
            raise EvidenceMismatch(f"promotion {promotion_id} is {row.state.value}, not DECIDED")
        recorded = json.loads(row.decision_json)
        definition, identity, artifact = _experiment(self.jobs, row.job_id)
        pinned = json.loads(row.selection_json)
        selection = select_challenger(definition, artifact, self._attempts(row.job_id))
        if selection != pinned["selection"]:
            raise EvidenceMismatch("stored validation runs no longer select the pinned challenger")
        if compatibility_identity(definition, artifact) != pinned["compatibility"]:
            raise EvidenceMismatch("the experiment's identity no longer matches the pinned one")
        heldout = None
        if selection["challenger"] is not None:
            heldout = heldout_record(definition, pinned, self.store.attempts(promotion_id))
        superseded = ReasonCode.INCUMBENT_SUPERSEDED.value in recorded["reason_codes"]
        rebuilt = decision_record(
            promotion=row,
            definition=definition,
            identity=identity,
            pinned=pinned,
            heldout=heldout,
            superseded=superseded,
            champion=recorded["champion"],
        )
        if rebuilt != recorded:
            raise EvidenceMismatch(f"stored evidence does not reproduce decision {promotion_id}")
        if superseded and rebuilt["gate"]["decision"] != Decision.PROMOTED:
            raise EvidenceMismatch("only a passing gate can lose compare-and-promote")
        if recorded["champion"] is not None:
            champ = self.store.champion(recorded["champion"]["champion_id"])
            if champ is None or champ.promotion_id != promotion_id:
                raise EvidenceMismatch("the promoted champion record is missing")
            assert heldout is not None
            expected = champion_record(
                lineage_id=row.lineage_id,
                champion_id=champ.champion_id,
                version=champ.version,
                promotion=row,
                identity=identity,
                pinned=pinned,
                heldout=heldout,
            )
            if json.loads(champ.record_json) != expected or champ.version != (
                row.incumbent_version + 1
            ):
                raise EvidenceMismatch("the champion record does not reproduce from evidence")
        return self.get(promotion_id)


# -- helpers ------------------------------------------------------------------------------------
def _valid_lineage(lineage: str) -> bool:
    return re.fullmatch(LINEAGE, lineage) is not None


def _usage(run: EvaluatedRun) -> dict[str, Any]:
    return asdict(MeasuredUsage.of(run))


def _experiment(jobs: SQLiteJobStore, job_id: str):
    """``(definition, identity, artifact)`` of a COMPLETED job, checked for consistency."""
    job = jobs.get_job(job_id)
    if job is None:
        raise JobNotFound(f"no experiment job {job_id}")
    if job.state is not JobState.COMPLETED or job.artifact_json is None:
        raise NotPromotable(f"experiment {job_id} is {job.state.value}, not COMPLETED")
    definition = ExperimentJobDefinition.model_validate_json(job.definition_json)
    identity = json.loads(job.identity_json)
    artifact = json.loads(job.artifact_json)
    if artifact["experiment_id"] != identity["experiment_id"]:
        raise EvidenceMismatch(f"experiment {job_id}'s artifact has a different identity")
    if artifact["test_runs"] != 0 or any(r["split_usage"]["test_runs"] for r in artifact["runs"]):
        raise EvidenceMismatch(f"experiment {job_id} executed test rows: it cannot be promoted")
    if any(u.state.value != "COMPLETED" for u in job.units):
        raise EvidenceMismatch(f"experiment {job_id} has unfinished units")
    return definition, identity, artifact


def _check_binding(
    definition: ExperimentJobDefinition, identity: Mapping[str, Any], binding: HeldoutBinding
) -> None:
    rows = definition.splits.rows_for(SplitRole.TEST, SplitUse.PROMOTION_GATE)
    if tuple(sorted(t.id for t in binding.tasks)) != tuple(sorted(rows)):
        raise ExperimentError("the held-out binding does not cover exactly the test split")
    if binding.evaluator_version != identity["evaluator_version"]:
        raise ExperimentError(
            f"held-out evaluator {binding.evaluator_version!r}, experiment was judged by "
            f"{identity['evaluator_version']!r}"
        )
    for task in binding.tasks:
        if task.contract_hash != definition.contract.contract_hash:
            raise ExperimentError(f"held-out row {task.id} is bound to another contract")


def _check_run(
    run: EvaluatedRun,
    definition: ExperimentJobDefinition,
    identity: Mapping[str, Any],
    versions: Sequence[Mapping[str, Any]],
    genome: Genome,
    task: ExecutionTask,
    trial: int,
    seed: int,
) -> None:
    key = run.execution.key
    role = definition.splits.role_of(key.task_id)
    if role is None:
        raise ExperimentError(f"held-out run on {key.task_id}, which is in no split")
    require_use(role, SplitUse.PROMOTION_GATE)  # only test rows may decide the gate
    if (key.genome_hash, key.task_id, key.trial, key.seed) != (
        genome.genome_hash,
        task.id,
        trial,
        seed,
    ):
        raise ExperimentError("the held-out evaluator returned a run for a different job")
    if run.evaluation.evaluator_version != identity["evaluator_version"]:
        raise ExperimentError(
            f"held-out run judged by {run.evaluation.evaluator_version!r}, experiment declares "
            f"{identity['evaluator_version']!r}"
        )
    if [key.versions.model_dump(mode="json")] != list(versions):
        raise ExperimentError(
            "held-out run reports different model / prompt / compiler / grammar versions than "
            "the experiment"
        )
