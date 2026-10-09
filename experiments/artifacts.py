"""Experiment artifacts as a product surface: provenance, trace, verify, reproduce.

    ExperimentArtifacts(jobs, champions)
      provenance(job_id)       the job's ProvenanceRecord (+ its promotion, if any)
      trace(job_id, path)      "where did this number come from?" for any field path
      verify(job_id)           integrity + identities + declared metrics + promotion citation
      reproduce(job_id)        replay every strategy run from stored execution evidence

    python -m experiments.artifacts verify    --data-dir .wynk-data --workspace ws-... --job j-...
    python -m experiments.artifacts trace     --data-dir .wynk-data --job j-... --path P
    python -m experiments.artifacts reproduce --data-dir .wynk-data --job j-... [--model-registry R]
    python -m experiments.artifacts verify    --frozen experiments/results/musique/protocol-v2

Reproduction semantics. ``reproduce`` re-runs the unchanged #23 ``run_strategy`` for every
(strategy, seed) of the job - same contract, splits, plan, seed, optimizer and evaluator binding
(whose identity must equal the job's: model registry entry, evaluator version, pricing) - with an
evaluator that answers ONLY from the job's stored write-ahead attempts (``experiments.jobs``):
no model is called, ever. Because optimizers, the ledger and selection are pure functions of the
results they are fed, the replay must propose the same candidates in the same order and select
the same workflows; the re-assembled body is then compared field class by field class:

    scientific          must be equal (scores, verdicts, candidate order, selections, tokens,
                        calls, cost, stop reasons, curves)
    measured_telemetry  must be equal too: it is replayed from the stored runs, not re-measured
    clock_telemetry     never compared (process clocks); counted only

A model backend that is nondeterministic is therefore never "re-run and hoped equal": its stored
outputs are the evidence, and what is verified is that the declared results follow from them.
``replay`` can also re-execute against a live evaluate function (``reexecute``) for backends
known to be deterministic; then only the scientific fields are compared. Any disagreement fails
closed (``ReproductionMismatch``).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from core.genome import Genome
from core.results import EvaluatedRun
from core.run_contract import ExecutionTask
from experiments.budget_ledger import BudgetLedger, run_reservation
from experiments.jobs import (
    ArtifactCorrupted,
    BackendUnavailable,
    ExperimentJobDefinition,
    ExperimentJobs,
    JobError,
    JobRuntime,
    attempt_identity,
    job_identity,
    load_canonical,
)
from experiments.learning_curves import EvaluateFn
from experiments.optimization_experiment import Strategy, assemble, make_strategy, run_strategy
from experiments.provenance import (
    CLOCK,
    MEASURED,
    SCIENTIFIC,
    ArtifactIntegrityError,
    CanonicalArtifact,
    TraceError,
    _walk,
    class_counts,
    differences,
    load_frozen,
    optimizer_config,
    project,
    recompute_declared,
    scientific,
    trace,
)
from store.champions import SQLiteChampionStore
from store.jobs import AttemptRow, AttemptState, SQLiteJobStore


class TraceNotFound(JobError):
    code = "trace_path_not_found"


class ReproductionMismatch(JobError):
    """The stored evidence does not reproduce the artifact (fail closed)."""

    code = "reproduction_mismatch"


class PromotionCitationMismatch(JobError):
    code = "artifact_integrity_error"


# -- evidence replay ----------------------------------------------------------------------------
class EvidenceEvaluate:
    """``EvaluateFn`` over a unit's stored attempts only: the n-th call for a job answers the
    n-th stored attempt (``attempt_identity``, as ``DurableEvaluate`` keys them). A run with no
    COMPLETED stored attempt, or one whose result disagrees with its recorded usage, fails
    closed - nothing is ever sent to a model."""

    def __init__(self, job_id: str, run_id: str, stored: Sequence[AttemptRow], measure) -> None:
        self._job_id, self._run_id, self._measure = job_id, run_id, measure
        self._stored = {a.attempt_id: a for a in stored if a.state is AttemptState.COMPLETED}
        self._open = [a.attempt_id for a in stored if a.state is not AttemptState.COMPLETED]
        self._counts: dict[tuple[str, str, int, int], int] = {}
        self._consumed: set[str] = set()
        self._timings: dict[tuple[str, str, int, int], tuple[float, float] | None] = {}
        self._lock = threading.Lock()

    def unconsumed(self) -> set[str]:
        return set(self._stored) - self._consumed

    def timing(self, genome: Genome, task: ExecutionTask, trial: int, seed: int):
        with self._lock:
            return self._timings.pop((genome.genome_hash, task.id, trial, seed), None)

    def __call__(self, genome: Genome, task: ExecutionTask, trial: int, seed: int) -> EvaluatedRun:
        key = (genome.genome_hash, task.id, trial, seed)
        with self._lock:
            n = self._counts.get(key, 0)
            self._counts[key] = n + 1
        attempt_id = attempt_identity(self._job_id, self._run_id, *key, n)
        row = self._stored.get(attempt_id)
        if row is None:
            raise ReproductionMismatch(
                f"the replay needs run {task.id} (trial {trial}, attempt {n}) of genome "
                f"{genome.genome_hash[:12]}, which has no stored result"
            )
        run = EvaluatedRun.model_validate_json(row.result_json or "")
        if self._measure(run) != json.loads(row.entry_json or "null"):
            raise ReproductionMismatch(f"stored run {attempt_id} disagrees with its recorded usage")
        timing = json.loads(row.timing_json) if row.timing_json else None
        with self._lock:
            self._consumed.add(attempt_id)
            self._timings[key] = tuple(timing) if timing is not None else None
        return run


@dataclass(frozen=True)
class Replay:
    body: dict[str, Any]  # the re-assembled result body
    units: list[dict[str, Any]]
    model_calls: int


def replay(
    store: SQLiteJobStore,
    job_id: str,
    runtime: JobRuntime,
    *,
    reexecute: EvaluateFn | None = None,
) -> tuple[CanonicalArtifact, Replay]:
    """Re-run every strategy run of a COMPLETED job from its stored evidence (or, with
    ``reexecute``, against that evaluate function) and re-assemble the result body."""
    job, art = load_canonical(store, job_id)
    definition = ExperimentJobDefinition.model_validate_json(job.definition_json)
    identity = json.loads(job.identity_json)
    record = art.provenance
    try:
        binding = runtime.bind(definition)
        rebound = job_identity(definition, binding)
    except ValueError as exc:
        raise ReproductionMismatch(f"the experiment cannot be rebound: {exc}") from exc
    if rebound != identity:
        raise ReproductionMismatch(
            "the runtime does not reproduce the job's identity (dataset, contract, evaluator, "
            "model, pricing or optimizer versions)"
        )
    plan = definition.plan
    if binding.models is not None:
        entry = binding.models.admit(plan.expected_model_hash)
        if entry.identity_hash != record.model.registry_entry_hash:
            raise ReproductionMismatch("the model registry entry differs from the provenance's")
    elif not definition.synthetic:
        raise ReproductionMismatch("a real experiment replays only with its model registry")
    per_run = run_reservation(
        plan.budget, binding.suite.policy, binding.pricing, plan.expected_model_hash
    )
    records, units, calls = [], [], 0
    for unit in job.units:
        strategy = Strategy(unit.strategy)
        if (
            optimizer_config(strategy, plan)
            != record.run(unit.strategy, unit.seed).optimizer_config
        ):
            raise ReproductionMismatch(f"{unit.strategy}/{unit.seed}: optimizer config changed")
        meter = BudgetLedger(plan.budget, binding.pricing, per_run)
        evidence = EvidenceEvaluate(
            job_id, unit.run_id, store.attempts(job_id, *unit_key(unit)), meter.measure
        )
        evaluate = reexecute if reexecute is not None else evidence
        out = run_strategy(
            plan,
            strategy,
            unit.seed,
            binding.suite,
            evaluate,
            checker=binding.checker,
            evaluator_version=binding.evaluator_version,
            pricing=binding.pricing,
            optimizer=make_strategy(strategy, plan),
        )
        if reexecute is None and evidence.unconsumed():
            raise ReproductionMismatch(
                f"{unit.strategy}/{unit.seed}: {len(evidence.unconsumed())} stored run(s) were "
                "never reached by the replay"
            )
        if out["run_id"] != unit.run_id:
            raise ReproductionMismatch(f"{unit.strategy}/{unit.seed}: replay has another run_id")
        calls += out["usage"]["workflow_runs"] if reexecute is not None else 0
        records.append(out)
        units.append(
            {
                "strategy": unit.strategy,
                "seed": unit.seed,
                "run_id": unit.run_id,
                "candidates": out["usage"]["candidate_evaluations"],
                "workflow_runs": out["usage"]["workflow_runs"],
                "selected_genome_hash": (out["champion"] or {}).get("genome_hash"),
            }
        )
    body = assemble(
        plan,
        binding.suite,
        records,
        evaluator_version=binding.evaluator_version,
        synthetic=definition.synthetic,
        pricing=binding.pricing,
        provenance=definition.provenance,
    )
    return art, Replay(json.loads(json.dumps(body)), units, calls)


def unit_key(unit) -> tuple[str, int]:
    return unit.strategy, unit.seed


def _paths(diffs: Sequence[tuple[Any, ...]]) -> list[str]:
    return ["/" + "/".join(str(p) for p in d) for d in diffs[:10]]


def compare(stored: Mapping[str, Any], replayed: Mapping[str, Any], *, evidence: bool):
    """Field-class comparison of a stored and a replayed body (``ReproductionMismatch``)."""
    sci = differences(scientific(replayed), scientific(stored))
    if sci:
        raise ReproductionMismatch(f"scientific fields do not reproduce: {_paths(sci)}")
    measured = differences(project(replayed, (MEASURED,)), project(stored, (MEASURED,)))
    if evidence and measured:
        raise ReproductionMismatch(
            f"measured telemetry does not replay from its stored evidence: {_paths(measured)}"
        )
    clock = differences(project(replayed, (CLOCK,)), project(stored, (CLOCK,)))
    counts = class_counts(stored)
    return {
        SCIENTIFIC: {"compared": counts[SCIENTIFIC], "equal": True},
        MEASURED: {
            "compared": counts[MEASURED] if evidence else 0,
            "equal": True if evidence else None,
            "policy": "replayed from stored runs" if evidence else "not compared (re-measured)",
            "differing": len(measured),
        },
        CLOCK: {
            "compared": 0,
            "equal": None,
            "policy": "excluded: process-clock values are never reproduced",
            "differing": len(clock),
        },
    }


# -- views --------------------------------------------------------------------------------------
class _View(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProvenanceView(_View):
    job_id: str
    artifact_id: str
    provenance_id: str
    experiment_sha256: str
    promotion_id: str | None
    provenance: dict[str, Any]


class TraceView(_View):
    job_id: str
    path: str
    artifact_id: str
    experiment_id: str
    experiment_sha256: str
    provenance_id: str
    document: str  # experiment | provenance | artifact | promotion
    pointer: str  # RFC 6901 pointer into that document
    value: Any
    field_class: str
    run: dict[str, Any] | None
    evidence: dict[str, Any] | None
    derivation: dict[str, Any] | None
    links: list[dict[str, Any]] = []


class VerificationView(_View):
    job_id: str
    artifact_id: str
    provenance_id: str
    experiment_sha256: str
    scientific_sha256: str
    verified: bool
    checks: list[str]
    declared_metric_checks: int
    field_classes: dict[str, int]
    promotion: dict[str, Any] | None


class ReproductionView(_View):
    job_id: str
    artifact_id: str
    provenance_id: str
    experiment_id: str
    mode: str
    reproduced: bool
    model_calls: int
    declared_metric_checks: int
    units: list[dict[str, Any]]
    fields: dict[str, Any]
    semantics: str


EVIDENCE_SEMANTICS = (
    "Every strategy run was re-executed by the unchanged runner from the job's stored "
    "write-ahead attempts: no model call. Scientific fields and measured telemetry reproduce "
    "exactly from that evidence; process-clock telemetry is excluded. Model outputs are not "
    "regenerated: a hosted model may be nondeterministic, so its stored outputs are the evidence."
)


# -- the service --------------------------------------------------------------------------------
@dataclass
class ExperimentArtifacts:
    jobs: ExperimentJobs
    champions: SQLiteChampionStore | None = None

    def _canonical(self, job_id: str) -> CanonicalArtifact:
        return load_canonical(self.jobs.store, job_id)[1]

    def _decision(self, job_id: str, art: CanonicalArtifact) -> dict[str, Any] | None:
        """The job's decided promotion record, checked to cite exactly this artifact."""
        if self.champions is None:
            return None
        row = self.champions.for_job(job_id)
        if row is None or row.decision_json is None:
            return None
        record = json.loads(row.decision_json)
        cited = record.get("artifact")
        if cited is not None and cited != art.ref():
            raise PromotionCitationMismatch(
                f"promotion {row.promotion_id} cites artifact {cited.get('artifact_id')}, not "
                f"{art.artifact_id}"
            )
        return record

    def _promotion_id(self, job_id: str) -> str | None:
        row = self.champions.for_job(job_id) if self.champions is not None else None
        return row.promotion_id if row is not None else None

    def provenance(self, job_id: str) -> ProvenanceView:
        art = self._canonical(job_id)
        self._decision(job_id, art)
        return ProvenanceView(
            job_id=job_id,
            artifact_id=art.artifact_id,
            provenance_id=art.provenance_id,
            experiment_sha256=art.experiment_sha256,
            promotion_id=self._promotion_id(job_id),
            provenance=art.envelope["provenance"],
        )

    def trace(self, job_id: str, path: str) -> TraceView:
        art = self._canonical(job_id)
        if path.split(".")[0] == "promotion":
            return self._trace_promotion(job_id, art, path)
        try:
            out = trace(art, path)
        except TraceError as exc:
            raise TraceNotFound(str(exc)) from None
        return TraceView(job_id=job_id, **out)

    def _trace_promotion(self, job_id: str, art: CanonicalArtifact, path: str) -> TraceView:
        record = self._decision(job_id, art)
        if record is None:
            raise TraceNotFound(f"experiment {job_id} has no decided promotion")
        try:
            value, parts = _walk(record, path.split(".")[1:])
        except TraceError as exc:
            raise TraceNotFound(str(exc)) from None
        links = []
        challenger = record.get("challenger")
        if challenger is not None and parts[:1] == ("challenger",):
            run_path = f"strategy.{challenger['strategy']}.seed.{challenger['seed']}"
            selected = trace(art, f"{run_path}.champion.genome_hash")["value"]
            links.append(
                {
                    "path": f"{run_path}.champion",
                    "relation": "validation-selected workflow of the challenger's strategy run",
                    "consistent": selected == challenger["genome_hash"]
                    and trace(art, f"{run_path}.run_id")["value"] == challenger["run_id"],
                }
            )
        return TraceView(
            job_id=job_id,
            path=path,
            artifact_id=art.artifact_id,
            experiment_id=art.body["experiment_id"],
            experiment_sha256=art.experiment_sha256,
            provenance_id=art.provenance_id,
            document="promotion",
            pointer="".join(f"/{p}" for p in parts),
            value=value,
            field_class=SCIENTIFIC,
            run=None,
            evidence={
                "promotion_id": record["promotion_id"],
                "decision_hash": record["decision_hash"],
            },
            derivation=None,
            links=links,
        )

    def verify(self, job_id: str) -> VerificationView:
        art = self._canonical(job_id)  # hashes, links, provenance vs body / job / definition
        declared = _declared(job_id, art)
        record = self._decision(job_id, art)
        promotion = None
        if record is not None:
            promotion = {
                "promotion_id": record["promotion_id"],
                "decision": record["decision"],
                "decision_hash": record["decision_hash"],
                "cites_artifact": record.get("artifact") is not None,
            }
        return VerificationView(
            job_id=job_id,
            artifact_id=art.artifact_id,
            provenance_id=art.provenance_id,
            experiment_sha256=art.experiment_sha256,
            scientific_sha256=art.envelope["experiment"]["scientific_sha256"],
            verified=True,
            checks=[
                "envelope hashes to artifact_id",
                "result body matches its sha256 link and canonical form",
                "scientific_sha256 hashes the body's scientific fields",
                "provenance hashes to provenance_id",
                "provenance agrees with the body, the job identity and the job definition",
                "declared metrics re-derive from stored execution evidence",
                "a decided promotion cites this artifact",
            ],
            declared_metric_checks=declared,
            field_classes=class_counts(art.body),
            promotion=promotion,
        )

    def reproduce(self, job_id: str) -> ReproductionView:
        runtime = self.jobs.runtime
        if runtime is None:
            raise BackendUnavailable(
                "reproduction rebinds the experiment (contract, evaluator, model registry entry); "
                "this server has no experiment runtime"
            )
        art, rep = replay(self.jobs.store, job_id, runtime)
        fields = compare(art.body, rep.body, evidence=True)
        declared = _declared(job_id, art)
        return ReproductionView(
            job_id=job_id,
            artifact_id=art.artifact_id,
            provenance_id=art.provenance_id,
            experiment_id=art.body["experiment_id"],
            mode="stored_evidence",
            reproduced=True,
            model_calls=rep.model_calls,
            declared_metric_checks=declared,
            units=rep.units,
            fields=fields,
            semantics=EVIDENCE_SEMANTICS,
        )


def _declared(job_id: str, art: CanonicalArtifact) -> int:
    try:
        return recompute_declared(art.body)
    except ArtifactIntegrityError as exc:
        raise ArtifactCorrupted(f"job {job_id}: {exc}") from exc


# -- command line -------------------------------------------------------------------------------
def _frozen(args: argparse.Namespace) -> dict[str, Any]:
    art = load_frozen(Path(args.frozen))
    if args.command == "trace":
        return trace(art, args.path)
    if args.command == "provenance":
        return {"artifact_id": art.artifact_id, "provenance": art.envelope["provenance"]}
    checks = recompute_declared(art.body)  # verify / reproduce: no stored attempts to replay
    return {
        "artifact_id": art.artifact_id,
        "provenance_id": art.provenance_id,
        "migration": art.envelope["provenance"]["migration"],
        "declared_metric_checks": checks,
        "field_classes": class_counts(art.body),
        "mode": "declared metrics from the per-run evidence in the artifact",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="verify, trace and reproduce experiment artifacts")
    p.add_argument("command", choices=("verify", "trace", "provenance", "reproduce"))
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", type=Path, help="a product server's data dir (jobs.sqlite3)")
    src.add_argument("--frozen", type=Path, help="a committed write_compact result directory")
    p.add_argument("--job", help="experiment job id (with --data-dir)")
    p.add_argument(
        "--workspace",
        help="#32: the workspace whose partition holds the job (with a multi-tenant --data-dir)",
    )
    p.add_argument("--path", help="field path to trace, e.g. strategy.aco.seed.1.usage.tokens")
    p.add_argument("--model-registry", type=Path, help="model registry (reproduce)")
    args = p.parse_args(argv)
    if args.command == "trace" and not args.path:
        p.error("trace needs --path")
    try:
        if args.frozen is not None:
            out = _frozen(args)
        else:
            if not args.job:
                p.error("--data-dir needs --job")
            out = (
                _service(args)
                .__getattribute__(args.command)(
                    *((args.job, args.path) if args.command == "trace" else (args.job,))
                )
                .model_dump(mode="json")
            )
    except (JobError, ValueError, LookupError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


def _service(args: argparse.Namespace) -> ExperimentArtifacts:
    root: Path = args.data_dir
    ws: str | None = getattr(args, "workspace", None)
    if ws is not None:  # every store of the partition is bound to (and checked against) ws
        root = root / "workspaces" / ws
    runtime = None
    if args.model_registry is not None:
        import os

        from api.product import registry_runtime
        from ingestion.service import DatasetService
        from store.blobs import LocalBlobStore
        from store.datasets import SQLiteDatasetRepository

        service = DatasetService(
            SQLiteDatasetRepository(root / "metadata.sqlite3", workspace_id=ws),
            LocalBlobStore(root / "blobs", workspace_id=ws),
        )
        runtime = registry_runtime(
            service, args.model_registry, os.environ.get("WYNK_MODEL_API_KEY")
        )
    champions = root / "champions.sqlite3"
    return ExperimentArtifacts(
        ExperimentJobs(SQLiteJobStore(root / "jobs.sqlite3", workspace_id=ws), runtime),
        SQLiteChampionStore(champions, workspace_id=ws) if champions.exists() else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
