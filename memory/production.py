"""Production workflow memory (#28): resolve, verify, learn and invalidate priors.

    resolve        job creation, for a plan with ``Strategy.ACO_WARM``: build the expected
                   validity keys from the job's authorities -> load the candidate priors ->
                   verify each one's provenance (record hash, dataset liveness, the #26 canonical
                   artifact it was learned from, the key re-derived from that provenance, the
                   learned pheromones re-derived from the job's checkpoint log) -> warm start only
                   on EXACT key equality and disjoint held-out rows -> otherwise a true cold start.
                   The result is a ``WarmStart`` pin with a structured reason (``StartReason``).
                   No model is called: resolving consumes no budget.
    verify_pin     execution of a warm unit: the pinned prior still exists, verifies, still
                   produces exactly the pinned state, and has not been invalidated.
    learn          after a job's canonical artifact is finalized: one dataset champion (with its
                   source run's feasible per-row champions, #10) and one task-family prior from
                   its VALIDATED evidence, or a recorded refusal.
    elite          product routing of one row: the compatible stored per-row champion, else the
                   dataset champion, else none - found and verified exactly like a warm start.
    invalidate_deleted
                   after a #33 deletion: every prior depending on a deleted dataset version is
                   retired and invalidated (history is kept).

Learning rules - production memory is never learned from:
    * a synthetic experiment, runtime or evaluator, or a migrated / frozen artifact (fixtures);
    * a job that is not COMPLETED with a verified canonical artifact (failed, cancelled,
      incomplete or corrupted jobs never reach ``load_canonical``'s success path);
    * an experiment that executed any test row, or whose source dataset has been deleted;
    * held-out evidence of any kind: only the job store (optimization-row feedback through the
      ACO checkpoint log, validation-row champion records) is read - never a promotion;
    * live web search (#12): a job bound with a websearch provider, or any candidate that
      gathered with one. Search results change over time and the returned payload is not
      persisted as a content-addressed snapshot, so its outcomes are not reproducible evidence
      (refusal ``live_websearch_evidence``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from core.canonical import canonical_hash
from core.evidence_authority import live_websearch
from core.provenance import ProvenanceRecord
from core.results import runtime_identity
from core.task_contract import TaskContract
from experiments.jobs import (
    ArtifactCorrupted,
    ExperimentJobDefinition,
    JobBinding,
    JobError,
    load_canonical,
)
from experiments.optimization_experiment import Strategy
from experiments.provenance import CanonicalArtifact
from experiments.task_champions import artifact_task_runs, rederive, usable
from memory.priors import (
    EVIDENCE_PROVIDER,
    WARM,
    Champion,
    DatasetRef,
    Diagnostic,
    ElitePick,
    EliteReason,
    Evidence,
    Pheromones,
    PriorKind,
    PriorRecord,
    PriorUnavailable,
    RetireReason,
    Source,
    StartMode,
    StartReason,
    TaskChampion,
    WarmStart,
    aco_config,
    anchor,
    classify,
    cold,
    elite_for,
    evidence_rows,
    heldout_rows,
    mismatches,
    row_hash,
    source_provider,
    validity_key,
    warm,
)
from optimizers.aco_mmas import MMASACO, ACOConfig
from optimizers.checkpoint import OptimizerStateError, restore
from store.jobs import JobState, SQLiteJobStore
from store.priors import Learned, PriorIntegrityError, SQLitePriorStore

log = logging.getLogger("wynk.memory.production")

ACO_STRATEGIES = (Strategy.ACO.value, Strategy.ACO_WARM.value)
SourceLive = Callable[[DatasetRef], bool]
_INVALID = {
    RetireReason.DATASET_DELETED: StartReason.COLD_DATASET_DELETED,
    RetireReason.PROVENANCE_INVALID: StartReason.COLD_PROVENANCE_INVALID,
}


class PriorService:
    """One workspace's production memory over its prior store and its job store.

    ``source_live(ref)`` is the #33 authority: is that dataset version's source still held?"""

    def __init__(
        self, store: SQLitePriorStore, jobs: SQLiteJobStore, *, source_live: SourceLive
    ) -> None:
        if store.workspace_id != getattr(jobs, "workspace_id", None):
            raise ValueError("the prior store and the job store belong to different workspaces")
        self.store = store
        self.jobs = jobs
        self.source_live = source_live
        self.workspace_id = store.workspace_id or "unscoped"

    # == keys =====================================================================================
    def expected_keys(
        self, definition: ExperimentJobDefinition, binding: JobBinding
    ) -> dict[PriorKind, dict[str, Any]] | None:
        """The keys a prior must have to warm-start this job (``None``: the runtime declares no
        RunVersions, so no runtime identity can be compared - never warm). The live evidence
        provider is the binding's (``JobBinding.evidence_provider``): the pinned JEV identity,
        or ``None`` when the runtime has no external evidence provider."""
        if binding.run_versions is None:
            return None
        plan = definition.plan
        return {
            kind: validity_key(
                kind,
                workspace_id=self.workspace_id,
                contract=definition.contract,
                splits_hash=definition.splits.identity_hash,
                evaluator_version=binding.evaluator_version,
                model_hash=plan.expected_model_hash,
                model_config_hash=plan.model.config_hash,
                run_versions=binding.run_versions,
                optimizer=(MMASACO.name, MMASACO.version),
                optimizer_config=aco_config(plan.lcb_z),
                evidence_provider=binding.evidence_provider,
            )
            for kind in PriorKind
        }

    def key_from_provenance(
        self,
        kind: PriorKind,
        record: ProvenanceRecord,
        strategy: str,
        seed: int,
        run_versions: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """The validity key of a prior learned from the strategy run ``(strategy, seed)`` of the
        experiment ``record`` describes - every component read from the #26 provenance. The
        evidence provider is the source job's bound authority the provenance pins
        (``versions.evidence_provider``, cross-checked with the job identity), checked against
        every candidate the experiment and that run reported (``run_versions``, verified with
        the artifact) and the warm start its protocol pins (``source_provider``)."""
        if record.contract.contract is None:
            raise ValueError("the provenance carries no TaskContract")
        try:  # every run of the experiment shares it; only the evidence source may differ
            shared = runtime_identity(record.versions.run_versions)
        except ValueError as exc:
            raise ValueError("the experiment reported no single runtime identity") from exc
        run = record.run(strategy, seed)
        warm = strategy == Strategy.ACO_WARM.value
        provider = source_provider(
            record.versions.evidence_provider,
            (*record.versions.run_versions, *run_versions),
            record.budget.protocol.get("warm_start") if warm else None,
        )
        if kind is PriorKind.TASK_FAMILY:
            optimizer, config = (run.optimizer, run.optimizer_version), run.optimizer_config
        else:  # a champion warm-starts THIS code's MMAS under the experiment's ACO parameters
            lcb_z = float(record.budget.protocol["lcb_z"])
            optimizer, config = (MMASACO.name, MMASACO.version), aco_config(lcb_z)
        return validity_key(
            kind,
            workspace_id=self.workspace_id,
            contract=TaskContract.model_validate(record.contract.contract),
            splits_hash=record.splits.splits_hash,
            evaluator_version=record.evaluator.run_version,
            model_hash=record.model.model_hash,
            model_config_hash=record.model.config_hash,
            run_versions=shared,
            optimizer=optimizer,
            optimizer_config=config or {},
            evidence_provider=provider,
        )

    # == resolve ==================================================================================
    def resolve(self, definition: ExperimentJobDefinition, binding: JobBinding) -> WarmStart:
        """How ``Strategy.ACO_WARM`` starts in this job (see the module docstring)."""
        if definition.synthetic:
            return cold(StartReason.COLD_SYNTHETIC)
        keys = self.expected_keys(definition, binding)
        if keys is None:
            return cold(StartReason.COLD_RUNTIME_UNDECLARED)
        expected = {kind.value: canonical_hash(key) for kind, key in keys.items()}
        heldout = heldout_rows(definition.contract, definition.splits)
        found: list[tuple[Diagnostic, PriorRecord | None]] = [
            self._find(kind, keys[kind], heldout) for kind in PriorKind
        ]
        diagnostics = tuple(d for d, _ in found)
        config = ACOConfig(lcb_z=definition.plan.lcb_z)
        for _, record in found:  # dataset champion first, then task family
            if record is not None:
                return warm(record, config, expected=expected, diagnostics=diagnostics)
        reason = next(
            (d.reason for d in diagnostics if d.reason is not StartReason.COLD_NO_PRIOR),
            StartReason.COLD_NO_PRIOR,
        )
        return cold(reason, expected=expected, diagnostics=diagnostics)

    def _find(
        self,
        kind: PriorKind,
        key: Mapping[str, Any],
        heldout: set[str],
        ignore: frozenset[str] = frozenset(),
    ) -> tuple[Diagnostic, PriorRecord | None]:
        """The usable prior of ``kind`` whose key is exactly ``key`` (except the ``ignore``d
        fields, which a caller judges itself), or the diagnostic of why there is none."""
        exact_failure: Diagnostic | None = None  # a prior with this exact key that is unusable
        other_failure: Diagnostic | None = None
        closest: Diagnostic | None = None
        for prior_id, row in self.store.candidates(kind, anchor(kind, key)):
            if row is None:  # the stored record does not hash to its id
                self.store.invalidate(prior_id, RetireReason.PROVENANCE_INVALID, "record hash")
                other_failure = other_failure or Diagnostic(
                    kind=kind, reason=StartReason.COLD_PROVENANCE_INVALID, prior_id=prior_id
                )
                continue
            bad = tuple(f for f in mismatches(row.record.key, key) if f not in ignore)
            if not row.invalidated and not row.active:
                continue  # superseded / dominated: history only
            reason: StartReason | None = None
            if row.invalidated:
                assert row.invalidated_reason is not None
                reason = _INVALID[row.invalidated_reason]
            elif problem := self._unusable(row.record):
                retire, reason = problem
                self.store.invalidate(prior_id, retire, reason.value)
                reason = _INVALID[retire]
            if reason is not None:
                diag = Diagnostic(kind=kind, reason=reason, prior_id=prior_id, mismatched=bad)
                if not bad:
                    exact_failure = exact_failure or diag
                else:
                    other_failure = other_failure or diag
                continue
            if bad:
                diag = Diagnostic(
                    kind=kind, reason=classify(bad), prior_id=prior_id, mismatched=bad
                )
                if closest is None or len(bad) < len(closest.mismatched):
                    closest = diag
                continue
            if set(row.record.evidence.rows) & heldout:  # its evidence is held out here
                closest = Diagnostic(
                    kind=kind, reason=StartReason.COLD_HELDOUT_OVERLAP, prior_id=prior_id
                )
                continue
            return Diagnostic(kind=kind, reason=WARM[kind][1], prior_id=prior_id), row.record
        for diag in (exact_failure, closest, other_failure):
            if diag is not None:
                return diag, None
        return Diagnostic(kind=kind, reason=StartReason.COLD_NO_PRIOR), None

    def _unusable(self, record: PriorRecord) -> tuple[RetireReason, StartReason] | None:
        """Does the prior's authority still hold? (dataset sources, then provenance)"""
        if not all(self.source_live(d) for d in record.evidence.datasets):
            return RetireReason.DATASET_DELETED, StartReason.COLD_DATASET_DELETED
        problem = self.verify_provenance(record)
        if problem is not None:
            log.warning("prior %s fails provenance: %s", record.prior_id, problem)
            return RetireReason.PROVENANCE_INVALID, StartReason.COLD_PROVENANCE_INVALID
        return None

    def verify_provenance(self, record: PriorRecord) -> str | None:
        """``None`` if ``record`` is exactly what its source experiment's VERIFIED canonical
        artifact (and, for pheromones, its job's checkpoint log) teaches; else why not."""
        src = record.source
        try:
            job, art = load_canonical(self.jobs, src.job_id)
        except JobError as exc:
            return f"source artifact: {exc}"
        prov = art.provenance
        pinned = {
            "artifact_id": (art.artifact_id, src.artifact_id),
            "provenance_id": (art.provenance_id, src.provenance_id),
            "experiment_id": (art.body["experiment_id"], src.experiment_id),
            "definition_hash": (job.definition_hash, src.definition_hash),
        }
        if bad := [k for k, (a, b) in pinned.items() if a != b]:
            return "source identity disagrees: " + ", ".join(bad)
        if refusal := _ineligible(prov, art):
            return refusal
        try:
            if prov.run(src.strategy, src.seed).run_id != src.run_id:
                return "source run identity disagrees"
            run = _run(art, src.strategy, src.seed)
            key = self.key_from_provenance(
                record.kind, prov, src.strategy, src.seed, run["run_versions"]
            )
            if key != record.key:
                return "the validity key does not derive from the provenance"
        except (KeyError, ValueError) as exc:
            return f"provenance: {exc}"
        if record.champion is not None:
            champ = run["champion"]
            if champ is None or champ["genome_hash"] != record.champion.genome_hash:
                return "the champion is not the source run's validation-selected workflow"
            # per-row champions must be what the run's own validation runs select (#10)
            contract = TaskContract.model_validate(prov.contract.contract)
            if bad := rederive(run, contract):
                return "task champions do not re-derive: " + "; ".join(bad[:3])
            try:
                learned = _champion(run, prov.dataset.dataset_id)
            except ValueError as exc:
                return f"champion: {exc}"
            if learned != record.champion:
                return "the champion's validation evidence disagrees with the artifact"
        if record.pheromones is not None:
            try:
                learned = self._final_pheromones(art, src.job_id, src.strategy, src.seed)
            except ValueError as exc:
                return f"pheromones: {exc}"
            if learned != record.pheromones:
                return "the pheromones do not derive from the source run's checkpoint log"
        return None

    def _final_pheromones(
        self, art: CanonicalArtifact, job_id: str, strategy: str, seed: int
    ) -> Pheromones:
        """The pheromone state the ACO unit ended with: its last checkpointed optimizer state,
        restored exactly and cross-checked against the artifact's candidate log."""
        state = None
        for row in self.jobs.checkpoints(job_id, strategy, seed):
            payload = json.loads(row.payload_json)
            if payload.get("optimizer_state") is not None:
                state = payload["optimizer_state"]
        if state is None:
            raise ValueError("the unit checkpointed no optimizer state")
        try:
            opt = restore(state)
        except (OptimizerStateError, KeyError, TypeError) as exc:
            raise ValueError(f"checkpoint does not restore: {exc}") from None
        if not isinstance(opt, MMASACO) or opt.version != MMASACO.version:
            raise ValueError("the unit's optimizer is not this MMAS")
        run = _run(art, strategy, seed)
        evaluated = set(run["evaluated_genome_hashes"])
        if not evaluated <= set(opt._genomes) or not set(opt._board.hashes()) <= evaluated:
            raise ValueError("the checkpointed optimizer disagrees with the artifact's candidates")
        return Pheromones.of(opt)

    # == elites (#10) =============================================================================
    def elite(
        self,
        definition: ExperimentJobDefinition,
        binding: JobBinding,
        task_id: str,
        *,
        evidence_source: str | None = None,
    ) -> ElitePick:
        """The stored workflow a run on row ``task_id`` of this job's dataset context starts
        from: that row's task champion, else the dataset champion, else none.

        The dataset-champion prior is found exactly as a warm start finds it (``_find``: exact
        validity key in THIS workspace, live dataset sources, verified provenance, no held-out
        overlap), so a deleted, stale, invalidated or incompatible prior never supplies an elite -
        except for the prior's evidence provider: every elite is judged by its OWN runs' provider
        against ``evidence_source``, the live evidence provider the runtime would use now (the
        pinned JEV identity, or ``None``), so a fetch elite of a mixed search stays usable. A hint
        only: whatever it returns is executed and measured
        like any other workflow; nothing here calls a model."""
        kind = PriorKind.DATASET_CHAMPION
        if definition.synthetic:
            diag = Diagnostic(kind=kind, reason=StartReason.COLD_SYNTHETIC)
            return ElitePick(reason=EliteReason.NO_COMPATIBLE_PRIOR, diagnostic=diag)
        keys = self.expected_keys(definition, binding)
        if keys is None:
            diag = Diagnostic(kind=kind, reason=StartReason.COLD_RUNTIME_UNDECLARED)
            return ElitePick(reason=EliteReason.NO_COMPATIBLE_PRIOR, diagnostic=diag)
        heldout = heldout_rows(definition.contract, definition.splits)
        diag, record = self._find(kind, keys[kind], heldout, frozenset({EVIDENCE_PROVIDER}))
        if record is None:
            return ElitePick(reason=EliteReason.NO_COMPATIBLE_PRIOR, diagnostic=diag)
        task = row_hash(definition.contract.dataset.dataset_id, task_id)
        pick = elite_for(record, task, evidence_source)
        return pick.model_copy(update={"diagnostic": diag})

    # == verify a pin at execution ================================================================
    def verify_pin(self, pin: WarmStart, lcb_z: float) -> None:
        """Raise ``PriorUnavailable`` unless a warm pin's prior still verifies and still produces
        exactly the pinned state, under the evidence provider the pin records. Never re-resolves:
        a pin is used exactly or not at all."""
        if pin.mode is StartMode.COLD:
            return
        assert pin.prior_id is not None and pin.state is not None
        state = self.store.state_of(pin.prior_id)
        if state is None:
            raise PriorUnavailable(f"pinned prior {pin.prior_id} does not exist")
        if state[1] is not None:
            raise PriorUnavailable(f"pinned prior {pin.prior_id} was invalidated ({state[1]})")
        try:
            row = self.store.get(pin.prior_id)
        except PriorIntegrityError as exc:
            raise PriorUnavailable(str(exc)) from None
        assert row is not None
        if not all(self.source_live(d) for d in row.record.evidence.datasets):
            self.store.invalidate(pin.prior_id, RetireReason.DATASET_DELETED)
            raise PriorUnavailable(f"pinned prior {pin.prior_id} depends on a deleted dataset")
        if row.record.state(ACOConfig(lcb_z=lcb_z)) != pin.state:
            raise PriorUnavailable(f"pinned prior {pin.prior_id} does not produce the pinned state")
        if row.record.key.get(EVIDENCE_PROVIDER) != pin.evidence_provider:
            raise PriorUnavailable(
                f"pinned prior {pin.prior_id} learned from another evidence provider than pinned"
            )

    # == learn ====================================================================================
    def learn(self, job_id: str) -> Learned | None:
        """Learn from one COMPLETED job's verified artifact - exactly once. ``None``: the job is
        not (yet) a finished experiment; nothing is recorded, so it can be learned later."""
        done = self.store.learned(job_id)
        if done is not None:
            return done
        try:
            job, art = load_canonical(self.jobs, job_id)
        except ArtifactCorrupted as exc:
            return self.store.learn(job_id, [], f"artifact_corrupted: {exc}")
        except JobError:
            return None
        definition = ExperimentJobDefinition.model_validate_json(job.definition_json)
        prov = art.provenance
        own = DatasetRef(
            dataset_id=prov.dataset.dataset_id,
            dataset_version=prov.dataset.dataset_version,
            content_hash=prov.dataset.content_hash,
        )
        refusal = ("synthetic" if definition.synthetic else _ineligible(prov, art)) or (
            None if self.source_live(own) else "source_dataset_deleted"
        )
        if refusal is not None:
            return self.store.learn(job_id, [], refusal)
        try:
            records = self._records(job_id, definition, art, own)
        except PriorUnavailable as exc:
            return self.store.learn(job_id, [], f"parent_prior_unusable: {exc}")
        except ValueError as exc:
            return self.store.learn(job_id, [], f"evidence_invalid: {exc}")
        return self.store.learn(job_id, records, None if records else "no_validated_champion")

    def _records(
        self,
        job_id: str,
        definition: ExperimentJobDefinition,
        art: CanonicalArtifact,
        own: DatasetRef,
    ) -> list[PriorRecord]:
        prov = art.provenance
        rows = evidence_rows(definition.contract, definition.splits)
        out: list[PriorRecord] = []
        for kind in PriorKind:
            eligible = [
                r
                for r in art.body["runs"]
                if r["champion"] is not None
                and r["champion"]["feasible"]
                and (kind is PriorKind.DATASET_CHAMPION or r["strategy"] in ACO_STRATEGIES)
            ]
            if not eligible:
                continue
            # highest contract rank; ties: first in artifact order (seed-major, plan order)
            best = max(eligible, key=lambda r: tuple(r["champion"]["rank_key"]))
            run = prov.run(best["strategy"], best["seed"])
            parent = self._parent(definition, best["strategy"])
            evidence = Evidence(
                datasets=tuple(
                    sorted(
                        {own, *(parent.evidence.datasets if parent else ())},
                        key=lambda d: (d.dataset_id, d.dataset_version, d.content_hash),
                    )
                ),
                rows=tuple(sorted(rows | set(parent.evidence.rows if parent else ()))),
                parent_prior_id=parent.prior_id if parent else None,
            )
            key = self.key_from_provenance(
                kind, prov, best["strategy"], best["seed"], best["run_versions"]
            )
            out.append(
                PriorRecord(
                    kind=kind,
                    key=key,
                    key_hash=canonical_hash(key),
                    source=Source(
                        job_id=job_id,
                        artifact_id=art.artifact_id,
                        experiment_id=art.body["experiment_id"],
                        provenance_id=art.provenance_id,
                        definition_hash=definition.definition_hash,
                        strategy=best["strategy"],
                        seed=best["seed"],
                        run_id=run.run_id,
                    ),
                    evidence=evidence,
                    champion=_champion(best, prov.dataset.dataset_id)
                    if kind is PriorKind.DATASET_CHAMPION
                    else None,
                    pheromones=self._final_pheromones(art, job_id, best["strategy"], best["seed"])
                    if kind is PriorKind.TASK_FAMILY
                    else None,
                )
            )
        return out

    def _parent(self, definition: ExperimentJobDefinition, strategy: str) -> PriorRecord | None:
        """The prior a warm-started source run carried forward (its evidence is inherited)."""
        pin = definition.plan.warm_start
        if strategy != Strategy.ACO_WARM.value or pin is None or pin.prior_id is None:
            return None
        self.verify_pin(pin, definition.plan.lcb_z)
        row = self.store.get(pin.prior_id)
        assert row is not None
        return row.record

    def learn_pending(self) -> list[Learned]:
        """Learn every COMPLETED job with a finalized artifact that learning never decided on
        (e.g. the process stopped between finalizing and learning)."""
        out = []
        for job_id in self.jobs.list_job_ids():
            job = self.jobs.get_job(job_id)
            if job is None or job.state is not JobState.COMPLETED:
                continue
            if self.store.learned(job_id) is None and self.jobs.artifact(job_id) is not None:
                learned = self.learn(job_id)
                if learned is not None:
                    out.append(learned)
        return out

    # == invalidation =============================================================================
    def invalidate_deleted(self) -> list[str]:
        """Retire + invalidate every prior that depends on a dataset version whose source is
        gone (#33). Deterministic and idempotent; nothing is deleted."""
        out = []
        for prior_id, refs in self.store.dataset_refs().items():
            if not all(self.source_live(r) for r in refs):
                if self.store.invalidate(prior_id, RetireReason.DATASET_DELETED):
                    out.append(prior_id)
        return out


# -- helpers ----------------------------------------------------------------------------------
def _run(art: CanonicalArtifact, strategy: str, seed: int) -> Mapping[str, Any]:
    for r in art.body["runs"]:
        if (r["strategy"], r["seed"]) == (strategy, seed):
            return r  # type: ignore[no-any-return]
    raise ValueError(f"the artifact has no run {strategy}/{seed}")


_STATS = ("runs", "score_mean", "pass_rate", "fitness_mean")


def _champion(run: Mapping[str, Any], dataset_id: str) -> Champion:
    """The dataset champion one strategy-run record teaches: its validation-selected workflow,
    the live evidence provider ITS runs used, and (#10) every FEASIBLE per-row champion of the
    same run with the provider ITS runs used - an infeasible per-row winner stays diagnostic
    evidence in the artifact only. One search may evaluate JEV and fetch/api genomes, so each
    elite keeps the evidence source of its own runs (``candidates[].run_versions``)."""
    champ = run["champion"]
    try:
        runtime_identity(run["run_versions"])
    except ValueError as exc:
        raise ValueError("the source run reported no single runtime identity") from exc
    runs = artifact_task_runs(run)  # each validation run with the RunVersions it ran under
    by_run = {tr.path: tr.evidence_source for tr in runs}
    champ_sources = {tr.evidence_source for tr in runs if tr.genome_hash == champ["genome_hash"]}
    if len(champ_sources) != 1:
        raise ValueError("the champion's runs report no single evidence source")
    tasks = []
    for t in run.get("task_champions") or ():  # absent: recorded before per-task selection
        sources = {by_run[p] for p in t["run_paths"]}
        if len(sources) != 1 or sorted(s for s in sources if s) != t["evidence_sources"]:
            raise ValueError(f"task {t['task_id']} disagrees with its runs' evidence source")
        if usable(t):
            tasks.append(
                TaskChampion(
                    task=row_hash(dataset_id, t["task_id"]),
                    genome=t["genome"],
                    genome_hash=t["genome_hash"],
                    rank_key=tuple(t["rank_key"]),
                    validation={k: t["validation"][k] for k in _STATS},
                    run_paths=tuple(t["run_paths"]),
                    evidence_source=sources.pop(),
                )
            )
    return Champion(
        genome=champ["genome"],
        genome_hash=champ["genome_hash"],
        rank_key=tuple(champ["rank_key"]),
        validation={k: champ["validation"][k] for k in _STATS},
        evidence_source=champ_sources.pop(),
        tasks=tuple(sorted(tasks, key=lambda t: t.task)),
    )


def _ineligible(prov: ProvenanceRecord, art: CanonicalArtifact) -> str | None:
    """Why this artifact is not real, validated, held-out-free production evidence."""
    if prov.synthetic or art.body.get("synthetic"):
        return "synthetic"
    if prov.migration is not None or art.envelope.get("parent", {}).get("kind") != "job":
        return "not_a_durable_job"  # frozen / migrated artifacts are fixtures, not evidence
    if prov.model.registry_entry is None:
        return "no_registry_entry"
    if not prov.evaluator.run_version.startswith(prov.evaluator.spec_version + "+"):
        return "synthetic_evaluator"
    if art.body.get("test_runs") != 0 or any(
        r["split_usage"]["test_runs"] for r in art.body["runs"]
    ):
        return "heldout_executed"
    if live_websearch(prov.versions.evidence_provider, prov.versions.run_versions):
        # live web-search results are not a frozen, content-addressed snapshot: never learned
        return "live_websearch_evidence"
    return None


def source_live_from(service: Any) -> SourceLive:
    """The #33 liveness authority over a ``DatasetService``: the version exists, its source
    upload is not deleted, and it still has the content the prior was learned from."""

    def live(ref: DatasetRef) -> bool:
        try:
            status = service.source_status(ref.dataset_id, ref.dataset_version)
        except Exception:  # NotFound, SourceDeleted, integrity errors: not live
            return False
        return status.status == "available" and status.content_hash == ref.content_hash

    return live


__all__ = ["PriorService", "source_live_from"]
