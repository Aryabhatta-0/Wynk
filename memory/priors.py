"""Production workflow-memory priors (#28): validity keys, prior records, warm-start pins.

Two levels of memory, both HINTS only - a warm-started search is still measured, validated,
held-out gated and promoted exactly like a cold one:

    dataset champion   the best VALIDATED workflow of one exact dataset context (dataset
                       version + splits + TaskContract + grammar + evaluator + model + runtime +
                       optimizer). Warm start: the MMAS fixed point of a colony converged on the
                       champion's path (``champion_state``) - its edges at ``tau_max``, every
                       other edge evaporated to ``tau_min``. The champion is re-evaluated under
                       the new experiment's budget like any other candidate; nothing is trusted.
    task family        the learned MMAS pheromone state (edges + epoch) of one ACO run, reusable
                       on any dataset of the same task family (same validity key minus the
                       dataset). Warm start: ``MMASACO.from_pheromones`` - exactly the saved
                       optimizer's pheromones; its score board is never carried over.

Validity key (``validity_key``). Every identity that can change search semantics, read from the
canonical authorities (TaskContract, DatasetSplits, ExperimentPlan, the bound evaluator and the
runtime's declared ``RunVersions``) - or, for a stored prior, from the #26 ``ProvenanceRecord``
of the artifact it was learned from (``key_from_provenance``). Both derivations give the same
dict for the same context. Loose names (``task_id``) only route the lookup of candidates and
the diagnosis of a cold start; compatibility is EXACT equality of the whole key.

Warm-start pin (``WarmStart``). Resolved once, when a job is created (``memory.production``),
and pinned into the ``ExperimentPlan``: the mode, a structured reason, the prior identity and
the exact pheromone state. The plan is hashed into the experiment and run identities, so a
warm run is reproducible from its plan alone and can never be mistaken for a cold one. Only
``Strategy.ACO_WARM`` ever reads a pin; ``fixed`` / ``random`` / ``aco`` are always cold.

This module is pure (no I/O). Persistence: ``store.priors``. Resolution, provenance
verification, learning and invalidation: ``memory.production``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from core.canonical import canonical_hash, canonical_json
from core.dataset import DatasetSplits, SplitRole
from core.genome import Genome
from core.provenance import grammar_hash
from core.task_contract import TaskContract, workflow_grammar
from optimizers.aco_mmas import MMASACO, ACOConfig
from optimizers.construct import path_edges

KEY_SCHEMA = "wynk-prior-key/1"
PRIOR_SCHEMA = "wynk-workflow-prior/1"
PIN_SCHEMA = "wynk-warm-start/1"


class PriorKind(StrEnum):
    DATASET_CHAMPION = "dataset_champion"
    TASK_FAMILY = "task_family"


class StartMode(StrEnum):
    COLD = "cold"
    WARM_DATASET_CHAMPION = "warm_dataset_champion"
    WARM_TASK_FAMILY = "warm_task_family"


class StartReason(StrEnum):
    """Why a search started the way it did. Stable strings: part of the artifact and API."""

    WARM_DATASET_CHAMPION = "warm_dataset_champion"
    WARM_TASK_FAMILY = "warm_task_family"
    COLD_BASELINE = "cold_baseline"  # the strategy never consults memory (fixed/random/aco)
    COLD_NO_PRIOR = "cold_no_prior"
    COLD_DATASET_CHANGED = "cold_dataset_changed"
    COLD_CONTRACT_CHANGED = "cold_contract_changed"
    COLD_GRAMMAR_CHANGED = "cold_grammar_changed"
    COLD_EVALUATOR_CHANGED = "cold_evaluator_changed"
    COLD_MODEL_CHANGED = "cold_model_changed"
    COLD_RUNTIME_CHANGED = "cold_runtime_changed"  # prompts / compiler / runtime versions
    COLD_OPTIMIZER_CHANGED = "cold_optimizer_changed"
    COLD_PROVENANCE_INVALID = "cold_provenance_invalid"
    COLD_DATASET_DELETED = "cold_dataset_deleted"
    COLD_HELDOUT_OVERLAP = "cold_heldout_overlap"  # a prior's evidence rows are test rows here
    COLD_SYNTHETIC = "cold_synthetic"  # a synthetic experiment never uses production memory
    COLD_RUNTIME_UNDECLARED = "cold_runtime_undeclared"  # the runtime declares no RunVersions
    COLD_MEMORY_UNAVAILABLE = "cold_memory_unavailable"  # no prior store in this process


class RetireReason(StrEnum):
    SUPERSEDED = "superseded"  # newer validated evidence for the same key
    DOMINATED = "dominated"  # a dataset champion that never beat the active one
    DATASET_DELETED = "dataset_deleted"  # a source dataset version was deleted (#33)
    PROVENANCE_INVALID = "provenance_invalid"  # its source artifact no longer verifies


# A prior retired for one of these reasons is unusable, also by a job that already pinned it.
INVALIDATING = frozenset({RetireReason.DATASET_DELETED, RetireReason.PROVENANCE_INVALID})

WARM: dict[PriorKind, tuple[StartMode, StartReason]] = {
    PriorKind.DATASET_CHAMPION: (
        StartMode.WARM_DATASET_CHAMPION,
        StartReason.WARM_DATASET_CHAMPION,
    ),
    PriorKind.TASK_FAMILY: (StartMode.WARM_TASK_FAMILY, StartReason.WARM_TASK_FAMILY),
}


class PriorUnavailable(ValueError):
    """A pinned prior can no longer be used (invalidated, missing or altered): fail closed."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# == validity key =================================================================================
# Key fields by the cold-start reason they explain, finest first. ``task_contract_hash`` is an
# aggregate of the contract components above it: it alone decides only if nothing finer differs.
CATEGORIES: tuple[tuple[StartReason, tuple[str, ...]], ...] = (
    (StartReason.COLD_PROVENANCE_INVALID, ("schema", "kind", "workspace_id")),
    (
        StartReason.COLD_DATASET_CHANGED,
        (
            "dataset_id",
            "dataset_version",
            "dataset_content_hash",
            "dataset_identity_hash",
            "splits_hash",
        ),
    ),
    (StartReason.COLD_CONTRACT_CHANGED, ("task_id", "task_semantics_hash")),
    (StartReason.COLD_GRAMMAR_CHANGED, ("workflow_hash", "grammar_version", "grammar_hash")),
    (StartReason.COLD_EVALUATOR_CHANGED, ("evaluation_hash", "evaluator_version")),
    (StartReason.COLD_MODEL_CHANGED, ("model_hash", "model_config_hash")),
    (
        StartReason.COLD_RUNTIME_CHANGED,
        (
            "prompt_template_version",
            "compiler_version",
            "runtime_grammar_version",
            "benchmark_hash",
        ),
    ),
    (StartReason.COLD_OPTIMIZER_CHANGED, ("optimizer", "optimizer_version", "aco_config_hash")),
    (StartReason.COLD_CONTRACT_CHANGED, ("task_contract_hash",)),
)


def aco_config(lcb_z: float) -> dict[str, Any]:
    """The MMAS parameters an experiment builds its ACO with, as the provenance records them
    (``experiments.provenance.optimizer_config``: canonical JSON round trip)."""
    return json.loads(canonical_json(asdict(ACOConfig(lcb_z=lcb_z))))


def validity_key(
    kind: PriorKind,
    *,
    workspace_id: str,
    contract: TaskContract,
    splits_hash: str,
    evaluator_version: str,
    model_hash: str,
    model_config_hash: str,
    run_versions: Mapping[str, Any],
    optimizer: tuple[str, str],
    optimizer_config: Mapping[str, Any],
) -> dict[str, Any]:
    """The exact compatibility key of a prior of ``kind`` in this context.

    Task family: workspace, task semantics (the TaskContract minus its dataset binding,
    evaluation and workflow, which are separate components), workflow vocabulary + grammar,
    evaluator spec + version, exact model + configuration, prompts / compiler / runtime grammar,
    optimizer name + version + MMAS config. Dataset champion: all of that plus the exact dataset
    version (content + DatasetSpec identity), the splits (same validation AND test rows, so a
    champion is never selected on rows that are held out here), the whole TaskContract hash and
    the runtime's benchmark binding."""
    semantics = contract.authoritative()
    for part in ("dataset", "evaluation", "workflow"):
        semantics.pop(part)
    grammar = workflow_grammar(contract)
    kinds = tuple(k.value for k in contract.workflow.stages)
    key: dict[str, Any] = {
        "schema": KEY_SCHEMA,
        "kind": kind.value,
        "workspace_id": workspace_id,
        "task_id": contract.task_id,
        "task_semantics_hash": canonical_hash(semantics),
        "workflow_hash": canonical_hash(contract.workflow.model_dump(mode="json")),
        "grammar_version": grammar.version,
        "grammar_hash": grammar_hash(grammar.version, kinds),
        "evaluation_hash": contract.evaluation.identity_hash,
        "evaluator_version": evaluator_version,
        "model_hash": model_hash,
        "model_config_hash": model_config_hash,
        "prompt_template_version": run_versions["prompt_template_version"],
        "compiler_version": run_versions["compiler_version"],
        "runtime_grammar_version": run_versions["grammar_version"],
        "optimizer": optimizer[0],
        "optimizer_version": optimizer[1],
        "aco_config_hash": canonical_hash(dict(optimizer_config)),
    }
    if kind is PriorKind.DATASET_CHAMPION:
        ds = contract.dataset
        key |= {
            "dataset_id": ds.dataset_id,
            "dataset_version": ds.dataset_version,
            "dataset_content_hash": ds.content_hash,
            "dataset_identity_hash": ds.identity_hash,
            "splits_hash": splits_hash,
            "task_contract_hash": contract.contract_hash,
            "benchmark_hash": run_versions["benchmark_hash"],
        }
    return key


def anchor(kind: PriorKind, key: Mapping[str, Any]) -> str:
    """Which priors are looked at for a search: a dataset champion by (dataset, task id), a task
    family by task id. Never a compatibility decision - only exact key equality is."""
    parts = [key["workspace_id"], key["task_id"]]
    if kind is PriorKind.DATASET_CHAMPION:
        parts.append(key["dataset_id"])
    return canonical_hash(["prior-anchor", kind.value, *parts])


def mismatches(have: Mapping[str, Any], want: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k)))


def classify(fields: Iterable[str]) -> StartReason:
    """The cold-start reason of a key mismatch (finest category first)."""
    differing = set(fields)
    for reason, names in CATEGORIES:
        if differing & set(names):
            return reason
    return StartReason.COLD_PROVENANCE_INVALID  # a field no category knows: not this schema


# == evidence rows (held-out isolation) ===========================================================
def row_hash(dataset_id: str, row_id: str) -> str:
    """A stored stand-in for one dataset row: identifiers are never kept in clear."""
    return canonical_hash(["prior-row", dataset_id, row_id])


def _rows(contract: TaskContract, splits: DatasetSplits, roles: Iterable[SplitRole]) -> set[str]:
    out: set[str] = set()
    for role in roles:
        split = splits.split(role)
        if split is not None:
            out |= {row_hash(contract.dataset.dataset_id, r) for r in split.row_ids}
    return out


def evidence_rows(contract: TaskContract, splits: DatasetSplits) -> set[str]:
    """Every row whose outcome can reach a prior: optimization rows (pheromones) and validation
    rows (champion selection). Test rows never can."""
    return _rows(contract, splits, (SplitRole.OPTIMIZATION, SplitRole.VALIDATION))


def heldout_rows(contract: TaskContract, splits: DatasetSplits) -> set[str]:
    return _rows(contract, splits, (SplitRole.TEST,))


# == prior records ================================================================================
class TaskChampion(_Frozen):
    """The feasible workflow the contract ranks best on ONE validation row (#10), with that
    row's validation stats. An infeasible per-task winner is never stored: it stays diagnostic
    evidence in the source artifact (``task_champions``), never a usable elite."""

    task: str  # row_hash(dataset_id, row id): identifiers are never kept in clear
    genome: dict[str, Any]  # Genome.canonical()
    genome_hash: str
    feasible: Literal[True] = True
    rank_key: tuple[float, ...]  # TaskContract ranking on that row's runs; higher is better
    validation: dict[str, float | int]  # runs, score_mean, pass_rate, fitness_mean
    run_paths: tuple[str, ...]  # its validation runs inside the source strategy run (trace)
    # RunVersions.evidence_source of THOSE runs: one search may evaluate JEV and fetch/api
    # genomes, so a task champion's provider can differ from the dataset champion's. ``None``
    # (fetch/api/inline) is left out of the serialized form, so stored priors keep their ids.
    evidence_source: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _genome_matches(self) -> TaskChampion:
        if Genome.from_canonical(self.genome).genome_hash != self.genome_hash:
            raise ValueError("task champion genome does not match its hash")
        return self

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if data.get("evidence_source") is None:
            data.pop("evidence_source", None)
        return data


class Champion(_Frozen):
    """A dataset champion: one validation-selected, feasible workflow and its VALIDATION stats,
    plus (#10, additive) the per-row champions of the same source run and the live evidence
    provider that run used. Both new fields are left out of the serialized form when empty, so
    a champion stored before them keeps its content - and therefore its ``prior_id``."""

    genome: dict[str, Any]  # Genome.canonical()
    genome_hash: str
    feasible: Literal[True] = True
    rank_key: tuple[float, ...]  # TaskContract ranking; higher is better
    validation: dict[str, float | int]  # runs, score_mean, pass_rate, fitness_mean
    # RunVersions.evidence_source of this champion's runs in the source run (a GATHER jev
    # provider, pinned by #52); each task champion records the provider of its own runs.
    evidence_source: str | None = Field(default=None, min_length=1)
    tasks: tuple[TaskChampion, ...] = ()

    @model_validator(mode="after")
    def _genome_matches(self) -> Champion:
        if Genome.from_canonical(self.genome).genome_hash != self.genome_hash:
            raise ValueError("champion genome does not match its hash")
        if [t.task for t in self.tasks] != sorted({t.task for t in self.tasks}):
            raise ValueError("task champions must be sorted and distinct")
        return self

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if data.get("evidence_source") is None:
            data.pop("evidence_source", None)
        if not data.get("tasks"):
            data.pop("tasks", None)
        return data

    def task(self, task: str) -> TaskChampion | None:
        """The champion of the row whose ``row_hash`` is ``task`` (``None``: none stored)."""
        return next((t for t in self.tasks if t.task == task), None)


Edge = tuple[str, str, float]


class Pheromones(_Frozen):
    """MMAS pheromone state: every explored edge's current tau, and the epoch it is at."""

    epoch: int = Field(ge=0)
    edges: tuple[Edge, ...]

    @model_validator(mode="after")
    def _well_formed(self) -> Pheromones:
        pairs = [(a, b) for a, b, _ in self.edges]
        if pairs != sorted(set(pairs)):
            raise ValueError("pheromone edges must be sorted and distinct")
        if any(not (math.isfinite(t) and t > 0) for _, _, t in self.edges):
            raise ValueError("pheromone values must be finite and positive")
        return self

    def as_map(self) -> dict[tuple[str, str], float]:
        return {(a, b): t for a, b, t in self.edges}

    @classmethod
    def of(cls, optimizer: MMASACO) -> Pheromones:
        """What ``optimizer`` learned: every explored edge at its current value + the epoch."""
        return cls(
            epoch=optimizer.epoch,
            edges=tuple((a, b, t) for (a, b), t in optimizer.explored_pheromones().items()),
        )

    @property
    def state_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class Source(_Frozen):
    """The verified experiment a prior was learned from (#26 canonical artifact)."""

    job_id: str
    artifact_id: str
    experiment_id: str
    provenance_id: str
    definition_hash: str
    strategy: str
    seed: int
    run_id: str


class DatasetRef(_Frozen):
    dataset_id: str
    dataset_version: int
    content_hash: str


class Evidence(_Frozen):
    """Everything a prior's content depends on, transitively through the prior its source run
    was warm-started from: dataset versions (deletion invalidates) and hashed rows."""

    datasets: tuple[DatasetRef, ...]
    rows: tuple[str, ...]
    parent_prior_id: str | None = None

    @model_validator(mode="after")
    def _canonical(self) -> Evidence:
        refs = [(d.dataset_id, d.dataset_version, d.content_hash) for d in self.datasets]
        if refs != sorted(set(refs)) or list(self.rows) != sorted(set(self.rows)):
            raise ValueError("evidence must be sorted and distinct")
        return self


class PriorRecord(_Frozen):
    """One immutable prior. ``prior_id`` is the hash of its content: no timestamps, no uuids -
    learning the same evidence twice gives the same prior."""

    schema_: Literal["wynk-workflow-prior/1"] = Field(PRIOR_SCHEMA, alias="schema")
    kind: PriorKind
    key: dict[str, Any]
    key_hash: str
    source: Source
    evidence: Evidence
    champion: Champion | None = None
    pheromones: Pheromones | None = None
    synthetic: Literal[False] = False  # production memory is never learned from a stand-in

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _consistent(self) -> PriorRecord:
        if canonical_hash(self.key) != self.key_hash:
            raise ValueError("key_hash does not hash the validity key")
        if self.key.get("kind") != self.kind.value:
            raise ValueError("the validity key is for another kind of prior")
        champion = self.kind is PriorKind.DATASET_CHAMPION
        if champion != (self.champion is not None) or champion == (self.pheromones is not None):
            raise ValueError("a dataset champion holds a champion, a task family pheromones")
        return self

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    @property
    def prior_id(self) -> str:
        return "prior-" + canonical_hash(self.dump())

    @property
    def anchor(self) -> str:
        return anchor(self.kind, self.key)

    def state(self, config: ACOConfig) -> Pheromones:
        """The MMAS state a warm start under ``config`` begins from."""
        if self.pheromones is not None:
            return self.pheromones
        assert self.champion is not None
        return champion_state(Genome.from_canonical(self.champion.genome), config)


def converged_epoch(config: ACOConfig) -> int:
    """Updates after which an edge never reinforced has evaporated from tau_max to tau_min."""
    n, tau = 0, config.tau_max
    while tau > config.tau_min:
        tau *= 1.0 - config.rho
        n += 1
    return n


def champion_state(genome: Genome, config: ACOConfig) -> Pheromones:
    """The MMAS fixed point of a colony converged on ``genome``: its path at ``tau_max``, every
    other edge evaporated to ``tau_min`` (``from_pheromones`` default rule at that epoch)."""
    edges = sorted(set(path_edges(genome)))
    return Pheromones(
        epoch=converged_epoch(config), edges=tuple((a, b, config.tau_max) for a, b in edges)
    )


# == warm-start pin ===============================================================================
class Diagnostic(_Frozen):
    """What one prior level found for a search (recorded whether or not it was used)."""

    kind: PriorKind
    reason: StartReason
    prior_id: str | None = None
    mismatched: tuple[str, ...] = ()


class WarmStart(_Frozen):
    """How ``Strategy.ACO_WARM`` starts in one experiment, resolved when the job was created."""

    schema_: Literal["wynk-warm-start/1"] = Field(PIN_SCHEMA, alias="schema")
    mode: StartMode
    reason: StartReason
    expected: dict[str, str] = Field(default_factory=dict)  # prior kind -> expected key hash
    prior_kind: PriorKind | None = None
    prior_id: str | None = None
    state: Pheromones | None = None
    run_versions: dict[str, str] | None = None  # what every warm run must report
    diagnostics: tuple[Diagnostic, ...] = ()

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _consistent(self) -> WarmStart:
        warm = self.mode is not StartMode.COLD
        pinned = (self.prior_kind, self.prior_id, self.state, self.run_versions)
        if warm and any(p is None for p in pinned):
            raise ValueError("a warm start pins its prior, state and run versions")
        if not warm and any(p is not None for p in pinned):
            raise ValueError("a cold start pins no prior")
        if warm:
            assert self.prior_kind is not None
            if WARM[self.prior_kind] != (self.mode, self.reason):
                raise ValueError("warm start mode/reason disagree with the prior kind")
        elif not self.reason.value.startswith("cold_"):
            raise ValueError("a cold start needs a cold_* reason")
        return self

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    def summary(self) -> dict[str, Any]:
        """The pin as the run protocol records it (the full state lives in the plan)."""
        return {
            "mode": self.mode.value,
            "reason": self.reason.value,
            "prior_kind": self.prior_kind.value if self.prior_kind else None,
            "prior_id": self.prior_id,
            "state_hash": self.state.state_hash if self.state is not None else None,
        }

    def optimizer(self, config: ACOConfig) -> MMASACO:
        """The MMAS this pin starts: ``from_pheromones`` on exactly the pinned state, or cold."""
        if self.state is None:
            return MMASACO(config)
        return MMASACO.from_pheromones(self.state.as_map(), config, epoch=self.state.epoch)


def cold(reason: StartReason, **kw: Any) -> WarmStart:
    return WarmStart(mode=StartMode.COLD, reason=reason, **kw)


def warm(
    record: PriorRecord,
    config: ACOConfig,
    *,
    expected: Mapping[str, str],
    diagnostics: tuple[Diagnostic, ...],
) -> WarmStart:
    mode, reason = WARM[record.kind]
    rv = {k: record.key[k] for k in ("model_hash", "prompt_template_version", "compiler_version")}
    rv["grammar_version"] = record.key["runtime_grammar_version"]
    if record.kind is PriorKind.DATASET_CHAMPION:
        rv["benchmark_hash"] = record.key["benchmark_hash"]
    return WarmStart(
        mode=mode,
        reason=reason,
        expected=dict(expected),
        prior_kind=record.kind,
        prior_id=record.prior_id,
        state=record.state(config),
        run_versions=rv,
        diagnostics=diagnostics,
    )


# == elites (#10) ================================================================================
class EliteRole(StrEnum):
    """Which stored workflow a run starts from. Stable strings: part of the API."""

    TASK = "task_elite"  # the champion of exactly this validation row
    DATASET = "dataset_elite"  # the dataset champion (suite level): every row


class EliteReason(StrEnum):
    TASK_ELITE = "task_elite"
    NO_TASK_CHAMPION = "no_task_champion"  # the row has no feasible champion: dataset elite
    EVIDENCE_SOURCE_CHANGED = "evidence_source_changed"  # another live provider (JEV) now
    # the row's task champion was measured with another live provider: dataset elite
    TASK_EVIDENCE_SOURCE_CHANGED = "task_evidence_source_changed"
    NO_COMPATIBLE_PRIOR = "no_compatible_prior"  # see the diagnostic for why


class Elite(_Frozen):
    role: EliteRole
    prior_id: str
    genome: dict[str, Any]  # Genome.canonical()
    genome_hash: str
    rank_key: tuple[float, ...]
    task: str | None = None  # the row hash of a task elite


class ElitePick(_Frozen):
    """What workflow memory offers a run on one row: an elite (or none) and why."""

    reason: EliteReason
    elite: Elite | None = None
    diagnostic: Diagnostic | None = None  # the dataset-champion lookup behind the pick


def elite_for(record: PriorRecord, task: str, evidence_source: str | None) -> ElitePick:
    """The elite a run on row ``task`` (its ``row_hash``) starts from, given a VERIFIED, exactly
    compatible dataset-champion ``record``: that row's task champion first, else the dataset
    champion. ``evidence_source`` is the live evidence provider the runtime would use now
    (``JevSource.identity`` or ``None``): an elite measured with another provider is never
    offered, since its evidence was not gathered the way a run would gather it now. Each elite
    is judged by its OWN runs' provider (one search may evaluate JEV and fetch/api genomes)."""
    champ = record.champion
    if champ is None:
        raise ValueError("only a dataset champion offers elites")

    def live(source: str | None) -> bool:
        return source is None or source == evidence_source

    best = champ.task(task)
    if best is not None and live(best.evidence_source):
        return ElitePick(
            reason=EliteReason.TASK_ELITE,
            elite=Elite(
                role=EliteRole.TASK,
                prior_id=record.prior_id,
                genome=best.genome,
                genome_hash=best.genome_hash,
                rank_key=best.rank_key,
                task=task,
            ),
        )
    if not live(champ.evidence_source):
        return ElitePick(reason=EliteReason.EVIDENCE_SOURCE_CHANGED)
    return ElitePick(
        reason=(
            EliteReason.NO_TASK_CHAMPION
            if best is None
            else EliteReason.TASK_EVIDENCE_SOURCE_CHANGED
        ),
        elite=Elite(
            role=EliteRole.DATASET,
            prior_id=record.prior_id,
            genome=champ.genome,
            genome_hash=champ.genome_hash,
            rank_key=champ.rank_key,
        ),
    )


__all__ = [
    "CATEGORIES",
    "INVALIDATING",
    "Champion",
    "DatasetRef",
    "Diagnostic",
    "Elite",
    "ElitePick",
    "EliteReason",
    "EliteRole",
    "Evidence",
    "Pheromones",
    "PriorKind",
    "PriorRecord",
    "PriorUnavailable",
    "RetireReason",
    "Source",
    "StartMode",
    "StartReason",
    "TaskChampion",
    "WarmStart",
    "aco_config",
    "anchor",
    "champion_state",
    "classify",
    "cold",
    "converged_epoch",
    "elite_for",
    "evidence_rows",
    "heldout_rows",
    "mismatches",
    "row_hash",
    "validity_key",
    "warm",
]
