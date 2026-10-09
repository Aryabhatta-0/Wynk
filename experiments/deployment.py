"""Champion deployment: an approved champion -> immutable workflow version -> staging ->
production -> inference -> rollback.

    ChampionDeployments.publish(champion_id)
      1. champion     only a #25 CHAMPION (a row of ``champions``, written by a PROMOTED
                      decision) can be published. Its promotion is re-verified from stored
                      evidence (``ChampionPromotions.verify``) before anything is written.
      2. provenance   every pinned identity is read from the experiment's #26 canonical artifact
                      (``load_canonical``: hashes, provenance, job identity) and cross-checked
                      against the champion record: artifact + provenance ids, compatibility
                      identity, TaskContract, grammar, model registry entry + model_hash, prompt /
                      compiler / grammar runtime versions, the strategy run that selected the
                      genome. Two authorities that disagree fail closed; nothing is repaired.
      3. version      a ``wynk-workflow-version/1`` document, content-addressed (``wv-<hash>``)
                      and immutable. Publishing creates it CREATED: it is not deployed anywhere.

    stage(version) -> promote(version)       explicit, fenced (``expected_revision``) and atomic
    rollback(lineage[, version])             a previous production version becomes active again

    invoke(version, inputs) / invoke_production(lineage, inputs)
      validate inputs against the pinned TaskContract input schema (before anything is bound) ->
      bind the pinned model (fail closed: entry gone / disabled / changed, model_hash differs,
      capability missing, grammar / compiler / prompt / runtime versions differ) -> run the frozen
      genome through compiler -> ``WorkflowRunner`` -> ``ModelClient`` -> check the run is the
      pinned one -> validate the answer against the pinned output schema -> record.

Inference never runs a search, an optimizer, an evaluator or a promotion: the champion was chosen
and judged before it was published, and the version holds only what execution needs. Which
optimizer found the genome (fixed, random or ACO) is provenance, never behaviour.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

from core.canonical import canonical_hash, canonical_json, sha256_hex
from core.dataset import DatasetFormat
from core.genome import Genome
from core.models import (
    AllowedModels,
    ModelEntry,
    ModelRegistry,
    ModelRegistryError,
    ModelRequirements,
)
from core.provenance import ProvenanceRecord, grammar_hash
from core.results import ExecutionResult, RunVersions
from core.run_contract import ExampleInput, ExecutionTask
from core.task_contract import ContractError, TaskContract, workflow_grammar
from core.violations import ViolationCode
from evaluation.schema import validate_answer
from experiments.jobs import ExperimentJobDefinition, JobError, load_canonical
from experiments.promotion import (
    CHAMPION_SCHEMA,
    LEGACY_CHAMPION_SCHEMA,
    CandidateStatus,
    ChampionNotFound,
    ChampionPromotions,
    Decision,
    EvidenceMismatch,
    compatibility_identity,
    lineage_identity,
)
from runtime.model_client import ModelClient
from store.deployments import (
    DeploymentRow,
    EventRow,
    InvalidTransition,
    SQLiteDeploymentStore,
    StaleDeployment,
    VersionRow,
    VersionState,
)

WORKFLOW_VERSION_SCHEMA = "wynk-workflow-version/1"
INFERENCE_SCHEMA = "wynk-inference/1"
INLINE_BENCHMARK = "inline"  # rows carry their own context (``InlineSource``): no page store
VERSION_ID = re.compile(r"^wv-[0-9a-f]{24}$")
DEPLOYABLE = frozenset({VersionState.STAGING, VersionState.PRODUCTION})


# -- errors -------------------------------------------------------------------------------------
class DeploymentError(Exception):
    code = "deployment_error"

    def __init__(self, message: str, **details: str | int | None) -> None:
        super().__init__(message)
        self.details = details


class WorkflowVersionNotFound(DeploymentError):
    code = "workflow_version_not_found"


class DeploymentNotFound(DeploymentError):
    code = "deployment_not_found"


class InferenceNotFound(DeploymentError):
    code = "inference_not_found"


class NotPublishable(DeploymentError):
    """The champion cannot become a workflow version (no canonical provenance, a disagreement
    between its authorities, a workflow that needs a page store, ...)."""

    code = "not_publishable"


class InvalidDeploymentTransition(DeploymentError):
    code = "invalid_deployment_transition"


class StaleDeploymentRevision(DeploymentError):
    """The deployment moved since the caller read it: nothing was written."""

    code = "stale_deployment"


class ChampionNotCurrent(DeploymentError):
    """Only a version of the lineage's CURRENT champion may be promoted to production."""

    code = "champion_not_current"


class NotDeployed(DeploymentError):
    code = "workflow_version_not_deployed"


class NoProductionVersion(DeploymentError):
    code = "no_production_version"


class InvalidInferenceRequest(DeploymentError):
    code = "invalid_inference_request"


class ModelBindingFailed(DeploymentError):
    """The pinned model / runtime cannot be bound exactly: nothing was invoked."""

    code = "model_binding_failed"


class InferenceBackendUnavailable(DeploymentError):
    code = "inference_backend_unavailable"


class InferenceFailed(DeploymentError):
    code = "inference_failed"


class OutputSchemaViolation(DeploymentError):
    code = "output_schema_violation"


class VersionIntegrityError(DeploymentError):
    """A stored workflow version does not verify against its own content address."""

    code = "workflow_version_integrity_error"


# -- views --------------------------------------------------------------------------------------
class _View(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WorkflowVersionView(_View):
    version_id: str
    lineage_id: str
    champion_id: str
    state: VersionState
    created_at: str
    state_updated_at: str
    document: dict[str, Any]  # the immutable wynk-workflow-version/1 document


class WorkflowVersionList(_View):
    versions: list[WorkflowVersionView]


class DeploymentView(_View):
    lineage_id: str
    revision: int  # the fence a stage / promote / rollback must name (expected_revision)
    staging_version_id: str | None
    production_version_id: str | None
    created_at: str
    updated_at: str


class DeploymentEventView(_View):
    seq: int
    action: str
    version_id: str
    replaced_version_id: str | None
    revision_before: int
    revision_after: int
    actor: str
    created_at: str


class DeploymentHistory(_View):
    lineage_id: str
    revision: int
    staging_version_id: str | None
    production_version_id: str | None
    events: list[DeploymentEventView]


class InferenceUsage(_View):
    model_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float  # measured wall time of the workflow run
    cost: float | None  # None unless the pinned registry entry prices the model
    cost_authoritative: bool


class InferenceView(_View):
    record_schema: str  # wynk-inference/1
    inference_id: str
    status: str  # SUCCEEDED | FAILED
    workflow_version: str
    lineage_id: str
    addressed_by: str  # "version" | "production"
    deployment_revision: int | None  # the revision that resolved "production"
    champion_id: str
    genome_hash: str
    provenance: dict[str, Any]  # artifact / provenance / experiment / promotion / strategy run
    model: dict[str, Any]  # model_hash, registry entry name + hash
    versions: dict[str, Any]  # model / prompt / compiler / grammar / benchmark versions
    run_id: str | None
    request_sha256: str
    output: dict[str, Any] | None  # only in the invoke response; records keep its hash
    output_sha256: str | None
    failure: dict[str, Any] | None
    usage: InferenceUsage
    created_at: str


# -- runtime ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class InferenceRuntime:
    """How this process binds a version's pinned model: ``registry()`` is re-read on EVERY
    invocation (an entry removed or disabled since publishing fails closed) and ``client(entry)``
    builds the #27 ``ModelClient`` for that exact entry (``runtime.backends.client_for``)."""

    registry: Callable[[], ModelRegistry]
    client: Callable[[ModelEntry], ModelClient]
    clock: Callable[[], float] = time.perf_counter


@dataclass(frozen=True)
class BoundVersion:
    """A version bound to this process's runtime, checked identity for identity."""

    document: Mapping[str, Any]
    genome: Genome
    contract: TaskContract
    entry: ModelEntry
    runner: Any  # runtime.runner.WorkflowRunner


def _document_hash(document: Mapping[str, Any]) -> str:
    return canonical_hash(dict(document))


def version_id_for(document: Mapping[str, Any]) -> str:
    return "wv-" + _document_hash(document)[:24]


def _record_sha256(record_json: str) -> str:
    return sha256_hex(canonical_json(json.loads(record_json)))


# -- the immutable workflow version -------------------------------------------------------------
def workflow_version_document(
    *,
    champion_id: str,
    champion_version: int,
    lineage_id: str,
    compat_hash: str,
    record_json: str,
    record: Mapping[str, Any],
    decision: Mapping[str, Any],
    definition: ExperimentJobDefinition,
    artifact_ref: Mapping[str, str],
    body: Mapping[str, Any],
    provenance: ProvenanceRecord,
) -> dict[str, Any]:
    """The version document of a champion, every identity read from the experiment's canonical
    provenance and cross-checked against the champion record. ``NotPublishable`` names every
    disagreement; nothing is taken from the caller."""
    problems: list[str] = []

    def agree(what: str, a: Any, b: Any) -> None:
        if a != b:
            problems.append(f"{what} disagrees")

    prov = record["provenance"]
    compat = record["compatibility"]
    # -- champion and its decision
    schema = record.get("schema")
    agree("champion schema", schema in (CHAMPION_SCHEMA, LEGACY_CHAMPION_SCHEMA), True)
    agree("champion id", record.get("champion_id"), champion_id)
    agree("champion lineage", record.get("lineage_id"), lineage_id)
    agree("champion version", record.get("version"), champion_version)
    if schema == CHAMPION_SCHEMA:  # champion/2: the row's compat_hash is the lineage hash
        lineage = record.get("lineage_identity") or {}
        agree("lineage_hash", lineage.get("lineage_hash"), compat_hash)
        agree("lineage identity", lineage_identity(definition, body, provenance), lineage)
        agree("decision lineage", decision.get("lineage_identity"), lineage)
        agree(
            "evaluation context",
            decision.get("evaluation_context"),
            record.get("evaluation_context"),
        )
    else:  # champion/1: the dataset-scoped compatibility hash, exactly as #30 pinned it
        agree("compat_hash", compat.get("compat_hash"), compat_hash)
    agree("decision", decision.get("decision"), Decision.PROMOTED.value)
    agree("decision champion", (decision.get("champion") or {}).get("champion_id"), champion_id)
    agree(
        "challenger status",
        (decision.get("challenger") or {}).get("status"),
        CandidateStatus.CHAMPION.value,
    )
    agree("decision artifact", decision.get("artifact"), dict(artifact_ref))
    # -- #26 canonical artifact and provenance
    agree("artifact_id", prov.get("artifact_id"), artifact_ref["artifact_id"])
    agree("provenance_id", prov.get("provenance_id"), artifact_ref["provenance_id"])
    agree("provenance_id (record)", provenance.provenance_id, artifact_ref["provenance_id"])
    agree("experiment_id", prov.get("experiment_id"), provenance.experiment_id)
    agree("definition_hash", prov.get("definition_hash"), provenance.definition_hash)
    agree(
        "compatibility identity",
        compatibility_identity(definition, body, provenance),
        dict(compat),
    )
    # -- the frozen genome and the strategy run that selected it
    genome = Genome.from_canonical(record["genome"])
    agree("genome_hash", genome.genome_hash, record.get("genome_hash"))
    try:
        run = provenance.run(prov["strategy"], prov["seed"])
    except KeyError:
        problems.append("the champion's strategy run is not in the provenance")
    else:
        agree("run_id", run.run_id, prov.get("run_id"))
        agree(
            "optimizer",
            (run.optimizer, run.optimizer_version),
            (prov.get("optimizer"), prov.get("optimizer_version")),
        )
        agree("selected genome", run.selected_genome_hash, genome.genome_hash)
    # -- TaskContract
    if provenance.contract.contract is None:
        raise NotPublishable("the experiment's provenance has no TaskContract")
    contract = TaskContract.model_validate(provenance.contract.contract)
    agree("contract_hash", contract.contract_hash, provenance.contract.contract_hash)
    agree("task contract", contract.contract_hash, compat.get("task_contract_hash"))
    if contract.dataset.format is DatasetFormat.WYNK_SNAPSHOT:
        problems.append("a workflow over benchmark snapshots needs a page store: not deployable")
    # -- grammar
    grammar = workflow_grammar(contract)
    agree("grammar version", grammar.version, provenance.grammar.version)
    agree("grammar version (compat)", grammar.version, compat.get("grammar_version"))
    agree(
        "grammar_hash",
        grammar_hash(provenance.grammar.version, provenance.grammar.stage_kinds),
        provenance.grammar.grammar_hash,
    )
    if grammar.validate(genome):
        problems.append("the genome is not a complete workflow of the contract's grammar")
    # -- model registry entry and model_hash
    model = provenance.model
    if model.registry_entry is None:
        raise NotPublishable("the experiment pinned no model registry entry; nothing to serve")
    entry = ModelEntry.model_validate(model.registry_entry)
    agree("registry entry hash", entry.identity_hash, model.registry_entry_hash)
    agree("registry model_hash", entry.model_hash, model.model_hash)
    agree("model_hash (compat)", model.model_hash, compat.get("model_hash"))
    agree("model configuration", model.configuration, compat.get("model"))
    # -- runtime versions: one identity for every run of the experiment
    run_versions = list(provenance.versions.run_versions)
    agree("run versions (compat)", run_versions, compat.get("run_versions"))
    if len(run_versions) != 1:
        raise NotPublishable(f"the experiment ran under {len(run_versions)} runtime identities")
    versions = RunVersions.model_validate(run_versions[0])
    agree("run model_hash", versions.model_hash, model.model_hash)
    agree("run grammar version", versions.grammar_version, grammar.version)
    if model.prompt_template_version is not None:
        agree("prompt version", versions.prompt_template_version, model.prompt_template_version)
    if versions.benchmark_hash != INLINE_BENCHMARK:
        problems.append(f"runs read benchmark {versions.benchmark_hash!r}, not inline rows")
    if problems:
        raise NotPublishable(
            f"champion {champion_id} cannot be published: " + "; ".join(sorted(set(problems)))
        )
    return {
        "schema": WORKFLOW_VERSION_SCHEMA,
        "lineage_id": lineage_id,
        "synthetic": provenance.synthetic,
        "champion": {
            "champion_id": champion_id,
            "champion_version": champion_version,
            "promotion_id": prov["promotion_id"],
            "decision_hash": decision["decision_hash"],
            "record_sha256": _record_sha256(record_json),
            "compat_hash": compat_hash,
        },
        "workflow": {"genome_hash": genome.genome_hash, "genome": genome.canonical()},
        "provenance": {
            "artifact_id": artifact_ref["artifact_id"],
            "provenance_id": artifact_ref["provenance_id"],
            "experiment_sha256": artifact_ref["experiment_sha256"],
            "experiment_id": provenance.experiment_id,
            "definition_hash": provenance.definition_hash,
            "job_id": prov["job_id"],
            "run_id": prov["run_id"],
            "strategy": prov["strategy"],
            "seed": prov["seed"],
            "optimizer": prov["optimizer"],
            "optimizer_version": prov["optimizer_version"],
        },
        "contract": {
            "contract_hash": contract.contract_hash,
            "task_id": contract.task_id,
            "contract_version": contract.contract_version,
            "task_contract": contract.model_dump(mode="json"),
            "input_schema": contract.input_schema.model_dump(mode="json"),
            "output_schema": contract.output_schema.model_dump(mode="json"),
        },
        "grammar": {
            "version": provenance.grammar.version,
            "stage_kinds": list(provenance.grammar.stage_kinds or ()),
            "grammar_hash": provenance.grammar.grammar_hash,
        },
        "model": {
            "model_hash": model.model_hash,
            "registry_entry": model.registry_entry,
            "registry_entry_hash": model.registry_entry_hash,
            "configuration": model.configuration,
            "config_hash": model.config_hash,
        },
        "versions": versions.model_dump(mode="json"),
    }


def check_document(row: VersionRow) -> dict[str, Any]:
    """The stored document, verified against its hash and content address (fail closed)."""
    try:
        document = json.loads(row.document_json)
    except ValueError:
        raise VersionIntegrityError(f"workflow version {row.version_id} is not JSON") from None
    digest = _document_hash(document)
    if (
        digest != row.document_sha256
        or version_id_for(document) != row.version_id
        or document.get("schema") != WORKFLOW_VERSION_SCHEMA
        or document["champion"]["champion_id"] != row.champion_id
        or document["lineage_id"] != row.lineage_id
    ):
        raise VersionIntegrityError(
            f"stored workflow version {row.version_id} does not verify (tampered or corrupt)"
        )
    return document


# -- binding: the pinned model and runtime, exactly, or nothing ---------------------------------
def bind_version(document: Mapping[str, Any], runtime: InferenceRuntime) -> BoundVersion:
    """Bind ``document`` to this process: the registry entry it pinned (present, enabled,
    byte-identical), a ``ModelClient`` reporting exactly the pinned ``model_hash``, a
    ``WorkflowRunner`` whose model / prompt / compiler / grammar / benchmark versions equal the
    pinned ones, and a genome the runtime admits. Any difference raises ``ModelBindingFailed``
    before a model is invoked; another model is never substituted."""
    from runtime.runner import WorkflowRunner

    pinned = document["model"]
    model_hash = pinned["model_hash"]

    def fail(reason: str, message: str) -> ModelBindingFailed:
        return ModelBindingFailed(message, reason=reason, model_hash=model_hash)

    try:
        registry = runtime.registry()
    except (OSError, ValueError) as exc:
        raise fail("registry_unavailable", f"the model registry cannot be read: {exc}") from None
    try:
        entry = registry.by_hash(model_hash)
    except ModelRegistryError:
        raise fail(
            "registry_entry_missing", f"model {model_hash!r} is no longer in the model registry"
        ) from None
    if not entry.enabled:
        raise fail("registry_entry_disabled", f"model {entry.name!r} is disabled")
    if entry.identity_hash != pinned["registry_entry_hash"]:
        raise fail(
            "registry_entry_changed",
            f"registry entry {entry.name!r} changed since the version was published",
        )
    try:
        client = runtime.client(entry)
    except Exception as exc:  # an adapter that cannot be built is a binding failure
        raise fail("client_unavailable", f"{type(exc).__name__}: {exc}") from None
    if client.model_hash != model_hash:
        raise fail(
            "model_hash_mismatch",
            f"client reports model_hash {client.model_hash!r}, the version pins {model_hash!r}",
        )
    contract = TaskContract.model_validate(document["contract"]["task_contract"])
    genome = Genome.from_canonical(document["workflow"]["genome"])
    if (
        contract.contract_hash != document["contract"]["contract_hash"]
        or genome.genome_hash != document["workflow"]["genome_hash"]
    ):
        raise VersionIntegrityError("the version's contract or genome does not hash as pinned")
    versions = document["versions"]
    try:  # the client must be bound to exactly this entry (`RegisteredModelClient`)
        runner = WorkflowRunner(
            model=client,
            benchmark_hash=versions["benchmark_hash"],
            allowed_models=AllowedModels((entry,), ModelRequirements(capabilities=())),
        )
    except ModelRegistryError as exc:
        raise fail("model_identity_mismatch", str(exc)) from None
    try:
        runner.check_model(genome)  # the capabilities the workflow's model stages need
    except ModelRegistryError as exc:
        raise fail("capability_unavailable", str(exc)) from None
    actual = runner.versions(_probe_task(contract)).model_dump(mode="json")
    if actual != versions:
        differs = sorted(k for k in versions if actual.get(k) != versions[k])
        raise fail(
            "runtime_incompatible",
            "this runtime's " + ", ".join(differs) + " differ from the pinned versions",
        )
    structural = [
        v
        for v in runner.checker.check(genome, contract, complete=True)
        if v.code != ViolationCode.BUDGET_INFEASIBLE
    ]
    if structural:
        raise fail("grammar_incompatible", "the runtime does not admit the pinned genome")
    return BoundVersion(document, genome, contract, entry, runner)


def _probe_task(contract: TaskContract) -> ExecutionTask:
    """A task of ``contract`` for reading the runtime's versions (never executed)."""
    values = {f.name: _probe_value(f.type.value) for f in contract.input_schema.fields}
    return ExecutionTask(contract=contract, example=ExampleInput(row_id="probe", values=values))


def _probe_value(kind: str) -> Any:
    return {
        "string": "x",
        "integer": 0,
        "number": 0.0,
        "boolean": False,
        "date": "2000-01-01",
        "string_list": [],
    }[kind]


def validate_inputs(contract: TaskContract, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The request's inputs against the pinned input schema: required fields present, types
    exact, no unknown field. ``null`` for an optional field means absent."""
    schema = contract.input_schema
    optional = {f.name for f in schema.fields if not f.required}
    values = {k: v for k, v in inputs.items() if not (v is None and k in optional)}
    problems = validate_answer(schema, values)
    if problems:
        name = sorted(problems)[0]
        raise InvalidInferenceRequest(
            f"inputs.{name}: {problems[name]}", field=f"inputs.{name}", problems=len(problems)
        )
    return values


# -- the service --------------------------------------------------------------------------------
@dataclass
class ChampionDeployments:
    """publish / stage / promote / rollback / invoke. ``runtime`` is ``None`` in a process that
    cannot bind a model (versions and deployments stay inspectable; nothing can be served)."""

    store: SQLiteDeploymentStore
    promotions: ChampionPromotions
    runtime: InferenceRuntime | None = None
    ids: Callable[[], str] = field(default=lambda: "inf-" + secrets.token_hex(12))

    # -- publish ----------------------------------------------------------------------------
    def publish(self, champion_id: str, actor: str = "api") -> tuple[WorkflowVersionView, bool]:
        """The champion's immutable workflow version (created CREATED), or the one already
        published for it. ``bool``: created now."""
        champions = self.promotions.store
        champ = champions.champion(champion_id)
        if champ is None:
            raise ChampionNotFound(
                f"no champion {champion_id}; only a promoted champion can be published"
            )
        verified = self.promotions.verify(champ.promotion_id)  # re-derived from evidence
        if verified.decision is not Decision.PROMOTED or verified.record is None:
            raise NotPublishable(f"promotion {champ.promotion_id} did not promote a champion")
        if (verified.challenger or {}).get("genome_hash") != json.loads(champ.record_json).get(
            "genome_hash"
        ):
            raise NotPublishable("the champion is not its promotion's challenger")
        if verified.artifact is None:
            raise NotPublishable(
                f"champion {champion_id} predates provenance artifacts; it cites no canonical "
                "artifact to pin"
            )
        try:
            job, art = load_canonical(self.promotions.jobs, verified.job_id)
        except JobError as exc:
            raise EvidenceMismatch(str(exc)) from exc
        if art.ref() != verified.artifact:
            raise NotPublishable("the experiment's artifact is not the one the promotion cited")
        definition = ExperimentJobDefinition.model_validate_json(job.definition_json)
        record = json.loads(champ.record_json)
        try:
            document = workflow_version_document(
                champion_id=champ.champion_id,
                champion_version=champ.version,
                lineage_id=champ.lineage_id,
                compat_hash=champ.compat_hash,
                record_json=champ.record_json,
                record=record,
                decision=verified.record,
                definition=definition,
                artifact_ref=art.ref(),
                body=art.body,
                provenance=art.provenance,
            )
        except (KeyError, TypeError, ValueError, ContractError) as exc:
            raise NotPublishable(f"champion {champion_id}: {type(exc).__name__}: {exc}") from exc
        version_id = version_id_for(document)
        row, created = self.store.publish(
            version_id,
            champ.lineage_id,
            champ.champion_id,
            canonical_json(document),
            _document_hash(document),
            actor,
        )
        stored = check_document(row)
        if stored != document:  # the same champion always publishes the same version
            raise VersionIntegrityError(
                f"champion {champion_id} was published as {row.version_id}, which differs from "
                "what its evidence pins now"
            )
        return self._view(row), created

    # -- inspection -------------------------------------------------------------------------
    def _row(self, version_id: str) -> VersionRow:
        row = self.store.version(version_id) if VERSION_ID.match(version_id) else None
        if row is None:
            raise WorkflowVersionNotFound(f"no workflow version {version_id}")
        return row

    def _view(self, row: VersionRow) -> WorkflowVersionView:
        return WorkflowVersionView(
            version_id=row.version_id,
            lineage_id=row.lineage_id,
            champion_id=row.champion_id,
            state=row.state,
            created_at=row.created_at,
            state_updated_at=row.state_updated_at,
            document=check_document(row),
        )

    def get(self, version_id: str) -> WorkflowVersionView:
        return self._view(self._row(version_id))

    def list(self, lineage_id: str | None = None) -> WorkflowVersionList:
        return WorkflowVersionList(
            versions=[self._view(r) for r in self.store.versions(lineage_id)]
        )

    def _deployment(self, lineage_id: str) -> DeploymentRow:
        row = self.store.deployment(lineage_id)
        if row is None:
            raise DeploymentNotFound(f"lineage {lineage_id} has no published workflow version")
        return row

    def deployment(self, lineage_id: str) -> DeploymentView:
        return _deployment_view(self._deployment(lineage_id))

    def history(self, lineage_id: str) -> DeploymentHistory:
        d = self._deployment(lineage_id)
        return DeploymentHistory(
            lineage_id=lineage_id,
            revision=d.revision,
            staging_version_id=d.staging_version_id,
            production_version_id=d.production_version_id,
            events=[_event_view(e) for e in self.store.history(lineage_id)],
        )

    # -- staging -> production -> rollback --------------------------------------------------
    def stage(self, version_id: str, expected_revision: int, actor: str = "api") -> DeploymentView:
        row = self._row(version_id)
        document = check_document(row)
        if row.state is not VersionState.CREATED:
            raise InvalidDeploymentTransition(
                f"workflow version {version_id} is {row.state.value}; only a CREATED version "
                "can be staged"
            )
        self._preflight(document)  # a version that cannot be served is never staged
        return self._fenced(lambda: self.store.stage(version_id, expected_revision, actor))

    def promote(
        self, version_id: str, expected_revision: int, actor: str = "api"
    ) -> DeploymentView:
        """STAGING -> PRODUCTION (explicit, fenced, atomic). The version must belong to its
        lineage's CURRENT champion, verified against the stored champion record, and must bind
        to this process's model runtime."""
        row = self._row(version_id)
        document = check_document(row)
        if row.state is not VersionState.STAGING:
            raise InvalidDeploymentTransition(
                f"workflow version {version_id} is {row.state.value}; only a STAGING version can "
                "be promoted to production (stage it first)"
            )
        self._check_current_champion(row, document)
        self._preflight(document)
        return self._fenced(lambda: self.store.promote(version_id, expected_revision, actor))

    def rollback(
        self,
        lineage_id: str,
        expected_revision: int,
        version_id: str | None = None,
        actor: str = "api",
    ) -> DeploymentView:
        """Make a PREVIOUS production version active again - by default the one the current
        production version replaced. No optimization, evaluation or promotion is re-run, and no
        version is modified: the old immutable version itself is served again."""
        d = self._deployment(lineage_id)
        if d.production_version_id is None:
            raise NoProductionVersion(f"lineage {lineage_id} has no production version")
        if version_id is None:
            version_id = _previous_production(self.store.history(lineage_id), d)
            if version_id is None:
                raise InvalidDeploymentTransition(
                    f"lineage {lineage_id} has no previous production version to roll back to"
                )
        row = self._row(version_id)
        if row.lineage_id != lineage_id:
            raise InvalidDeploymentTransition(
                f"workflow version {version_id} belongs to lineage {row.lineage_id}"
            )
        check_document(row)
        return self._fenced(lambda: self.store.rollback(version_id, expected_revision, actor))

    def _check_current_champion(self, row: VersionRow, document: Mapping[str, Any]) -> None:
        champions = self.promotions.store
        current = champions.current(row.lineage_id)
        pinned = document["champion"]
        if current is None or current.champion_id != row.champion_id:
            raise ChampionNotCurrent(
                f"workflow version {row.version_id} is champion {row.champion_id}; lineage "
                f"{row.lineage_id}'s current champion is "
                f"{current.champion_id if current else 'none'}"
            )
        if (
            current.version != pinned["champion_version"]
            or current.compat_hash != pinned["compat_hash"]
            or _record_sha256(current.record_json) != pinned["record_sha256"]
        ):
            raise VersionIntegrityError(
                f"champion {row.champion_id} no longer matches what version {row.version_id} pinned"
            )

    def _fenced(self, write: Callable[[], DeploymentRow]) -> DeploymentView:
        try:
            return _deployment_view(write())
        except StaleDeployment as exc:
            raise StaleDeploymentRevision(str(exc)) from None
        except InvalidTransition as exc:
            raise InvalidDeploymentTransition(str(exc)) from None

    def _runtime(self) -> InferenceRuntime:
        if self.runtime is None:
            raise InferenceBackendUnavailable("this process has no model runtime to serve with")
        return self.runtime

    def _preflight(self, document: Mapping[str, Any]) -> BoundVersion:
        return bind_version(document, self._runtime())

    # -- inference --------------------------------------------------------------------------
    def invoke(self, version_id: str, inputs: Mapping[str, Any]) -> InferenceView:
        """Run exactly ``version_id`` (STAGING or PRODUCTION)."""
        row = self._row(version_id)
        return self._invoke(row, inputs, addressed_by="version", revision=None)

    def invoke_production(self, lineage_id: str, inputs: Mapping[str, Any]) -> InferenceView:
        """Run the lineage's current production version; the response names which one."""
        d = self._deployment(lineage_id)
        if d.production_version_id is None:
            raise NoProductionVersion(f"lineage {lineage_id} has no production version")
        row = self._row(d.production_version_id)
        return self._invoke(row, inputs, addressed_by="production", revision=d.revision)

    def inference(self, inference_id: str) -> InferenceView:
        row = self.store.inference(inference_id)
        if row is None:
            raise InferenceNotFound(f"no inference {inference_id}")
        return InferenceView.model_validate(
            json.loads(row.record_json) | {"output": None, "created_at": row.created_at}
        )

    def _invoke(
        self,
        row: VersionRow,
        inputs: Mapping[str, Any],
        *,
        addressed_by: str,
        revision: int | None,
    ) -> InferenceView:
        document = check_document(row)
        contract = TaskContract.model_validate(document["contract"]["task_contract"])
        values = validate_inputs(contract, inputs)  # before anything is bound or invoked
        if row.state not in DEPLOYABLE:
            raise NotDeployed(
                f"workflow version {row.version_id} is {row.state.value}; only a STAGING or "
                "PRODUCTION version serves inference",
                state=row.state.value,
            )
        bound = bind_version(document, self._runtime())
        inference_id = self.ids()
        task = ExecutionTask(
            contract=bound.contract, example=ExampleInput(row_id=inference_id, values=values)
        )
        clock = self._runtime().clock
        t0 = clock()
        result: ExecutionResult | None = None
        failure: dict[str, Any] | None = None
        try:
            result = bound.runner.run_sync(bound.genome, task, trial=0, seed=0)
        except Exception as exc:  # recorded, then reported: never a silent retry elsewhere
            failure = {"kind": "runtime_error", "message": f"{type(exc).__name__}: {exc}"[:500]}
        latency = clock() - t0
        output = None
        if result is not None:
            failure, output = _outcome(result, bound, task)
        record = _inference_record(
            inference_id=inference_id,
            row=row,
            document=document,
            entry=bound.entry,
            addressed_by=addressed_by,
            revision=revision,
            request=values,
            result=result,
            output=output,
            failure=failure,
            latency=latency,
        )
        status = "SUCCEEDED" if failure is None else "FAILED"
        stored = self.store.record_inference(
            inference_id, row.version_id, row.lineage_id, status, canonical_json(record)
        )
        details = {"inference_id": inference_id, "workflow_version": row.version_id}
        if failure is not None:
            kind = failure["kind"]
            if kind == "output_schema_invalid":
                raise OutputSchemaViolation(failure["message"], **details, failure_kind=kind)
            raise InferenceFailed(failure["message"], **details, failure_kind=kind)
        return InferenceView.model_validate(
            record | {"output": output, "created_at": stored.created_at}
        )


def _outcome(
    result: ExecutionResult, bound: BoundVersion, task: ExecutionTask
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """``(failure, output)`` of a finished run: the run must be the pinned workflow under the
    pinned versions, and its answer must satisfy the pinned output schema - or no output."""
    key = result.key
    pinned = bound.document
    if (
        key.genome_hash != pinned["workflow"]["genome_hash"]
        or key.contract_hash != pinned["contract"]["contract_hash"]
        or key.task_id != task.id
        or key.versions.model_dump(mode="json") != pinned["versions"]
    ):
        return {"kind": "run_identity_mismatch", "message": "the run is not the pinned one"}, None
    if result.failure is not None:
        return {"kind": result.failure.kind.value, "message": result.failure.message[:500]}, None
    if result.answer is None:
        return {"kind": "no_answer", "message": "the workflow produced no answer"}, None
    problems = validate_answer(bound.contract.output_schema, dict(result.answer.values))
    if problems:
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(problems.items()))
        return {"kind": "output_schema_invalid", "message": f"output {detail}"[:500]}, None
    return None, dict(result.answer.values)


def _inference_record(
    *,
    inference_id: str,
    row: VersionRow,
    document: Mapping[str, Any],
    entry: ModelEntry,
    addressed_by: str,
    revision: int | None,
    request: Mapping[str, Any],
    result: ExecutionResult | None,
    output: Mapping[str, Any] | None,
    failure: Mapping[str, Any] | None,
    latency: float,
) -> dict[str, Any]:
    metrics = result.metrics if result is not None else None
    prompt = metrics.prompt_tokens if metrics else 0
    completion = metrics.completion_tokens if metrics else 0
    cost = entry.pricing.cost(prompt, completion) if entry.pricing is not None else None
    prov = document["provenance"]
    return {
        "record_schema": INFERENCE_SCHEMA,
        "inference_id": inference_id,
        "status": "SUCCEEDED" if failure is None else "FAILED",
        "workflow_version": row.version_id,
        "lineage_id": row.lineage_id,
        "addressed_by": addressed_by,
        "deployment_revision": revision,
        "champion_id": row.champion_id,
        "genome_hash": document["workflow"]["genome_hash"],
        "provenance": {
            "promotion_id": document["champion"]["promotion_id"],
            **{
                k: prov[k]
                for k in (
                    "artifact_id",
                    "provenance_id",
                    "experiment_id",
                    "job_id",
                    "run_id",
                    "strategy",
                    "seed",
                )
            },
        },
        "model": {
            "model_hash": document["model"]["model_hash"],
            "registry_entry_hash": document["model"]["registry_entry_hash"],
            "name": entry.name,
        },
        "versions": dict(document["versions"]),
        "run_id": result.run_id if result is not None else None,
        "request_sha256": canonical_hash(dict(request)),
        "output_sha256": canonical_hash(dict(output)) if output is not None else None,
        "failure": dict(failure) if failure is not None else None,
        "usage": {
            "model_calls": metrics.model_calls if metrics else 0,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "latency_s": round(latency, 6),
            "cost": cost,
            "cost_authoritative": cost is not None,
        },
    }


def _previous_production(events: list[EventRow], d: DeploymentRow) -> str | None:
    """The version the current production version replaced when it became active."""
    for e in reversed(events):
        if e.action.value in ("promote", "rollback") and e.version_id == d.production_version_id:
            return e.replaced_version_id
    return None


def _deployment_view(row: DeploymentRow) -> DeploymentView:
    return DeploymentView(
        lineage_id=row.lineage_id,
        revision=row.revision,
        staging_version_id=row.staging_version_id,
        production_version_id=row.production_version_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _event_view(e: EventRow) -> DeploymentEventView:
    return DeploymentEventView(
        seq=e.seq,
        action=e.action.value,
        version_id=e.version_id,
        replaced_version_id=e.replaced_version_id,
        revision_before=e.revision_before,
        revision_after=e.revision_after,
        actor=e.actor,
        created_at=e.created_at,
    )


__all__ = [
    "ChampionDeployments",
    "DeploymentError",
    "InferenceRuntime",
    "bind_version",
    "check_document",
    "validate_inputs",
    "workflow_version_document",
]
