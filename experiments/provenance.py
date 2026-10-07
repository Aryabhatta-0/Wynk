"""Canonical experiment artifacts: provenance, integrity, traceability, declared-metric checks.

    #23 result body (``assemble``, ``wynk-optimization-experiment/2``)   - unchanged, stored once
      + ProvenanceRecord (``core.provenance``)                           - built from authorities
      = canonical artifact envelope (``wynk-experiment-artifact/1``)     - small, hashed

There is no second result format. The envelope holds the provenance and a hash LINK to the body
(``experiment.sha256`` = sha256 of the body's canonical JSON), so the body is never duplicated and
neither half can change without the other noticing. ``artifact_id`` is the canonical hash of the
envelope; it is the one identity a job, a promotion decision and every trace refer to.

    build_provenance      authorities (TaskContract, DatasetSplits, ExperimentPlan, registry
                          entry) -> ProvenanceRecord, cross-checked against the identities the
                          body recorded while it ran (``check_provenance``: fail closed)
    canonical_envelope    body + provenance + parent (job or frozen directory) -> envelope
    verify_artifact       envelope + stored body -> ``CanonicalArtifact``; any hash, schema or
                          identity disagreement raises ``ArtifactIntegrityError`` (never repaired)
    trace                 a stable field path -> (artifact id, JSON pointer, value, field class,
                          provenance identity, how a derived number is derived from its sources)
    recompute_declared    every declared aggregate re-derived from the per-run execution evidence
                          the body carries (summary, run usage, curves, champions)
    load_frozen           a committed ``write_compact`` directory (MuSiQue / MMLU-Pro) -> an
                          explicitly migrated canonical artifact (``Migration``: hash-only parts)

Field classes. Every leaf of a body is exactly one of
    scientific          what the experiment computed: scores, verdicts, candidate order, selected
                        workflows, tokens, calls, cost, stop reasons, declared budgets/limits.
                        Reproducible from stored evidence, and by re-execution on a
                        deterministic backend.
    measured_telemetry  runtime-measured wall time of workflow runs (latency, execution time).
                        Reproduced exactly from stored evidence; never expected to reproduce when
                        a model is called again.
    clock_telemetry     process-clock values (end-to-end time, overheads, throughput; the existing
                        ``CLOCK_KEYS``). Never compared.
Deterministic equality is equality of the scientific projection only.
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.canonical import canonical_hash, canonical_json, sha256_hex
from core.dataset import DatasetSplits, SplitRole
from core.experiment import ModelConfiguration
from core.models import ModelEntry
from core.provenance import BudgetProvenance as _Budget
from core.provenance import (
    ContractProvenance,
    DatasetProvenance,
    EvaluatorProvenance,
    GrammarProvenance,
    Migration,
    ModelProvenance,
    ProvenanceMismatch,
    ProvenanceRecord,
    SplitsProvenance,
    StrategyRunProvenance,
    VersionsProvenance,
    grammar_hash,
)
from core.task_contract import TaskContract, workflow_grammar
from experiments.optimization_experiment import (
    ARTIFACT_SCHEMA,
    CLOCK_KEYS,
    ExperimentPlan,
    Strategy,
    load_compact,
    make_strategy,
    run_seed,
    summarize,
)

CANONICAL_SCHEMA = "wynk-experiment-artifact/1"
FIELD_CLASSES_VERSION = "wynk-field-classes/1"
MIGRATION_METHOD = "legacy-identity/1"
# What an artifact from before provenance existed only carries as a hash (or not at all).
LEGACY_MISSING = (
    "contract.contract",
    "contract.schema_version",
    "splits.method",
    "splits.plan",
    "grammar.stage_kinds",
    "model.registry_entry",
    "model.registry_entry_hash",
)

SCIENTIFIC = "scientific"
MEASURED = "measured_telemetry"
CLOCK = "clock_telemetry"
MEASURED_KEYS = frozenset(
    {
        "latency_s",
        "wall_time_s",
        "execution_s",
        "cumulative_execution_s",
        "mean_latency_s",
        "p50_latency_s",
        "p95_latency_s",
        "max_latency_s",
    }
)
# Containers of declared configuration and limits: never telemetry, whatever their keys are.
DECLARED_KEYS = frozenset({"budget", "reservation_per_run", "fairness", "identity", "protocol"})


class ArtifactIntegrityError(ValueError):
    """A stored artifact does not verify (hash, schema, identity): fail closed, never repair."""


class TraceError(LookupError):
    """A field path that does not resolve to a stored field."""


# == provenance ===================================================================================
def optimizer_config(strategy: Strategy, plan: ExperimentPlan) -> dict[str, Any] | None:
    """The parameters the experiment builds ``strategy``'s optimizer with (``make_strategy``)."""
    cfg = getattr(make_strategy(strategy, plan), "config", None)
    if cfg is None or not dataclasses.is_dataclass(cfg) or isinstance(cfg, type):
        return None
    return json.loads(canonical_json(dataclasses.asdict(cfg)))


def _plan_of(body: Mapping[str, Any]) -> ExperimentPlan:
    return ExperimentPlan.model_validate(body["identity"]["plan"])


def _runs(body: Mapping[str, Any], plan: ExperimentPlan) -> tuple[StrategyRunProvenance, ...]:
    out = []
    for r in body["runs"]:
        champion = r["champion"]
        out.append(
            StrategyRunProvenance(
                strategy=r["strategy"],
                seed=r["seed"],
                run_id=r["run_id"],
                optimizer=r["optimizer"],
                optimizer_version=r["optimizer_version"],
                optimizer_config=optimizer_config(Strategy(r["strategy"]), plan),
                candidates=r["usage"]["candidate_evaluations"],
                candidate_order_hash=canonical_hash(r["evaluated_genome_hashes"]),
                selected_genome_hash=champion["genome_hash"] if champion else None,
            )
        )
    return tuple(out)


def _budget(body: Mapping[str, Any], plan: ExperimentPlan) -> _Budget:
    problem = body["identity"]["problem"]
    return _Budget(
        budget=plan.budget.model_dump(mode="json"),
        budget_hash=plan.budget.identity_hash,
        pricing=problem["pricing"],
        protocol=plan.protocol(),
        protocol_hash=canonical_hash(plan.protocol()),
        protocol_id=plan.protocol_id,
        fixed_baseline_rule=plan.fixed_baseline_rule,
        trials=plan.trials,
        batch_size=plan.batch_size,
    )


def recorded_provenance(
    body: Mapping[str, Any],
    *,
    definition_hash: str | None = None,
    job_schema: str | None = None,
    migration: Migration | None = None,
) -> ProvenanceRecord:
    """The provenance the body itself RECORDED while it ran (its problem identity, plan and
    strategy-run identities) - hash-level only for the contract, splits and registry entry.
    Used as is for a migrated artifact, and as the cross-check of ``build_provenance``."""
    problem = body["identity"]["problem"]
    plan = _plan_of(body)
    return ProvenanceRecord(
        experiment_id=body["experiment_id"],
        problem_id=problem["problem_id"],
        definition_hash=definition_hash,
        synthetic=body["synthetic"],
        dataset=DatasetProvenance(
            dataset_id=problem["dataset_id"],
            dataset_version=problem["dataset_version"],
            content_hash=problem["dataset_content_hash"],
            identity_hash=problem["dataset_hash"],
            manifest_hash=problem["dataset_manifest_hash"],
        ),
        splits=SplitsProvenance(
            splits_hash=problem["splits_hash"],
            dataset_hash=problem["dataset_hash"],
            method=None,
            plan=None,
            rows=dict(body["splits"]),
        ),
        contract=ContractProvenance(
            contract_hash=problem["task_contract_hash"],
            schema_version=None,
            task_id=problem["task_id"],
            contract_version=problem["contract_version"],
            objective_hash=problem["objective_hash"],
            constraints_hash=problem["constraints_hash"],
            contract=None,
        ),
        grammar=GrammarProvenance(
            version=problem["grammar_version"],
            stage_kinds=None,
            grammar_hash=grammar_hash(problem["grammar_version"], None),
        ),
        evaluator=EvaluatorProvenance(
            kind=problem["evaluator_kind"],
            spec_version=problem["evaluator_version"],
            spec_hash=problem["evaluation_hash"],
            config=problem["evaluator_config"],
            run_version=problem["evaluator_run_version"],
        ),
        model=ModelProvenance(
            configuration=problem["model"],
            config_hash=problem["model_config_hash"],
            model_hash=problem["model_hash"],
            registry_entry=None,
            registry_entry_hash=None,
            prompt_template_version=problem["prompt_template_version"],
        ),
        budget=_budget(body, plan),
        seeds=tuple(plan.seeds),
        strategies=tuple(s.value for s in plan.strategies),
        runs=_runs(body, plan),
        versions=VersionsProvenance(
            experiment_schema=body["schema"],
            experiment_identity_schema=problem["schema_version"],
            runner_version=problem["runner_version"],
            ledger_version=problem["ledger_version"],
            job_schema=job_schema,
            run_versions=tuple(body["run_versions"]),
        ),
        migration=migration,
    )


def build_provenance(
    body: Mapping[str, Any],
    *,
    contract: TaskContract,
    splits: DatasetSplits,
    plan: ExperimentPlan,
    registry_entry: ModelEntry | None,
    definition_hash: str | None = None,
    job_schema: str | None = None,
) -> ProvenanceRecord:
    """The ProvenanceRecord of ``body`` from its authorities. Fails closed
    (``ProvenanceMismatch``) when any authority disagrees with what the body recorded."""
    grammar = workflow_grammar(contract)
    kinds = tuple(k.value for k in contract.workflow.stages)
    ev = contract.evaluation
    record = recorded_provenance(body, definition_hash=definition_hash, job_schema=job_schema)
    built = record.model_copy(
        update={
            "dataset": DatasetProvenance(
                dataset_id=contract.dataset.dataset_id,
                dataset_version=contract.dataset.dataset_version,
                content_hash=contract.dataset.content_hash,
                identity_hash=contract.dataset.identity_hash,
                manifest_hash=plan.dataset_manifest_hash,
            ),
            "splits": SplitsProvenance(
                splits_hash=splits.identity_hash,
                dataset_hash=splits.dataset_hash,
                method=splits.method.value,
                plan=splits.plan.model_dump(mode="json") if splits.plan is not None else None,
                rows={
                    role.value: len(s.row_ids) if (s := splits.split(role)) else 0
                    for role in SplitRole
                },
            ),
            "contract": ContractProvenance(
                contract_hash=contract.contract_hash,
                schema_version=contract.schema_version,
                task_id=contract.task_id,
                contract_version=contract.contract_version,
                objective_hash=contract.objective.identity_hash,
                constraints_hash=contract.constraints.identity_hash,
                contract=contract.model_dump(mode="json"),
            ),
            "grammar": GrammarProvenance(
                version=grammar.version,
                stage_kinds=kinds,
                grammar_hash=grammar_hash(grammar.version, kinds),
            ),
            "evaluator": EvaluatorProvenance(
                kind=ev.evaluator.value,
                spec_version=ev.evaluator_version,
                spec_hash=ev.identity_hash,
                config=ev.config,
                run_version=record.evaluator.run_version,  # the binding's: checked below
            ),
            "model": ModelProvenance(
                configuration=plan.model.model_dump(mode="json"),
                config_hash=plan.model.config_hash,
                model_hash=plan.expected_model_hash,
                registry_entry=registry_entry.model_dump(mode="json") if registry_entry else None,
                registry_entry_hash=registry_entry.identity_hash if registry_entry else None,
                prompt_template_version=plan.expected_prompt_version,
            ),
            "budget": _budget(body, plan),
            "seeds": tuple(plan.seeds),
            "strategies": tuple(s.value for s in plan.strategies),
            "runs": _runs(body, plan),
        }
    )
    built = ProvenanceRecord.model_validate(built.dump())  # re-validated, never a shallow copy
    check_provenance(built, body)
    return built


def _diff(a: Any, b: Any, path: str, out: list[str], skip: frozenset[str]) -> None:
    if path in skip:
        return
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for k in sorted(set(a) | set(b)):
            _diff(a.get(k), b.get(k), f"{path}.{k}" if path else k, out, skip)
    elif a != b:
        out.append(path)


def check_provenance(
    record: ProvenanceRecord, body: Mapping[str, Any], identity: Mapping[str, Any] | None = None
) -> None:
    """Every authority in ``record`` agrees with every other and with the body's recorded
    identities (and the durable job's bound ``identity``). Raises ``ProvenanceMismatch``
    naming each disagreeing component; nothing is repaired."""
    problems: list[str] = []
    recorded = recorded_provenance(
        body,
        definition_hash=record.definition_hash,
        job_schema=record.versions.job_schema,
        migration=record.migration,
    )
    # grammar_hash hashes stage_kinds (hash-only in a recorded identity); checked below instead
    skip = frozenset((*LEGACY_MISSING, "grammar.grammar_hash"))
    diffs: list[str] = []
    _diff(record.dump(), recorded.dump(), "", diffs, skip)
    problems += [f"{p} disagrees with the identity the experiment recorded" for p in diffs]

    problem = body["identity"]["problem"]
    if (
        canonical_hash({k: v for k, v in problem.items() if k != "problem_id"})
        != problem["problem_id"]
    ):
        problems.append("problem_id does not hash its problem identity")
    if canonical_hash(body["identity"]) != body["experiment_id"]:
        problems.append("experiment_id does not hash the experiment identity")
    if record.splits.dataset_hash != record.dataset.identity_hash:
        problems.append("splits were made for a different dataset")
    if record.grammar.grammar_hash != grammar_hash(
        record.grammar.version, record.grammar.stage_kinds
    ):
        problems.append("grammar_hash does not hash the grammar")
    problems += _check_contract(record)
    problems += _check_model(record, body)
    problems += _check_runs(record, body)
    if identity is not None:
        problems += _check_job_identity(record, identity)
    if problems:
        raise ProvenanceMismatch("; ".join(problems))


def _check_contract(record: ProvenanceRecord) -> list[str]:
    c = record.contract
    if c.contract is None:
        return [] if record.migration is not None else ["the TaskContract is missing"]
    try:
        contract = TaskContract.model_validate(c.contract)
    except ValueError as exc:
        return [f"the stored TaskContract is invalid: {exc}"]
    out = []
    checks = {
        "contract.contract_hash": (contract.contract_hash, c.contract_hash),
        "contract.task_id": (contract.task_id, c.task_id),
        "contract.objective_hash": (contract.objective.identity_hash, c.objective_hash),
        "contract.constraints_hash": (contract.constraints.identity_hash, c.constraints_hash),
        "dataset.dataset_id": (contract.dataset.dataset_id, record.dataset.dataset_id),
        "dataset.dataset_version": (
            contract.dataset.dataset_version,
            record.dataset.dataset_version,
        ),
        "dataset.content_hash": (contract.dataset.content_hash, record.dataset.content_hash),
        "dataset.identity_hash": (contract.dataset.identity_hash, record.dataset.identity_hash),
        "evaluator.spec_hash": (contract.evaluation.identity_hash, record.evaluator.spec_hash),
        "evaluator.spec_version": (
            contract.evaluation.evaluator_version,
            record.evaluator.spec_version,
        ),
        "grammar.version": (workflow_grammar(contract).version, record.grammar.version),
    }
    out += [f"{k} disagrees with the TaskContract" for k, (a, b) in checks.items() if a != b]
    if not record.synthetic and not record.evaluator.run_version.startswith(
        record.evaluator.spec_version + "+"
    ):
        out.append("evaluator.run_version is not the contract evaluator's version")
    return out


def _check_model(record: ProvenanceRecord, body: Mapping[str, Any]) -> list[str]:
    m = record.model
    out = []
    if ModelConfiguration.model_validate(m.configuration).config_hash != m.config_hash:
        out.append("model.config_hash does not hash the model configuration")
    if body["model_hashes"] != [m.model_hash]:
        out.append("the runs executed on a model other than model.model_hash")
    for v in record.versions.run_versions:
        if v.get("model_hash") != m.model_hash:
            out.append("a run reported a model_hash other than model.model_hash")
        if m.prompt_template_version is not None and (
            v.get("prompt_template_version") != m.prompt_template_version
        ):
            out.append("a run reported prompts other than model.prompt_template_version")
    if m.registry_entry is None:
        if not record.synthetic and record.migration is None:
            out.append("a real experiment must pin its model registry entry")
        return out
    try:
        entry = ModelEntry.model_validate(m.registry_entry)
        entry.check_configuration(ModelConfiguration.model_validate(m.configuration))
    except ValueError as exc:
        return [*out, f"model.registry_entry: {exc}"]
    if entry.identity_hash != m.registry_entry_hash:
        out.append("model.registry_entry_hash does not hash the registry entry")
    if entry.model_hash != m.model_hash:
        out.append("model.registry_entry pins a different model_hash")
    return out


def _check_runs(record: ProvenanceRecord, body: Mapping[str, Any]) -> list[str]:
    out = []
    expected = {(s, seed) for s in record.strategies for seed in record.seeds}
    got = [(r.strategy, r.seed) for r in record.runs]
    if sorted(got) != sorted(expected) or len(got) != len(set(got)):
        out.append("strategy runs do not cover every (strategy, seed) exactly once")
    protocol = record.budget.protocol
    for r in record.runs:
        if (r.optimizer, r.optimizer_version) != _optimizer_identity(r.strategy, body):
            out.append(f"{r.strategy}/{r.seed}: optimizer identity disagrees")
        ident = {
            "problem_id": record.problem_id,
            "protocol": protocol,
            "budget_hash": record.budget.budget_hash,
            "strategy": r.strategy,
            "optimizer": r.optimizer,
            "optimizer_version": r.optimizer_version,
            "seed": r.seed,
        }
        if canonical_hash(ident) != r.run_id:
            out.append(f"{r.strategy}/{r.seed}: run_id does not hash its run identity")
    return out


def _optimizer_identity(strategy: str, body: Mapping[str, Any]) -> tuple[str, str]:
    optimizer = make_strategy(Strategy(strategy), _plan_of(body))
    return optimizer.name, str(getattr(optimizer, "version", optimizer.name))


def _check_job_identity(record: ProvenanceRecord, identity: Mapping[str, Any]) -> list[str]:
    checks = {
        "definition_hash": (identity["definition_hash"], record.definition_hash),
        "contract_hash": (identity["contract_hash"], record.contract.contract_hash),
        "protocol_hash": (identity["protocol_hash"], record.budget.protocol_hash),
        "budget_hash": (identity["budget_hash"], record.budget.budget_hash),
        "problem_id": (identity["problem_id"], record.problem_id),
        "experiment_id": (identity["experiment_id"], record.experiment_id),
        "evaluator_version": (identity["evaluator_version"], record.evaluator.run_version),
        "pricing": (identity["pricing"], record.budget.pricing),
        "model_hash": (identity["model_hash"], record.model.model_hash),
        "synthetic": (identity["synthetic"], record.synthetic),
        "units": (
            [(u["strategy"], u["seed"], u["run_id"]) for u in identity["units"]],
            [(r.strategy, r.seed, r.run_id) for r in record.runs],
        ),
    }
    return [
        f"job identity {k} disagrees with the provenance" for k, (a, b) in checks.items() if a != b
    ]


# == the canonical artifact =======================================================================
def body_text(body: Mapping[str, Any]) -> str:
    """The canonical JSON of a result body (what is stored, hashed and linked)."""
    return canonical_json(body)


def canonical_envelope(
    body: Mapping[str, Any],
    record: ProvenanceRecord,
    parent: Mapping[str, Any],
    *,
    gzip_sha256: str | None = None,
) -> dict[str, Any]:
    text = body_text(body)
    return {
        "schema": CANONICAL_SCHEMA,
        "experiment_id": body["experiment_id"],
        "parent": dict(parent),
        "provenance_id": record.provenance_id,
        "provenance": record.dump(),
        "experiment": {
            "schema": body["schema"],
            "sha256": sha256_hex(text),
            "bytes": len(text.encode("utf-8")),
            # the deterministic projection only: equal for two runs that computed the same
            # science, whatever their clocks measured
            "scientific_sha256": canonical_hash(scientific(body)),
            "gzip_sha256": gzip_sha256,
        },
        "field_classes": FIELD_CLASSES_VERSION,
    }


def job_parent(job_id: str, definition_hash: str) -> dict[str, Any]:
    return {"kind": "job", "job_id": job_id, "definition_hash": definition_hash}


@dataclass(frozen=True)
class CanonicalArtifact:
    """A verified canonical artifact: the envelope and the body it links to."""

    envelope: dict[str, Any]
    body: dict[str, Any]

    @property
    def artifact_id(self) -> str:
        return canonical_hash(self.envelope)

    @property
    def provenance(self) -> ProvenanceRecord:
        return ProvenanceRecord.model_validate(self.envelope["provenance"])

    @property
    def provenance_id(self) -> str:
        return self.envelope["provenance_id"]

    @property
    def experiment_sha256(self) -> str:
        return self.envelope["experiment"]["sha256"]

    def ref(self) -> dict[str, str]:
        """What a promotion decision (or anything else) cites: the artifact, its provenance and
        the exact result body."""
        return {
            "artifact_id": self.artifact_id,
            "provenance_id": self.provenance_id,
            "experiment_sha256": self.experiment_sha256,
            "scientific_sha256": self.envelope["experiment"]["scientific_sha256"],
        }


def verify_artifact(
    envelope: Mapping[str, Any] | str,
    body: Mapping[str, Any] | str,
    *,
    artifact_id: str | None = None,
    identity: Mapping[str, Any] | None = None,
) -> CanonicalArtifact:
    """Load + verify, failing closed on any disagreement: the envelope schema, the body's
    canonical form and sha256 link, the artifact id, the provenance id and every provenance
    identity against the body (and the durable job's bound identity)."""
    env = (
        json.loads(envelope) if isinstance(envelope, str) else json.loads(canonical_json(envelope))
    )
    if env.get("schema") != CANONICAL_SCHEMA:
        raise ArtifactIntegrityError(f"unsupported artifact schema {env.get('schema')!r}")
    if isinstance(body, str):
        text, data = body, json.loads(body)
        if canonical_json(data) != text:
            raise ArtifactIntegrityError("the stored result body is not in canonical form")
    else:
        data = json.loads(canonical_json(body))
        text = canonical_json(data)
    link = env.get("experiment") or {}
    if sha256_hex(text) != link.get("sha256") or len(text.encode("utf-8")) != link.get("bytes"):
        raise ArtifactIntegrityError("the result body does not match the artifact's sha256 link")
    if link.get("scientific_sha256") != canonical_hash(scientific(data)):
        raise ArtifactIntegrityError("scientific_sha256 does not hash the body's scientific fields")
    if link.get("schema") != data.get("schema") or data.get("schema") != ARTIFACT_SCHEMA:
        raise ArtifactIntegrityError(f"unsupported result body schema {data.get('schema')!r}")
    if artifact_id is not None and canonical_hash(env) != artifact_id:
        raise ArtifactIntegrityError("the artifact envelope does not hash to its artifact_id")
    try:
        record = ProvenanceRecord.model_validate(env["provenance"])
    except ValueError as exc:
        raise ArtifactIntegrityError(f"invalid provenance record: {exc}") from exc
    if record.dump() != env["provenance"] or record.provenance_id != env.get("provenance_id"):
        raise ArtifactIntegrityError("the provenance record does not hash to its provenance_id")
    if not (env.get("experiment_id") == data["experiment_id"] == record.experiment_id):
        raise ArtifactIntegrityError("envelope, body and provenance name different experiments")
    if env.get("field_classes") != FIELD_CLASSES_VERSION:
        raise ArtifactIntegrityError(f"unknown field classes {env.get('field_classes')!r}")
    try:
        check_provenance(record, data, identity)
    except ProvenanceMismatch as exc:
        raise ArtifactIntegrityError(f"provenance does not verify: {exc}") from exc
    return CanonicalArtifact(env, data)


# == field classes ================================================================================
def field_class(parts: Sequence[str | int]) -> str:
    """The class of one leaf, from its path inside a result body (see the module docstring)."""
    keys = [p for p in parts if isinstance(p, str)]
    if any(k in DECLARED_KEYS for k in keys):
        return SCIENTIFIC
    if keys and keys[-1] in MEASURED_KEYS:
        return MEASURED
    if any(k in CLOCK_KEYS for k in keys):
        return CLOCK
    if any(k in MEASURED_KEYS for k in keys):
        return MEASURED
    return SCIENTIFIC


def leaves(obj: Any, prefix: tuple[str | int, ...] = ()):
    """``(path parts, value)`` of every leaf (dicts and lists are containers)."""
    if isinstance(obj, Mapping):
        for k in sorted(obj):
            yield from leaves(obj[k], (*prefix, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, (*prefix, i))
    else:
        yield prefix, obj


def project(obj: Any, keep: Sequence[str], prefix: tuple[str | int, ...] = ()) -> Any:
    """``obj`` with only the leaves whose ``field_class`` is in ``keep`` (containers kept)."""
    if isinstance(obj, Mapping):
        return {k: project(v, keep, (*prefix, k)) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [project(v, keep, (*prefix, i)) for i, v in enumerate(obj)]
    return obj if field_class(prefix) in keep else None


def scientific(obj: Any) -> Any:
    """The deterministic projection: what must reproduce exactly."""
    return project(obj, (SCIENTIFIC,))


def class_counts(body: Mapping[str, Any]) -> dict[str, int]:
    out = {SCIENTIFIC: 0, MEASURED: 0, CLOCK: 0}
    for parts, _ in leaves(body):
        out[field_class(parts)] += 1
    return out


def differences(a: Any, b: Any, prefix: tuple[str | int, ...] = ()) -> list[tuple[Any, ...]]:
    """Paths (as part tuples) where two JSON documents differ."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((*prefix, k))
            else:
                out += differences(a[k], b[k], (*prefix, k))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [prefix]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += differences(x, y, (*prefix, i))
        return out
    return [] if a == b else [prefix]


# == traceability =================================================================================
# ``summary.by_strategy.<s>.metrics.<metric>`` / ``per_seed.<seed>.<metric>``: where each per-seed
# value lives inside that seed's strategy run (``optimization_experiment._per_seed``).
PER_SEED_SOURCE: dict[str, str] = {
    "champion_genome_hash": "champion.genome_hash",
    "champion_feasible": "champion.feasible",
    "champion_validation_score": "champion.validation.score_mean",
    "champion_validation_pass_rate": "champion.validation.pass_rate",
    "candidate_evaluations": "usage.candidate_evaluations",
    "model_calls": "usage.model_calls",
    "prompt_tokens": "usage.prompt_tokens",
    "completion_tokens": "usage.completion_tokens",
    "tokens": "usage.tokens",
    "cost": "usage.cost",
    "tokens_per_example": "efficiency.tokens_per_example",
    "tokens_per_candidate": "efficiency.tokens_per_candidate",
    "mean_latency_s": "latency_s.mean",
    "p50_latency_s": "latency_s.p50",
    "p95_latency_s": "latency_s.p95",
    "max_latency_s": "latency_s.max",
    "execution_s": "timing.execution_s",
    "mean_candidate_e2e_s": "timing.candidate_e2e_s.mean",
    "e2e_wall_s": "timing.e2e_wall_s",
    "optimizer_overhead_s": "timing.optimizer_overhead_s",
    "evaluator_overhead_s": "timing.evaluator_overhead_s",
    "examples_per_s": "timing.examples_per_s",
    "candidates_per_min": "timing.candidates_per_min",
    "seed": "seed",
    "run_id": "run_id",
    "stop_reason": "stop_reason",
}
_COUNTERS = ("model_calls", "prompt_tokens", "completion_tokens", "tokens")
_MISSING = object()


def _get(obj: Any, parts: Sequence[str | int]) -> Any:
    """The value at raw path ``parts`` (dict keys / list indices), or ``_MISSING``."""
    for p in parts:
        if isinstance(obj, Mapping) and isinstance(p, str) and p in obj:
            obj = obj[p]
        elif isinstance(obj, list) and isinstance(p, int) and 0 <= p < len(obj):
            obj = obj[p]
        else:
            return _MISSING
    return obj


def _pointer(parts: Sequence[str | int]) -> str:
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def _step(node: Any, seg: str, where: str) -> tuple[Any, str | int]:
    if isinstance(node, Mapping):
        if seg not in node:
            raise TraceError(f"{where}: no field {seg!r}")
        return node[seg], seg
    if isinstance(node, list):
        if not seg.isdigit() or int(seg) >= len(node):
            raise TraceError(f"{where}: {seg!r} is not an index of a list of {len(node)}")
        return node[int(seg)], int(seg)
    raise TraceError(f"{where}: {seg!r} goes below a value")


def _keyed(items: list[Any], field: str, seg: str, where: str) -> int:
    for i, item in enumerate(items):
        if str(item.get(field)) == seg:
            return i
    raise TraceError(f"{where}: no entry with {field} {seg}")


def _walk(root: Any, segs: Sequence[str], base: tuple[str | int, ...] = ()):
    """Resolve ``segs`` under ``root``: ``candidate.<evaluation>`` / ``curve.<evaluation>`` /
    ``per_seed.<seed>`` / ``best_so_far_curve.<evaluation>`` address list items by their stable
    key; anything else is a dict key or a list index."""
    node, parts, i = root, list(base), 0
    while i < len(segs):
        seg, where = segs[i], ".".join(segs[: i + 1])
        keyed = {"candidate": ("candidates", "evaluation"), "curve": ("curve", "evaluation")}
        keyed |= {
            "per_seed": ("per_seed", "seed"),
            "best_so_far_curve": ("best_so_far_curve", "evaluation"),
        }
        if seg in keyed and isinstance(node, Mapping) and i + 1 < len(segs):
            field, key = keyed[seg]
            items = node.get(field)
            if isinstance(items, list):
                j = _keyed(items, key, segs[i + 1], where)
                node, parts, i = items[j], [*parts, field, j], i + 2
                continue
        node, part = _step(node, seg, where)
        parts.append(part)
        i += 1
    return node, tuple(parts)


def _run_index(body: Mapping[str, Any], strategy: str, seed: str) -> int:
    for i, r in enumerate(body["runs"]):
        if r["strategy"] == strategy and str(r["seed"]) == seed:
            return i
    raise TraceError(f"no strategy run {strategy}/seed {seed}")


def resolve(art: CanonicalArtifact, path: str) -> dict[str, Any]:
    """``path`` -> the stored field it names. Raises ``TraceError`` if it names nothing."""
    segs = path.split(".")
    if not path or any(not s for s in segs):
        raise TraceError(f"invalid field path {path!r}")
    if segs[0] == "provenance":
        value, parts = _walk(art.envelope["provenance"], segs[1:])
        return {"document": "provenance", "parts": parts, "value": value, "field_class": SCIENTIFIC}
    if segs[0] == "artifact":
        value, parts = _walk(art.envelope, segs[1:])
        return {"document": "artifact", "parts": parts, "value": value, "field_class": SCIENTIFIC}
    body = art.body
    if segs[0] == "strategy":
        if len(segs) < 4 or segs[2] != "seed":
            raise TraceError("a strategy-run path is strategy.<strategy>.seed.<seed>[.field...]")
        i = _run_index(body, segs[1], segs[3])
        value, parts = _walk(body["runs"][i], segs[4:], ("runs", i))
    else:
        value, parts = _walk(body, segs)
    return {
        "document": "experiment",
        "parts": parts,
        "value": value,
        "field_class": field_class(parts),
    }


def trace(art: CanonicalArtifact, path: str) -> dict[str, Any]:
    """Where a number comes from: the artifact and provenance it belongs to, its canonical JSON
    pointer, its value and field class, the strategy run it belongs to, the stored attempt that
    measured it (durable jobs), and - for a derived number - its sources and the recomputation."""
    hit = resolve(art, path)
    parts = hit["parts"]
    out: dict[str, Any] = {
        "path": path,
        "artifact_id": art.artifact_id,
        "experiment_id": art.body["experiment_id"],
        "experiment_sha256": art.experiment_sha256,
        "provenance_id": art.provenance_id,
        "document": hit["document"],
        "pointer": _pointer(parts),
        "value": hit["value"],
        "field_class": hit["field_class"],
        "run": None,
        "evidence": None,
        "derivation": None,
    }
    if hit["document"] != "experiment":
        return out
    if parts[:1] == ("runs",):
        run = art.body["runs"][parts[1]]
        out["run"] = {"strategy": run["strategy"], "seed": run["seed"], "run_id": run["run_id"]}
        out["evidence"] = _evidence(art, run, parts[2:])
    out["derivation"] = derivation(art.body, parts)
    return out


def _evidence(art: CanonicalArtifact, run: Mapping[str, Any], rest: Sequence[Any]):
    """For a field of one stored workflow run (``candidates/<i>/runs/<j>/...``): the run's
    deterministic identity and, for a durable job, its write-ahead ``attempt_id``."""
    if len(rest) < 4 or rest[0] != "candidates" or rest[2] != "runs":
        return None
    cand = run["candidates"][rest[1]]
    entry = cand["runs"][rest[3]]
    seed = run_seed(run["seed"], cand["genome_hash"], entry["row_id"], entry["trial"])
    out = {
        "genome_hash": cand["genome_hash"],
        "row_id": entry["row_id"],
        "split": entry["split"],
        "trial": entry["trial"],
        "run_seed": seed,
        "attempt": entry["attempt"],
        "attempt_id": None,
    }
    parent = art.envelope["parent"]
    if parent.get("kind") == "job":
        out["attempt_id"] = canonical_hash(
            [
                parent["job_id"],
                run["run_id"],
                cand["genome_hash"],
                entry["row_id"],
                entry["trial"],
                seed,
                entry["attempt"],
            ]
        )  # = experiments.jobs.attempt_identity
    return out


def _path_of(body: Mapping[str, Any], i: int, suffix: str) -> str:
    run = body["runs"][i]
    return f"strategy.{run['strategy']}.seed.{run['seed']}.{suffix}"


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _stats(values: Sequence[Any]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def derivation(body: Mapping[str, Any], parts: Sequence[Any]) -> dict[str, Any] | None:
    """How a derived number is computed from stored sources (``None`` for a stored fact)."""
    p = list(parts)
    if p[:2] == ["summary", "by_strategy"] and len(p) >= 4:
        return _summary_derivation(body, p)
    if (
        p[:1] == ["runs"]
        and len(p) >= 4
        and p[2] == "usage"
        and p[3] in (*_COUNTERS, "workflow_runs")
    ):
        i, key = p[1], p[3]
        cands = body["runs"][i]["candidates"]
        field = "workflow_runs" if key == "workflow_runs" else key
        sources = [_path_of(body, i, f"candidate.{c['evaluation']}.usage.{field}") for c in cands]
        return _result("sum/1", sources, sum(c["usage"][field] for c in cands), body, p)
    if p[:1] == ["runs"] and len(p) >= 4 and p[2:4] == ["usage", "candidate_evaluations"]:
        cands = body["runs"][p[1]]["candidates"]
        sources = [_path_of(body, p[1], f"candidate.{c['evaluation']}") for c in cands]
        return _result("count/1", sources, len(cands), body, p)
    if p[:1] == ["runs"] and len(p) >= 6 and p[2] == "candidates" and p[4] == "usage":
        cand = body["runs"][p[1]]["candidates"][p[3]]
        if p[5] in _COUNTERS:
            prefix = f"candidate.{cand['evaluation']}.runs"
            sources = [
                _path_of(body, p[1], f"{prefix}.{j}.{p[5]}") for j in range(len(cand["runs"]))
            ]
            return _result("sum/1", sources, sum(e[p[5]] for e in cand["runs"]), body, p)
    if p[:1] == ["runs"] and len(p) >= 5 and p[2] == "curve":
        return _curve_derivation(body, p)
    if p[:1] == ["runs"] and len(p) >= 4 and p[2:4] == ["champion", "genome_hash"]:
        run = body["runs"][p[1]]
        last = run["curve"][-1]
        src = _path_of(body, p[1], f"curve.{last['evaluation']}.best_so_far_genome_hash")
        return _result("final_best_so_far/1", [src], last["best_so_far_genome_hash"], body, p)
    if p[:1] == ["runs"] and len(p) >= 5 and p[2:5] == ["champion", "validation", "score_mean"]:
        run = body["runs"][p[1]]
        last = run["curve"][-1]
        src = _path_of(body, p[1], f"curve.{last['evaluation']}.best_so_far_score")
        return _result("final_best_so_far/1", [src], last["best_so_far_score"], body, p)
    return None


def _source(run: Mapping[str, Any], src: str) -> Any:
    value = _get(run, src.split("."))
    return None if value is _MISSING else value


def _result(rule: str, sources: list[str], recomputed: Any, body: Any, parts: Sequence[Any]):
    value = _get(body, parts)
    return {
        "rule": rule,
        "sources": sources,
        "recomputed": recomputed,
        "consistent": _same(value, recomputed),
    }


def _summary_derivation(body: Mapping[str, Any], p: list[Any]) -> dict[str, Any] | None:
    strategy = p[2]
    runs = sorted(
        ((i, r) for i, r in enumerate(body["runs"]) if r["strategy"] == strategy),
        key=lambda x: x[1]["seed"],
    )
    if len(p) >= 6 and p[3] == "metrics" and p[4] in PER_SEED_SOURCE:
        sources, values = [], []
        for i, r in runs:
            src = PER_SEED_SOURCE[p[4]]
            value = _source(r, src)
            if value is not None:
                sources.append(_path_of(body, i, src))
                values.append(value)
        return _result("seed_stats/1", sources, _stats(values)[p[5]], body, p)
    if len(p) >= 6 and p[3] == "per_seed":
        i, r = runs[p[4]]
        src = PER_SEED_SOURCE.get(p[5])
        if src is None:
            return None
        value = _source(r, src)
        return _result("copy/1", [_path_of(body, i, src)], value, body, p)
    if len(p) == 6 and p[3] == "best_so_far_curve" and p[5] == "evaluation":
        x = body["summary"]["by_strategy"][strategy]["best_so_far_curve"][p[4]]["evaluation"]
        sources = [
            _path_of(body, i, f"curve.{x}.evaluation") for i, r in runs if len(r["curve"]) >= x
        ]
        return _result("axis/1", sources, x if sources else None, body, p)
    if len(p) >= 7 and p[3] == "best_so_far_curve" and p[5] == "best_so_far_score":
        x = body["summary"]["by_strategy"][strategy]["best_so_far_curve"][p[4]]["evaluation"]
        sources, values = [], []
        for i, r in runs:
            if len(r["curve"]) >= x:
                sources.append(_path_of(body, i, f"curve.{x}.best_so_far_score"))
                values.append(r["curve"][x - 1]["best_so_far_score"])
        return _result("seed_stats/1", sources, _stats(values)[p[6]], body, p)
    if len(p) == 4 and p[3] == "feasible_champions":
        sources = [_path_of(body, i, "champion.feasible") for i, _ in runs]
        count = sum(bool(r["champion"] and r["champion"]["feasible"]) for _, r in runs)
        return _result("count_true/1", sources, count, body, p)
    return None


def _curve_derivation(body: Mapping[str, Any], p: list[Any]) -> dict[str, Any] | None:
    i, j, key = p[1], p[3], p[4]
    cands = body["runs"][i]["candidates"][: j + 1]
    if key.startswith("cumulative_") and key[len("cumulative_") :] in _COUNTERS:
        field = key[len("cumulative_") :]
        sources = [_path_of(body, i, f"candidate.{c['evaluation']}.usage.{field}") for c in cands]
        return _result("running_sum/1", sources, sum(c["usage"][field] for c in cands), body, p)
    if key == "score":
        c = cands[-1]
        src = _path_of(body, i, f"candidate.{c['evaluation']}.validation.score_mean")
        return _result("copy/1", [src], c["validation"]["score_mean"], body, p)
    if key == "max_score_so_far":
        sources = [
            _path_of(body, i, f"candidate.{c['evaluation']}.validation.score_mean") for c in cands
        ]
        return _result(
            "running_max/1", sources, max(c["validation"]["score_mean"] for c in cands), body, p
        )
    return None


# == declared metrics from stored execution evidence ==============================================
def recompute_declared(body: Mapping[str, Any]) -> int:
    """Re-derive every declared aggregate from the per-run execution evidence the body carries -
    the summary from the strategy runs, each run's usage and curve from its candidates, each
    candidate's usage from its workflow runs, every selected workflow from its curve - and fail
    closed (``ArtifactIntegrityError``) on the first disagreement. Returns the number of checks."""
    checks = 0
    problems: list[str] = []

    def expect(ok: bool, what: str) -> None:
        nonlocal checks
        checks += 1
        if not ok:
            problems.append(what)

    runs = body["runs"]
    # Every metric the summary DECLARES must re-derive; metrics ``summarize`` gained after an
    # artifact was frozen (e.g. p50 latency) are not part of what that artifact declared.
    recomputed = summarize(runs)
    for parts, value in leaves(body["summary"]):
        expect(
            _get(recomputed, parts) == value,
            f"summary {_pointer(parts)} does not re-derive from the runs",
        )
    expect(
        body["model_hashes"] == sorted({h for r in runs for h in r["model_hashes"]}),
        "model_hashes",
    )
    expect(body["test_runs"] == sum(r["split_usage"]["test_runs"] for r in runs), "test_runs")
    for r in runs:
        name = f"{r['strategy']}/{r['seed']}"
        cands = r["candidates"]
        expect(r["evaluated_genome_hashes"] == [c["genome_hash"] for c in cands], f"{name} order")
        expect(r["distinct_genomes"] == len({c["genome_hash"] for c in cands}), f"{name} distinct")
        expect(r["usage"]["candidate_evaluations"] == len(cands), f"{name} candidate count")
        for k in (*_COUNTERS, "workflow_runs"):
            expect(r["usage"][k] == sum(c["usage"][k] for c in cands), f"{name} usage.{k}")
        expect(
            [p["evaluation"] for p in r["curve"]] == [c["evaluation"] for c in cands],
            f"{name} curve",
        )
        running = dict.fromkeys(_COUNTERS, 0)
        best = -math.inf
        for c, point in zip(cands, r["curve"], strict=True):
            ce = f"{name} candidate {c['evaluation']}"
            for k in _COUNTERS:
                expect(c["usage"][k] == sum(e[k] for e in c["runs"]), f"{ce} usage.{k}")
                running[k] += c["usage"][k]
                expect(point[f"cumulative_{k}"] == running[k], f"{ce} curve cumulative_{k}")
            expect(c["usage"]["workflow_runs"] == len(c["runs"]), f"{ce} workflow_runs")
            final = _final_entries(c["runs"], SplitRole.VALIDATION.value)
            scores = [e["score"] for e in final]
            expect(
                bool(scores) and _same(c["validation"]["score_mean"], statistics.fmean(scores)),
                f"{ce} validation score_mean",
            )
            expect(_same(point["score"], c["validation"]["score_mean"]), f"{ce} curve score")
            best = max(best, c["validation"]["score_mean"])
            expect(_same(point["max_score_so_far"], best), f"{ce} curve max_score_so_far")
        champ = r["champion"]
        if r["curve"]:
            last = r["curve"][-1]
            expect(
                champ is not None and champ["genome_hash"] == last["best_so_far_genome_hash"],
                f"{name} champion is not the final best-so-far workflow",
            )
            expect(
                champ is not None
                and _same(champ["validation"]["score_mean"], last["best_so_far_score"]),
                f"{name} champion validation score",
            )
        else:
            expect(champ is None, f"{name} champion without candidates")
    if problems:
        raise ArtifactIntegrityError(
            "declared metrics do not re-derive from the stored evidence: " + "; ".join(problems[:5])
        )
    return checks


def _final_entries(entries: Sequence[Mapping[str, Any]], split: str) -> list[Mapping[str, Any]]:
    """The last attempt of every (row, trial) on ``split`` (MODEL_ERROR retries superseded)."""
    last: dict[tuple[str, int], Mapping[str, Any]] = {}
    for e in entries:
        if e["split"] == split:
            last[(e["row_id"], e["trial"])] = e
    return list(last.values())


# == frozen artifacts (written by ``write_compact`` before provenance existed) ===================
def load_frozen(out_dir: Path) -> CanonicalArtifact:
    """A committed ``summary.json`` + ``experiment.json.gz`` directory as an explicitly migrated
    canonical artifact. Both recorded digests are verified first (``load_compact``). The body is
    unchanged; its provenance is the identity it recorded, marked ``Migration`` with every
    authority that survives as a hash only. Deterministic: the same directory always migrates
    to the same ``artifact_id``."""
    summary, body = load_compact(out_dir)
    meta = summary["artifact"]
    migration = Migration(
        source_schema=f"{body['schema']} via {summary['schema']}",
        method=MIGRATION_METHOD,
        missing=LEGACY_MISSING,
    )
    record = recorded_provenance(body, migration=migration)
    parent = {
        "kind": "frozen",
        "summary_schema": summary["schema"],
        "json_sha256": meta["json_sha256"],
        "gz_sha256": meta["gz_sha256"],
    }
    env = canonical_envelope(body, record, parent, gzip_sha256=meta["gz_sha256"])
    return verify_artifact(env, body, artifact_id=canonical_hash(env))
