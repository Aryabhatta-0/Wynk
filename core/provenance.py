"""ProvenanceRecord: the one immutable answer to "what exactly produced this experiment?".

    dataset -> splits -> TaskContract -> grammar -> evaluator -> model -> budget/pricing
            -> optimizer + seed per strategy run -> candidates -> selected workflow

Every component is copied from its existing authority and never re-derived by hand:
``DatasetSpec`` (inside the ``TaskContract``), ``DatasetSplits``, ``TaskContract.contract_hash``,
``workflow_grammar(contract)``, ``EvaluationSpec``, the ``ModelRegistry`` entry pinned by
``ExperimentPlan.expected_model_hash``, ``ExperimentBudget`` + pricing identity, the optimizer
each ``Strategy`` is built with, and the strategy-run identities (``run_id``) of the experiment.
``experiments.provenance`` builds the record and cross-checks every component against the others
(``ProvenanceMismatch``: two identities that disagree fail closed, nothing is repaired).

The record is versioned (``schema``) and hashed (``provenance_id``) through ``core.canonical``,
so it never depends on dict order, timestamps or the environment. It holds no clock-derived
value and no secret (a registry entry has no API key; ``endpoint`` is the registry's own data).

Pure data: no I/O, no optimizer or experiment imports.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.canonical import canonical_hash

PROVENANCE_SCHEMA = "wynk-provenance/1"


class ProvenanceMismatch(ValueError):
    """Two authorities of one experiment disagree, or a record does not match its evidence."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetProvenance(_Frozen):
    dataset_id: str
    dataset_version: int
    content_hash: str  # sha256 of the dataset bytes
    identity_hash: str  # DatasetSpec.identity_hash (what splits are made for)
    manifest_hash: str | None = None  # frozen dataset manifest (benchmarks), when there is one


class SplitsProvenance(_Frozen):
    splits_hash: str  # DatasetSplits.identity_hash
    dataset_hash: str  # the DatasetSpec identity the splits were made for
    method: str | None  # e.g. seeded_hash/1; None for a migrated artifact (hash only)
    plan: dict[str, Any] | None  # the split config (seed, fractions), when seeded
    rows: dict[str, int]  # rows per role


class ContractProvenance(_Frozen):
    contract_hash: str
    schema_version: str | None
    task_id: str
    contract_version: int
    objective_hash: str
    constraints_hash: str
    # The full TaskContract. ``None`` only for an artifact migrated from before provenance
    # existed, whose contract survives as its hash alone (``migration`` says so).
    contract: dict[str, Any] | None


class GrammarProvenance(_Frozen):
    version: str  # workflow_grammar(contract).version
    stage_kinds: tuple[str, ...] | None  # the contract's stage vocabulary
    grammar_hash: str  # canonical_hash({version, stage_kinds})


class EvaluatorProvenance(_Frozen):
    kind: str  # EvaluationSpec.evaluator
    spec_version: str  # EvaluationSpec.evaluator_version
    spec_hash: str  # EvaluationSpec.identity_hash
    config: dict[str, Any]
    run_version: str  # what every EvaluatedRun reports (spec version + fitness, or synthetic)


class ModelProvenance(_Frozen):
    configuration: dict[str, Any]  # ExperimentPlan.model (ModelConfiguration)
    config_hash: str
    model_hash: str  # ExperimentPlan.expected_model_hash = every RunVersions.model_hash
    registry_entry: dict[str, Any] | None  # the pinned ModelEntry (None: synthetic stand-in)
    registry_entry_hash: str | None
    prompt_template_version: str | None


class BudgetProvenance(_Frozen):
    budget: dict[str, Any]  # ExperimentBudget
    budget_hash: str
    pricing: str | None  # PricingPolicy.identity; None = cost unknown
    protocol: dict[str, Any]  # ExperimentPlan.protocol(): what every strategy run is held to
    protocol_hash: str
    protocol_id: str | None  # hash of the pre-registered protocol document, when there is one
    fixed_baseline_rule: str
    trials: int
    batch_size: int


class StrategyRunProvenance(_Frozen):
    strategy: str
    seed: int
    run_id: str
    optimizer: str
    optimizer_version: str
    optimizer_config: dict[str, Any] | None  # e.g. the MMAS ACOConfig; None: no parameters
    candidates: int  # candidate evaluations
    candidate_order_hash: str  # canonical_hash(evaluated genome hashes, in order)
    selected_genome_hash: str | None  # the run's validation-selected workflow


class VersionsProvenance(_Frozen):
    experiment_schema: str  # the result body's schema (wynk-optimization-experiment/N)
    experiment_identity_schema: str  # core.experiment ExperimentIdentity schema
    runner_version: str
    ledger_version: str
    job_schema: str | None  # wynk-experiment-job/N for a durable job
    run_versions: tuple[dict[str, Any], ...]  # RunVersions: model, prompts, compiler, grammar


class Migration(_Frozen):
    """Set only on a record built from an artifact older than provenance (never on a new one)."""

    source_schema: str
    method: str
    missing: tuple[str, ...]  # authorities that survive as a hash only


class ProvenanceRecord(_Frozen):
    schema_: Literal["wynk-provenance/1"] = Field(PROVENANCE_SCHEMA, alias="schema")
    experiment_id: str
    problem_id: str
    definition_hash: str | None  # the durable job's immutable definition, when there is one
    synthetic: bool
    dataset: DatasetProvenance
    splits: SplitsProvenance
    contract: ContractProvenance
    grammar: GrammarProvenance
    evaluator: EvaluatorProvenance
    model: ModelProvenance
    budget: BudgetProvenance
    seeds: tuple[int, ...]
    strategies: tuple[str, ...]
    runs: tuple[StrategyRunProvenance, ...]  # one per (strategy, seed), artifact order
    versions: VersionsProvenance
    migration: Migration | None = None

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    @property
    def provenance_id(self) -> str:
        return canonical_hash(self.dump())

    def run(self, strategy: str, seed: int) -> StrategyRunProvenance:
        for r in self.runs:
            if (r.strategy, r.seed) == (strategy, seed):
                return r
        raise KeyError(f"no strategy run {strategy}/seed {seed}")


def grammar_hash(version: str, stage_kinds: tuple[str, ...] | None) -> str:
    return canonical_hash({"version": version, "stage_kinds": stage_kinds})
