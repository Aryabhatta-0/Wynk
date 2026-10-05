"""Canonical experiment identity: one hash for "the same optimization problem, set up the same way".

Built only from immutable, authoritative inputs - dataset identity, splits, task contract,
workflow grammar version, model configuration, evaluator, objective and constraints - and hashed
through ``core.canonical`` (key-sorted JSON), so it never depends on dict ordering, timestamps,
filesystem paths or descriptive metadata (dataset name / metadata are excluded by
``DatasetSpec.authoritative``). Intended consumers: reproducibility, caching, optimizer resume,
memory isolation and champion comparison.

The components are kept side by side (not only folded into ``contract_hash``) so two identities
can be diffed to see WHAT changed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.canonical import canonical_hash
from core.dataset import DatasetSplits
from core.task_contract import TaskContract

EXPERIMENT_IDENTITY_SCHEMA_VERSION = "experiment/1"


class ModelConfiguration(BaseModel):
    """The model setup a workflow executes with. Hashed; never interpreted here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class ExperimentIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["experiment/1"] = EXPERIMENT_IDENTITY_SCHEMA_VERSION
    dataset_hash: str
    splits_hash: str
    task_id: str
    contract_version: int
    task_contract_hash: str
    grammar_version: str = Field(min_length=1)
    model_config_hash: str = Field(min_length=1)
    evaluator_version: str
    evaluation_hash: str
    objective_hash: str
    constraints_hash: str

    @field_validator("dataset_hash", "splits_hash", "task_contract_hash")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("identity components must be non-empty")
        return v

    @property
    def experiment_id(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))

    def differences(self, other: ExperimentIdentity) -> tuple[str, ...]:
        """Names of the components that differ (empty iff the identities are equal)."""
        a: dict[str, Any] = self.model_dump()
        b: dict[str, Any] = other.model_dump()
        return tuple(k for k in sorted(a) if a[k] != b[k])


def experiment_identity(
    contract: TaskContract,
    splits: DatasetSplits,
    *,
    grammar_version: str,
    model: ModelConfiguration,
) -> ExperimentIdentity:
    """Identity of optimizing ``contract`` on ``splits`` under a grammar and model setup."""
    if splits.dataset_hash != contract.dataset.identity_hash:
        raise ValueError("splits were made for a different dataset than the contract's")
    return ExperimentIdentity(
        dataset_hash=contract.dataset.identity_hash,
        splits_hash=splits.identity_hash,
        task_id=contract.task_id,
        contract_version=contract.contract_version,
        task_contract_hash=contract.contract_hash,
        grammar_version=grammar_version,
        model_config_hash=model.config_hash,
        evaluator_version=contract.evaluation.evaluator_version,
        evaluation_hash=contract.evaluation.identity_hash,
        objective_hash=contract.objective.identity_hash,
        constraints_hash=contract.constraints.identity_hash,
    )
