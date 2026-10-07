"""Dataset contract: what a dataset IS (columns, roles, content identity) and how it is SPLIT.

``DatasetSpec`` describes a user-provided dataset independently of any benchmark. The dataset
bytes stay outside this model; ``content_hash`` (sha256 of those bytes, computed by whoever holds
them) pins the contract to exactly one version of the data. There is deliberately no path field:
a filesystem location is never identity.

``DatasetSplits`` assigns row ids to the three split roles. The role decides what a split's
results may be used for, and that policy is fixed here, in code:

    optimization -> optimizer feedback, selection, reporting
    validation   -> selection / promotion, reporting      (never optimizer feedback)
    test         -> reporting only                         (final, held out)

``DatasetSplits.optimizer_view()`` is the only split object optimizer-side code needs. It has no
test field at all, so final-test rows cannot reach an optimizer through this abstraction.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

from core.canonical import canonical_hash, canonical_json, sha256_hex
from core.task_spec import FieldType

DATASET_SPEC_SCHEMA_VERSION = "datasetspec/1"
DATASET_SPLITS_SCHEMA_VERSION = "datasetsplits/1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SLUG = r"^[a-z0-9][a-z0-9_.-]{0,127}$"  # ids: never a path, never whitespace
COLUMN_NAME = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"  # identifier-like: maps 1:1 onto schema fields
MAX_METADATA_KEYS = 32

# Fields that describe a dataset for humans but never change what an experiment computes.
NON_AUTHORITATIVE_FIELDS = frozenset({"name", "metadata"})


def _check_sha256(value: str, what: str) -> str:
    if not _SHA256.match(value):
        raise ValueError(f"{what} must be a lowercase sha256 hex digest")
    return value


class DatasetFormat(StrEnum):
    CSV = "csv"
    JSONL = "jsonl"
    # The frozen benchmark snapshot layout (benchmarks/snapshots). Legacy adapter only.
    WYNK_SNAPSHOT = "wynk_snapshot"


class ColumnType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    STRING_LIST = "string_list"
    JSON = "json"  # opaque structured value; cannot be mapped onto a schema field

    def field_type(self) -> FieldType | None:
        """The schema field type this column carries, or ``None`` if it has none (``json``)."""
        return None if self is ColumnType.JSON else FieldType(self.value)


class ColumnSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=COLUMN_NAME)
    type: ColumnType
    nullable: bool = False


class DatasetSpec(BaseModel):
    """Immutable, versioned description + identity of one dataset version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["datasetspec/1"] = DATASET_SPEC_SCHEMA_VERSION
    dataset_id: str = Field(pattern=SLUG)
    dataset_version: PositiveInt
    name: str = Field(min_length=1, max_length=200)  # display only (non-authoritative)
    content_hash: str  # sha256 hex of the dataset bytes

    format: DatasetFormat
    columns: tuple[ColumnSpec, ...] = Field(min_length=1)

    id_column: str | None = None  # stable row ids; required to split the dataset
    input_columns: tuple[str, ...] = Field(min_length=1)
    target_columns: tuple[str, ...] = Field(min_length=1)  # evaluation-side only
    context_columns: tuple[str, ...] = ()

    row_count: PositiveInt

    # Free-form descriptive data. Never part of identity, never read by execution.
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("content_hash")
    @classmethod
    def _content_hash(cls, v: str) -> str:
        return _check_sha256(v, "content_hash")

    @field_validator("metadata")
    @classmethod
    def _metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        if len(v) > MAX_METADATA_KEYS:
            raise ValueError(f"metadata may hold at most {MAX_METADATA_KEYS} keys")
        for key, value in v.items():
            if not re.match(COLUMN_NAME, key):
                raise ValueError(f"metadata key {key!r} is not an identifier")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"metadata value for {key!r} must be finite")
        return v

    @model_validator(mode="after")
    def _column_roles(self) -> DatasetSpec:
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("column names must be unique")
        known = set(names)
        roles = {
            "input_columns": self.input_columns,
            "target_columns": self.target_columns,
            "context_columns": self.context_columns,
            "id_column": (self.id_column,) if self.id_column is not None else (),
        }
        seen: dict[str, str] = {}
        for role, cols in roles.items():
            if len(set(cols)) != len(cols):
                raise ValueError(f"{role} repeats a column")
            for col in cols:
                if col not in known:
                    raise ValueError(f"{role} references unknown column {col!r}")
                if col in seen:
                    raise ValueError(f"column {col!r} is both {seen[col]} and {role}")
                seen[col] = role
        by_name = {c.name: c for c in self.columns}
        if self.id_column is not None:
            id_col = by_name[self.id_column]
            if id_col.nullable or id_col.type not in (ColumnType.STRING, ColumnType.INTEGER):
                raise ValueError("id_column must be a non-nullable string or integer column")
        return self

    # -- views --------------------------------------------------------------------------------
    def column(self, name: str) -> ColumnSpec:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    # -- identity -----------------------------------------------------------------------------
    def authoritative(self) -> dict[str, Any]:
        """Everything that defines the dataset contract; excludes display name and metadata."""
        return self.model_dump(mode="json", exclude=set(NON_AUTHORITATIVE_FIELDS))

    @property
    def identity_hash(self) -> str:
        """Changes iff content, format, columns, roles, row count, id or version change."""
        return canonical_hash(self.authoritative())

    def canonical_json(self) -> str:
        """Full serialization (including non-authoritative fields), key-order independent."""
        return canonical_json(self.model_dump(mode="json"))


# -- splits -------------------------------------------------------------------------------------
class SplitRole(StrEnum):
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    TEST = "test"


class SplitUse(StrEnum):
    OPTIMIZER_FEEDBACK = "optimizer_feedback"  # results may update optimizer state
    SELECTION = "selection"  # results may choose a candidate among several
    REPORTING = "reporting"  # results may be reported
    # results may decide the binary held-out gate of ONE already-selected challenger against the
    # incumbent champion (``experiments.promotion``): never choose among several candidates
    PROMOTION_GATE = "promotion_gate"


ALLOWED_USES: dict[SplitRole, frozenset[SplitUse]] = {
    SplitRole.OPTIMIZATION: frozenset(
        {SplitUse.OPTIMIZER_FEEDBACK, SplitUse.SELECTION, SplitUse.REPORTING}
    ),
    SplitRole.VALIDATION: frozenset({SplitUse.SELECTION, SplitUse.REPORTING}),
    SplitRole.TEST: frozenset({SplitUse.REPORTING, SplitUse.PROMOTION_GATE}),
}


class SplitAccessError(PermissionError):
    """A split's results were about to be used for something its role forbids."""


def require_use(role: SplitRole, use: SplitUse) -> None:
    """Fail closed unless ``role`` permits ``use`` (e.g. test results -> optimizer: refused)."""
    if use not in ALLOWED_USES[SplitRole(role)]:
        raise SplitAccessError(f"{SplitRole(role).value} split may not be used for {use.value}")


class DatasetSplit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split_id: str = Field(pattern=SLUG)
    role: SplitRole
    row_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("row_ids")
    @classmethod
    def _sorted_unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if any(not r for r in v):
            raise ValueError("row ids must be non-empty")
        if len(set(v)) != len(v):
            raise ValueError("row ids must be unique within a split")
        return tuple(sorted(v))  # canonical order: identity never depends on input order

    @property
    def is_final(self) -> bool:
        """The final held-out split: reporting only."""
        return self.role is SplitRole.TEST


class SplitMethod(StrEnum):
    SEEDED_HASH = "seeded_hash/1"  # reproducible from (row ids, plan); verified on load
    EXPLICIT = "explicit"  # assignment supplied as-is (e.g. a frozen benchmark split file)


BPS = 10_000  # fractions are integer basis points: no float rounding in split sizes


class SplitPlan(BaseModel):
    """Seeded split recipe. Sizes: ``n * bps // 10000`` rows each; optimization gets the rest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    validation_bps: int = Field(ge=0, lt=BPS)
    test_bps: int = Field(ge=0, lt=BPS)

    @model_validator(mode="after")
    def _leaves_optimization_rows(self) -> SplitPlan:
        if self.validation_bps + self.test_bps >= BPS:
            raise ValueError("validation + test fractions must leave rows for optimization")
        return self


def split_order_key(seed: int, row_id: str) -> str:
    return sha256_hex(f"wynk-split/1:{seed}:{row_id}")


def _assign(row_ids: tuple[str, ...], plan: SplitPlan) -> dict[SplitRole, tuple[str, ...]]:
    ordered = sorted(row_ids, key=lambda r: (split_order_key(plan.seed, r), r))
    n = len(ordered)
    n_test = n * plan.test_bps // BPS
    n_val = n * plan.validation_bps // BPS
    if n - n_test - n_val < 1:
        raise ValueError("split plan leaves no optimization rows")
    out = {
        SplitRole.TEST: tuple(ordered[:n_test]),
        SplitRole.VALIDATION: tuple(ordered[n_test : n_test + n_val]),
        SplitRole.OPTIMIZATION: tuple(ordered[n_test + n_val :]),
    }
    return {role: rows for role, rows in out.items() if rows}


class OptimizerSplitView(BaseModel):
    """What optimizer-side code may know about the splits. There is no test field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_hash: str
    optimization_row_ids: tuple[str, ...]
    validation_row_ids: tuple[str, ...] = ()


class DatasetSplits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["datasetsplits/1"] = DATASET_SPLITS_SCHEMA_VERSION
    dataset_hash: str  # identity of what is being split (``DatasetSpec.identity_hash``)
    method: SplitMethod
    plan: SplitPlan | None = None
    splits: tuple[DatasetSplit, ...] = Field(min_length=1)

    @field_validator("dataset_hash")
    @classmethod
    def _dataset_hash(cls, v: str) -> str:
        return _check_sha256(v, "dataset_hash")

    @model_validator(mode="after")
    def _consistent(self) -> DatasetSplits:
        ids = [s.split_id for s in self.splits]
        if len(set(ids)) != len(ids):
            raise ValueError("split ids must be unique")
        roles = [s.role for s in self.splits]
        if roles.count(SplitRole.OPTIMIZATION) != 1:
            raise ValueError("exactly one optimization split is required")
        for role in (SplitRole.VALIDATION, SplitRole.TEST):
            if roles.count(role) > 1:
                raise ValueError(f"at most one {role.value} split is allowed")
        seen: set[str] = set()
        for s in self.splits:
            if overlap := seen & set(s.row_ids):
                raise ValueError(f"row {sorted(overlap)[0]!r} appears in more than one split")
            seen |= set(s.row_ids)
        if (self.method is SplitMethod.SEEDED_HASH) != (self.plan is not None):
            raise ValueError("a plan is required for seeded_hash splits and only for them")
        if self.plan is not None:  # fail closed: the assignment must be the plan's assignment
            expected = _assign(tuple(seen), self.plan)
            actual = {s.role: s.row_ids for s in self.splits}
            if actual != {role: tuple(sorted(rows)) for role, rows in expected.items()}:
                raise ValueError("splits do not match their seeded plan")
        return self

    def split(self, role: SplitRole) -> DatasetSplit | None:
        return next((s for s in self.splits if s.role is role), None)

    def rows_for(self, role: SplitRole, use: SplitUse) -> tuple[str, ...]:
        """Row ids of ``role``'s split, after checking ``use`` is permitted for that role."""
        require_use(role, use)
        s = self.split(role)
        return s.row_ids if s else ()

    def optimizer_view(self) -> OptimizerSplitView:
        return OptimizerSplitView(
            dataset_hash=self.dataset_hash,
            optimization_row_ids=self.rows_for(SplitRole.OPTIMIZATION, SplitUse.OPTIMIZER_FEEDBACK),
            validation_row_ids=self.rows_for(SplitRole.VALIDATION, SplitUse.SELECTION),
        )

    def role_of(self, row_id: str) -> SplitRole | None:
        return next((s.role for s in self.splits if row_id in s.row_ids), None)

    def check_feedback(self, row_ids: Iterable[str]) -> None:
        """Gate for optimizer feedback: fail closed unless every row is an optimization row.

        Validation rows (selection only), final-test rows (reporting only) and rows that are in
        no split at all are refused with ``SplitAccessError``.
        """
        for row_id in sorted(set(row_ids)):
            role = self.role_of(row_id)
            if role is None:
                raise SplitAccessError(f"row {row_id!r} is in no split; refusing as feedback")
            require_use(role, SplitUse.OPTIMIZER_FEEDBACK)

    @property
    def identity_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


def seeded_splits(dataset_hash: str, row_ids: tuple[str, ...], plan: SplitPlan) -> DatasetSplits:
    """Deterministic assignment: same (row ids, plan) -> same splits, whatever the input order."""
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("row ids must be unique")
    assigned = _assign(tuple(row_ids), plan)
    return DatasetSplits(
        dataset_hash=dataset_hash,
        method=SplitMethod.SEEDED_HASH,
        plan=plan,
        splits=tuple(
            DatasetSplit(split_id=role.value, role=role, row_ids=rows)
            for role, rows in sorted(assigned.items())
        ),
    )
