"""The canonical workflow genome.

A genome is an ORDERED tuple of typed stage specs. Order is part of identity:
``EXTRACT -> VERIFY -> SYNTHESIZE`` and ``EXTRACT -> SYNTHESIZE -> VERIFY`` are different
genomes with different hashes.

A Genome is plain data and may be partial (optimizers build it stage by stage). Whether a
genome is type-valid / complete / allowed is decided by ``core.grammar`` and
``core.constraints`` - never by the optimizer.

Equality semantics: two genomes are equal iff their canonical JSON (hence hash) is equal.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.canonical import canonical_hash, canonical_json
from core.stages import StageSpec

# Bump when the canonical representation changes; it is part of the hashed payload.
GENOME_SCHEMA_VERSION = "genome/1"


class Genome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stages: tuple[StageSpec, ...] = Field(default=())

    @classmethod
    def of(cls, *stages: Any) -> Genome:
        return cls(stages=tuple(stages))

    @classmethod
    def from_stages(cls, stages: Iterable[Any]) -> Genome:
        return cls(stages=tuple(stages))

    def extend(self, stage: Any) -> Genome:
        """Return a new genome with ``stage`` appended (genomes are immutable)."""
        return Genome(stages=(*self.stages, stage))

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": GENOME_SCHEMA_VERSION,
            "stages": [s.model_dump(mode="json") for s in self.stages],
        }

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> Genome:
        """Inverse of ``canonical`` (refuses another schema version)."""
        if data.get("schema") != GENOME_SCHEMA_VERSION:
            raise ValueError(f"genome schema {data.get('schema')!r} is not {GENOME_SCHEMA_VERSION}")
        return cls.model_validate({"stages": data["stages"]})

    def canonical_json(self) -> str:
        return canonical_json(self.canonical())

    @property
    def genome_hash(self) -> str:
        return canonical_hash(self.canonical())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Genome):
            return NotImplemented
        return self.canonical_json() == other.canonical_json()

    def __hash__(self) -> int:
        return hash(self.canonical_json())

    def __len__(self) -> int:
        return len(self.stages)
