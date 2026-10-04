"""Workflow-memory data model (one ``WorkflowMemory`` per task class).

Everything here is plain, frozen data. Identity fields (``MemoryKey``) decide whether a memory
may be reused: a memory is only compatible with a search that has the same key.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.genome import Genome

MEMORY_SCHEMA = "wynk-workflow-memory/1"
MAX_WORKFLOWS = 3

# real = deterministic evaluator on the real runtime; synthetic = the labelled fake objective;
# TEST/FIXTURE = hand-made test data. Only "real" is a learned benchmark result.
Source = Literal["real", "synthetic", "TEST/FIXTURE"]


class MemoryKey(BaseModel):
    """Compatibility policy: a memory is reused only if every field is equal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_class: str
    grammar_version: str
    optimizer_version: str
    benchmark_hash: str
    model_hash: str | None = None
    evaluator_version: str
    prompt_template_version: str | None = None
    compiler_version: str | None = None
    aco_config_hash: str | None = None


class WorkflowRecord(BaseModel):
    """One remembered workflow and its measured VALIDATION stats."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    genome_hash: str
    path: str  # readable stage path, derived from ``genome``
    genome: Genome
    validation_fitness: float = Field(allow_inf_nan=False)
    pass_rate: float = Field(ge=0.0, le=1.0)
    mean_tokens: float = Field(ge=0.0)
    mean_wall_time_s: float = Field(ge=0.0)
    validation_runs: int = Field(ge=1)


class EdgePheromone(BaseModel):
    """MMAS pheromone on one construction-graph edge (node = canonical stage JSON / START / END)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    src: str
    dst: str
    tau: float = Field(gt=0.0, allow_inf_nan=False)
    label: str  # readable, derived from src/dst


class WorkflowMemory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_schema: Literal["wynk-workflow-memory/1"] = MEMORY_SCHEMA
    key: MemoryKey
    source: Source
    updated_at: str
    runs_merged: int = Field(ge=1)
    workflows: tuple[WorkflowRecord, ...] = Field(max_length=MAX_WORKFLOWS)
    pheromones: tuple[EdgePheromone, ...] = ()
    # MMAS updates behind ``pheromones``; edges not listed evaporate from tau_max over this many.
    aco_epoch: int = Field(0, ge=0)

    @property
    def task_class(self) -> str:
        return self.key.task_class

    def pheromone_map(self) -> dict[tuple[str, str], float]:
        return {(e.src, e.dst): e.tau for e in self.pheromones}


# -- readable labels -----------------------------------------------------------------------
def stage_label(stage: dict) -> str:
    """``{"kind":"GATHER","source":"api","mode":"parallel-2"}`` -> ``GATHER(api,parallel-2)``."""
    opts = [str(v) for k, v in sorted(stage.items()) if k != "kind"]
    if stage.get("kind") == "GATHER":  # source first reads better than alphabetical
        opts = [str(stage["source"]), str(stage["mode"])]
    return f"{stage['kind']}({','.join(opts)})"


def node_label(node: str) -> str:
    """Construction-graph node (canonical stage JSON, or START/END) -> readable label."""
    return node if node in ("START", "END") else stage_label(json.loads(node))


def genome_path(genome: Genome) -> str:
    return " -> ".join(stage_label(s.model_dump(mode="json")) for s in genome.stages)
