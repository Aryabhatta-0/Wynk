"""Typed construction grammar.

Owns: stage input/output types, which stages may follow a (partial) genome, required stages,
and structural placement rules. Does NOT own hard constraints (jev/parallel-4, verifier
counts, budgets) - those live in ``core.constraints``.

Optimizers ask ``Grammar.valid_successors(partial)`` instead of duplicating these rules.
"""

from __future__ import annotations

from enum import StrEnum

from core.genome import Genome
from core.stages import StageKind, StageSpec, all_stage_specs
from core.violations import Violation, ViolationCode

GRAMMAR_VERSION = "grammar/1"


class DataType(StrEnum):
    TASK = "Task"
    PAGES = "Pages"
    FACTS = "Facts"
    ANSWER = "Answer"


# stage kind -> {input type: output type}. VERIFY is polymorphic: its type is set by position.
SIGNATURES: dict[StageKind, dict[DataType, DataType]] = {
    StageKind.GATHER: {DataType.TASK: DataType.PAGES},
    StageKind.FILTER: {DataType.PAGES: DataType.PAGES},
    StageKind.EXTRACT: {DataType.PAGES: DataType.FACTS},
    StageKind.REASON: {DataType.FACTS: DataType.FACTS},
    StageKind.VERIFY: {DataType.FACTS: DataType.FACTS, DataType.ANSWER: DataType.ANSWER},
    StageKind.SYNTHESIZE: {DataType.FACTS: DataType.ANSWER},
}

START_TYPE = DataType.TASK
FINAL_TYPE = DataType.ANSWER
REQUIRED_KINDS = (StageKind.GATHER, StageKind.EXTRACT, StageKind.SYNTHESIZE)
OPTIONAL_KINDS = (StageKind.FILTER, StageKind.REASON, StageKind.VERIFY)
# Placement: at most this many of a kind (VERIFY count is a hard constraint, not grammar).
MAX_PER_GENOME = {StageKind.FILTER: 1, StageKind.REASON: 1}


class GrammarError(ValueError):
    """Raised when asked about a partial genome that is already grammar-invalid."""


class Grammar:
    version = GRAMMAR_VERSION

    # -- typing ---------------------------------------------------------------
    def output_type(self, genome: Genome) -> DataType:
        """Data type after the last stage. Raises ``GrammarError`` if the genome is invalid."""
        violations, current = self._walk(genome)
        if violations:
            raise GrammarError(violations[0].message)
        return current

    # -- construction (what optimizers call) ----------------------------------
    def valid_successors(self, partial: Genome) -> tuple[StageKind, ...]:
        """Stage kinds that may legally be appended to ``partial`` (deterministic order)."""
        violations, current = self._walk(partial)
        if violations:
            raise GrammarError(violations[0].message)
        kinds = [
            k for k in StageKind if current in SIGNATURES[k] and self._placement_ok(partial, k)
        ]
        return tuple(kinds)

    def valid_successor_specs(self, partial: Genome) -> tuple[StageSpec, ...]:
        """Every concrete configured stage that may be appended (deterministic order)."""
        return tuple(
            spec for kind in self.valid_successors(partial) for spec in all_stage_specs(kind)
        )

    def can_terminate(self, partial: Genome) -> bool:
        """True iff ``partial`` is a complete, grammar-valid Answer-producing workflow."""
        return not self.validate(partial, complete=True)

    # -- validation -----------------------------------------------------------
    def validate(self, genome: Genome, *, complete: bool = True) -> tuple[Violation, ...]:
        violations, final = self._walk(genome)
        if complete and not violations:
            kinds = {s.kind for s in genome.stages}
            for req in REQUIRED_KINDS:
                if req not in kinds:
                    violations.append(
                        Violation(
                            code=ViolationCode.MISSING_REQUIRED_STAGE,
                            message=f"required stage {req.value} is missing",
                        )
                    )
            if final != FINAL_TYPE:
                violations.append(
                    Violation(
                        code=ViolationCode.NO_ANSWER_TERMINAL,
                        message=f"workflow ends in {final.value}, not {FINAL_TYPE.value}",
                    )
                )
        return tuple(violations)

    # -- internals ------------------------------------------------------------
    def _walk(self, genome: Genome) -> tuple[list[Violation], DataType]:
        violations: list[Violation] = []
        current = START_TYPE
        for i, stage in enumerate(genome.stages):
            kind = StageKind(stage.kind)
            out = SIGNATURES[kind].get(current)
            if out is None:
                violations.append(
                    Violation(
                        code=ViolationCode.INVALID_TRANSITION,
                        message=f"{kind.value} cannot consume {current.value}",
                        stage_index=i,
                    )
                )
                return violations, current  # later stages would only cascade
            if not self._placement_ok(Genome(stages=genome.stages[:i]), kind):
                violations.append(
                    Violation(
                        code=ViolationCode.PLACEMENT,
                        message=f"{kind.value} is not allowed at position {i}",
                        stage_index=i,
                    )
                )
            current = out
        return violations, current

    @staticmethod
    def _placement_ok(prefix: Genome, kind: StageKind) -> bool:
        limit = MAX_PER_GENOME.get(kind)
        if limit is not None and sum(1 for s in prefix.stages if s.kind == kind) >= limit:
            return False
        # No back-to-back verifiers: keeps the grammar finite (<= 3 verify slots).
        return not (
            kind is StageKind.VERIFY and prefix.stages and prefix.stages[-1].kind == "VERIFY"
        )
