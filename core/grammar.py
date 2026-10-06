"""Typed construction grammar.

Owns: stage input/output types, which stages may follow a (partial) genome, the producers a
complete workflow needs, structural placement rules (per-kind limits, no adjacent verifiers,
terminal stages), stage dependencies (a stage that needs gathered pages needs an upstream
GATHER) and the stage vocabulary a task supports. Does NOT own hard constraints (jev/parallel-4,
verifier counts, sources, budgets) - those live in ``core.constraints``.

Optimizers ask ``Grammar.valid_successors(partial)`` instead of duplicating these rules.

Vocabulary. A grammar admits a fixed set of stage kinds. ``Grammar()`` admits exactly the frozen
benchmark's six kinds and reports ``GRAMMAR_VERSION`` (``grammar/1``): its language is unchanged.
``Grammar(ALL_STAGE_KINDS)`` adds DIRECT (Task -> Answer, no retrieval) and CONFIDENCE_GATE
(terminal Answer -> Answer); a task contract selects its own vocabulary
(``core.task_contract.workflow_grammar``). A kind outside the vocabulary is rejected, never
silently dropped.

Requested capability -> representation (see ``capabilities``):

    Direct           DIRECT                    Task -> Answer
    Gather           GATHER                    Task -> Pages
    Filter           FILTER                    Pages -> Pages
    Reason           REASON(single)            Facts -> Facts
    Decompose        REASON(decompose)         Facts -> Facts
    Parallel         GATHER(parallel-2|4)      bounded fan-out width of page reads
    Verify           VERIFY                    Facts -> Facts | Answer -> Answer
    Synthesize       SYNTHESIZE                Facts -> Answer
    Confidence Gate  CONFIDENCE_GATE           Answer -> Answer, terminal
    (bridge)         EXTRACT                   Pages -> Facts

Parallel and Decompose are bounded configurations of linear stages, so a genome always compiles
to a linear DAG: there is no fan-out structure that could be malformed.

Finiteness. Every option is an enum, FILTER/REASON appear at most once, two VERIFY stages are
never adjacent, nothing follows a terminal stage, and every type moves forward
(Task -> Pages -> Facts -> Answer). So the language is finite and no genome is longer than
``MAX_GENOME_STAGES``; ``language_size`` counts it exactly without materializing it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from functools import cache

from core.genome import Genome
from core.stages import (
    LEGACY_STAGE_KINDS,
    FailureStrategy,
    GatherMode,
    ReasonMethod,
    StageKind,
    StageSpec,
    VerifyMethod,
    all_stage_specs,
)
from core.violations import Violation, ViolationCode

GRAMMAR_VERSION = "grammar/1"  # the legacy vocabulary; its language is pinned by tests
EXTENDED_GRAMMAR_VERSION = "grammar/2"  # any other vocabulary (reported with its kinds)


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
    StageKind.DIRECT: {DataType.TASK: DataType.ANSWER},
    StageKind.CONFIDENCE_GATE: {DataType.ANSWER: DataType.ANSWER},
}

START_TYPE = DataType.TASK
FINAL_TYPE = DataType.ANSWER
# The two producer chains from Task to Answer. A complete workflow contains exactly one.
ANSWER_PATHS: tuple[tuple[StageKind, ...], ...] = (
    (StageKind.GATHER, StageKind.EXTRACT, StageKind.SYNTHESIZE),
    (StageKind.DIRECT,),
)
_PRODUCERS = frozenset(k for path in ANSWER_PATHS for k in path)
# Kinds that only occur on the retrieval path: they read, or depend on, gathered pages.
RETRIEVAL_KINDS = frozenset(
    {
        StageKind.GATHER,
        StageKind.FILTER,
        StageKind.EXTRACT,
        StageKind.REASON,
        StageKind.SYNTHESIZE,
        StageKind.CONFIDENCE_GATE,
    }
)
# Placement: at most this many of a kind (VERIFY count is a hard constraint, not grammar).
MAX_PER_GENOME = {StageKind.FILTER: 1, StageKind.REASON: 1}
# Nothing may follow a terminal stage (so a second one is rejected too).
TERMINAL_KINDS = frozenset({StageKind.CONFIDENCE_GATE})
# Longest grammar-valid genome: GATHER FILTER EXTRACT VERIFY REASON VERIFY SYNTHESIZE VERIFY GATE.
MAX_GENOME_STAGES = 9


class Capability(StrEnum):
    DIRECT = "Direct"
    GATHER = "Gather"
    FILTER = "Filter"
    EXTRACT = "Extract"
    REASON = "Reason"
    DECOMPOSE = "Decompose"
    PARALLEL = "Parallel"
    VERIFY = "Verify"
    SYNTHESIZE = "Synthesize"
    CONFIDENCE_GATE = "Confidence Gate"


_KIND_CAPABILITY = {
    StageKind.DIRECT: Capability.DIRECT,
    StageKind.GATHER: Capability.GATHER,
    StageKind.FILTER: Capability.FILTER,
    StageKind.EXTRACT: Capability.EXTRACT,
    StageKind.REASON: Capability.REASON,
    StageKind.VERIFY: Capability.VERIFY,
    StageKind.SYNTHESIZE: Capability.SYNTHESIZE,
    StageKind.CONFIDENCE_GATE: Capability.CONFIDENCE_GATE,
}


def capabilities(spec: StageSpec) -> frozenset[Capability]:
    """The requested workflow capabilities a configured stage provides."""
    kind = StageKind(spec.kind)
    if kind is StageKind.GATHER and spec.mode is not GatherMode.SEQUENTIAL:
        return frozenset({Capability.GATHER, Capability.PARALLEL})
    if kind is StageKind.REASON and spec.method is ReasonMethod.DECOMPOSE:
        return frozenset({Capability.DECOMPOSE})
    return frozenset({_KIND_CAPABILITY[kind]})


def requires(spec: StageSpec) -> StageKind | None:
    """The upstream stage kind a configured stage depends on, if any.

    Evidence checks, the confidence gate and ``regather`` all need pages from a GATHER earlier in
    the workflow; on the DIRECT path there are none, so such a stage could never succeed.
    """
    if spec.kind == StageKind.VERIFY and (
        spec.method == VerifyMethod.EVIDENCE_SPAN or spec.on_failure == FailureStrategy.REGATHER
    ):
        return StageKind.GATHER
    if spec.kind == StageKind.CONFIDENCE_GATE:
        return StageKind.GATHER
    return None


def completions(
    stages: Iterable[StageSpec], kinds: Iterable[StageKind]
) -> tuple[tuple[StageKind, ...], ...]:
    """The producer kinds any completion of the (valid) prefix ``stages`` must still add - one
    tuple per Answer path the vocabulary ``kinds`` enables. Static lower bounds (cost, workflow
    steps, model calls) take the minimum over these, so they stay sound for every completion."""
    present = {_KIND[s.kind] for s in stages}
    if present & {StageKind.DIRECT, StageKind.SYNTHESIZE}:
        return ((),)  # the prefix already produces the Answer
    retrieval = tuple(k for k in ANSWER_PATHS[0] if k not in present)
    if present:
        return (retrieval,)
    allowed = set(kinds)
    paths = tuple(p for p in (retrieval, (StageKind.DIRECT,)) if set(p) <= allowed)
    return paths or (retrieval,)


class GrammarError(ValueError):
    """Raised when asked about a partial genome that is already grammar-invalid."""


@dataclass(frozen=True)
class _State:
    """Everything the rules depend on after a prefix. Hashable: the language is counted by
    memoizing on it."""

    current: DataType = START_TYPE
    counts: tuple[int, ...] = (0,) * len(MAX_PER_GENOME)  # aligned with MAX_PER_GENOME
    seen: frozenset[StageKind] = frozenset()
    prev: StageKind | None = None
    terminated: bool = False

    def count(self, kind: StageKind) -> int:
        return self.counts[_LIMIT_SLOT[kind]]


_LIMIT_SLOT = {kind: i for i, kind in enumerate(MAX_PER_GENOME)}
_KIND = {kind.value: kind for kind in StageKind}  # str -> StageKind without Enum call overhead


class Grammar:
    def __init__(self, kinds: Iterable[StageKind] = LEGACY_STAGE_KINDS) -> None:
        allowed = frozenset(StageKind(k) for k in kinds)
        if not allowed:
            raise ValueError("a grammar needs at least one stage kind")
        self.kinds: tuple[StageKind, ...] = tuple(k for k in StageKind if k in allowed)
        self._allowed = allowed
        self._specs = {k: all_stage_specs(k) for k in self.kinds}

    @property
    def version(self) -> str:
        if self._allowed == frozenset(LEGACY_STAGE_KINDS):
            return GRAMMAR_VERSION
        return f"{EXTENDED_GRAMMAR_VERSION}[{','.join(k.value for k in self.kinds)}]"

    # -- typing ---------------------------------------------------------------
    def output_type(self, genome: Genome) -> DataType:
        """Data type after the last stage. Raises ``GrammarError`` if the genome is invalid."""
        return self._valid_state(genome).current

    # -- construction (what optimizers call) ----------------------------------
    def valid_successors(self, partial: Genome) -> tuple[StageKind, ...]:
        """Stage kinds with at least one configuration that may legally be appended to
        ``partial`` (deterministic order)."""
        kinds = {_KIND[s.kind] for s in self.valid_successor_specs(partial)}
        return tuple(k for k in self.kinds if k in kinds)

    def valid_successor_specs(self, partial: Genome) -> tuple[StageSpec, ...]:
        """Every concrete configured stage that may be appended (deterministic order)."""
        state = self._valid_state(partial)
        return self._successor_specs(state)

    def can_terminate(self, partial: Genome) -> bool:
        """True iff ``partial`` is a complete, grammar-valid Answer-producing workflow."""
        return not self.validate(partial, complete=True)

    # -- validation -----------------------------------------------------------
    def validate(self, genome: Genome, *, complete: bool = True) -> tuple[Violation, ...]:
        violations, state = self._walk(genome.stages)
        if complete and not violations and state.current != FINAL_TYPE:
            violations += self._missing_producers(state)
            violations.append(
                Violation(
                    code=ViolationCode.NO_ANSWER_TERMINAL,
                    message=f"workflow ends in {state.current.value}, not {FINAL_TYPE.value}",
                )
            )
        return tuple(violations)

    # -- the bounded language ---------------------------------------------------
    def enumerate(self) -> Iterator[Genome]:
        """Every complete grammar-valid genome, depth first in successor order (finite)."""

        def dfs(genome: Genome, state: _State) -> Iterator[Genome]:
            if state.current == FINAL_TYPE:
                yield genome
            for spec in self._successor_specs(state):
                yield from dfs(genome.extend(spec), self._advance(state, _KIND[spec.kind]))

        yield from dfs(Genome(), _State())

    def language_size(self) -> int:
        """Exact number of complete grammar-valid genomes, without materializing them."""

        @cache
        def size(state: _State) -> int:
            done = 1 if state.current == FINAL_TYPE else 0
            return done + sum(
                size(self._advance(state, _KIND[spec.kind]))
                for spec in self._successor_specs(state)
            )

        return size(_State())

    def max_genome_length(self) -> int:
        """Length of the longest complete grammar-valid genome (0 if there is none)."""

        @cache
        def longest(state: _State) -> int:  # -1: no completion from here
            best = 0 if state.current == FINAL_TYPE else -1
            for spec in self._successor_specs(state):
                sub = longest(self._advance(state, _KIND[spec.kind]))
                if sub >= 0:
                    best = max(best, sub + 1)
            return best

        return max(0, longest(_State()))

    # -- internals ------------------------------------------------------------
    def _valid_state(self, genome: Genome) -> _State:
        violations, state = self._walk(genome.stages)
        if violations:
            raise GrammarError(violations[0].message)
        return state

    def _walk(self, stages: Iterable[StageSpec]) -> tuple[list[Violation], _State]:
        violations: list[Violation] = []
        state = _State()
        for i, stage in enumerate(stages):
            kind = _KIND[stage.kind]
            if kind not in self._allowed:
                violations.append(
                    Violation(
                        code=ViolationCode.STAGE_UNSUPPORTED,
                        message=f"{kind.value} is not in this task's stage vocabulary",
                        stage_index=i,
                    )
                )
            if state.current not in SIGNATURES[kind]:
                violations.append(
                    Violation(
                        code=ViolationCode.INVALID_TRANSITION,
                        message=f"{kind.value} cannot consume {state.current.value}",
                        stage_index=i,
                    )
                )
                return violations, state  # later stages would only cascade
            placement = self._placement_problem(state, kind)
            if placement is not None:
                code, message = placement
                violations.append(Violation(code=code, message=message, stage_index=i))
            needed = requires(stage)
            if needed is not None and needed not in state.seen:
                violations.append(
                    Violation(
                        code=ViolationCode.UNSATISFIED_DEPENDENCY,
                        message=f"{kind.value} ({_options(stage)}) needs an upstream "
                        f"{needed.value}",
                        stage_index=i,
                    )
                )
            state = self._advance(state, kind)
        return violations, state

    @staticmethod
    def _advance(state: _State, kind: StageKind) -> _State:
        """The state after appending a stage of ``kind`` (rules depend on the kind only)."""
        counts = state.counts
        slot = _LIMIT_SLOT.get(kind)
        if slot is not None:
            counts = (*counts[:slot], counts[slot] + 1, *counts[slot + 1 :])
        return _State(
            current=SIGNATURES[kind][state.current],
            counts=counts,
            seen=state.seen if kind in state.seen else state.seen | {kind},
            prev=kind,
            terminated=state.terminated or kind in TERMINAL_KINDS,
        )

    @staticmethod
    def _placement_problem(state: _State, kind: StageKind) -> tuple[ViolationCode, str] | None:
        if state.terminated:
            what = "a second terminal stage" if kind in TERMINAL_KINDS else kind.value
            return ViolationCode.AFTER_TERMINAL, f"{what} cannot follow a terminal stage"
        limit = MAX_PER_GENOME.get(kind)
        if limit is not None and state.count(kind) >= limit:
            return ViolationCode.PLACEMENT, f"{kind.value} may appear at most {limit} time(s)"
        # No back-to-back verifiers: keeps the grammar finite (<= 3 verify slots).
        if kind is StageKind.VERIFY and state.prev is StageKind.VERIFY:
            return ViolationCode.PLACEMENT, "VERIFY cannot directly follow VERIFY"
        return None

    def _kind_problem(self, state: _State, kind: StageKind) -> str | None:
        if kind not in self._allowed:
            return "unsupported"
        if state.current not in SIGNATURES[kind]:
            return "type"
        return None if self._placement_problem(state, kind) is None else "placement"

    def _successor_specs(self, state: _State) -> tuple[StageSpec, ...]:
        return tuple(
            spec
            for kind in self.kinds
            if self._kind_problem(state, kind) is None
            for spec in self._specs[kind]
            if (needed := requires(spec)) is None or needed in state.seen
        )

    def _missing_producers(self, state: _State) -> list[Violation]:
        """Why a type-valid genome does not reach Answer: the producers it still lacks."""
        # paths this grammar enables that are consistent with the producers already present
        paths = [
            p
            for p in ANSWER_PATHS
            if set(p) <= self._allowed and (state.seen & _PRODUCERS) <= set(p)
        ]
        if len(paths) == 1:
            return [
                Violation(
                    code=ViolationCode.MISSING_REQUIRED_STAGE,
                    message=f"required stage {k.value} is missing",
                )
                for k in paths[0]
                if k not in state.seen
            ]
        options = " or ".join(" -> ".join(k.value for k in p) for p in paths) or "none enabled"
        return [
            Violation(
                code=ViolationCode.MISSING_REQUIRED_STAGE,
                message=f"workflow has no Answer producer (options: {options})",
            )
        ]


@cache
def grammar_for(kinds: tuple[StageKind, ...]) -> Grammar:
    """The (shared, immutable) grammar admitting exactly ``kinds``."""
    return Grammar(kinds)


def _options(stage: StageSpec) -> str:
    return ", ".join(str(v) for k, v in stage.model_dump(mode="json").items() if k != "kind")
