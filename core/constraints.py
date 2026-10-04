"""The single authoritative hard-constraint layer.

Reused unchanged by ACO, random search, exhaustive search, PBIL, bandit and hand-built
workflows. Optimizers must NOT re-implement any of these rules; they call ``check`` or
``admissible_successors``. Everything here is deterministic and LLM-free.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, PositiveInt

from core.cost_model import CostModel, StaticCostModel, exceeded_caps
from core.genome import Genome
from core.grammar import Grammar, GrammarError
from core.stages import GatherMode, GatherSource, StageSpec, VerifyMethod
from core.task_spec import RuntimeTask
from core.violations import Violation, ViolationCode

CONSTRAINTS_VERSION = "constraints/1"


class ConstraintConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_active_verifiers: PositiveInt = 2
    max_self_consistency: PositiveInt = 1


class ConstraintChecker:
    def __init__(
        self,
        grammar: Grammar | None = None,
        cost_model: CostModel | None = None,
        config: ConstraintConfig | None = None,
    ) -> None:
        self.grammar = grammar or Grammar()
        self.cost_model = cost_model or StaticCostModel()
        self.config = config or ConstraintConfig()

    def check(
        self, genome: Genome, task: RuntimeTask | None = None, *, complete: bool = True
    ) -> tuple[Violation, ...]:
        """All violations of ``genome``. ``complete=False`` accepts partial genomes (prefix
        checking): every rule below is monotone, so a prefix violation can never be repaired
        by appending more stages."""
        violations = list(self.grammar.validate(genome, complete=complete))
        violations += self._stage_rules(genome)
        if task is not None:
            violations += self._task_rules(genome, task)
        return tuple(violations)

    def is_valid(
        self, genome: Genome, task: RuntimeTask | None = None, *, complete: bool = True
    ) -> bool:
        return not self.check(genome, task, complete=complete)

    def admissible_successors(self, partial: Genome, task: RuntimeTask) -> tuple[StageSpec, ...]:
        """Stages that may be appended to ``partial`` such that the prefix stays valid for
        ``task``: grammar-legal AND constraint-clean AND still within provable budget."""
        try:
            candidates = self.grammar.valid_successor_specs(partial)
        except GrammarError:
            return ()
        return tuple(
            s for s in candidates if self.is_valid(partial.extend(s), task, complete=False)
        )

    # -- rules ----------------------------------------------------------------
    def _stage_rules(self, genome: Genome) -> list[Violation]:
        out: list[Violation] = []
        gather = next((s for s in genome.stages if s.kind == "GATHER"), None)
        jev = gather is not None and gather.source == GatherSource.JEV
        if jev and gather.mode == GatherMode.PARALLEL_4:
            out.append(
                Violation(
                    code=ViolationCode.JEV_PARALLEL_4,
                    message="GATHER source jev cannot use mode parallel-4",
                    stage_index=0,
                )
            )
        verifiers = [(i, s) for i, s in enumerate(genome.stages) if s.kind == "VERIFY"]
        if len(verifiers) > self.config.max_active_verifiers:
            out.append(
                Violation(
                    code=ViolationCode.TOO_MANY_VERIFIERS,
                    message=f"{len(verifiers)} verifiers; at most "
                    f"{self.config.max_active_verifiers} allowed",
                    stage_index=verifiers[self.config.max_active_verifiers][0],
                )
            )
        consistency = [i for i, s in verifiers if s.method == VerifyMethod.SELF_CONSISTENCY]
        if len(consistency) > self.config.max_self_consistency:
            out.append(
                Violation(
                    code=ViolationCode.SELF_CONSISTENCY_REPEATED,
                    message="self_consistency may appear at most "
                    f"{self.config.max_self_consistency} time(s)",
                    stage_index=consistency[self.config.max_self_consistency],
                )
            )
        if jev:
            for i, s in verifiers:
                if s.on_failure.value == "regather":
                    out.append(
                        Violation(
                            code=ViolationCode.JEV_REGATHER,
                            message="regather cannot be used when GATHER source is jev",
                            stage_index=i,
                        )
                    )
        return out

    def _task_rules(self, genome: Genome, task: RuntimeTask) -> list[Violation]:
        out: list[Violation] = []
        gather = next((s for s in genome.stages if s.kind == "GATHER"), None)
        if gather is not None:
            if gather.source not in task.allowed_sources:
                out.append(
                    Violation(
                        code=ViolationCode.SOURCE_NOT_ALLOWED,
                        message=f"source {gather.source.value} not allowed for task {task.id}",
                        stage_index=0,
                    )
                )
            if task.interaction_required and gather.source != GatherSource.JEV:
                out.append(
                    Violation(
                        code=ViolationCode.INTERACTION_REQUIRES_JEV,
                        message="task requires interaction; GATHER source must be jev",
                        stage_index=0,
                    )
                )
        over = exceeded_caps(self.cost_model.estimate(genome, task), task.caps)
        if over:
            out.append(
                Violation(
                    code=ViolationCode.BUDGET_INFEASIBLE,
                    message="best-case estimate exceeds caps: " + ", ".join(c.value for c in over),
                )
            )
        return out
