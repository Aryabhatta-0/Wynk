"""Three hand-written genomes for the MVP. All run through the same entry point,
``runtime.runner.WorkflowRunner.run`` - they are ordinary ``Genome``s, nothing special-cased.
"""

from __future__ import annotations

from core.genome import Genome
from core.stages import (
    ExtractMethod,
    ExtractStage,
    FailureStrategy,
    FilterMethod,
    FilterStage,
    GatherMode,
    GatherSource,
    GatherStage,
    ReasonMethod,
    ReasonStage,
    SynthesizeMethod,
    SynthesizeStage,
    VerifyMethod,
    VerifyStage,
)

# A. minimal cheap workflow: GATHER -> EXTRACT -> SYNTHESIZE
GENOME_A = Genome.of(
    GatherStage(source=GatherSource.FETCH, mode=GatherMode.SEQUENTIAL),
    ExtractStage(method=ExtractMethod.DIRECT),
    SynthesizeStage(method=SynthesizeMethod.DIRECT),
)

# B. FILTER + REASON: GATHER(parallel-2) -> FILTER -> EXTRACT -> REASON -> SYNTHESIZE
GENOME_B = Genome.of(
    GatherStage(source=GatherSource.FETCH, mode=GatherMode.PARALLEL_2),
    FilterStage(method=FilterMethod.KEYWORD_CHUNK),
    ExtractStage(method=ExtractMethod.SCHEMA_GUIDED),
    ReasonStage(method=ReasonMethod.SINGLE),
    SynthesizeStage(method=SynthesizeMethod.CITE_EVIDENCE),
)

# C. VERIFY: GATHER -> EXTRACT -> VERIFY(evidence_span) -> SYNTHESIZE -> VERIFY(schema_check)
GENOME_C = Genome.of(
    GatherStage(source=GatherSource.FETCH, mode=GatherMode.SEQUENTIAL),
    ExtractStage(method=ExtractMethod.DIRECT),
    VerifyStage(method=VerifyMethod.EVIDENCE_SPAN, on_failure=FailureStrategy.RETRY_1),
    SynthesizeStage(method=SynthesizeMethod.DIRECT),
    VerifyStage(method=VerifyMethod.SCHEMA_CHECK, on_failure=FailureStrategy.RETRY_1),
)

MVP_GENOMES = {"A": GENOME_A, "B": GENOME_B, "C": GENOME_C}
