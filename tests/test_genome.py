import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.genome import Genome
from core.stages import ExtractMethod, FailureStrategy, GatherMode, GatherSource, VerifyMethod
from tests.conftest import extract, gather, minimal_genome, synth, verify

ROOT = Path(__file__).resolve().parent.parent

# Golden value: changes ONLY if the canonical representation (or schema version) changes.
# If this fails you changed identity for every stored run - bump GENOME_SCHEMA_VERSION.
MINIMAL_GENOME_HASH = "aeceed9675ed3b110b5228474f3e04bd8fefdc43f31b3101b6a07f772d763669"


def test_same_genome_same_canonical_and_hash():
    a, b = minimal_genome(), minimal_genome()
    assert a.canonical_json() == b.canonical_json()
    assert a.genome_hash == b.genome_hash
    assert a == b and hash(a) == hash(b)


def test_json_roundtrip_preserves_identity():
    g = Genome.of(gather(), extract(), verify(), synth())
    back = Genome.model_validate_json(g.model_dump_json())
    assert back == g
    assert back.genome_hash == g.genome_hash


def test_stage_order_changes_hash():
    verify_before_synth = Genome.of(gather(), extract(), verify(), synth())
    verify_after_synth = Genome.of(gather(), extract(), synth(), verify())
    assert verify_before_synth != verify_after_synth
    assert verify_before_synth.genome_hash != verify_after_synth.genome_hash


@pytest.mark.parametrize(
    "other",
    [
        Genome.of(gather(mode=GatherMode.PARALLEL_2), extract(), synth()),
        Genome.of(gather(source=GatherSource.API), extract(), synth()),
        Genome.of(gather(), extract(ExtractMethod.COT), synth()),
    ],
)
def test_stage_config_changes_hash(other):
    assert other.genome_hash != minimal_genome().genome_hash
    assert other != minimal_genome()


def test_verify_method_and_failure_strategy_both_matter():
    base = Genome.of(gather(), extract(), verify(), synth())
    other_method = Genome.of(gather(), extract(), verify(VerifyMethod.EVIDENCE_SPAN), synth())
    other_failure = Genome.of(
        gather(), extract(), verify(on_failure=FailureStrategy.RETRY_2), synth()
    )
    assert len({base.genome_hash, other_method.genome_hash, other_failure.genome_hash}) == 3


def test_canonical_form_is_plain_sorted_json_without_volatile_fields():
    text = minimal_genome().canonical_json()
    assert text.startswith('{"schema":"genome/1","stages":[{"kind":"GATHER"')
    assert " " not in text


def test_hash_is_stable_across_processes_and_hash_seeds():
    code = "from tests.conftest import minimal_genome;print(minimal_genome().genome_hash)"
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT), **_base_env()},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert outs == {minimal_genome().genome_hash}


def test_golden_hash_pins_the_canonical_format():
    assert minimal_genome().genome_hash == MINIMAL_GENOME_HASH


def test_genome_is_immutable_and_extend_returns_new_genome():
    g = Genome.of(gather())
    longer = g.extend(extract())
    assert len(g) == 1 and len(longer) == 2
    with pytest.raises(ValidationError):
        g.stages = ()


def test_unknown_stage_kind_or_option_rejected():
    with pytest.raises(ValidationError):
        Genome.model_validate({"stages": [{"kind": "MAGIC"}]})
    with pytest.raises(ValidationError):
        Genome.model_validate(
            {"stages": [{"kind": "GATHER", "source": "fetch", "mode": "parallel-8"}]}
        )
    with pytest.raises(ValidationError):  # extra keys are not silently dropped
        Genome.model_validate(
            {"stages": [{"kind": "GATHER", "source": "fetch", "mode": "sequential", "x": 1}]}
        )


def _base_env():
    import os

    return {k: v for k, v in os.environ.items() if k in ("SYSTEMROOT", "PATH", "TEMP", "TMP")}
