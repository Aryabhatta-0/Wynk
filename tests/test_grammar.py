import pytest

from core.genome import Genome
from core.grammar import DataType, Grammar, GrammarError
from core.stages import StageKind
from core.violations import ViolationCode
from tests.conftest import extract, flt, gather, minimal_genome, reason, synth, verify

G = Grammar()


def codes(genome: Genome, complete: bool = True) -> set[ViolationCode]:
    return {v.code for v in G.validate(genome, complete=complete)}


@pytest.mark.parametrize(
    "stages",
    [
        (gather(), extract(), synth()),
        (gather(), flt(), extract(), synth()),
        (gather(), extract(), reason(), synth()),
        (gather(), extract(), verify(), synth()),  # Facts -> Facts verifier
        (gather(), extract(), synth(), verify()),  # Answer -> Answer verifier
        (gather(), flt(), extract(), verify(), reason(), verify(), synth(), verify()),  # longest
    ],
)
def test_accepts_valid_complete_paths(stages):
    g = Genome.from_stages(stages)
    assert G.validate(g) == ()
    assert G.can_terminate(g)
    assert G.output_type(g) == DataType.ANSWER


@pytest.mark.parametrize(
    ("stages", "expected"),
    [
        ((synth(),), ViolationCode.INVALID_TRANSITION),  # Task cannot feed SYNTHESIZE
        ((extract(), synth()), ViolationCode.INVALID_TRANSITION),  # EXTRACT before GATHER
        ((gather(), synth()), ViolationCode.INVALID_TRANSITION),  # Pages cannot feed SYNTHESIZE
        ((gather(), reason(), extract(), synth()), ViolationCode.INVALID_TRANSITION),
        ((gather(), verify(), extract(), synth()), ViolationCode.INVALID_TRANSITION),  # Pages
        ((gather(), extract(), extract(), synth()), ViolationCode.INVALID_TRANSITION),
        ((gather(), extract(), synth(), synth()), ViolationCode.INVALID_TRANSITION),
        ((gather(), gather(), extract(), synth()), ViolationCode.INVALID_TRANSITION),
        ((gather(), extract(), synth(), reason()), ViolationCode.INVALID_TRANSITION),
        ((gather(), flt(), flt(), extract(), synth()), ViolationCode.PLACEMENT),
        ((gather(), extract(), reason(), reason(), synth()), ViolationCode.PLACEMENT),
        ((gather(), extract(), verify(), verify(), synth()), ViolationCode.PLACEMENT),
    ],
)
def test_rejects_wrong_transitions_and_placements(stages, expected):
    assert expected in codes(Genome.from_stages(stages))


def test_required_stages_are_enforced():
    assert ViolationCode.MISSING_REQUIRED_STAGE in codes(Genome())
    incomplete = Genome.of(gather(), extract())
    assert codes(incomplete) == {
        ViolationCode.MISSING_REQUIRED_STAGE,
        ViolationCode.NO_ANSWER_TERMINAL,
    }
    assert not G.can_terminate(incomplete)
    # the same genome is fine as a PARTIAL one
    assert codes(incomplete, complete=False) == set()


def test_valid_successors_follow_the_types():
    k = StageKind
    assert G.valid_successors(Genome()) == (k.GATHER,)
    g = Genome.of(gather())
    assert G.valid_successors(g) == (k.FILTER, k.EXTRACT)
    g = g.extend(flt())
    assert G.valid_successors(g) == (k.EXTRACT,)  # one FILTER max
    g = g.extend(extract())
    assert G.valid_successors(g) == (k.REASON, k.VERIFY, k.SYNTHESIZE)
    assert G.valid_successors(g.extend(reason())) == (k.VERIFY, k.SYNTHESIZE)
    assert G.valid_successors(g.extend(verify())) == (k.REASON, k.SYNTHESIZE)  # no VERIFY twice
    g = g.extend(synth())
    assert G.valid_successors(g) == (k.VERIFY,)
    assert G.valid_successors(g.extend(verify())) == ()


def test_successor_specs_enumerate_every_configuration_deterministically():
    specs = G.valid_successor_specs(Genome())
    assert len(specs) == 9  # 3 sources x 3 modes
    assert specs == G.valid_successor_specs(Genome())
    assert len({Genome.of(s).genome_hash for s in specs}) == 9


def test_every_successor_spec_keeps_the_prefix_grammar_valid():
    g = minimal_genome()
    for spec in G.valid_successor_specs(Genome.of(gather(), extract())):
        assert G.validate(Genome.of(gather(), extract(), spec), complete=False) == ()
    assert G.valid_successor_specs(g) != ()


def test_asking_about_an_invalid_partial_raises():
    with pytest.raises(GrammarError):
        G.valid_successors(Genome.of(gather(), synth()))
    with pytest.raises(GrammarError):
        G.output_type(Genome.of(synth()))
