"""MuSiQue-Answerable: provenance, frozen sampling, sanitization and gold-annotation leakage.

The fixture rows below follow the official MuSiQue schema but are invented, with marker strings
in every gold annotation so any leak into execution is detectable. Tests that need the official
file run only where it has been downloaded (``data/external/musique``); it is never committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.canonical import canonical_hash
from core.dataset import (
    DatasetSplit,
    DatasetSplits,
    SplitMethod,
    SplitPlan,
    SplitRole,
    seeded_splits,
)
from experiments.contract_run import contract_suite
from experiments.musique import (
    GOLD_FIELDS,
    HOPS,
    PREPROCESSING_HASH,
    SOURCE,
    SourceMismatch,
    check_manifest,
    dataset_bytes,
    hop_count,
    load_source,
    musique_contract,
    prepare,
    sample_ids,
    sanitize,
)
from experiments.musique_run import BLOCKED, MANIFEST, load_protocol, plan_from, real_runner

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL = ROOT / "data" / "external" / "musique" / "musique_ans_v1.0_dev.jsonl"
MARKERS = ("ALIAS-MARKER", "DECOMP-MARKER", "INTERMEDIATE-MARKER", "SUPPORT-IDX-MARKER")
ANSWER = "ZEBRA-ANSWER"


def fixture_row(rid: str) -> dict:
    return {
        "id": rid,
        "question": f"Which river flows past the birthplace of person {rid}?",
        "paragraphs": [
            {
                "idx": 1,
                "title": "Town",
                "paragraph_text": "The town lies on a river.",
                "is_supporting": True,
            },
            {
                "idx": 0,
                "title": "Person",
                "paragraph_text": "The person was born in a town.",
                "is_supporting": False,
            },
        ],
        "answer": f"{ANSWER}-{rid}",
        "answer_aliases": [f"ALIAS-MARKER-{rid}"],
        "question_decomposition": [
            {
                "id": 1,
                "question": "DECOMP-MARKER birthplace",
                "answer": "INTERMEDIATE-MARKER",
                "paragraph_support_idx": "SUPPORT-IDX-MARKER",
            },
        ],
        "answerable": True,
    }


ROWS = [fixture_row(f"{h}hop__{i}_{i + 1}") for h in (2, 3, 4) for i in range(2)]


# -- sanitization -------------------------------------------------------------------------------
def test_sanitize_keeps_only_id_question_context_and_target():
    row = fixture_row("2hop__1_2")
    clean = sanitize(row)
    assert set(clean) == {"id", "question", "context", "answer"}
    # every paragraph (supporting or not), raw title + text, in official idx order
    assert (
        clean["context"]
        == "Person\nThe person was born in a town.\n\nTown\nThe town lies on a river."
    )
    exposed = json.dumps({k: v for k, v in clean.items() if k != "answer"})
    for marker in (*MARKERS, "is_supporting", ANSWER):
        assert marker not in exposed
    assert set(GOLD_FIELDS).isdisjoint(clean)


def _fixture_suite():
    data = dataset_bytes(ROWS, [r["id"] for r in ROWS])
    contract = musique_contract(data, load_protocol())
    ids = sorted(r["id"] for r in ROWS)
    splits = DatasetSplits(
        dataset_hash=contract.dataset.identity_hash,
        method=SplitMethod.EXPLICIT,
        splits=(
            DatasetSplit(split_id="opt", role=SplitRole.OPTIMIZATION, row_ids=tuple(ids[:3])),
            DatasetSplit(split_id="val", role=SplitRole.VALIDATION, row_ids=tuple(ids[3:5])),
            DatasetSplit(split_id="test", role=SplitRole.TEST, row_ids=tuple(ids[5:])),
        ),
    )
    suite, references = contract_suite(contract, splits, data)
    return contract, suite, references, data


def test_gold_annotations_never_reach_execution_tasks():
    contract, suite, references, data = _fixture_suite()
    assert all(m.encode() not in data for m in MARKERS)  # not even in the dataset bytes
    for task in suite.tasks:
        assert set(task.example.values) == {"question", "context"}
        visible = json.dumps(task.example.values) + task.question
        for marker in (*MARKERS, ANSWER):
            assert marker not in visible
        assert references.expected(task.id)["answer"].startswith(ANSWER)  # evaluator side only


class RecordingModel:
    """TEST DOUBLE: records every prompt; answers with an empty JSON object."""

    model_hash = "recording/1"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, request):
        from runtime.gemma_client import GenerationResponse

        self.prompts.append(request.input_text)
        body = {"facts": [], "answer": {"answer": "x"}}
        return GenerationResponse(
            text=json.dumps(body),
            parsed=body,
            prompt_tokens=1,
            completion_tokens=1,
            model_hash=self.model_hash,
        )


def test_gold_annotations_never_reach_a_model_prompt():
    pytest.importorskip("agent_framework")
    from core.genome import Genome
    from core.stages import (
        DirectStage,
        ExtractStage,
        FilterStage,
        GatherStage,
        ReasonStage,
        SynthesizeStage,
    )
    from runtime.runner import WorkflowRunner

    _, suite, _, _ = _fixture_suite()
    model = RecordingModel()
    runner = WorkflowRunner(model=model, benchmark_hash="inline")
    genomes = [
        Genome.of(DirectStage(method="answer")),
        Genome.of(
            GatherStage(source="fetch", mode="sequential"),
            FilterStage(method="keyword_chunk"),
            ExtractStage(method="schema_guided"),
            ReasonStage(method="decompose"),
            SynthesizeStage(method="cite_evidence"),
        ),
    ]
    for genome in genomes:
        for task in suite.tasks:
            runner.run_sync(genome, task)
    assert len(model.prompts) >= len(suite.tasks) * 4
    for prompt in model.prompts:
        for marker in (*MARKERS, ANSWER, "is_supporting"):
            assert marker not in prompt
    assert any("The person was born in a town." in p for p in model.prompts)  # context did reach


# -- frozen sampling ----------------------------------------------------------------------------
def test_sampling_is_equal_per_hop_reproducible_and_reads_ids_only():
    ids = [f"{h}hop{s}__{i}" for h in HOPS for s in ("", "1", "2") for i in range(30)]
    a = sample_ids(ids, seed=7, per_hop=10)
    assert {h: len(v) for h, v in a.items()} == {2: 10, 3: 10, 4: 10}
    assert all(hop_count(r) == h for h, v in a.items() for r in v)
    assert sample_ids(list(reversed(ids)), seed=7, per_hop=10) == a  # order-free
    assert sample_ids(ids, seed=8, per_hop=10) != a  # the seed decides
    with pytest.raises(ValueError):
        sample_ids(ids[:5], seed=7, per_hop=10)


def test_the_committed_manifest_is_self_consistent_and_its_split_reproduces():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    body = {k: v for k, v in m.items() if k != "manifest_hash"}
    assert m["manifest_hash"] == canonical_hash(body)
    assert m["source"] == SOURCE and m["preprocessing"]["hash"] == PREPROCESSING_HASH
    protocol = load_protocol()
    assert m["sampling"]["seed"] == protocol["sampling"]["seed"]
    selected = [r for h in HOPS for r in m["selected_ids"][str(h)]]
    assert {h: len(m["selected_ids"][str(h)]) for h in HOPS} == {
        h: protocol["sampling"]["per_hop"] for h in HOPS
    }
    assert all(hop_count(r) == h for h in HOPS for r in m["selected_ids"][str(h)])
    split_rows = [r for rows in m["splits"]["rows"].values() for r in rows]
    assert sorted(split_rows) == sorted(selected)  # a partition of exactly the selected ids
    again = seeded_splits(
        m["dataset"]["identity_hash"], tuple(selected), SplitPlan(**protocol["split_plan"])
    )
    assert again.identity_hash == m["splits"]["identity_hash"]
    plan = plan_from(protocol, m["manifest_hash"])
    assert plan.expected_model_hash == protocol["expected_model_hash"]
    assert plan.dataset_manifest_hash == m["manifest_hash"]


def test_the_protocol_freezes_token_f1_as_the_primary_evaluator():
    contract = load_protocol()["contract"]
    assert contract["evaluation"] == {
        "evaluator": "token_f1",
        "evaluator_version": "token_f1/1",
        "config": {"pass_threshold": 0.5},
    }


@pytest.mark.skipif(not OFFICIAL.is_file(), reason="official MuSiQue file not downloaded")
def test_the_official_file_rebuilds_the_frozen_subset_exactly():
    protocol = load_protocol()
    contract, splits, data, manifest, rows = prepare(OFFICIAL, protocol)
    check_manifest(manifest, json.loads(MANIFEST.read_text(encoding="utf-8")))
    _, _, data2, manifest2, _ = prepare(OFFICIAL, protocol)
    assert data2 == data and manifest2 == manifest
    suite, _ = contract_suite(contract, splits, data)
    for task in suite.tasks:
        assert set(task.example.values) == {"question", "context"}
    for line in data.decode("utf-8").splitlines():
        assert set(json.loads(line)) == {"id", "question", "context", "answer"}


@pytest.mark.skipif(not OFFICIAL.is_file(), reason="official MuSiQue file not downloaded")
def test_other_bytes_than_the_pinned_official_file_are_refused(tmp_path):
    bad = tmp_path / "musique_ans_v1.0_dev.jsonl"
    bad.write_bytes(OFFICIAL.read_bytes()[:-2] + b"\n")
    with pytest.raises(SourceMismatch):
        load_source(bad)


def test_without_a_real_backend_the_run_is_blocked(monkeypatch, tmp_path):
    for key in ("GEMMA_BASE_URL", "GEMMA_MODEL", "GEMMA_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SystemExit) as exc:
        real_runner(tmp_path / "missing.env")
    assert exc.value.code == BLOCKED
