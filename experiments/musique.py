"""MuSiQue-Answerable v1.0 (dev) -> a frozen, sanitized Wynk dataset + TaskContract + splits.

Source: the official release (https://github.com/StonyBrookNLP/musique, ``download_data.sh``):
``musique_v1.0.zip`` -> ``data/musique_ans_v1.0_dev.jsonl``. Both SHA-256 digests are pinned in
``SOURCE``; ``load_source`` refuses any other bytes. The dataset itself is never committed.

Sanitization (``sanitize``) keeps exactly four things per row:

    id                       -> ID       (row id)
    question                 -> INPUT
    paragraphs[].title/text  -> CONTEXT  (every paragraph, in official ``idx`` order, distractors
                                          included; nothing marks which ones are supporting)
    answer                   -> TARGET   (evaluator side only: never reaches execution)

Everything else is dropped before a row exists: ``answer_aliases``, ``question_decomposition``
(sub-questions, intermediate answers, ``paragraph_support_idx``), ``is_supporting``,
``answerable``. Only ``title`` and ``paragraph_text`` are ever read from a paragraph.

Sampling (``sample_ids``) is frozen BEFORE any strategy runs and never looks at answers or
outcomes: per hop count (2 / 3 / 4, read from the official id prefix), ids are ordered by
``sha256("wynk-musique-sample/1:<seed>:<id>")`` and the first ``per_hop`` are taken - equal
counts per hop. ``manifest`` records the provenance, the preprocessing identity, the sampling
rule + seed, the selected official ids and the resulting dataset / split hashes, and is hashed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from core.canonical import canonical_hash, sha256_hex
from core.constraints import ConstraintLimits
from core.dataset import (
    ColumnSpec,
    ColumnType,
    DatasetFormat,
    DatasetSpec,
    DatasetSplits,
    SplitPlan,
    seeded_splits,
)
from core.evaluation_spec import EvaluationSpec
from core.task_contract import TaskContract, TaskType
from core.task_spec import AnswerField, AnswerSchema, FieldType
from ingestion.parse import sha256_bytes

SOURCE: dict[str, Any] = {
    "name": "MuSiQue-Answerable",
    "version": "v1.0",
    "split": "dev",
    "repository": "https://github.com/StonyBrookNLP/musique",
    "revision": "922ac98f19a201998dbdae6d7f2887a5258dbdeb",
    "download_script": "download_data.sh",
    "archive": "musique_v1.0.zip",
    "archive_url": "https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h",
    "archive_sha256": "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd",
    "file": "data/musique_ans_v1.0_dev.jsonl",
    "file_sha256": "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b",
    "rows": 2417,
    "license": "CC-BY-4.0",
}

PREPROCESSING = {
    "version": "musique-wynk/1",
    "columns": {"id": "id", "input": "question", "context": "context", "target": "answer"},
    "context": (
        "for p in paragraphs sorted by idx: title + '\\n' + paragraph_text; joined by '\\n\\n'"
    ),
    "dropped": [
        "answer_aliases",
        "question_decomposition",
        "paragraphs[].is_supporting",
        "paragraphs[].idx",
        "answerable",
    ],
    "row_order": "sorted by official id",
    "encoding": "jsonl, utf-8, keys sorted, one row per line",
}
PREPROCESSING_HASH = canonical_hash(PREPROCESSING)

SAMPLING_RULE = "wynk-musique-sample/1"
HOPS = (2, 3, 4)
GOLD_FIELDS = ("answer_aliases", "question_decomposition", "is_supporting", "answerable")
_HOP = re.compile(r"^(\d)hop")


class SourceMismatch(ValueError):
    """The bytes are not the pinned official file."""


def load_source(path: Path) -> list[dict[str, Any]]:
    """The official dev rows, after checking the file's SHA-256 against ``SOURCE``."""
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if digest != SOURCE["file_sha256"]:
        raise SourceMismatch(f"{path.name} has sha256 {digest}, expected {SOURCE['file_sha256']}")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != SOURCE["rows"]:
        raise SourceMismatch(f"{len(rows)} rows, expected {SOURCE['rows']}")
    return rows


def hop_count(row_id: str) -> int:
    m = _HOP.match(row_id)
    if not m:
        raise ValueError(f"not a MuSiQue id: {row_id!r}")
    return int(m.group(1))


def render_context(paragraphs: Iterable[Mapping[str, Any]]) -> str:
    """Every paragraph's raw title + text, in official order. Reads nothing else."""
    ordered = sorted(paragraphs, key=lambda p: p["idx"])
    return "\n\n".join(f"{p['title']}\n{p['paragraph_text']}" for p in ordered)


def sanitize(row: Mapping[str, Any]) -> dict[str, str]:
    """The only view of a MuSiQue row that enters Wynk (``answer`` = evaluator-side target)."""
    return {
        "id": str(row["id"]),
        "question": str(row["question"]),
        "context": render_context(row["paragraphs"]),
        "answer": str(row["answer"]),
    }


def sample_key(seed: int, row_id: str) -> str:
    return sha256_hex(f"{SAMPLING_RULE}:{seed}:{row_id}")


def sample_ids(row_ids: Iterable[str], *, seed: int, per_hop: int) -> dict[int, list[str]]:
    """Equal-count, hash-ordered sample per hop count. Uses ids only - never answers/outcomes."""
    by_hop: dict[int, list[str]] = {h: [] for h in HOPS}
    for rid in row_ids:
        by_hop[hop_count(rid)].append(rid)
    out: dict[int, list[str]] = {}
    for h in HOPS:
        if len(by_hop[h]) < per_hop:
            raise ValueError(f"only {len(by_hop[h])} {h}-hop rows, need {per_hop}")
        ranked = sorted(by_hop[h], key=lambda r: (sample_key(seed, r), r))
        out[h] = sorted(ranked[:per_hop])
    return out


def dataset_bytes(rows: Sequence[Mapping[str, Any]], ids: Iterable[str]) -> bytes:
    """The sanitized subset as Wynk JSONL (deterministic bytes)."""
    by_id = {r["id"]: r for r in rows}
    lines = [
        json.dumps(sanitize(by_id[rid]), sort_keys=True, ensure_ascii=False)
        for rid in sorted(set(ids))
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _schema(*names: str) -> AnswerSchema:
    return AnswerSchema(fields=tuple(AnswerField(name=n, type=FieldType.STRING) for n in names))


def musique_contract(
    data: bytes, protocol: Mapping[str, Any], *, dataset_version: int = 1
) -> TaskContract:
    """The frozen TaskContract for the subset bytes, from the frozen protocol."""
    c = protocol["contract"]
    rows = sum(1 for line in data.splitlines() if line.strip())
    dataset = DatasetSpec(
        dataset_id=c["dataset_id"],
        dataset_version=dataset_version,
        name="MuSiQue-Answerable v1.0 dev (frozen Wynk subset)",
        content_hash=sha256_bytes(data),
        format=DatasetFormat.JSONL,
        columns=tuple(
            ColumnSpec(name=n, type=ColumnType.STRING)
            for n in ("id", "question", "context", "answer")
        ),
        id_column="id",
        input_columns=("question",),
        context_columns=("context",),
        target_columns=("answer",),
        row_count=rows,
    )
    return TaskContract(
        task_id=c["task_id"],
        contract_version=c["contract_version"],
        task_type=TaskType.QUESTION_ANSWERING,
        instructions=c["instructions"],
        input_schema=_schema("question", "context"),
        output_schema=_schema("answer"),
        dataset=dataset,
        evaluation=EvaluationSpec(**c["evaluation"]),
        constraints=ConstraintLimits(**c["constraints"]),
    )


def musique_splits(
    contract: TaskContract, row_ids: Iterable[str], plan: SplitPlan
) -> DatasetSplits:
    return seeded_splits(contract.dataset.identity_hash, tuple(row_ids), plan)


def manifest(
    selected: Mapping[int, Sequence[str]],
    *,
    protocol: Mapping[str, Any],
    contract: TaskContract,
    splits: DatasetSplits,
) -> dict[str, Any]:
    """Everything needed to rebuild and verify the frozen subset, plus its own hash."""
    sampling = protocol["sampling"]
    body = {
        "schema": "wynk-musique-manifest/1",
        "source": SOURCE,
        "preprocessing": {**PREPROCESSING, "hash": PREPROCESSING_HASH},
        "sampling": {
            "rule": SAMPLING_RULE,
            "seed": sampling["seed"],
            "per_hop": sampling["per_hop"],
            "hops": list(HOPS),
            "selection_inputs": "official ids only (hop count from the id prefix)",
        },
        "selected_ids": {str(h): list(ids) for h, ids in sorted(selected.items())},
        "dataset": {
            "rows": contract.dataset.row_count,
            "content_hash": contract.dataset.content_hash,
            "identity_hash": contract.dataset.identity_hash,
        },
        "splits": {
            "plan": splits.plan.model_dump(mode="json") if splits.plan else None,
            "identity_hash": splits.identity_hash,
            "rows": {s.role.value: list(s.row_ids) for s in splits.splits},
        },
        "contract_hash": contract.contract_hash,
    }
    return {**body, "manifest_hash": canonical_hash(body)}


def prepare(
    source: Path, protocol: Mapping[str, Any]
) -> tuple[TaskContract, DatasetSplits, bytes, dict[str, Any], list[dict[str, Any]]]:
    """``(contract, splits, subset bytes, manifest, official rows)`` from the official file."""
    rows = load_source(source)
    sampling = protocol["sampling"]
    selected = sample_ids(
        (r["id"] for r in rows), seed=sampling["seed"], per_hop=sampling["per_hop"]
    )
    ids = sorted(rid for group in selected.values() for rid in group)
    data = dataset_bytes(rows, ids)
    contract = musique_contract(data, protocol)
    splits = musique_splits(contract, ids, SplitPlan(**protocol["split_plan"]))
    frozen = manifest(selected, protocol=protocol, contract=contract, splits=splits)
    return contract, splits, data, frozen, rows


def check_manifest(fresh: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    """The rebuilt manifest must equal the committed one, hash for hash."""
    if dict(fresh) != dict(frozen):
        diff = sorted(k for k in set(fresh) | set(frozen) if fresh.get(k) != frozen.get(k))
        raise SourceMismatch(f"rebuilt manifest differs from the frozen one in {diff}")
