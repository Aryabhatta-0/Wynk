"""MMLU-Pro (test split) behind the shared adapter contract.

Source: the official release on the Hugging Face Hub, ``TIGER-Lab/MMLU-Pro`` at an exact
revision; the file ``data/test-00000-of-00001.parquet`` is pinned by SHA-256 (equal to the Hub's
LFS object id). The dataset itself is never committed; reading parquet needs ``pyarrow``.

Sanitization keeps exactly four things per row:

    question_id   -> id        (row id)
    question      -> question  (INPUT)
    options       -> options   (INPUT; "A. <text>" lines, official order, nothing marks the answer)
    answer        -> answer    (TARGET letter A-J; evaluator side only, never reaches execution)

Dropped before a row exists: ``answer_index`` (the target as an index), ``cot_content``
(a gold rationale; empty in the test split, dropped anyway), ``src`` (origin metadata) and
``category`` (used ONLY as the sampling stratum, recorded in the manifest).

Sampling: per dataset-native ``category`` (all 14), ids are ordered by
``sha256("wynk-mmlu-pro-sample/1:<seed>:<question_id>")`` and the first ``per_category`` kept -
equal counts per category. Nothing reads an answer, a rationale or a model outcome.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.task_contract import TaskType
from experiments.external.adapter import Benchmark, Columns

ROOT = Path(__file__).resolve().parent.parent
LETTERS = "ABCDEFGHIJ"


def render_options(options: list[str]) -> str:
    """Every option, lettered in official order. Reads nothing but the option texts."""
    if not 1 < len(options) <= len(LETTERS):
        raise ValueError(f"MMLU-Pro rows have 2-10 options, got {len(options)}")
    return "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(options))


class MMLUPro(Benchmark):
    key = "mmlu-pro"
    title = "MMLU-Pro (test)"
    task_family = "context-free multiple-choice reasoning (10-way classification)"
    metric = "accuracy"
    source = {
        "name": "MMLU-Pro",
        "split": "test",
        "repository": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "file": "data/test-00000-of-00001.parquet",
        "file_url": (
            "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/"
            "b189ec765aa7ed75c8acfea42df31fdae71f97be/data/test-00000-of-00001.parquet"
        ),
        "file_sha256": "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
        "file_bytes": 4144185,
        "rows": 12032,
        "license": "MIT",
        "paper": "Wang et al. 2024, MMLU-Pro, arXiv:2406.01574",
    }
    preprocessing = {
        "version": "mmlu-pro-wynk/1",
        "columns": {
            "id": "question_id",
            "question": "question",
            "options": "options",
            "answer": "answer",
        },
        "options": "for i, text in enumerate(options): 'ABCDEFGHIJ'[i] + '. ' + text; '\\n'-joined",
        "dropped": ["answer_index", "cot_content", "src", "category"],
        "stratum": "category (dataset-native; sampling only, never a column)",
        "row_order": "sorted by str(question_id)",
        "encoding": "jsonl, utf-8, keys sorted, one row per line",
    }
    sampling_rule = "wynk-mmlu-pro-sample/1"
    manifest_schema = "wynk-benchmark-manifest/1"
    per_stratum_key = "per_category"
    columns = Columns(id="id", inputs=("question", "options"), context=(), targets=("answer",))
    task_type = TaskType.CLASSIFICATION
    gold_fields = ("answer_index", "cot_content", "src", "category")
    dataset_name = "MMLU-Pro test (frozen Wynk subset)"
    frozen_dir = ROOT / "external" / "frozen" / "mmlu-pro"
    results_dir = ROOT / "results" / "mmlu-pro"

    def parse(self, data: bytes) -> list[dict[str, Any]]:
        import pyarrow.parquet as pq  # optional dependency: only the official file needs it

        return pq.read_table(io.BytesIO(data)).to_pylist()

    def row_id(self, row: Mapping[str, Any]) -> str:
        return str(row["question_id"])

    def stratum(self, row: Mapping[str, Any]) -> str:
        return str(row["category"])

    def sanitize(self, row: Mapping[str, Any]) -> dict[str, str]:
        answer = str(row["answer"])
        if answer not in LETTERS[: len(row["options"])]:
            raise ValueError(f"row {row['question_id']}: answer {answer!r} is not an option")
        return {
            "id": str(row["question_id"]),
            "question": str(row["question"]),
            "options": render_options(list(row["options"])),
            "answer": answer,
        }


MMLU_PRO = MMLUPro()
