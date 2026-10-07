"""MuSiQue-Answerable behind the shared adapter contract.

Every benchmark-specific value is the frozen one in ``experiments.musique`` (source, sanitizer,
preprocessing, sampling rule): this module only exposes it through ``Benchmark``. The generic
``prepare`` must rebuild exactly the manifest MuSiQue protocol-v1/v2 ran on
(``tests/test_benchmark_adapter.py``); the MuSiQue results themselves are never rerun.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.task_contract import TaskType
from experiments import musique as m
from experiments.external.adapter import Benchmark, Columns

ROOT = Path(__file__).resolve().parent.parent


class MuSiQue(Benchmark):
    key = "musique"
    title = "MuSiQue-Answerable v1.0 dev"
    task_family = "multi-hop extractive QA over provided paragraphs"
    metric = "token F1"
    source = m.SOURCE
    preprocessing = m.PREPROCESSING
    sampling_rule = m.SAMPLING_RULE
    manifest_schema = "wynk-musique-manifest/1"
    per_stratum_key = "per_hop"
    columns = Columns(id="id", inputs=("question",), context=("context",), targets=("answer",))
    task_type = TaskType.QUESTION_ANSWERING
    gold_fields = (*m.GOLD_FIELDS, "paragraph_support_idx")
    dataset_name = "MuSiQue-Answerable v1.0 dev (frozen Wynk subset)"
    frozen_dir = ROOT / "musique_frozen"
    results_dir = ROOT / "results" / "musique"

    def parse(self, data: bytes) -> list[dict[str, Any]]:
        return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]

    def row_id(self, row: Mapping[str, Any]) -> str:
        return str(row["id"])

    def stratum(self, row: Mapping[str, Any]) -> str:
        return str(m.hop_count(str(row["id"])))  # official id prefix, never the answer

    def strata(self, sampling: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(h) for h in m.HOPS)

    def sanitize(self, row: Mapping[str, Any]) -> dict[str, str]:
        return m.sanitize(row)

    def sampling_manifest(self, sampling: Mapping[str, Any], strata: Sequence[str]) -> dict:
        return {
            "rule": self.sampling_rule,
            "seed": sampling["seed"],
            "per_hop": sampling["per_hop"],
            "hops": [int(s) for s in strata],
            "selection_inputs": "official ids only (hop count from the id prefix)",
        }


MUSIQUE = MuSiQue()
