"""Canonical JSON + hashing shared by every identity in the framework.

Identity (genome hash, run key, DAG hash) must never depend on timestamps, random
ids, dict insertion order, or the environment. Everything hashes through here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def canonical_json(obj: Any) -> str:
    """Compact, key-sorted, ASCII-only JSON. Pydantic models are dumped in JSON mode."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))
