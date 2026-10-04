"""Local JSON persistence: one file per task class, ``<root>/<task_class>.json``.

Serialization is deterministic: same ``WorkflowMemory`` -> byte-identical file (sorted keys,
fixed indent, ASCII, LF newlines). No database, no cloud, no extra dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from memory.models import WorkflowMemory

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results" / "memory"


def dumps(memory: WorkflowMemory) -> bytes:
    data = memory.model_dump(mode="json")
    text = json.dumps(data, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return (text + "\n").encode("ascii")


def loads(raw: bytes) -> WorkflowMemory:
    return WorkflowMemory.model_validate(json.loads(raw.decode("ascii")))


class WorkflowMemoryStore:
    def __init__(self, root: Path | str = DEFAULT_DIR) -> None:
        self.root = Path(root)

    def path(self, task_class: str) -> Path:
        if not task_class.isalnum():
            raise ValueError(f"bad task class {task_class!r}")
        return self.root / f"{task_class}.json"

    def load(self, task_class: str) -> WorkflowMemory | None:
        """The stored memory for ``task_class``, or ``None`` if there is none."""
        p = self.path(task_class)
        if not p.exists():
            return None
        memory = loads(p.read_bytes())
        if memory.task_class != task_class:
            raise ValueError(f"{p} holds class {memory.task_class}, not {task_class}")
        return memory

    def save(self, memory: WorkflowMemory) -> Path:
        """Write atomically (temp file + replace) so a crash never leaves half a memory."""
        p = self.path(memory.task_class)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_bytes(dumps(memory))
        os.replace(tmp, p)
        return p

    def clear(self, task_class: str) -> bool:
        """Delete the memory for ``task_class``. Returns whether one existed."""
        p = self.path(task_class)
        if p.exists():
            p.unlink()
            return True
        return False
