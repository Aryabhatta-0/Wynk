"""Content-addressed blob storage for uploaded dataset bytes.

A blob is named by the sha256 of its bytes and nothing else, so identical uploads share one blob
and no filesystem path ever becomes part of a dataset's identity. ``BlobStore`` is the narrow
interface the ingestion service uses; ``LocalBlobStore`` is the durable local implementation (an
object-storage implementation can replace it without touching callers).

Integrity: a blob is only ever reported healthy after its bytes were re-hashed; there is no
existence-only check. ``put`` reuses a stored blob only if it still hashes to its name. A missing
or corrupt blob is atomically replaced by the supplied bytes, which hash to that name by
construction, so the result is exactly the content the name promises.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import NamedTuple, Protocol

from store.tenancy import bind_directory

_CHUNK = 1024 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BlobError(Exception):
    """Base class for blob storage failures."""


class BlobNotFound(BlobError):
    pass


class BlobCorrupted(BlobError):
    """Stored bytes no longer hash to their name: refuse to serve them."""


class BlobRef(NamedTuple):
    sha256: str
    size_bytes: int
    created: bool  # False iff a healthy identical blob was already stored (deduplicated)


class BlobStore(Protocol):
    def put(self, data: bytes) -> BlobRef:
        """Store ``data`` under its sha256, atomically. A stored blob is reused only if it
        verifies; a missing or corrupt one is replaced by ``data``."""
        ...

    def get(self, sha256: str) -> bytes:
        """The stored bytes, verified against their hash."""
        ...

    def verify(self, sha256: str) -> None:
        """Re-hash the stored bytes: ``BlobNotFound`` / ``BlobCorrupted`` unless they match."""
        ...


def _check(sha256: str) -> str:
    if not _SHA256.match(sha256):
        raise ValueError("blob names are lowercase sha256 hex digests")
    return sha256


class LocalBlobStore:
    """``<root>/sha256/ab/cd/<digest>``; written via temp file + fsync + atomic rename."""

    def __init__(self, root: Path | str, *, workspace_id: str | None = None) -> None:
        self.root = Path(root)
        bind_directory(self.root, workspace_id)  # #32: before any read or write
        self.workspace_id = workspace_id

    def _path(self, sha256: str) -> Path:
        h = _check(sha256)
        return self.root / "sha256" / h[:2] / h[2:4] / h

    def verify(self, sha256: str) -> None:
        h = hashlib.sha256()
        try:
            with self._path(sha256).open("rb") as f:
                while chunk := f.read(_CHUNK):
                    h.update(chunk)
        except FileNotFoundError:
            raise BlobNotFound(sha256) from None
        if h.hexdigest() != sha256:
            raise BlobCorrupted(sha256)

    def put(self, data: bytes) -> BlobRef:
        digest = hashlib.sha256(data).hexdigest()
        final = self._path(digest)
        try:
            self.verify(digest)
            return BlobRef(digest, len(data), created=False)  # healthy duplicate: dedup
        except (BlobNotFound, BlobCorrupted):
            pass  # missing or corrupt: atomically (re)write the exact supplied bytes
        final.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=final.parent, prefix=".tmp-", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, final)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        _fsync_dir(final.parent)
        return BlobRef(digest, len(data), created=True)

    def get(self, sha256: str) -> bytes:
        path = self._path(sha256)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise BlobNotFound(sha256) from None
        if hashlib.sha256(data).hexdigest() != sha256:
            raise BlobCorrupted(sha256)
        return data


def _fsync_dir(path: Path) -> None:
    """Make the rename durable. Windows cannot open directories for fsync; NTFS journals it."""
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
