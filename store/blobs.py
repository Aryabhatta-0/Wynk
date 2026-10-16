"""Content-addressed, encrypted blob storage for uploaded dataset bytes.

A blob is named by the sha256 of its *plaintext* bytes and nothing else, so identical uploads
share one blob and no filesystem path ever becomes part of a dataset's identity. ``BlobStore`` is
the narrow interface the ingestion service uses; ``LocalBlobStore`` is the durable local
implementation (an object-storage implementation can replace it without touching callers).

Encryption at rest (#33): every file is an AES-256-GCM envelope (``store.crypto``) under the
workspace's data key, authenticated together with the workspace id and the blob's name. The
keyring comes from the environment (never from the data dir); without one the store refuses to
open. Callers only ever see verified plaintext.

Integrity: a blob is only ever reported healthy after it was decrypted (authenticated) and its
plaintext re-hashed; there is no existence-only check. ``put`` reuses a stored blob only if it
verifies. A missing, corrupt or pre-#33 plaintext blob is atomically replaced by the supplied
bytes (which hash to that name by construction), freshly encrypted, so the result is exactly the
content the name promises. A blob under a key id the keyring no longer holds fails closed
(``BlobKeyUnavailable``).

Deletion: ``delete`` unlinks the envelope; afterwards ``get`` / ``verify`` raise
``BlobNotFound``. The ingestion service decides WHEN a blob may go (no live upload references it).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple, Protocol

from store.crypto import BlobCipher, DecryptionError, Keyring, NotEncrypted, UnknownKey
from store.tenancy import bind_directory

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BlobError(Exception):
    """Base class for blob storage failures."""


class BlobNotFound(BlobError):
    pass


class BlobCorrupted(BlobError):
    """Stored bytes fail authentication or no longer hash to their name: refuse to serve them."""


class BlobNotEncrypted(BlobError):
    """A pre-#33 plaintext blob: refused until ``python -m api.tenancy secure-data`` (or an
    identical re-upload) encrypts it."""


class BlobKeyUnavailable(BlobError):
    """The blob's key id is not in the configured keyring (rotated away / wrong keyring)."""


class BlobRef(NamedTuple):
    sha256: str
    size_bytes: int  # plaintext size
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

    def delete(self, sha256: str) -> bool:
        """Remove the blob. ``True`` iff something was removed."""
        ...


def _check(sha256: str) -> str:
    if not _SHA256.match(sha256):
        raise ValueError("blob names are lowercase sha256 hex digests")
    return sha256


class SecureReport(NamedTuple):
    encrypted: int  # pre-#33 plaintext blobs encrypted in place
    rekeyed: int  # envelopes re-encrypted under the active key
    healthy: int  # already encrypted under the active key
    refused: list[str]  # blobs left untouched: corrupt, or under an unknown key


class LocalBlobStore:
    """``<root>/sha256/ab/cd/<digest>``; written via temp file + fsync + atomic rename."""

    def __init__(
        self,
        root: Path | str,
        *,
        workspace_id: str | None = None,
        keyring: Keyring | None = None,
    ) -> None:
        self.root = Path(root)
        # no key -> no store: resolved before the directory is bound or anything is read
        keyring = keyring if keyring is not None else Keyring.from_env(data_dir=self.root.parent)
        bind_directory(self.root, workspace_id)  # #32: before any read or write
        self.workspace_id = workspace_id
        self._cipher = BlobCipher(keyring, workspace_id)

    def __repr__(self) -> str:
        return f"LocalBlobStore({str(self.root)!r}, workspace_id={self.workspace_id!r})"

    def _path(self, sha256: str) -> Path:
        h = _check(sha256)
        return self.root / "sha256" / h[:2] / h[2:4] / h

    def _open(self, sha256: str, envelope: bytes) -> bytes:
        try:
            data = self._cipher.open(envelope, sha256)
        except NotEncrypted:  # plaintext: a pre-#33 blob if it hashes to its name
            if hashlib.sha256(envelope).hexdigest() == sha256:
                raise BlobNotEncrypted(sha256) from None
            raise BlobCorrupted(sha256) from None
        except UnknownKey:
            raise BlobKeyUnavailable(sha256) from None
        except DecryptionError:
            raise BlobCorrupted(sha256) from None
        if hashlib.sha256(data).hexdigest() != sha256:
            raise BlobCorrupted(sha256)
        return data

    def _read(self, sha256: str) -> bytes:
        try:
            return self._path(sha256).read_bytes()
        except FileNotFoundError:
            raise BlobNotFound(sha256) from None

    def verify(self, sha256: str) -> None:
        self._open(sha256, self._read(sha256))

    def get(self, sha256: str) -> bytes:
        return self._open(sha256, self._read(sha256))

    def put(self, data: bytes) -> BlobRef:
        digest = hashlib.sha256(data).hexdigest()
        try:
            self.verify(digest)
            return BlobRef(digest, len(data), created=False)  # healthy duplicate: dedup
        except (BlobNotFound, BlobCorrupted, BlobNotEncrypted, BlobKeyUnavailable):
            pass  # missing / corrupt / plaintext / old key: atomically (re)write the bytes
        self._write(digest, self._cipher.seal(data, digest))
        return BlobRef(digest, len(data), created=True)

    def _write(self, digest: str, envelope: bytes) -> None:
        final = self._path(digest)
        final.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=final.parent, prefix=".tmp-", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(envelope)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, final)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        _fsync_dir(final.parent)

    def delete(self, sha256: str) -> bool:
        path = self._path(sha256)
        try:
            os.unlink(path)
        except FileNotFoundError:
            return False
        _fsync_dir(path.parent)
        return True

    def names(self) -> Iterator[tuple[str, float]]:
        """Every stored blob name with its modification time (for garbage collection)."""
        base = self.root / "sha256"
        if not base.exists():
            return
        for path in sorted(base.glob("*/*/*")):
            if _SHA256.match(path.name):
                yield path.name, path.stat().st_mtime

    def secure(self) -> SecureReport:
        """Encrypt every pre-#33 plaintext blob (only if it still hashes to its name) and
        re-encrypt envelopes under an older key with the active one. Idempotent; never touches
        a blob it cannot verify."""
        encrypted = rekeyed = healthy = 0
        refused: list[str] = []
        for name, _ in list(self.names()):
            raw = self._read(name)
            try:
                kid = BlobCipher.key_id_of(raw)
            except NotEncrypted:
                if hashlib.sha256(raw).hexdigest() != name:
                    refused.append(name)
                    continue
                self._write(name, self._cipher.seal(raw, name))
                encrypted += 1
                continue
            except DecryptionError:
                refused.append(name)
                continue
            try:
                data = self._open(name, raw)
            except BlobError:
                refused.append(name)
                continue
            if kid == self._cipher.keyring.active_id:
                healthy += 1
            else:
                self._write(name, self._cipher.seal(data, name))
                rekeyed += 1
        return SecureReport(encrypted, rekeyed, healthy, refused)

    def collect(self, live: set[str], *, older_than_s: float = 3600.0) -> list[str]:
        """Delete blobs no live upload references, if older than ``older_than_s`` (an upload
        in flight writes its blob before its metadata row). Returns the removed names."""
        cutoff = time.time() - older_than_s
        removed = []
        for name, mtime in list(self.names()):
            if name not in live and mtime < cutoff and self.delete(name):
                removed.append(name)
        return removed


def _fsync_dir(path: Path) -> None:
    """Make the rename durable. Windows cannot open directories for fsync; NTFS journals it."""
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
