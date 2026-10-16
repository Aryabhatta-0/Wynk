"""Encryption at rest for uploaded dataset bytes (#33): AES-256-GCM, keys outside the data dir.

Key material
------------
A *keyring* is one or more 256-bit master keys, each with a short id. The FIRST key encrypts
new blobs; the others only decrypt (rotation: prepend a new key, run ``python -m api.tenancy
secure-data`` to re-encrypt, then drop the old one). It comes from exactly one of:

    WYNK_DATA_KEYS="k2026a:<base64url 32 bytes>[,k2025b:<...>]"
    WYNK_DATA_KEYS_FILE=/run/secrets/wynk-data-keys   # same format; must be outside the data dir

``python -m api.tenancy generate-data-key`` prints a fresh entry. There is no default key, no
plaintext mode and no "development" fallback: a missing, empty or malformed keyring raises
``KeyConfigError`` and the server does not start (fail closed). Tests configure a fixed,
clearly-labelled test keyring explicitly (``tests/conftest.py``).

Keys are never written to the data dir, a database, a log or an error. Each workspace gets its
own data key: ``HKDF-SHA256(master, salt="wynk-blob-key/1", info=workspace_id)``, so a blob
copied into another workspace's partition does not decrypt there.

Envelope (one blob file)
------------------------
    b"WYNKBLOB" | 0x01 (format) | 0x01 (AES-256-GCM) | len(key_id) | key_id | nonce (12 bytes)
    | ciphertext || tag (16 bytes)

AAD = the header bytes above || 0x00 || workspace id || 0x00 || sha256 of the plaintext. The
header, the workspace and the blob's name (its plaintext hash) are therefore authenticated: a
flipped bit, a truncated file, a blob swapped for another one's file or moved to another
workspace fails with ``InvalidTag`` and is refused as corrupt.

Hashes: dataset content hashes, upload ids and every dataset / split identity are computed over
the canonical *plaintext* bytes and never over ciphertext; the random nonce makes every write's
ciphertext different, and nothing scientific ever depends on it.
"""

from __future__ import annotations

import base64
import os
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MAGIC = b"WYNKBLOB"
FORMAT_V1 = 1
ALG_AES_256_GCM = 1
NONCE_BYTES = 12
_KEY_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_SALT = b"wynk-blob-key/1"
ENV_KEYS = "WYNK_DATA_KEYS"
ENV_KEYS_FILE = "WYNK_DATA_KEYS_FILE"


class KeyConfigError(Exception):
    """The encryption keyring is missing or invalid. The message never contains key bytes."""


class DecryptionError(Exception):
    """A ciphertext failed authentication (tampered, truncated, wrong workspace / name)."""


class UnknownKey(Exception):
    """The blob was encrypted under a key id this keyring does not hold."""


class NotEncrypted(Exception):
    """The stored bytes are not a Wynk envelope (a pre-#33 plaintext blob)."""


def generate_key_entry(key_id: str) -> str:
    """A fresh ``key_id:<base64url>`` keyring entry (256 bits from ``secrets``)."""
    if not _KEY_ID.match(key_id):
        raise KeyConfigError("key ids are 1-32 lowercase letters, digits or -")
    return f"{key_id}:{base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip('=')}"


@dataclass(frozen=True)
class Keyring:
    keys: tuple[tuple[str, bytes], ...]  # (key_id, 32-byte master key); the first is active
    source: str = "explicit"  # where it came from, for operators ("env", "file", ...)

    def __post_init__(self) -> None:
        if not self.keys:
            raise KeyConfigError("the keyring holds no key")
        seen: set[str] = set()
        for kid, key in self.keys:
            if not _KEY_ID.match(kid):
                raise KeyConfigError(f"invalid key id {kid[:40]!r}")
            if kid in seen:
                raise KeyConfigError(f"key id {kid!r} appears twice")
            if len(key) != 32:
                raise KeyConfigError(f"key {kid!r} is not 32 bytes")
            seen.add(kid)

    def __repr__(self) -> str:  # never print key bytes
        return f"Keyring(key_ids={[k for k, _ in self.keys]}, source={self.source!r})"

    @property
    def active_id(self) -> str:
        return self.keys[0][0]

    @classmethod
    def parse(cls, spec: str, source: str = "explicit") -> Keyring:
        entries = []
        for item in filter(None, (p.strip() for p in spec.replace("\n", ",").split(","))):
            kid, sep, b64 = item.partition(":")
            if not sep:
                raise KeyConfigError("keyring entries are key_id:<base64url 32 bytes>")
            try:
                key = base64.urlsafe_b64decode(b64.strip() + "=" * (-len(b64.strip()) % 4))
            except (ValueError, TypeError):
                raise KeyConfigError(f"key {kid.strip()!r} is not base64url") from None
            entries.append((kid.strip(), key))
        return cls(tuple(entries), source)

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, *, data_dir: Path | str | None = None
    ) -> Keyring:
        """The production keyring. Fails closed: exactly one of ``WYNK_DATA_KEYS`` /
        ``WYNK_DATA_KEYS_FILE`` must be set, and a key file inside ``data_dir`` (beside the
        ciphertext it protects) is refused."""
        env = os.environ if env is None else env
        inline, path = env.get(ENV_KEYS, "").strip(), env.get(ENV_KEYS_FILE, "").strip()
        if inline and path:
            raise KeyConfigError(f"set {ENV_KEYS} or {ENV_KEYS_FILE}, not both")
        if path:
            file = Path(path).resolve()
            if data_dir is not None and file.is_relative_to(Path(data_dir).resolve()):
                raise KeyConfigError(
                    f"{ENV_KEYS_FILE} is inside the data directory; keep keys away from the "
                    "ciphertext they protect"
                )
            try:
                return cls.parse(file.read_text(encoding="utf-8"), "file")
            except OSError as exc:
                raise KeyConfigError(f"cannot read {ENV_KEYS_FILE}: {exc.strerror}") from None
        if inline:
            return cls.parse(inline, "env")
        raise KeyConfigError(
            f"uploaded datasets are encrypted at rest and no key is configured: set {ENV_KEYS} "
            f"(or {ENV_KEYS_FILE}); generate one with `python -m api.tenancy generate-data-key`"
        )

    def cipher(self, workspace_id: str | None) -> BlobCipher:
        return BlobCipher(self, workspace_id)

    def _key(self, key_id: str) -> bytes | None:
        return next((k for kid, k in self.keys if kid == key_id), None)


class BlobCipher:
    """Seal / open blob envelopes for ONE workspace (``None``: an unscoped store)."""

    def __init__(self, keyring: Keyring, workspace_id: str | None) -> None:
        self.keyring = keyring
        self.workspace_id = workspace_id
        self._aeads: dict[str, AESGCM] = {}

    def _aead(self, key_id: str) -> AESGCM:
        aead = self._aeads.get(key_id)
        if aead is None:
            master = self.keyring._key(key_id)
            if master is None:
                raise UnknownKey(key_id)
            info = b"workspace:" + (self.workspace_id or "-").encode()
            dek = HKDF(algorithm=hashes.SHA256(), length=32, salt=_SALT, info=info).derive(master)
            aead = self._aeads[key_id] = AESGCM(dek)
        return aead

    def _aad(self, header: bytes, content_sha256: str) -> bytes:
        ws = (self.workspace_id or "").encode()
        return header + b"\0" + ws + b"\0" + content_sha256.encode()

    def seal(self, plaintext: bytes, content_sha256: str) -> bytes:
        kid = self.keyring.active_id
        nonce = secrets.token_bytes(NONCE_BYTES)
        header = MAGIC + bytes((FORMAT_V1, ALG_AES_256_GCM, len(kid))) + kid.encode() + nonce
        return header + self._aead(kid).encrypt(nonce, plaintext, self._aad(header, content_sha256))

    @staticmethod
    def key_id_of(envelope: bytes) -> str:
        if not envelope.startswith(MAGIC):
            raise NotEncrypted()
        if len(envelope) < len(MAGIC) + 3:
            raise DecryptionError("truncated envelope header")
        n = envelope[len(MAGIC) + 2]
        return envelope[len(MAGIC) + 3 : len(MAGIC) + 3 + n].decode("ascii", "replace")

    def open(self, envelope: bytes, content_sha256: str) -> bytes:
        kid = self.key_id_of(envelope)
        fmt, alg, n = envelope[len(MAGIC) : len(MAGIC) + 3]
        if fmt != FORMAT_V1 or alg != ALG_AES_256_GCM:
            raise DecryptionError("unsupported envelope format")
        start = len(MAGIC) + 3 + n
        nonce = envelope[start : start + NONCE_BYTES]
        if len(nonce) != NONCE_BYTES or len(envelope) < start + NONCE_BYTES + 16:
            raise DecryptionError("truncated envelope")
        header = envelope[: start + NONCE_BYTES]
        try:
            return self._aead(kid).decrypt(
                nonce, envelope[start + NONCE_BYTES :], self._aad(header, content_sha256)
            )
        except InvalidTag:
            raise DecryptionError("ciphertext failed authentication") from None
