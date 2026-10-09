"""Authentication, workspaces and tenant isolation (#32).

``authenticated principal -> workspace -> project -> tenant-scoped resources``

* **Identity** (``store/identity.py``, ``<data>/identity.sqlite3``): users, workspaces,
  memberships (``OWNER`` / ``MEMBER``) and workspace API keys. No OAuth / social login: internal
  ids (``usr-…``, ``ws-…``) that a later identity provider can map onto.
* **Partition = tenant authority.** Every resource of a workspace lives in that workspace's own
  stores under ``<data>/workspaces/<workspace_id>/`` and every store is bound to it
  (``store/tenancy.py``): a store bound to another workspace - or to any workspace, when opened
  unscoped - refuses to open. Projects, uploads, datasets, splits, jobs, attempts, artifacts,
  promotions, champions, workflow versions, deployments, inferences, feedback, policies,
  triggers and re-optimizations are created INSIDE that partition, so each inherits its workspace
  from the chain it was derived from and can never change workspace. Overlapping human-readable
  ids (two workspaces' ``capitals`` datasets) never collide, and a foreign id is simply absent:
  the response is the same ``*_not_found`` as for an id that never existed (no oracle).
* **Requests.** ``TenantRouter.handle`` authenticates FIRST (``Authorization: Bearer <key>``),
  resolves ``principal + workspace + scopes`` from the key alone (never from a caller-supplied
  workspace or project id), checks the route's scope, checks quotas for creating routes, then
  hands the request to that workspace's ``ProductAPI``. Only ``GET /api/v1/health`` is public.

API keys: ``wynk_sk_<key_id:16 hex>_<secret:43 url-safe base64 chars>`` (256-bit secret). The
plaintext is returned once, at creation; the store keeps ``key_id`` (lookup), the scopes, and
``sha256("wynk-api-key/1" NUL key_id NUL secret)`` compared in constant time. A revoked or
expired key, a key whose user is no longer a member, or a key of a workspace that does not exist
fails closed with the same ``unauthenticated`` 401.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from api import product
from api.product import (
    API_PREFIX,
    ApiFailure,
    ApiResponse,
    ProductAPI,
    _error,
    _ok,
    _Request,
)
from experiments.deployment import ChampionDeployments, InferenceRuntime
from experiments.jobs import ExperimentJobs, JobRuntime, JobWorker
from experiments.monitoring import ProductionMonitor
from experiments.promotion import ChampionPromotions
from ingestion.service import DatasetService, IngestLimits
from store.blobs import LocalBlobStore
from store.champions import SQLiteChampionStore
from store.datasets import SQLiteDatasetRepository
from store.deployments import SQLiteDeploymentStore
from store.identity import (
    IdentityConflict,
    KeyRow,
    MembershipRow,
    MigrationRow,
    Role,
    SQLiteIdentityStore,
    UserRow,
    WorkspaceRow,
)
from store.jobs import TERMINAL, SQLiteJobStore
from store.monitoring import SQLiteMonitoringStore
from store.tenancy import TenantBindingError, bind_directory, bind_sqlite, check_workspace_id

log = logging.getLogger("wynk.api.tenancy")

SCOPES = ("read", "write", "invoke", "admin")
KEY_PREFIX = "wynk_sk_"
KEY_RE = re.compile(r"^wynk_sk_([0-9a-f]{16})_([A-Za-z0-9_-]{43})$")
_KEY_HASH_DOMAIN = b"wynk-api-key/1"

# Every store file / directory of one workspace, relative to its partition (and, before #32,
# to the data dir itself - the legacy single-tenant layout).
STORE_FILES = (
    "metadata.sqlite3",
    "jobs.sqlite3",
    "champions.sqlite3",
    "deployments.sqlite3",
    "monitoring.sqlite3",
)
STORE_DIRS = ("blobs",)
LEGACY_SOURCE = "legacy-single-tenant-data-dir"
DEFAULT_LEGACY_WORKSPACE = "ws-legacy"

# -- scopes ---------------------------------------------------------------------------------
# Product routes (api.product._ROUTES) that change state need ``write``; running a deployed
# workflow needs ``invoke``; every other product route is a read. ``admin`` is only for
# workspace administration (members, keys, new workspaces) and only counts for an OWNER.
WRITE_ROUTES = frozenset(
    {
        "create_project",
        "upload",
        "register",
        "create_splits",
        "create_experiment",
        "cancel_experiment",
        "resume_experiment",
        "promote_experiment",
        "publish_workflow",
        "stage_workflow",
        "promote_workflow",
        "rollback_deployment",
        "submit_feedback",
        "register_policy",
        "evaluate_trigger",
        "reoptimize",
    }
)
INVOKE_ROUTES = frozenset({"invoke_workflow", "invoke_production"})


def route_scope(name: str, method: str) -> str:
    if name in INVOKE_ROUTES:
        return "invoke"
    if name in WRITE_ROUTES:
        return "write"
    if method != "GET":  # a state-changing product route nobody classified: refuse loudly
        raise AssertionError(f"route {name} has no scope")
    return "read"


# -- keys -----------------------------------------------------------------------------------
def hash_secret(key_id: str, secret: str) -> str:
    return hashlib.sha256(
        _KEY_HASH_DOMAIN + b"\0" + key_id.encode() + b"\0" + secret.encode()
    ).hexdigest()


def new_key() -> tuple[str, str, str]:
    """``(plaintext, key_id, secret)`` of a fresh key."""
    key_id = secrets.token_hex(8)
    secret = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    return f"{KEY_PREFIX}{key_id}_{secret}", key_id, secret


def parse_key(plaintext: str) -> tuple[str, str] | None:
    m = KEY_RE.match(plaintext)
    return (m.group(1), m.group(2)) if m else None


# -- quotas ---------------------------------------------------------------------------------
class Quotas(BaseModel):
    """Per-workspace limits, checked BEFORE a resource is created (a refusal writes nothing).
    The authorities are the workspace's own stores - a count of what exists - so no separate
    counter can drift from the data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_projects: PositiveInt = 50
    max_uploads: PositiveInt = 500
    max_stored_bytes: PositiveInt = 1024 * 1024 * 1024  # sum of stored upload sizes
    max_dataset_versions: PositiveInt = 500
    max_active_jobs: PositiveInt = 4  # experiment jobs not COMPLETED / CANCELLED / FAILED
    max_workflow_versions: PositiveInt = 200
    max_api_keys: PositiveInt = 50  # active (not revoked, not expired)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Quotas:
        env = os.environ if env is None else env
        values = {}
        for name in cls.model_fields:
            raw = env.get("WYNK_QUOTA_" + name.removeprefix("max_").upper())
            if raw is not None:
                values[name] = int(raw)
        return cls(**values)


# quota-checked routes -> the usage counter they grow
QUOTA_ROUTES: dict[str, tuple[str, ...]] = {
    "create_project": ("projects",),
    "upload": ("uploads", "stored_bytes"),
    "register": ("dataset_versions",),
    "create_experiment": ("active_jobs",),
    "resume_experiment": ("active_jobs",),
    "reoptimize": ("active_jobs",),
    "publish_workflow": ("workflow_versions",),
}
_LIMIT = {
    "projects": "max_projects",
    "uploads": "max_uploads",
    "stored_bytes": "max_stored_bytes",
    "dataset_versions": "max_dataset_versions",
    "active_jobs": "max_active_jobs",
    "workflow_versions": "max_workflow_versions",
    "api_keys": "max_api_keys",
}


# -- principal ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Principal:
    """Who is calling, resolved by the server from an API key and nothing else."""

    workspace_id: str
    user_id: str
    key_id: str
    role: Role
    scopes: frozenset[str]

    def record(self) -> dict[str, Any]:
        """What #30 / #31 records store as ``authenticated_principal``."""
        return {
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "key_id": self.key_id,
            "authenticated_by": "api_key",
        }


class LegacyDataNotMigrated(RuntimeError):
    pass


class TenantIntegrityError(RuntimeError):
    pass


# -- request / response models --------------------------------------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


Scope = Field(pattern=r"^(read|write|invoke|admin)$")


class NewWorkspace(_Strict):
    name: str = Field(min_length=1, max_length=200)


class NewMember(_Strict):
    email: str = Field(pattern=r"^[^@\s]{1,64}@[^@\s]{1,190}$")
    display_name: str = Field(default="", max_length=200)
    role: Role = Role.MEMBER


class NewKey(_Strict):
    name: str = Field(min_length=1, max_length=100)
    scopes: list[str] = Field(min_length=1, max_length=4)
    user_id: str | None = Field(default=None, pattern=r"^usr-[0-9a-f]{20}$")
    expires_in_s: PositiveInt | None = Field(default=None, le=10 * 366 * 86400)


class WorkspaceView(_Strict):
    workspace_id: str
    name: str
    created_by: str
    created_at: str


class PrincipalView(_Strict):
    user_id: str
    key_id: str
    role: str
    scopes: list[str]


class CurrentWorkspace(_Strict):
    workspace: WorkspaceView
    principal: PrincipalView


class MemberView(_Strict):
    membership_id: str
    user_id: str
    email: str
    display_name: str
    role: str
    created_by: str
    created_at: str


class MemberList(_Strict):
    members: list[MemberView]


class KeyView(_Strict):
    """An API key's metadata. The secret is never part of it (and never stored)."""

    key_id: str
    prefix: str  # what the key starts with: ``wynk_sk_<key_id>_``
    workspace_id: str
    user_id: str
    name: str
    scopes: list[str]
    created_by: str
    created_at: str
    expires_at: str | None
    revoked_at: str | None
    revoked_by: str | None
    last_used_at: str | None
    active: bool


class CreatedKey(_Strict):
    key: KeyView
    secret: str  # the plaintext key - shown ONCE, here, and stored nowhere


class KeyList(_Strict):
    keys: list[KeyView]


class CreatedWorkspace(_Strict):
    workspace: WorkspaceView
    membership: MemberView
    api_key: CreatedKey


class RemovedMember(_Strict):
    user_id: str
    removed: bool


class QuotaView(_Strict):
    workspace_id: str
    limits: dict[str, int]
    usage: dict[str, int]


class Health(_Strict):
    status: str


# -- one workspace --------------------------------------------------------------------------
@dataclass
class Tenant:
    """One workspace's partition: its bound stores and the ProductAPI over them."""

    workspace_id: str
    root: Path
    api: ProductAPI
    lock: threading.Lock = field(default_factory=threading.Lock)
    worker: JobWorker | None = None

    def usage(self) -> dict[str, int]:
        """Counted from the partition's own stores: the quota authority."""

        def one(name: str, sql: str) -> int:
            conn = sqlite3.connect(f"file:{self.root / name}?mode=ro", uri=True, timeout=30)
            try:
                return int(conn.execute(sql).fetchone()[0] or 0)
            finally:
                conn.close()

        terminal = ",".join(f"'{s.value}'" for s in TERMINAL)
        return {
            "projects": one("metadata.sqlite3", "SELECT COUNT(*) FROM projects"),
            "uploads": one("metadata.sqlite3", "SELECT COUNT(*) FROM uploads"),
            "stored_bytes": one(
                "metadata.sqlite3",
                "SELECT COALESCE(SUM(json_extract(record_json, '$.size_bytes')), 0) FROM uploads",
            ),
            "dataset_versions": one("metadata.sqlite3", "SELECT COUNT(*) FROM dataset_versions"),
            "active_jobs": one(
                "jobs.sqlite3",
                f"SELECT COUNT(*) FROM experiment_jobs WHERE state NOT IN ({terminal})",
            ),
            "workflow_versions": one(
                "deployments.sqlite3", "SELECT COUNT(*) FROM workflow_versions"
            ),
        }


RuntimeFactory = Callable[[DatasetService], JobRuntime]


class WorkerGroup:
    """The per-workspace job workers of one router."""

    def __init__(self, router: TenantRouter) -> None:
        self.router = router

    def stop(self) -> None:
        self.router.serve_workers = False
        for tenant in list(self.router._tenants.values()):
            if tenant.worker is not None:
                tenant.worker.stop()
                tenant.worker = None


class TenantRouter:
    """The product API for many workspaces: authenticate, scope, quota, then dispatch into
    exactly one workspace's partition. Same ``handle`` signature as ``ProductAPI``."""

    def __init__(
        self,
        data_dir: Path | str,
        *,
        max_upload_bytes: int | None = None,
        runtime: RuntimeFactory | None = None,
        inference: InferenceRuntime | None = None,
        quotas: Quotas | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        legacy = legacy_items(self.root)
        if legacy:
            raise LegacyDataNotMigrated(
                f"{self.root} holds single-tenant data ({', '.join(legacy)}); assign it to a "
                "workspace first: python -m api.tenancy migrate-legacy --data-dir "
                f"{self.root} --owner-email <email> (see docs/tenancy.md)"
            )
        self.identity = SQLiteIdentityStore(self.root / "identity.sqlite3")
        self.limits = (
            IngestLimits()
            if max_upload_bytes is None
            else IngestLimits(max_upload_bytes=max_upload_bytes)
        )
        self.runtime = runtime
        self.inference = inference
        self.quotas = quotas or Quotas()
        self.now = now
        self.serve_workers = False  # start_workers() turns this on
        self._tenants: dict[str, Tenant] = {}
        self._tenants_lock = threading.Lock()
        self._identity_lock = threading.Lock()

    # -- partitions -------------------------------------------------------------------------
    def tenant(self, workspace_id: str) -> Tenant:
        """The workspace's partition (opened once per process). Every store checks its binding
        to ``workspace_id`` before it reads or writes anything."""
        check_workspace_id(workspace_id)
        with self._tenants_lock:
            cached = self._tenants.get(workspace_id)
            if cached is not None:
                return cached
            if self.identity.workspace(workspace_id) is None:
                raise TenantIntegrityError(f"no workspace {workspace_id}")
            root = self.root / "workspaces" / workspace_id
            try:
                tenant = self._open(workspace_id, root)
            except TenantBindingError as exc:
                raise TenantIntegrityError(str(exc)) from exc
            self._tenants[workspace_id] = tenant
            if self.serve_workers:
                tenant.worker = _start(tenant.api)
            return tenant

    def _open(self, workspace_id: str, root: Path) -> Tenant:
        ws = workspace_id
        service = DatasetService(
            SQLiteDatasetRepository(root / "metadata.sqlite3", workspace_id=ws),
            LocalBlobStore(root / "blobs", workspace_id=ws),
            self.limits,
        )
        bound = self.runtime(service) if self.runtime else None
        jobs = ExperimentJobs(SQLiteJobStore(root / "jobs.sqlite3", workspace_id=ws), bound)
        promotions = ChampionPromotions(
            SQLiteChampionStore(root / "champions.sqlite3", workspace_id=ws),
            jobs.store,
            bound if bound is not None and hasattr(bound, "bind_heldout") else None,  # type: ignore[arg-type]
        )
        deployments = ChampionDeployments(
            SQLiteDeploymentStore(root / "deployments.sqlite3", workspace_id=ws),
            promotions,
            self.inference,
        )
        monitor = ProductionMonitor(
            deployments,
            SQLiteMonitoringStore(root / "monitoring.sqlite3", workspace_id=ws),
            jobs,
            service,
        )
        # every store of the partition must name the same workspace: cross-check the handles
        stores = (
            service.repo,
            service.blobs,
            jobs.store,
            promotions.store,
            deployments.store,
            monitor.store,
        )
        if {getattr(s, "workspace_id", None) for s in stores} != {ws}:
            raise TenantBindingError(f"the stores of {ws} do not agree on their workspace")
        api = ProductAPI(service, jobs, promotions, deployments, monitor)
        return Tenant(ws, root, api)

    def start_workers(self) -> WorkerGroup | None:
        """Recover and run experiment jobs for every workspace (and any opened later): one
        worker per partition, each over that workspace's own job store only."""
        self.serve_workers = True
        for ws in self.identity.workspaces():
            tenant = self.tenant(ws.workspace_id)
            if tenant.worker is None:
                tenant.worker = _start(tenant.api)
        return WorkerGroup(self) if self.runtime is not None else None

    # -- identity administration (programmatic; the API wraps these) ------------------------
    def _now(self) -> str:
        return self.now().astimezone(UTC).isoformat(timespec="milliseconds")

    def _user(self, c: sqlite3.Connection, email: str, display_name: str) -> UserRow:
        r = c.execute(
            "SELECT user_id, email, display_name, created_at FROM users WHERE email=?", (email,)
        ).fetchone()
        if r is not None:
            return UserRow(*r)
        user = UserRow("usr-" + secrets.token_hex(10), email, display_name or email, self._now())
        SQLiteIdentityStore.insert_user(c, user)
        return user

    def _key_row(
        self,
        workspace_id: str,
        user_id: str,
        name: str,
        scopes: list[str],
        created_by: str,
        expires_in_s: int | None,
        plaintext: str | None = None,
    ) -> tuple[KeyRow, str]:
        if plaintext is None:
            plaintext, key_id, secret = new_key()
        else:
            parsed = parse_key(plaintext)
            if parsed is None:
                raise ValueError(f"an API key looks like {KEY_PREFIX}<16 hex>_<43 chars>")
            key_id, secret = parsed
        now = self.now().astimezone(UTC)
        row = KeyRow(
            key_id=key_id,
            workspace_id=workspace_id,
            user_id=user_id,
            name=name,
            scopes=tuple(s for s in SCOPES if s in scopes),
            secret_hash=hash_secret(key_id, secret),
            created_by=created_by,
            created_at=now.isoformat(timespec="milliseconds"),
            expires_at=(now + timedelta(seconds=expires_in_s)).isoformat(timespec="milliseconds")
            if expires_in_s
            else None,
            revoked_by=None,
            revoked_at=None,
            last_used_at=None,
        )
        return row, plaintext

    def create_workspace(
        self,
        name: str,
        owner_email: str,
        owner_name: str = "",
        *,
        workspace_id: str | None = None,
        key_name: str = "owner",
        key_plaintext: str | None = None,
    ) -> tuple[WorkspaceRow, MembershipRow, KeyRow, str]:
        """A workspace, its first OWNER and an all-scope key for them - one transaction."""
        ws_id = check_workspace_id(workspace_id or "ws-" + secrets.token_hex(10))

        def create(c: sqlite3.Connection) -> tuple[WorkspaceRow, MembershipRow, KeyRow, str]:
            user = self._user(c, owner_email, owner_name)
            ws = WorkspaceRow(ws_id, name, user.user_id, self._now())
            SQLiteIdentityStore.insert_workspace(c, ws)
            member = MembershipRow(
                "mem-" + secrets.token_hex(10), ws_id, user.user_id, Role.OWNER,
                user.user_id, self._now(), None, None,
            )  # fmt: skip
            SQLiteIdentityStore.insert_membership(c, member)
            key, plaintext = self._key_row(
                ws_id, user.user_id, key_name, list(SCOPES), user.user_id, None, key_plaintext
            )
            SQLiteIdentityStore.insert_key(c, key)
            return ws, member, key, plaintext

        with self._identity_lock:
            out = self.identity.write(create)
        self.tenant(ws_id)  # bind the partition now
        return out

    def bootstrap(
        self,
        owner_email: str,
        workspace_name: str = "Default workspace",
        *,
        workspace_id: str = "ws-default",
        key_plaintext: str | None = None,
    ) -> tuple[str, str | None]:
        """Idempotent dev / first-run bootstrap: the workspace (created once) and, when
        ``key_plaintext`` is given, that exact key (registered once, hash only). Returns the
        workspace id and the plaintext of a key created NOW (``None`` if nothing new)."""
        if self.identity.workspace(workspace_id) is None:
            ws, _, _, plaintext = self.create_workspace(
                workspace_name,
                owner_email,
                workspace_id=workspace_id,
                key_name="bootstrap",
                key_plaintext=key_plaintext,
            )
            return ws.workspace_id, plaintext
        if key_plaintext is None:
            return workspace_id, None
        parsed = parse_key(key_plaintext)
        if parsed is None:
            raise ValueError(f"an API key looks like {KEY_PREFIX}<16 hex>_<43 chars>")
        existing = self.identity.key(parsed[0])
        if existing is not None:
            if existing.workspace_id != workspace_id or not hmac.compare_digest(
                existing.secret_hash, hash_secret(*parsed)
            ):
                raise IdentityConflict("that bootstrap key id is already taken")
            return workspace_id, None
        owner = self.identity.user_by_email(owner_email)
        if owner is None or self.identity.membership(workspace_id, owner.user_id) is None:
            raise IdentityConflict(f"{owner_email} is not a member of {workspace_id}")
        key, plaintext = self._key_row(
            workspace_id, owner.user_id, "bootstrap", list(SCOPES), owner.user_id, None,
            key_plaintext,
        )  # fmt: skip
        self.identity.write(lambda c: SQLiteIdentityStore.insert_key(c, key))
        return workspace_id, plaintext

    def issue_key(
        self,
        workspace_id: str,
        user_id: str,
        name: str,
        scopes: list[str],
        *,
        created_by: str,
        expires_in_s: int | None = None,
    ) -> tuple[KeyRow, str]:
        """A new key of ``user_id`` (an active member). An admin scope needs an OWNER."""
        bad = sorted(set(scopes) - set(SCOPES))
        if bad or not scopes:
            raise ApiFailure("invalid_scopes", f"unknown scopes {bad}" if bad else "no scopes")
        member = self.identity.membership(workspace_id, user_id)
        if member is None:
            raise ApiFailure("member_not_found", f"no member {user_id}")
        if "admin" in scopes and member.role is not Role.OWNER:
            raise ApiFailure("invalid_scopes", "only an OWNER can hold the admin scope")
        with self._identity_lock:
            active = sum(1 for k in self.identity.keys(workspace_id) if self._active(k))
            if active + 1 > self.quotas.max_api_keys:
                raise ApiFailure(
                    "quota_exceeded",
                    "this workspace has its maximum number of active API keys",
                    resource="api_keys",
                    limit=self.quotas.max_api_keys,
                    used=active,
                )
            key, plaintext = self._key_row(
                workspace_id, user_id, name, scopes, created_by, expires_in_s
            )
            self.identity.write(lambda c: SQLiteIdentityStore.insert_key(c, key))
        return key, plaintext

    def _active(self, key: KeyRow) -> bool:
        if key.revoked_at is not None:
            return False
        return key.expires_at is None or _when(key.expires_at) > self.now().astimezone(UTC)

    # -- authentication ---------------------------------------------------------------------
    def authenticate(self, headers: Mapping[str, str]) -> Principal:
        """``Authorization: Bearer <key>`` -> the principal, or ``unauthenticated``. Every
        failure looks the same: an unknown, malformed, wrong, revoked or expired key, a removed
        member and a missing workspace are indistinguishable to the caller."""
        values = [v for k, v in headers.items() if k.lower() == "authorization"]
        denied = ApiFailure("unauthenticated", "a valid API key is required")
        if len(values) != 1:
            raise denied
        scheme, _, token = values[0].strip().partition(" ")
        if scheme.lower() != "bearer":
            raise denied
        parsed = parse_key(token.strip())
        if parsed is None:
            raise denied
        key_id, secret = parsed
        key = self.identity.key(key_id)
        expected = key.secret_hash if key is not None else hash_secret("0" * 16, "x" * 43)
        if not hmac.compare_digest(expected, hash_secret(key_id, secret)) or key is None:
            raise denied
        if not self._active(key):
            raise denied
        if self.identity.workspace(key.workspace_id) is None:
            raise denied
        member = self.identity.membership(key.workspace_id, key.user_id)
        if member is None:
            raise denied
        scopes = frozenset(key.scopes)
        if member.role is not Role.OWNER:
            scopes -= {"admin"}  # admin is an OWNER's scope only
        now = self._now()
        if key.last_used_at is None or key.last_used_at[:16] != now[:16]:  # once a minute
            try:
                self.identity.touch_key(key.key_id, now)
            except sqlite3.Error:  # bookkeeping only; never blocks an authenticated request
                log.warning("could not record use of key %s", key.key_id)
        return Principal(key.workspace_id, key.user_id, key.key_id, member.role, scopes)

    # -- dispatch ---------------------------------------------------------------------------
    def handle(
        self, method: str, target: str, headers: Mapping[str, str], rfile: BinaryIO
    ) -> ApiResponse:
        try:
            return self._handle(method, target, headers, rfile)
        except ApiFailure as exc:
            return _error(exc.code, exc.message, exc.details)
        except (TenantIntegrityError, TenantBindingError):
            log.exception("tenant integrity failure on %s %s", method, urlsplit(target).path)
            return _error("tenant_integrity_error", "this workspace's stores are inconsistent")
        except Exception:
            log.exception("unhandled error on %s %s", method, urlsplit(target).path)
            return _error("internal_error", "internal error")

    def _handle(
        self, method: str, target: str, headers: Mapping[str, str], rfile: BinaryIO
    ) -> ApiResponse:
        parts = urlsplit(target)
        if not parts.path.startswith(API_PREFIX):
            raise ApiFailure("not_found", "no such endpoint")
        path = parts.path[len(API_PREFIX) :]
        if path == "health":  # the only public endpoint: no tenant data
            if method != "GET":
                raise ApiFailure("method_not_allowed", "use GET")
            return _ok(Health(status="ok"))
        principal = self.authenticate(headers)  # before any resource is touched
        request = _Request(method, [], {k.lower(): v for k, v in headers.items()}, rfile)
        admin_methods = []
        for route_method, pattern, name in _ADMIN_ROUTES:
            m = pattern.match(path)
            if m and route_method != method:
                admin_methods.append(route_method)
            if m and route_method == method:
                self._require(principal, _ADMIN_SCOPE[name])
                if parts.query:
                    raise ApiFailure("unknown_parameter", "this endpoint takes no parameters")
                return getattr(self, f"_{name}")(principal, request, m.groups())
        if admin_methods:
            raise ApiFailure("method_not_allowed", f"use {' or '.join(sorted(admin_methods))}")
        name = _product_route(method, path)
        tenant = self.tenant(principal.workspace_id)
        if name is None:  # unknown endpoint / wrong method: the product API's own answer
            return tenant.api.handle(method, target, headers, rfile, principal.record())
        self._require(principal, route_scope(name, method))
        counters = QUOTA_ROUTES.get(name)
        if counters is None:
            return tenant.api.handle(method, target, headers, rfile, principal.record())
        with tenant.lock:  # check + create, serialized per workspace
            self._check_quota(tenant, counters, headers)
            return tenant.api.handle(method, target, headers, rfile, principal.record())

    @staticmethod
    def _require(principal: Principal, scope: str) -> None:
        if scope not in principal.scopes:
            raise ApiFailure(
                "insufficient_scope", f"this API key lacks the {scope!r} scope", scope=scope
            )

    def _check_quota(
        self, tenant: Tenant, counters: tuple[str, ...], headers: Mapping[str, str]
    ) -> None:
        usage = tenant.usage()
        for counter in counters:
            limit = getattr(self.quotas, _LIMIT[counter])
            used = usage[counter]
            grow = 1
            if counter == "stored_bytes":
                raw = {k.lower(): v for k, v in headers.items()}.get("content-length", "0")
                grow = int(raw) if raw.strip().isdigit() else 0
            if used + grow > limit:
                raise ApiFailure(
                    "quota_exceeded",
                    f"this workspace's {counter} quota is exhausted",
                    resource=counter,
                    limit=limit,
                    used=used,
                )

    # -- admin endpoints --------------------------------------------------------------------
    def _member_view(self, m: MembershipRow) -> MemberView:
        user = self.identity.user(m.user_id)
        if user is None:
            raise TenantIntegrityError(f"membership {m.membership_id} names no user")
        return MemberView(
            membership_id=m.membership_id,
            user_id=m.user_id,
            email=user.email,
            display_name=user.display_name,
            role=m.role.value,
            created_by=m.created_by,
            created_at=m.created_at,
        )

    def _key_view(self, k: KeyRow) -> KeyView:
        return KeyView(
            key_id=k.key_id,
            prefix=f"{KEY_PREFIX}{k.key_id}_",
            workspace_id=k.workspace_id,
            user_id=k.user_id,
            name=k.name,
            scopes=list(k.scopes),
            created_by=k.created_by,
            created_at=k.created_at,
            expires_at=k.expires_at,
            revoked_at=k.revoked_at,
            revoked_by=k.revoked_by,
            last_used_at=k.last_used_at,
            active=self._active(k),
        )

    def _workspace_view(self, workspace_id: str) -> WorkspaceView:
        ws = self.identity.workspace(workspace_id)
        if ws is None:
            raise ApiFailure("workspace_not_found", f"no workspace {workspace_id}")
        return WorkspaceView(**ws.__dict__)

    def _create_workspace(self, p: Principal, request: _Request, _: tuple[str, ...]) -> ApiResponse:
        body: NewWorkspace = _PARSER._json(request, NewWorkspace)
        owner = self.identity.user(p.user_id)
        assert owner is not None
        ws, member, key, plaintext = self.create_workspace(body.name, owner.email)
        return _ok(
            CreatedWorkspace(
                workspace=WorkspaceView(**ws.__dict__),
                membership=self._member_view(member),
                api_key=CreatedKey(key=self._key_view(key), secret=plaintext),
            ),
            created=True,
        )

    def _current_workspace(
        self, p: Principal, request: _Request, _: tuple[str, ...]
    ) -> ApiResponse:
        return _ok(self._current(p))

    def _get_workspace(self, p: Principal, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        if args[0] != p.workspace_id:  # another workspace: exactly as if it did not exist
            raise ApiFailure("workspace_not_found", f"no workspace {args[0]}")
        return _ok(self._current(p))

    def _current(self, p: Principal) -> CurrentWorkspace:
        return CurrentWorkspace(
            workspace=self._workspace_view(p.workspace_id),
            principal=PrincipalView(
                user_id=p.user_id,
                key_id=p.key_id,
                role=p.role.value,
                scopes=[s for s in SCOPES if s in p.scopes],
            ),
        )

    def _list_members(self, p: Principal, request: _Request, _: tuple[str, ...]) -> ApiResponse:
        return _ok(
            MemberList(
                members=[self._member_view(m) for m in self.identity.memberships(p.workspace_id)]
            )
        )

    def _add_member(self, p: Principal, request: _Request, _: tuple[str, ...]) -> ApiResponse:
        body: NewMember = _PARSER._json(request, NewMember)

        def add(c: sqlite3.Connection) -> MembershipRow:
            user = self._user(c, body.email, body.display_name)
            m = MembershipRow(
                "mem-" + secrets.token_hex(10), p.workspace_id, user.user_id, body.role,
                p.user_id, self._now(), None, None,
            )  # fmt: skip
            SQLiteIdentityStore.insert_membership(c, m)
            return m

        try:
            with self._identity_lock:
                member = self.identity.write(add)
        except IdentityConflict:
            raise ApiFailure("member_conflict", f"{body.email} is already a member") from None
        return _ok(self._member_view(member), created=True)

    def _remove_member(self, p: Principal, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        try:
            removed = self.identity.remove_membership(p.workspace_id, args[0], p.user_id)
        except IdentityConflict as exc:
            raise ApiFailure("member_conflict", str(exc)) from None
        if not removed:
            raise ApiFailure("member_not_found", f"no member {args[0]}")
        return _ok(RemovedMember(user_id=args[0], removed=True))

    def _create_key(self, p: Principal, request: _Request, _: tuple[str, ...]) -> ApiResponse:
        body: NewKey = _PARSER._json(request, NewKey)
        if not set(body.scopes) <= p.scopes:  # never mint more than the caller holds
            raise ApiFailure("invalid_scopes", "a new key cannot exceed the caller's scopes")
        key, plaintext = self.issue_key(
            p.workspace_id,
            body.user_id or p.user_id,
            body.name,
            body.scopes,
            created_by=p.user_id,
            expires_in_s=body.expires_in_s,
        )
        return _ok(CreatedKey(key=self._key_view(key), secret=plaintext), created=True)

    def _list_keys(self, p: Principal, request: _Request, _: tuple[str, ...]) -> ApiResponse:
        return _ok(KeyList(keys=[self._key_view(k) for k in self.identity.keys(p.workspace_id)]))

    def _revoke_key(self, p: Principal, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        key = self.identity.revoke_key(p.workspace_id, args[0], p.user_id)
        if key is None:  # unknown here (or another workspace's): the same answer
            raise ApiFailure("api_key_not_found", f"no API key {args[0]}")
        return _ok(self._key_view(key))

    def _get_quota(self, p: Principal, request: _Request, _: tuple[str, ...]) -> ApiResponse:
        usage = self.tenant(p.workspace_id).usage()
        usage["api_keys"] = sum(1 for k in self.identity.keys(p.workspace_id) if self._active(k))
        limits = {c: getattr(self.quotas, _LIMIT[c]) for c in _LIMIT}
        return _ok(QuotaView(workspace_id=p.workspace_id, limits=limits, usage=usage))


_PARSER = ProductAPI.__new__(ProductAPI)  # its stateless body / JSON parsing helpers

_ADMIN_ROUTES: list[tuple[str, re.Pattern[str], str]] = [
    ("POST", re.compile(r"^workspaces$"), "create_workspace"),
    ("GET", re.compile(r"^workspace$"), "current_workspace"),
    ("GET", re.compile(r"^workspaces/(ws-[0-9a-z][0-9a-z-]{1,62})$"), "get_workspace"),
    ("GET", re.compile(r"^workspace/members$"), "list_members"),
    ("POST", re.compile(r"^workspace/members$"), "add_member"),
    ("POST", re.compile(r"^workspace/members/(usr-[0-9a-f]{20})/remove$"), "remove_member"),
    ("POST", re.compile(r"^workspace/keys$"), "create_key"),
    ("GET", re.compile(r"^workspace/keys$"), "list_keys"),
    ("POST", re.compile(r"^workspace/keys/([0-9a-f]{16})/revoke$"), "revoke_key"),
    ("GET", re.compile(r"^workspace/quota$"), "get_quota"),
]
_ADMIN_SCOPE = {
    "create_workspace": "admin",
    "current_workspace": "read",
    "get_workspace": "read",
    "list_members": "read",
    "add_member": "admin",
    "remove_member": "admin",
    "create_key": "admin",
    "list_keys": "admin",
    "revoke_key": "admin",
    "get_quota": "read",
}


def _product_route(method: str, path: str) -> str | None:
    for route_method, pattern, name in product._ROUTES:
        if route_method == method and pattern.match(path):
            return name
    return None


def _when(stamp: str) -> datetime:
    value = datetime.fromisoformat(stamp)
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _start(api: ProductAPI) -> JobWorker | None:
    from api.product import start_worker

    return start_worker(api)


# -- legacy single-tenant data --------------------------------------------------------------
def legacy_items(root: Path) -> list[str]:
    """Store files / directories of the pre-#32 layout still at the data-dir root."""
    return [n for n in (*STORE_FILES, *STORE_DIRS) if (root / n).exists()]


def migrate_legacy(
    data_dir: Path | str,
    owner_email: str,
    *,
    workspace_id: str = DEFAULT_LEGACY_WORKSPACE,
    workspace_name: str = "Legacy workspace",
) -> dict[str, Any]:
    """Assign a pre-#32 single-tenant data dir to ONE explicitly named workspace.

    Nothing is guessed: the operator names the workspace id and its OWNER. Every legacy store
    moves into ``workspaces/<workspace_id>/`` unchanged (WAL checkpointed first, files renamed,
    never rewritten) and is then bound to the workspace. Idempotent and resumable: a re-run
    after completion changes nothing; a re-run after a crash finishes the remaining steps; a
    run naming a different workspace than the recorded one, or finding a store both at the root
    and in the partition, refuses. No API key is issued (``issue-key`` does that, once)."""
    root = Path(data_dir)
    check_workspace_id(workspace_id)
    partition = root / "workspaces" / workspace_id
    identity = SQLiteIdentityStore(root / "identity.sqlite3")
    recorded = identity.migration(LEGACY_SOURCE)
    if recorded is not None and recorded.workspace_id != workspace_id:
        raise IdentityConflict(
            f"this data dir was already assigned to {recorded.workspace_id}, not {workspace_id}"
        )
    if recorded is not None and recorded.completed_at is not None:
        if legacy_items(root):
            raise IdentityConflict("legacy stores reappeared after a completed migration")
        return {"status": "already_migrated", "workspace_id": workspace_id, "moved": []}
    if recorded is None:
        if not legacy_items(root):
            raise IdentityConflict(f"{root} holds no single-tenant data to migrate")
        if identity.workspace(workspace_id) is not None:
            raise IdentityConflict(f"{workspace_id} already exists; name a new workspace")
        now = utc()

        def start(c: sqlite3.Connection) -> None:
            r = c.execute("SELECT user_id FROM users WHERE email=?", (owner_email,)).fetchone()
            if r is None:
                user_id = "usr-" + secrets.token_hex(10)
                SQLiteIdentityStore.insert_user(c, UserRow(user_id, owner_email, owner_email, now))
            else:
                user_id = r[0]
            SQLiteIdentityStore.insert_workspace(
                c, WorkspaceRow(workspace_id, workspace_name, user_id, now)
            )
            SQLiteIdentityStore.insert_membership(
                c,
                MembershipRow(
                    "mem-" + secrets.token_hex(10), workspace_id, user_id, Role.OWNER, user_id,
                    now, None, None,
                ),
            )  # fmt: skip
            SQLiteIdentityStore.insert_migration(
                c, MigrationRow(LEGACY_SOURCE, workspace_id, user_id, now, None)
            )

        identity.write(start)
    partition.mkdir(parents=True, exist_ok=True)
    moved = []
    for name in (*STORE_FILES, *STORE_DIRS):
        src, dst = root / name, partition / name
        if src.exists() and dst.exists():
            raise IdentityConflict(f"{name} exists both at the root and in {workspace_id}")
        if not src.exists():
            continue
        if name in STORE_FILES:
            conn = sqlite3.connect(src, timeout=30)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                conn.close()
            for suffix in ("-wal", "-shm"):
                side = root / (name + suffix)
                if side.exists() and side.stat().st_size == 0:
                    side.unlink()
                elif side.exists():
                    side.rename(partition / side.name)
        src.rename(dst)
        moved.append(name)
    for name in STORE_FILES:
        if (partition / name).exists():
            bind_sqlite(partition / name, workspace_id)
    bind_directory(partition / "blobs", workspace_id)
    identity.complete_migration(LEGACY_SOURCE)
    return {"status": "migrated", "workspace_id": workspace_id, "moved": moved}


def utc() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


# -- CLI ------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m api.tenancy", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("migrate-legacy", help="assign single-tenant data to one workspace")
    m.add_argument("--data-dir", type=Path, required=True)
    m.add_argument("--owner-email", required=True)
    m.add_argument("--workspace-id", default=DEFAULT_LEGACY_WORKSPACE)
    m.add_argument("--workspace-name", default="Legacy workspace")
    b = sub.add_parser("bootstrap", help="create a workspace + owner key (prints the key once)")
    b.add_argument("--data-dir", type=Path, required=True)
    b.add_argument("--owner-email", required=True)
    b.add_argument("--workspace-id", default="ws-default")
    b.add_argument("--workspace-name", default="Default workspace")
    k = sub.add_parser("issue-key", help="a new key for a member (prints it once)")
    k.add_argument("--data-dir", type=Path, required=True)
    k.add_argument("--workspace-id", required=True)
    k.add_argument("--email", required=True)
    k.add_argument("--name", default="cli")
    k.add_argument("--scopes", default="read,write,invoke")
    args = p.parse_args(argv)
    if args.command == "migrate-legacy":
        out = migrate_legacy(
            args.data_dir,
            args.owner_email,
            workspace_id=args.workspace_id,
            workspace_name=args.workspace_name,
        )
        print(json.dumps(out, indent=1))
        return 0
    router = TenantRouter(args.data_dir)
    if args.command == "bootstrap":
        ws, plaintext = router.bootstrap(
            args.owner_email, args.workspace_name, workspace_id=args.workspace_id
        )
        print(json.dumps({"workspace_id": ws, "api_key": plaintext}, indent=1))
        return 0
    user = router.identity.user_by_email(args.email)
    if user is None:
        print(json.dumps({"error": "no such user"}))
        return 1
    key, plaintext = router.issue_key(
        args.workspace_id,
        user.user_id,
        args.name,
        [s for s in args.scopes.split(",") if s],
        created_by=user.user_id,
    )
    print(json.dumps({"key_id": key.key_id, "api_key": plaintext}, indent=1))
    return 0


__all__ = [
    "KEY_PREFIX",
    "SCOPES",
    "LegacyDataNotMigrated",
    "Principal",
    "Quotas",
    "Tenant",
    "TenantIntegrityError",
    "TenantRouter",
    "hash_secret",
    "migrate_legacy",
    "new_key",
    "parse_key",
    "route_scope",
]


if __name__ == "__main__":
    raise SystemExit(main())
