"""Product API v1: projects, dataset upload / inspection / registration, deterministic splits.

    python -m api.product --data-dir .wynk-data      # serves http://127.0.0.1:8788/api/v1/...

The chat server (``api.chat``) mounts the same routes, so the UI's ``/api`` proxy reaches them.
Stdlib ``http.server`` like the chat endpoint: no framework. ``ProductAPI.handle`` is a pure
request -> response function; the HTTP adapter only moves bytes.

    POST /api/v1/projects                                         {name, description?}
    GET  /api/v1/projects
    GET  /api/v1/projects/{project_id}
    POST /api/v1/projects/{project_id}/uploads?format=csv|jsonl&filename=...   raw file bytes
    GET  /api/v1/uploads/{upload_id}
    POST /api/v1/uploads/{upload_id}/register                     RegisterDataset
    GET  /api/v1/projects/{project_id}/datasets
    GET  /api/v1/datasets/{dataset_id}
    GET  /api/v1/datasets/{dataset_id}/versions/{version}
    POST /api/v1/datasets/{dataset_id}/versions/{version}/splits  SplitPlan
    GET  /api/v1/datasets/{dataset_id}/versions/{version}/splits
    GET  /api/v1/datasets/{dataset_id}/versions/{version}/splits/{splits_hash}
    POST /api/v1/experiments                                      CreateExperiment
    GET  /api/v1/experiments
    GET  /api/v1/experiments/{job_id}
    POST /api/v1/experiments/{job_id}/cancel
    POST /api/v1/experiments/{job_id}/resume
    GET  /api/v1/experiments/{job_id}/artifact
    GET  /api/v1/experiments/{job_id}/provenance                  canonical provenance record
    GET  /api/v1/experiments/{job_id}/trace?path=...               where a number comes from
    GET  /api/v1/experiments/{job_id}/verify                       artifact integrity + evidence
    GET  /api/v1/experiments/{job_id}/reproduce                    replay from stored evidence
    POST /api/v1/experiments/{job_id}/promote?lineage=...          held-out champion promotion
    GET  /api/v1/experiments/{job_id}/promotion
    GET  /api/v1/promotions?lineage=...
    GET  /api/v1/promotions/{promotion_id}
    GET  /api/v1/promotions/{promotion_id}/verify
    GET  /api/v1/champions/{lineage}
    GET  /api/v1/champions/{lineage}/history
    POST /api/v1/workflows                                        PublishWorkflow {champion_id}
    GET  /api/v1/workflows?lineage=...
    GET  /api/v1/workflows/{version_id}                           immutable workflow version
    POST /api/v1/workflows/{version_id}/stage                     DeploymentChange
    POST /api/v1/workflows/{version_id}/promote                   DeploymentChange
    POST /api/v1/workflows/{version_id}/invoke                    InvokeWorkflow {inputs}
    GET  /api/v1/deployments/{lineage}
    GET  /api/v1/deployments/{lineage}/history
    POST /api/v1/deployments/{lineage}/rollback                   RollbackDeployment
    POST /api/v1/deployments/{lineage}/invoke                     InvokeWorkflow (production)
    GET  /api/v1/inferences/{inference_id}
    POST /api/v1/inferences/{inference_id}/feedback                FeedbackRequest
    GET  /api/v1/feedback/{feedback_id}
    POST /api/v1/monitoring/policies                              MonitoringPolicy
    GET  /api/v1/monitoring/policies/{policy_id}
    GET  /api/v1/workflows/{version_id}/monitoring?since=&until=&policy_id=
    GET  /api/v1/workflows/{version_id}/drift?since=&until=&policy_id=
    POST /api/v1/workflows/{version_id}/triggers/evaluate         EvaluateTrigger
    GET  /api/v1/workflows/{version_id}/triggers                  trigger history + evidence
    GET  /api/v1/triggers/{trigger_id}
    GET  /api/v1/triggers/{trigger_id}/challenger                 new dataset version + job
    POST /api/v1/triggers/{trigger_id}/reoptimize                 resume a TRIGGERED decision

Experiments are durable jobs (``experiments.jobs``, stored in ``jobs.sqlite3`` next to the dataset
metadata) over a registered dataset version and its stored splits. A create returns the job
(201) once it is persisted; a worker thread in this process executes it, and a restarted
server recovers and resumes it. Without a model registry (``--model-registry``) jobs stay
inspectable and cancellable, but creating one answers ``experiment_backend_unavailable``.

Champion promotion (``experiments.promotion``, stored in ``champions.sqlite3``) takes a COMPLETED
experiment, selects ONE challenger on validation, opens the test split once for it (and the
lineage's incumbent), and promotes or rejects it with an immutable decision. The promote request
runs the held-out evaluation before it answers.

Champion deployment (``experiments.deployment``, stored in ``deployments.sqlite3``) publishes a
champion as an immutable workflow version pinned to its #26 provenance, stages it, promotes it to
production (explicit, fenced by ``expected_revision``, atomic) and rolls back to a previous
production version. ``invoke`` validates the request against the pinned TaskContract input schema,
binds exactly the pinned model (``--model-registry``; fail closed otherwise), runs the frozen
genome through the ``WorkflowRunner`` and validates the answer against the pinned output schema.

Every COMPLETED experiment has one canonical artifact (``experiments.provenance``): its result
body plus an immutable ProvenanceRecord, hashed and verified on every read (tampering answers
``artifact_integrity_error``). ``trace`` maps any field path - e.g.
``strategy.aco.seed.1.usage.tokens`` - to the artifact id, JSON pointer, provenance and, for a
derived number, its sources. ``reproduce`` replays every strategy run from the stored
write-ahead evidence without a model call (``experiments.artifacts``).

Every response body is a strict pydantic model dump. Errors are ``{"error": {"code", "message",
"details"}}`` with a stable ``code`` (``ERROR_STATUS``). A create returns 201, or 200 when the
identical resource already existed. Request bodies are validated in strict mode with unknown
fields refused, so a client cannot supply a hash, row count, type or row id.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, ValidationError

from core.dataset import SplitPlan
from core.task_contract import TaskContract
from experiments.artifacts import ExperimentArtifacts
from experiments.deployment import ChampionDeployments, DeploymentError, InferenceRuntime
from experiments.jobs import (
    DatasetRuntime,
    ExperimentJobDefinition,
    ExperimentJobs,
    JobError,
    JobRuntime,
    JobWorker,
    WorkflowBackend,
)
from experiments.monitoring import (
    FeedbackRequest,
    MonitoringError,
    MonitoringPolicy,
    ProductionMonitor,
)
from experiments.optimization_experiment import ExperimentPlan
from experiments.promotion import ChampionPromotions, PromotionError
from ingestion.parse import IngestError
from ingestion.service import (
    DatasetService,
    NewProject,
    NotFound,
    RegisterDataset,
    StorageFailure,
)
from store.blobs import BlobError
from store.datasets import (
    DatasetVersionRecord,
    ProjectRecord,
    RepositoryError,
    SplitsRecord,
)
from store.tenancy import QuotaExceeded

log = logging.getLogger("wynk.api.product")

API_PREFIX = "/api/v1/"
MAX_JSON_BYTES = 1024 * 1024
DEFAULT_DATA_DIR = ".wynk-data"
LINGER_SECONDS = 5.0  # how long a refused body is drained after the response (see ``_linger``)

# Stable error codes -> HTTP status. Codes are part of the API; never rename one.
ERROR_STATUS: dict[str, int] = {
    "invalid_request": 400,
    "invalid_json": 400,
    "unknown_parameter": 400,
    "not_found": 404,
    "project_not_found": 404,
    "upload_not_found": 404,
    "dataset_not_found": 404,
    "dataset_version_not_found": 404,
    "splits_not_found": 404,
    "method_not_allowed": 405,
    "dataset_conflict": 409,
    "length_required": 411,
    "payload_too_large": 413,
    "unsupported_media_type": 415,
    "unsupported_format": 422,
    "empty_file": 422,
    "invalid_encoding": 422,
    "malformed_csv": 422,
    "malformed_jsonl": 422,
    "no_rows": 422,
    "too_many_columns": 422,
    "invalid_column_name": 422,
    "duplicate_column": 422,
    "unknown_column": 422,
    "json_column_role": 422,
    "invalid_mapping": 422,
    "invalid_id_column": 422,
    "duplicate_row_id": 422,
    "invalid_split_plan": 422,
    "job_not_found": 404,
    "invalid_experiment": 422,
    "job_not_cancellable": 409,
    "job_not_resumable": 409,
    "ambiguous_attempt": 409,
    "job_not_completed": 409,
    "experiment_backend_unavailable": 503,
    "experiment_not_promotable": 409,
    "incompatible_incumbent": 409,
    "promotion_in_progress": 409,
    "promotion_ambiguous_attempt": 409,
    "promotion_failed": 409,
    "promotion_not_found": 404,
    "champion_not_found": 404,
    "heldout_backend_unavailable": 503,
    "promotion_evidence_mismatch": 500,
    "artifact_not_finalized": 409,
    "trace_path_not_found": 404,
    "artifact_integrity_error": 500,
    "reproduction_mismatch": 500,
    "promotion_error": 500,
    "workflow_version_not_found": 404,
    "deployment_not_found": 404,
    "inference_not_found": 404,
    "not_publishable": 409,
    "invalid_deployment_transition": 409,
    "stale_deployment": 409,
    "champion_not_current": 409,
    "workflow_version_not_deployed": 409,
    "no_production_version": 409,
    "invalid_inference_request": 422,
    "model_binding_failed": 503,
    "inference_backend_unavailable": 503,
    "inference_failed": 502,
    "output_schema_violation": 502,
    "workflow_version_integrity_error": 500,
    "invalid_feedback": 422,
    "feedback_mismatch": 422,
    "feedback_conflict": 409,
    "feedback_not_found": 404,
    "monitoring_policy_not_found": 404,
    "invalid_monitoring_window": 422,
    "trigger_not_found": 404,
    "trigger_not_triggered": 409,
    "reoptimization_unavailable": 503,
    "monitoring_integrity_error": 500,
    "monitoring_error": 500,
    "deployment_error": 500,
    "storage_error": 500,
    "internal_error": 500,
    # -- #32 authentication / workspaces / tenancy --
    "unauthenticated": 401,
    "insufficient_scope": 403,
    "workspace_not_found": 404,
    "member_not_found": 404,
    "api_key_not_found": 404,
    "member_conflict": 409,
    "invalid_scopes": 422,
    "quota_exceeded": 429,
    "tenant_integrity_error": 500,
}


# -- response schemas ---------------------------------------------------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ErrorBody(_Strict):
    code: str
    message: str
    details: dict[str, str | int | None] = {}


class ErrorResponse(_Strict):
    error: ErrorBody


class ProjectList(_Strict):
    projects: list[ProjectRecord]


class DatasetView(_Strict):
    dataset_id: str
    project_id: str
    name: str  # the latest version's display name
    latest_version: int
    versions: list[DatasetVersionRecord]


class DatasetList(_Strict):
    datasets: list[DatasetView]


class SplitsList(_Strict):
    splits: list[SplitsRecord]


class CreateExperiment(_Strict):
    """A durable experiment over a registered dataset version and its stored splits. The
    contract's dataset must be exactly that version; the server decides whether the model is
    a stand-in (``synthetic``), never the client."""

    dataset_id: str
    dataset_version: PositiveInt
    splits_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract: TaskContract
    plan: ExperimentPlan
    workers: PositiveInt = Field(default=1, le=16)


Actor = Field(default="api", pattern=r"^[A-Za-z0-9_.@-]{1,64}$")  # recorded, not authenticated


class PublishWorkflow(_Strict):
    champion_id: str = Field(pattern=r"^c-[0-9a-f]{24}$")
    actor: str = Actor


class DeploymentChange(_Strict):
    """A stage / promote, made against the deployment revision the caller read."""

    expected_revision: NonNegativeInt
    actor: str = Actor


class RollbackDeployment(_Strict):
    """`version_id` omitted: the version the current production version replaced."""

    expected_revision: NonNegativeInt
    version_id: str | None = Field(default=None, pattern=r"^wv-[0-9a-f]{24}$")
    actor: str = Actor


class InvokeWorkflow(_Strict):
    inputs: dict[str, Any]


class EvaluateTrigger(_Strict):
    """A manual trigger evaluation. Window bounds default to the policy's window ending now;
    the decision's identity is its evidence (the exact records), not the bounds."""

    since: str | None = None
    until: str | None = None
    policy_id: str | None = Field(default=None, pattern=r"^mp-[0-9a-f]{24}$")


class PolicyRegistered(_Strict):
    policy_id: str
    policy: MonitoringPolicy


def _dataset_view(versions: list[DatasetVersionRecord]) -> DatasetView:
    latest = versions[-1]
    return DatasetView(
        dataset_id=latest.dataset_id,
        project_id=latest.project_id,
        name=latest.spec.name,
        latest_version=latest.dataset_version,
        versions=versions,
    )


@dataclass(frozen=True)
class ApiResponse:
    status: int
    body: dict[str, Any]


class ApiFailure(Exception):
    def __init__(self, code: str, message: str, **details: str | int | None) -> None:
        super().__init__(message)
        self.code, self.message, self.details = code, message, details


def _error(
    code: str, message: str, details: Mapping[str, str | int | None] | None = None
) -> ApiResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=dict(details or {})))
    return ApiResponse(ERROR_STATUS[code], body.model_dump(mode="json"))


def _ok(model: BaseModel, created: bool | None = None) -> ApiResponse:
    return ApiResponse(201 if created else 200, model.model_dump(mode="json"))


# -- routing ------------------------------------------------------------------------------------
_ID = r"([a-z0-9][a-z0-9_.-]{0,127})"
_VERSION = r"([1-9][0-9]{0,8})"
_HASH = r"([0-9a-f]{64})"
_LINEAGE = r"([a-z0-9][a-z0-9_.-]{0,255})"
_WORKFLOW = r"(wv-[0-9a-f]{24})"
_INFERENCE = r"(inf-[0-9a-f]{24})"
_FEEDBACK = r"(fb-[0-9a-f]{24})"
_TRIGGER = r"(tr-[0-9a-f]{24})"
_POLICY = r"(mp-[0-9a-f]{24})"


@dataclass(frozen=True)
class _Request:
    method: str
    query: list[tuple[str, str]]
    headers: Mapping[str, str]  # lower-cased names
    rfile: BinaryIO
    principal: Mapping[str, Any] | None = None  # #32: set by the tenant router, never a header


_ROUTES: list[tuple[str, re.Pattern[str], str]] = [
    ("POST", re.compile(r"^projects$"), "create_project"),
    ("GET", re.compile(r"^projects$"), "list_projects"),
    ("GET", re.compile(rf"^projects/{_ID}$"), "get_project"),
    ("POST", re.compile(rf"^projects/{_ID}/uploads$"), "upload"),
    ("GET", re.compile(rf"^projects/{_ID}/datasets$"), "list_datasets"),
    ("GET", re.compile(rf"^uploads/{_ID}$"), "get_upload"),
    ("POST", re.compile(rf"^uploads/{_ID}/register$"), "register"),
    ("GET", re.compile(rf"^datasets/{_ID}$"), "get_dataset"),
    ("GET", re.compile(rf"^datasets/{_ID}/versions/{_VERSION}$"), "get_version"),
    ("POST", re.compile(rf"^datasets/{_ID}/versions/{_VERSION}/splits$"), "create_splits"),
    ("GET", re.compile(rf"^datasets/{_ID}/versions/{_VERSION}/splits$"), "list_splits"),
    ("GET", re.compile(rf"^datasets/{_ID}/versions/{_VERSION}/splits/{_HASH}$"), "get_splits"),
    ("POST", re.compile(r"^experiments$"), "create_experiment"),
    ("GET", re.compile(r"^experiments$"), "list_experiments"),
    ("GET", re.compile(rf"^experiments/{_ID}$"), "get_experiment"),
    ("POST", re.compile(rf"^experiments/{_ID}/cancel$"), "cancel_experiment"),
    ("POST", re.compile(rf"^experiments/{_ID}/resume$"), "resume_experiment"),
    ("GET", re.compile(rf"^experiments/{_ID}/artifact$"), "get_artifact"),
    ("GET", re.compile(rf"^experiments/{_ID}/provenance$"), "get_provenance"),
    ("GET", re.compile(rf"^experiments/{_ID}/trace$"), "trace_field"),
    ("GET", re.compile(rf"^experiments/{_ID}/verify$"), "verify_artifact"),
    ("GET", re.compile(rf"^experiments/{_ID}/reproduce$"), "reproduce_experiment"),
    ("POST", re.compile(rf"^experiments/{_ID}/promote$"), "promote_experiment"),
    ("GET", re.compile(rf"^experiments/{_ID}/promotion$"), "get_experiment_promotion"),
    ("GET", re.compile(r"^promotions$"), "list_promotions"),
    ("GET", re.compile(rf"^promotions/{_ID}$"), "get_promotion"),
    ("GET", re.compile(rf"^promotions/{_ID}/verify$"), "verify_promotion"),
    ("GET", re.compile(rf"^champions/{_LINEAGE}$"), "get_champion"),
    ("GET", re.compile(rf"^champions/{_LINEAGE}/history$"), "get_champion_history"),
    ("POST", re.compile(r"^workflows$"), "publish_workflow"),
    ("GET", re.compile(r"^workflows$"), "list_workflows"),
    ("GET", re.compile(rf"^workflows/{_WORKFLOW}$"), "get_workflow"),
    ("POST", re.compile(rf"^workflows/{_WORKFLOW}/stage$"), "stage_workflow"),
    ("POST", re.compile(rf"^workflows/{_WORKFLOW}/promote$"), "promote_workflow"),
    ("POST", re.compile(rf"^workflows/{_WORKFLOW}/invoke$"), "invoke_workflow"),
    ("GET", re.compile(rf"^deployments/{_LINEAGE}$"), "get_deployment"),
    ("GET", re.compile(rf"^deployments/{_LINEAGE}/history$"), "get_deployment_history"),
    ("POST", re.compile(rf"^deployments/{_LINEAGE}/rollback$"), "rollback_deployment"),
    ("POST", re.compile(rf"^deployments/{_LINEAGE}/invoke$"), "invoke_production"),
    ("GET", re.compile(rf"^inferences/{_INFERENCE}$"), "get_inference"),
    ("POST", re.compile(rf"^inferences/{_INFERENCE}/feedback$"), "submit_feedback"),
    ("GET", re.compile(rf"^feedback/{_FEEDBACK}$"), "get_feedback"),
    ("POST", re.compile(r"^monitoring/policies$"), "register_policy"),
    ("GET", re.compile(rf"^monitoring/policies/{_POLICY}$"), "get_policy"),
    ("GET", re.compile(rf"^workflows/{_WORKFLOW}/monitoring$"), "get_monitoring"),
    ("GET", re.compile(rf"^workflows/{_WORKFLOW}/drift$"), "get_drift"),
    ("POST", re.compile(rf"^workflows/{_WORKFLOW}/triggers/evaluate$"), "evaluate_trigger"),
    ("GET", re.compile(rf"^workflows/{_WORKFLOW}/triggers$"), "list_triggers"),
    ("GET", re.compile(rf"^triggers/{_TRIGGER}$"), "get_trigger"),
    ("GET", re.compile(rf"^triggers/{_TRIGGER}/challenger$"), "get_challenger"),
    ("POST", re.compile(rf"^triggers/{_TRIGGER}/reoptimize$"), "reoptimize"),
]


def is_product_path(target: str) -> bool:
    return urlsplit(target).path.startswith(API_PREFIX)


class ProductAPI:
    def __init__(
        self,
        service: DatasetService,
        jobs: ExperimentJobs | None = None,
        promotions: ChampionPromotions | None = None,
        deployments: ChampionDeployments | None = None,
        monitor: ProductionMonitor | None = None,
    ) -> None:
        self.service = service
        self.jobs = jobs
        self.promotions = promotions
        self.deployments = deployments
        self.monitor = monitor

    def handle(
        self,
        method: str,
        target: str,
        headers: Mapping[str, str],
        rfile: BinaryIO,
        principal: Mapping[str, Any] | None = None,
    ) -> ApiResponse:
        """One request -> one response. Never raises; unexpected failures are 500s.
        ``principal``: the authenticated caller (#32 ``TenantRouter``), recorded on inferences
        and feedback; this object only ever sees one workspace's stores."""
        try:
            return self._dispatch(method, target, headers, rfile, principal)
        except ApiFailure as exc:
            return _error(exc.code, exc.message, exc.details)
        except IngestError as exc:
            code = exc.code if exc.code in ERROR_STATUS else "invalid_request"
            return _error(code, exc.message, exc.details)
        except NotFound as exc:
            return _error(exc.code, exc.message)
        except (JobError, PromotionError) as exc:
            return _error(exc.code, str(exc))
        except DeploymentError as exc:
            return _error(exc.code, str(exc), exc.details)
        except MonitoringError as exc:
            return _error(exc.code, str(exc), exc.details)
        except QuotaExceeded as exc:  # #32: refused inside the store's insert transaction
            return _error("quota_exceeded", str(exc), exc.details())
        except (StorageFailure, RepositoryError, BlobError, OSError):
            log.exception("storage failure on %s %s", method, urlsplit(target).path)
            return _error("storage_error", "stored dataset state is unavailable or inconsistent")
        except Exception:
            log.exception("unhandled error on %s %s", method, urlsplit(target).path)
            return _error("internal_error", "internal error")

    def _dispatch(
        self,
        method: str,
        target: str,
        headers: Mapping[str, str],
        rfile: BinaryIO,
        principal: Mapping[str, Any] | None = None,
    ) -> ApiResponse:
        parts = urlsplit(target)
        if not parts.path.startswith(API_PREFIX):
            raise ApiFailure("not_found", "no such endpoint")
        path = parts.path[len(API_PREFIX) :]
        try:
            query = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=bool(parts.query))
        except ValueError:
            raise ApiFailure("invalid_request", "malformed query string") from None
        request = _Request(
            method, query, {k.lower(): v for k, v in headers.items()}, rfile, principal
        )
        allowed = []
        for route_method, pattern, name in _ROUTES:
            m = pattern.match(path)
            if m:
                if route_method == method:
                    return getattr(self, f"_{name}")(request, m.groups())
                allowed.append(route_method)
        if allowed:
            raise ApiFailure("method_not_allowed", f"use {' or '.join(sorted(allowed))}")
        raise ApiFailure("not_found", "no such endpoint")

    # -- request helpers --------------------------------------------------------------------
    @staticmethod
    def _params(request: _Request, allowed: set[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in request.query:
            if key not in allowed:
                raise ApiFailure("unknown_parameter", f"unknown query parameter {key!r}", name=key)
            if key in out:
                raise ApiFailure("invalid_request", f"query parameter {key!r} is repeated")
            out[key] = value
        return out

    @staticmethod
    def _body(request: _Request, limit: int) -> bytes:
        if "transfer-encoding" in request.headers:
            raise ApiFailure("length_required", "send Content-Length; chunked bodies are refused")
        raw = request.headers.get("content-length")
        if raw is None:
            raise ApiFailure("length_required", "Content-Length is required")
        if not raw.strip().isdigit():
            raise ApiFailure("invalid_request", "Content-Length must be a non-negative integer")
        n = int(raw)
        if n > limit:  # refused before a single body byte is read
            raise ApiFailure(
                "payload_too_large", f"body is larger than {limit} bytes", limit_bytes=limit
            )
        try:
            data = request.rfile.read(n) if n else b""
        except OSError:  # the client went away mid-body
            raise ApiFailure("invalid_request", "the request body could not be read") from None
        if len(data) != n:
            raise ApiFailure("invalid_request", "body is shorter than Content-Length")
        return data

    def _json(self, request: _Request, model: type[BaseModel]) -> Any:
        ctype = request.headers.get("content-type", "")
        if ctype.split(";")[0].strip().lower() != "application/json":
            raise ApiFailure("unsupported_media_type", "send Content-Type: application/json")
        raw = self._body(request, MAX_JSON_BYTES)

        def no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            if len({k for k, _ in pairs}) != len(pairs):
                raise ValueError("duplicate key")
            return dict(pairs)

        try:
            obj = json.loads(raw.decode("utf-8"), object_pairs_hook=no_dupes)
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise ApiFailure("invalid_json", "body is not valid JSON") from None
        if not isinstance(obj, dict):
            raise ApiFailure("invalid_json", "body must be a JSON object")
        try:
            return model.model_validate_json(raw, strict=True)
        except ValidationError as exc:
            err = exc.errors()[0]
            field = ".".join(str(p) for p in err["loc"]) or None
            raise ApiFailure(
                "invalid_request", f"{field + ': ' if field else ''}{err['msg']}", field=field
            ) from None

    # -- projects ---------------------------------------------------------------------------
    def _create_project(self, request: _Request, _: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self.service.create_project(self._json(request, NewProject)), created=True)

    def _list_projects(self, request: _Request, _: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(ProjectList(projects=self.service.list_projects()))

    def _get_project(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self.service.get_project(args[0]))

    # -- uploads ----------------------------------------------------------------------------
    def _upload(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        params = self._params(request, {"format", "filename"})
        if "format" not in params:
            raise ApiFailure("invalid_request", "query parameter 'format' (csv|jsonl) is required")
        ctype = request.headers.get("content-type", "").lower()
        if ctype.startswith("multipart/"):  # form framing would be parsed as file content
            raise ApiFailure(
                "unsupported_media_type", "send the raw file bytes as the body, not a form"
            )
        self.service.get_project(args[0])  # 404 before reading the body
        data = self._body(request, self.service.limits.max_upload_bytes)
        record, created = self.service.upload(
            args[0], data, params["format"], params.get("filename")
        )
        return _ok(record, created)

    def _get_upload(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self.service.get_upload(args[0]))

    def _register(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        body = self._json(request, RegisterDataset)
        record, created = self.service.register(args[0], body)
        return _ok(record, created)

    # -- datasets ---------------------------------------------------------------------------
    def _list_datasets(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        views = [_dataset_view(v) for v in self.service.list_datasets(args[0])]
        return _ok(DatasetList(datasets=views))

    def _get_dataset(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(_dataset_view(self.service.get_dataset(args[0])))

    def _get_version(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self.service.get_version(args[0], int(args[1])))

    # -- splits -----------------------------------------------------------------------------
    def _create_splits(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        plan = self._json(request, SplitPlan)
        record, created = self.service.create_splits(args[0], int(args[1]), plan)
        return _ok(record, created)

    def _list_splits(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(SplitsList(splits=self.service.list_splits(args[0], int(args[1]))))

    def _get_splits(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self.service.get_splits(args[0], int(args[1]), args[2]))

    # -- experiments ------------------------------------------------------------------------
    def _jobs(self) -> ExperimentJobs:
        if self.jobs is None:
            raise ApiFailure("experiment_backend_unavailable", "this server stores no experiments")
        return self.jobs

    def _create_experiment(self, request: _Request, _: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        jobs = self._jobs()
        body: CreateExperiment = self._json(request, CreateExperiment)
        version = self.service.get_version(body.dataset_id, body.dataset_version)
        if body.contract.dataset.identity_hash != version.spec.identity_hash:
            raise ApiFailure(
                "invalid_experiment",
                f"the contract's dataset is not {body.dataset_id} version {body.dataset_version}",
            )
        splits = self.service.get_splits(body.dataset_id, body.dataset_version, body.splits_hash)
        runtime = jobs.runtime
        definition = ExperimentJobDefinition(
            contract=body.contract,
            splits=splits.splits,
            plan=body.plan,
            synthetic=bool(runtime is not None and runtime.synthetic),
            workers=body.workers,
            provenance={
                "dataset_id": body.dataset_id,
                "dataset_version": body.dataset_version,
                "splits_hash": body.splits_hash,
                "upload_id": version.upload_id,
            },
        )
        return _ok(jobs.create(definition), created=True)

    def _list_experiments(self, request: _Request, _: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._jobs().list())

    def _get_experiment(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._jobs().get(args[0]))

    def _cancel_experiment(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._jobs().cancel(args[0]))

    def _resume_experiment(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._jobs().resume(args[0]))

    def _get_artifact(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._jobs().artifact(args[0]))

    # -- experiment artifacts ---------------------------------------------------------------
    def _artifacts(self) -> ExperimentArtifacts:
        champions = self.promotions.store if self.promotions is not None else None
        return ExperimentArtifacts(self._jobs(), champions)

    def _get_provenance(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._artifacts().provenance(args[0]))

    def _trace_field(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        params = self._params(request, {"path"})
        path = params.get("path", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+){0,31}", path):
            raise ApiFailure("invalid_request", "path must be a dotted field path", name="path")
        return _ok(self._artifacts().trace(args[0], path))

    def _verify_artifact(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._artifacts().verify(args[0]))

    def _reproduce_experiment(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._artifacts().reproduce(args[0]))

    # -- champion promotion -----------------------------------------------------------------
    def _promotions(self) -> ChampionPromotions:
        if self.promotions is None:
            raise ApiFailure("experiment_backend_unavailable", "this server stores no champions")
        return self.promotions

    def _promote_experiment(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        params = self._params(request, {"lineage"})
        promotions = self._promotions()
        existed = promotions.store.for_job(args[0]) is not None
        view = promotions.promote(args[0], params.get("lineage"))
        return _ok(view, created=not existed)

    def _get_experiment_promotion(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._promotions().for_job(args[0]))

    def _list_promotions(self, request: _Request, _: tuple[str, ...]) -> ApiResponse:
        params = self._params(request, {"lineage"})
        return _ok(self._promotions().list(params.get("lineage")))

    def _get_promotion(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._promotions().get(args[0]))

    def _verify_promotion(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._promotions().verify(args[0]))

    def _get_champion(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._promotions().current(args[0]))

    def _get_champion_history(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._promotions().history(args[0]))

    # -- champion deployment ----------------------------------------------------------------
    def _deployments(self) -> ChampionDeployments:
        if self.deployments is None:
            raise ApiFailure("experiment_backend_unavailable", "this server deploys no workflows")
        return self.deployments

    def _publish_workflow(self, request: _Request, _: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        body: PublishWorkflow = self._json(request, PublishWorkflow)
        view, created = self._deployments().publish(body.champion_id, body.actor)
        return _ok(view, created=created)

    def _list_workflows(self, request: _Request, _: tuple[str, ...]) -> ApiResponse:
        params = self._params(request, {"lineage"})
        return _ok(self._deployments().list(params.get("lineage")))

    def _get_workflow(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._deployments().get(args[0]))

    def _stage_workflow(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        body: DeploymentChange = self._json(request, DeploymentChange)
        return _ok(self._deployments().stage(args[0], body.expected_revision, body.actor))

    def _promote_workflow(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        body: DeploymentChange = self._json(request, DeploymentChange)
        return _ok(self._deployments().promote(args[0], body.expected_revision, body.actor))

    def _invoke_workflow(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        body: InvokeWorkflow = self._json(request, InvokeWorkflow)
        return _ok(self._deployments().invoke(args[0], body.inputs, request.principal))

    def _get_deployment(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._deployments().deployment(args[0]))

    def _get_deployment_history(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._deployments().history(args[0]))

    def _rollback_deployment(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        body: RollbackDeployment = self._json(request, RollbackDeployment)
        return _ok(
            self._deployments().rollback(
                args[0], body.expected_revision, body.version_id, body.actor
            )
        )

    def _invoke_production(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        body: InvokeWorkflow = self._json(request, InvokeWorkflow)
        return _ok(self._deployments().invoke_production(args[0], body.inputs, request.principal))

    def _get_inference(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._deployments().inference(args[0]))

    # -- production monitoring (#31) --------------------------------------------------------
    def _monitor(self) -> ProductionMonitor:
        if self.monitor is None:
            raise ApiFailure("experiment_backend_unavailable", "this server monitors nothing")
        return self.monitor

    def _submit_feedback(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        body: FeedbackRequest = self._json(request, FeedbackRequest)
        view, created = self._monitor().submit_feedback(args[0], body, request.principal)
        return _ok(view, created=created)

    def _get_feedback(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._monitor().feedback(args[0]))

    def _register_policy(self, request: _Request, _: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        body: MonitoringPolicy = self._json(request, MonitoringPolicy)
        policy, policy_id, created = self._monitor().register_policy(body)
        return _ok(PolicyRegistered(policy_id=policy_id, policy=policy), created=created)

    def _get_policy(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        policy, policy_id = self._monitor().policy(args[0])
        return _ok(PolicyRegistered(policy_id=policy_id, policy=policy))

    def _window(self, request: _Request) -> dict[str, str | None]:
        params = self._params(request, {"since", "until", "policy_id"})
        policy_id = params.get("policy_id")
        if policy_id is not None and not re.fullmatch(r"mp-[0-9a-f]{24}", policy_id):
            raise ApiFailure("invalid_request", "policy_id must be mp-<24 hex>", name="policy_id")
        return {"since": params.get("since"), "until": params.get("until"), "policy_id": policy_id}

    def _get_monitoring(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        return _ok(self._monitor().summary(args[0], **self._window(request)))

    def _get_drift(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        return _ok(self._monitor().drift(args[0], **self._window(request)))

    def _evaluate_trigger(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        body: EvaluateTrigger = self._json(request, EvaluateTrigger)
        view = self._monitor().evaluate(args[0], body.since, body.until, body.policy_id)
        return _ok(view, created=view.created)

    def _list_triggers(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._monitor().history(args[0]))

    def _get_trigger(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._monitor().decision(args[0]))

    def _get_challenger(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._monitor().challenger(args[0]))

    def _reoptimize(self, request: _Request, args: tuple[str, ...]) -> ApiResponse:
        self._params(request, set())
        return _ok(self._monitor().reoptimize(args[0]))


# -- HTTP adapter -------------------------------------------------------------------------------
class _CountingReader:
    """The request body stream, counting what the API consumed of it."""

    def __init__(self, raw: BinaryIO) -> None:
        self.raw, self.consumed = raw, 0

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        self.consumed += len(data)
        return data


def _unread_body(headers: Mapping[str, str], consumed: int) -> int | None:
    """Body bytes the client sent that nobody read; ``None`` when the length is unknown."""
    if "transfer-encoding" in headers:
        return None
    raw = headers.get("content-length", "").strip()
    return max(0, int(raw) - consumed) if raw.isdigit() else 0


def _linger(handler: BaseHTTPRequestHandler, unread: int | None) -> None:
    """Read and discard the rest of a refused request body, after the response was sent.

    A body refused before it was read (413, 411, 404, ...) is still in flight. Closing a socket
    with unread input makes the OS send a TCP reset, which can destroy the response before the
    client reads it: a browser or proxy then reports a network error instead of the error body.
    Like a server's lingering close, drain for at most ``LINGER_SECONDS``, then close regardless.
    """
    if unread == 0:
        return
    deadline = time.monotonic() + LINGER_SECONDS
    try:
        handler.wfile.flush()
        handler.connection.settimeout(1.0)
        while (unread is None or unread > 0) and time.monotonic() < deadline:
            chunk = handler.rfile.read1(65536 if unread is None else min(65536, unread))  # type: ignore[attr-defined]
            if not chunk:
                return
            if unread is not None:
                unread -= len(chunk)
    except OSError:  # timeout, reset: the response is already sent
        return


def respond(handler: BaseHTTPRequestHandler, api: ProductAPI) -> None:
    """Serve one product API request on a stdlib handler (used here and by ``api.chat``)."""
    headers = {k.lower(): v for k, v in handler.headers.items()}
    body = _CountingReader(handler.rfile)
    res = api.handle(handler.command, handler.path, dict(handler.headers.items()), body)  # type: ignore[arg-type]
    data = json.dumps(res.body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    handler.close_connection = True  # an unread (refused) body must not be parsed as a request
    handler.send_response(res.status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(data)
    _linger(handler, _unread_body(headers, body.consumed))


class Server(ThreadingHTTPServer):
    """Threaded stdlib server with a listen backlog for a browser's parallel requests.

    The stdlib default backlog is 5; a page that loads several resources at once (and more than
    one tab) overflows it, and the OS then refuses connections (seen on Windows as 502s from a
    proxy in front of the API).
    """

    daemon_threads = True
    request_queue_size = 128


def make_handler(api: ProductAPI) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            respond(self, api)

        def do_POST(self) -> None:  # noqa: N802
            respond(self, api)

        def do_PUT(self) -> None:  # noqa: N802
            respond(self, api)

        def do_DELETE(self) -> None:  # noqa: N802
            respond(self, api)

        def log_message(self, *args: Any) -> None:
            pass

    return Handler


def registry_runtime(
    service: DatasetService, registry_path: Path, api_key: str | None = None
) -> DatasetRuntime:
    """Real experiment backend: each job's ``plan.expected_model_hash`` is resolved in the model
    registry, bound (``bind_model``) to a client for that pinned entry, and executed by the MAF
    ``WorkflowRunner``; registry prices, when every allowed model has them, price the runs."""
    from core.models import ModelRegistry, registry_pricing
    from experiments.optimization_experiment import bind_model

    registry = ModelRegistry.load(registry_path)

    def backend(definition: ExperimentJobDefinition) -> WorkflowBackend:
        from runtime.backends import client_for
        from runtime.runner import WorkflowRunner

        plan = definition.plan
        client = client_for(registry.by_hash(plan.expected_model_hash), api_key)
        models = bind_model(plan, registry, client)
        runner = WorkflowRunner(model=client, benchmark_hash="inline", allowed_models=models)
        return WorkflowBackend(
            run_workflow=lambda g, t, tr, s: runner.run_sync(g, t, trial=tr, seed=s),
            checker=runner.checker,
            pricing=registry_pricing(models),
            models=models,
        )

    return DatasetRuntime(data=service.blobs.get, backend=backend, synthetic=False)


def registry_inference(registry_path: Path, api_key: str | None = None) -> InferenceRuntime:
    """Real inference backend: the registry file is re-read on every invocation (an entry
    removed, disabled or changed since publishing fails closed) and each pinned entry is served
    by its own registry-bound client (``client_for``)."""
    from core.models import ModelRegistry
    from runtime.backends import client_for

    return InferenceRuntime(
        registry=lambda: ModelRegistry.load(registry_path),
        client=lambda entry: client_for(entry, api_key),
    )


RuntimeFactory = Callable[[DatasetService], JobRuntime]


def build_api(
    data_dir: Path | str,
    max_upload_bytes: int | None = None,
    runtime: RuntimeFactory | None = None,
    inference: InferenceRuntime | None = None,
    quotas: Any = None,
) -> Any:
    """The multi-tenant product API (#32 ``api.tenancy.TenantRouter``) over ``data_dir``:
    ``identity.sqlite3`` plus one partition of durable stores per workspace. ``runtime``
    builds each workspace's experiment backend from its dataset service; without one, creating
    an experiment is refused. ``inference`` binds deployed workflow versions to their pinned
    model; without one, nothing is staged, promoted or served. A data dir still in the
    single-tenant layout is refused until ``python -m api.tenancy migrate-legacy`` assigns it."""
    from api.tenancy import TenantRouter

    return TenantRouter(
        data_dir,
        max_upload_bytes=max_upload_bytes,
        runtime=runtime,
        inference=inference,
        quotas=quotas,
    )


def start_worker(api: Any) -> Any:
    """Recover, then execute experiment jobs in background threads: one ``ProductAPI``'s (a
    single workspace), or every workspace of a ``TenantRouter``."""
    if not isinstance(api, ProductAPI):
        return api.start_workers()  # type: ignore[no-any-return]
    if api.jobs is None or api.jobs.runtime is None:
        return None
    if api.promotions is not None:
        try:
            api.promotions.recover()  # held-out evaluations whose process is gone
        except Exception:
            log.exception("promotion recovery failed")
    worker = JobWorker(api.jobs.store, api.jobs.runtime)
    worker.start()
    return worker


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("WYNK_DATA_DIR", DEFAULT_DATA_DIR)),
        help="dataset metadata + uploaded blobs (default: $WYNK_DATA_DIR or .wynk-data)",
    )
    registry = os.environ.get("WYNK_MODEL_REGISTRY")
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path(registry) if registry else None,
        help="model registry JSON; enables experiment jobs (default: $WYNK_MODEL_REGISTRY; the"
        " API key comes from $WYNK_MODEL_API_KEY)",
    )
    parser.add_argument(
        "--max-upload-mb",
        type=float,
        default=float(os.environ.get("WYNK_MAX_UPLOAD_MB", 50)),
        help="largest accepted dataset upload in MiB (default: $WYNK_MAX_UPLOAD_MB or 50)",
    )


def api_from_args(args: argparse.Namespace) -> Any:
    registry: Path | None = args.model_registry
    key = os.environ.get("WYNK_MODEL_API_KEY")

    def runtime(service: DatasetService) -> JobRuntime:
        assert registry is not None
        return registry_runtime(service, registry, key)

    from api.tenancy import Quotas

    max_bytes = max(1, int(args.max_upload_mb * 1024 * 1024))
    api = build_api(
        args.data_dir,
        max_bytes,
        runtime if registry is not None else None,
        registry_inference(registry, key) if registry is not None else None,
        Quotas.from_env(),
    )
    bootstrap = os.environ.get("WYNK_BOOTSTRAP_API_KEY")
    if bootstrap:  # dev / E2E: an operator-chosen key for a default workspace (hash stored)
        api.bootstrap(
            os.environ.get("WYNK_BOOTSTRAP_EMAIL", "owner@localhost"),
            workspace_id=os.environ.get("WYNK_BOOTSTRAP_WORKSPACE", "ws-default"),
            key_plaintext=bootstrap,
        )
    start_worker(api)
    return api


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_arguments(p)
    p.add_argument("--port", type=int, default=8788)
    args = p.parse_args(argv)
    api = api_from_args(args)
    server = Server(("127.0.0.1", args.port), make_handler(api))
    print(f"wynk product API on http://127.0.0.1:{args.port}{API_PREFIX} (data: {args.data_dir})")
    server.serve_forever()


if __name__ == "__main__":
    main()
