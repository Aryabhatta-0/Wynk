"""Production monitoring and re-optimization triggers: close the production loop.

    deployed workflow version (#30)
      -> immutable inference records (#30, the ONLY telemetry authority) + quality feedback
      -> deterministic windowed aggregates, drift statistics and trigger rules
      -> NOT_TRIGGERED / TRIGGERED / SUPPRESSED, with machine-readable reasons and evidence
      -> TRIGGERED: freeze the eligible new labelled examples -> NEW immutable dataset version
         -> deterministic splits that keep every parent row in its parent role
         -> NEW challenger experiment job (#23 / #24): strong Fixed vs Random vs ACO
      -> nothing is promoted or deployed: the challenger still needs #25 promotion and #30
         stage / promote, explicitly.

Authorities. Telemetry is read from ``inference_records`` (workflow version, provenance, model
hash, tokens, latency, model calls, cost when the pinned registry entry prices it, failures);
this module never writes a competing telemetry record and never writes the deployment store.
Feedback is bound to ONE inference record: its ``inputs`` must hash to the record's
``request_sha256`` and its ``output`` to ``output_sha256``, so a label can only describe the
request that was actually served. Quality is the pinned TaskContract evaluator
(``evaluation.dispatch.evaluate_prediction``), deterministic, never an LLM. Drift and trigger
rules are transparent statistics (presence rates, total-variation distance, two-sample
Kolmogorov-Smirnov) under a content-addressed, persisted ``wynk-monitoring-policy/1``.

Actor / source metadata on feedback is UNTRUSTED (no authentication before #32): it is stored
verbatim under ``untrusted_metadata``, and never enters an identity, an aggregate, a drift
statistic or a trigger decision.

Idempotency. A trigger decision's identity is a hash of its evidence (workflow version, policy,
the exact inference and feedback ids): the same evidence is decided once and that decision is
returned forever after. Decisions are inserted under ``BEGIN IMMEDIATE`` with the history they
were decided against, so concurrent monitors serialise; at most one TRIGGERED decision per
workflow version is open within the policy cooldown; a feedback example is frozen into at most
one new dataset version. The re-optimization is a fenced claim that moves forward only, the new
dataset version is content-addressed (identical bytes register as the identical version) and
the challenger job id is derived from the trigger id, so a crash or a concurrent monitor can
never create a second dataset version or job for the same trigger.
"""

from __future__ import annotations

import json
import math
import secrets
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from core.canonical import canonical_hash, canonical_json
from core.dataset import (
    BPS,
    DatasetSplit,
    DatasetSplits,
    SplitMethod,
    SplitPlan,
    SplitRole,
    SplitUse,
    split_order_key,
)
from core.models import ModelEntry
from core.results import EvaluatedRun, Verdict
from core.task_contract import TaskContract
from evaluation.dispatch import evaluate_prediction
from evaluation.schema import validate_answer
from experiments.contract_run import load_rows
from experiments.deployment import (
    INPUT_SCHEMA_REJECTED,
    ChampionDeployments,
    InferenceNotFound,
    InvalidInferenceRequest,
    check_document,
    validate_inputs,
)
from experiments.jobs import (
    BackendUnavailable,
    ExperimentJobDefinition,
    ExperimentJobs,
    JobNotFound,
    load_canonical,
)
from experiments.optimization_experiment import Strategy
from ingestion.service import DatasetService, NotFound, RegisterDataset
from store.datasets import Conflict, RowIdSource, SplitsRecord
from store.monitoring import (
    DecisionRow,
    FeedbackConflict,
    FeedbackRow,
    ReoptRow,
    ReoptState,
    SQLiteMonitoringStore,
    TriggerOutcome,
)

POLICY_SCHEMA = "wynk-monitoring-policy/1"
FEEDBACK_SCHEMA = "wynk-feedback/1"
SUMMARY_SCHEMA = "wynk-monitoring-summary/1"
DRIFT_SCHEMA = "wynk-drift-report/1"
DECISION_SCHEMA = "wynk-trigger-decision/1"
REOPT_SCHEMA = "wynk-reoptimization/1"
FAIR_STRATEGIES = (Strategy.FIXED, Strategy.RANDOM, Strategy.ACO)
DAY_S = 86_400
DIGITS = 6  # every reported statistic is rounded here: identical inputs, identical bytes


# -- errors -------------------------------------------------------------------------------------
class MonitoringError(Exception):
    code = "monitoring_error"

    def __init__(self, message: str, **details: str | int | None) -> None:
        super().__init__(message)
        self.details = details


class InvalidFeedback(MonitoringError):
    code = "invalid_feedback"


class FeedbackMismatch(MonitoringError):
    """The feedback does not describe the request / response the inference record pins."""

    code = "feedback_mismatch"


class FeedbackAlreadyRecorded(MonitoringError):
    code = "feedback_conflict"


class FeedbackNotFound(MonitoringError):
    code = "feedback_not_found"


class PolicyNotFound(MonitoringError):
    code = "monitoring_policy_not_found"


class InvalidWindow(MonitoringError):
    code = "invalid_monitoring_window"


class TriggerNotFound(MonitoringError):
    code = "trigger_not_found"


class NotTriggered(MonitoringError):
    code = "trigger_not_triggered"


class MonitoringIntegrityError(MonitoringError):
    """A stored record disagrees with the authority it cites: refused, never repaired."""

    code = "monitoring_integrity_error"


class ReoptimizationUnavailable(MonitoringError):
    code = "reoptimization_unavailable"


class _ReoptBlocked(Exception):
    """A permanent reason this trigger cannot produce a challenger (recorded as FAILED)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


# -- policy -------------------------------------------------------------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _fraction(default: float) -> Any:
    return Field(default=default, ge=0.0, le=1.0, allow_inf_nan=False)


class DriftConfig(_Strict):
    """Transparent statistics, each with its own threshold. Reference: the optimization +
    validation rows of the dataset version the champion was selected on (the test split stays
    sealed for #25) and the champion's own selection-time runs (tokens, latency, quality)."""

    min_samples: int = Field(default=30, ge=2)  # observations needed before any data verdict
    presence_delta: float = _fraction(0.2)  # presence-rate change of an optional field (TVD)
    categorical_tvd: float = _fraction(0.25)  # total-variation distance of a categorical field
    numeric_ks: float = _fraction(0.3)  # two-sample KS statistic of a numeric / date field
    length_ks: float = _fraction(0.3)  # KS of a free-text / list field's length, input length
    token_ks: float = _fraction(0.3)  # KS of prompt tokens per inference
    max_categories: int = Field(default=20, ge=2)  # a string field with more is free text
    material_features: int = Field(default=1, ge=1)  # drifted data features = material drift
    # share of ALL requests (served + rejected by the input schema) with a given violation
    schema_violation_rate: float = _fraction(0.05)
    min_labelled: int = Field(default=20, ge=2)  # labels needed before a quality verdict
    quality_max_drop: float = _fraction(0.15)  # selection-time minus production quality


class TriggerRules(_Strict):
    """When a monitoring window starts a re-optimization. Every rule has a minimum sample count:
    one bad request can never trigger anything."""

    min_inferences: int = Field(default=30, ge=2)
    min_labelled: int = Field(default=20, ge=2)
    min_quality: float | None = Field(default=0.7, ge=0.0, le=1.0)  # labelled mean quality
    max_failure_rate: float | None = Field(default=0.2, ge=0.0, le=1.0)
    on_material_drift: bool = True
    on_quality_drift: bool = True
    max_p95_latency_ratio: float | None = Field(default=None, gt=0.0)  # vs the champion's runs
    max_mean_cost_ratio: float | None = Field(default=None, gt=0.0)  # authoritative cost only
    min_new_labelled_examples: int = Field(default=10, ge=1)  # to build a new dataset version
    cooldown_s: int = Field(default=7 * DAY_S, ge=0)  # one open TRIGGERED per version


class MonitoringPolicy(_Strict):
    policy_schema: Literal["wynk-monitoring-policy/1"] = POLICY_SCHEMA
    name: str = Field(default="default", min_length=1, max_length=100)
    window_s: PositiveInt = 7 * DAY_S  # default window when the caller names none
    drift: DriftConfig = Field(default_factory=DriftConfig)
    trigger: TriggerRules = Field(default_factory=TriggerRules)
    # How the NEW rows of a re-optimization dataset are split when the parent splits carry no
    # seeded plan of their own. Parent rows always keep their parent role.
    new_row_splits: SplitPlan = Field(
        default_factory=lambda: SplitPlan(seed=0, validation_bps=2000, test_bps=2000)
    )

    @property
    def policy_id(self) -> str:
        return "mp-" + canonical_hash(self.model_dump(mode="json"))[:24]


# -- requests and views -------------------------------------------------------------------------
class FeedbackRequest(_Strict):
    """Feedback on ONE inference record. ``inputs`` / ``output`` must be exactly what was sent
    and returned (hash-checked against the record). ``expected`` omitted: an unlabelled input
    observation (drift only). ``actor`` / ``source`` are untrusted, opaque metadata."""

    inputs: dict[str, Any]
    output: dict[str, Any] | None = None
    expected: dict[str, Any] | None = None
    actor: str | None = Field(default=None, max_length=200)
    source: dict[str, str] = Field(default_factory=dict, max_length=20)


class FeedbackView(_Strict):
    record_schema: str
    feedback_id: str
    inference_id: str
    workflow_version: str
    lineage_id: str
    labelled: bool
    lineage: dict[str, Any]  # champion, artifact / provenance, source dataset version
    evaluation: dict[str, Any] | None  # the deterministic EvaluatorRecord (no target values)
    inference_created_at: str
    received_at: str
    untrusted_metadata: dict[str, Any]
    metadata_trusted: bool  # always False before #32


class MonitoringSummary(_Strict):
    record_schema: str
    workflow_version: str
    lineage_id: str
    window: dict[str, str]
    pins: dict[str, Any]
    inferences: dict[str, Any]
    latency_s: dict[str, float | None]
    model_calls: dict[str, float | int | None]
    tokens: dict[str, Any]
    cost: dict[str, Any]
    feedback: dict[str, Any]
    evidence: dict[str, Any]


class DriftReport(_Strict):
    record_schema: str
    workflow_version: str
    policy_id: str
    window: dict[str, str]
    reference: dict[str, Any]
    features: list[dict[str, Any]]
    quality: dict[str, Any]
    material: bool
    drifted_features: list[str]
    evidence: dict[str, Any]


class TriggerDecisionView(_Strict):
    record_schema: str
    trigger_id: str
    workflow_version: str
    lineage_id: str
    policy_id: str
    outcome: TriggerOutcome
    reasons: list[dict[str, Any]]
    checks: list[dict[str, Any]]
    evidence: dict[str, Any]
    frozen_feedback_ids: list[str]
    links: dict[str, Any]
    created_at: str
    created: bool = False  # True only on the call that decided it


class TriggerHistory(_Strict):
    workflow_version: str
    decisions: list[TriggerDecisionView]


class ReoptimizationView(_Strict):
    record_schema: str
    trigger_id: str
    state: ReoptState | None  # None: TRIGGERED, not yet claimed
    workflow_version: str
    lineage_id: str
    production: dict[str, Any]  # what triggered it: champion, artifact, provenance, source job
    dataset: dict[str, Any] | None  # the NEW dataset version + its splits and derivation
    challenger_job_id: str | None
    challenger_job_state: str | None
    strategies: list[str]
    deployed: bool  # always False: deployment is #25 + #30, explicitly
    detail: str | None


# -- statistics (pure, deterministic) -----------------------------------------------------------
def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), DIGITS)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least ``q`` of the data at or
    below it. ``None`` for no data."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(1, math.ceil(q * len(ordered)))
    return ordered[k - 1]


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max |F_a(x) - F_b(x)| over every point."""
    if not a or not b:
        raise ValueError("both samples need data")
    xs, ys = sorted(a), sorted(b)
    i = j = 0
    d = 0.0
    while i < len(xs) and j < len(ys):
        x = min(xs[i], ys[j])
        while i < len(xs) and xs[i] == x:
            i += 1
        while j < len(ys) and ys[j] == x:
            j += 1
        d = max(d, abs(i / len(xs) - j / len(ys)))
    return d


def total_variation(a: Iterable[str], b: Iterable[str]) -> float:
    """Total-variation distance of two categorical samples (an unseen category is mass too)."""
    ca, cb = Counter(a), Counter(b)
    na, nb = sum(ca.values()), sum(cb.values())
    if not na or not nb:
        raise ValueError("both samples need data")
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in set(ca) | set(cb))


def _when(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise InvalidWindow(f"{value!r} is not an ISO-8601 timestamp") from None
    if parsed.tzinfo is None:
        raise InvalidWindow(f"{value!r} has no timezone")
    return parsed.astimezone(UTC)


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds")


# -- the version under monitoring ---------------------------------------------------------------
@dataclass(frozen=True)
class _Context:
    version_id: str
    lineage_id: str
    document: Mapping[str, Any]
    contract: TaskContract
    source_job_id: str
    source_definition: ExperimentJobDefinition | None

    @property
    def pins(self) -> dict[str, Any]:
        d = self.document
        return {
            "champion_id": d["champion"]["champion_id"],
            "genome_hash": d["workflow"]["genome_hash"],
            "artifact_id": d["provenance"]["artifact_id"],
            "provenance_id": d["provenance"]["provenance_id"],
            "experiment_id": d["provenance"]["experiment_id"],
            "source_job_id": self.source_job_id,
            "model_hash": d["model"]["model_hash"],
            "contract_hash": d["contract"]["contract_hash"],
            "dataset_id": self.contract.dataset.dataset_id,
            "dataset_version": self.contract.dataset.dataset_version,
            "dataset_hash": self.contract.dataset.identity_hash,
        }


@dataclass
class ProductionMonitor:
    """Feedback, aggregates, drift, trigger decisions and re-optimization for deployed workflow
    versions. ``jobs`` / ``datasets`` are ``None`` in a process without an experiment backend or
    dataset store: monitoring still works; re-optimization answers ``reoptimization_unavailable``
    and stays resumable."""

    deployments: ChampionDeployments
    store: SQLiteMonitoringStore
    jobs: ExperimentJobs | None = None
    datasets: DatasetService | None = None
    owner: str = field(default_factory=lambda: "monitor-" + secrets.token_hex(6))
    lease_s: float = 600.0
    clock: Callable[[], float] = time.time  # leases only
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))  # default window end

    # -- policies ---------------------------------------------------------------------------
    def register_policy(self, policy: MonitoringPolicy) -> tuple[MonitoringPolicy, str, bool]:
        row, created = self.store.put_policy(
            policy.policy_id, canonical_json(policy.model_dump(mode="json"))
        )
        return self._policy_from(row.policy_id, row.policy_json), row.policy_id, created

    def policy(self, policy_id: str | None) -> tuple[MonitoringPolicy, str]:
        if policy_id is None:
            policy, pid, _ = self.register_policy(MonitoringPolicy())
            return policy, pid
        row = self.store.policy(policy_id)
        if row is None:
            raise PolicyNotFound(f"no monitoring policy {policy_id}")
        return self._policy_from(row.policy_id, row.policy_json), row.policy_id

    @staticmethod
    def _policy_from(policy_id: str, policy_json: str) -> MonitoringPolicy:
        policy = MonitoringPolicy.model_validate_json(policy_json)
        if policy.policy_id != policy_id:
            raise MonitoringIntegrityError(f"stored policy {policy_id} no longer hashes to its id")
        return policy

    # -- context ----------------------------------------------------------------------------
    def _context(self, version_id: str) -> _Context:
        row = self.deployments._row(version_id)  # workflow_version_not_found
        document = check_document(row)  # workflow_version_integrity_error
        contract = TaskContract.model_validate(document["contract"]["task_contract"])
        job_id = document["provenance"]["job_id"]
        definition = None
        if self.jobs is not None:
            # The ONLY tolerated failure is absence: the champion's job lives in another
            # process's job store. A job that IS here but whose artifact is missing, not
            # finalized, corrupt, hash-invalid or inconsistent with its provenance (or with the
            # version's pins) fails closed; corruption is never reported as "not here".
            try:
                job, artifact = load_canonical(self.jobs.store, job_id)
            except JobNotFound:
                job, artifact = None, None
            except Exception as exc:
                raise MonitoringIntegrityError(
                    f"source job {job_id} of workflow version {version_id} does not verify: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if job is not None:
                pinned = document["provenance"]
                problems = [
                    name
                    for name, (a, b) in {
                        "artifact_id": (artifact.artifact_id, pinned["artifact_id"]),
                        "provenance_id": (artifact.provenance_id, pinned["provenance_id"]),
                    }.items()
                    if a != b
                ]
                if problems:
                    raise MonitoringIntegrityError(
                        f"job {job_id}'s artifact is not the one version {version_id} pins: "
                        + ", ".join(problems)
                    )
                try:
                    definition = ExperimentJobDefinition.model_validate_json(job.definition_json)
                except ValueError as exc:
                    raise MonitoringIntegrityError(
                        f"source job {job_id}'s definition does not parse"
                    ) from exc
        return _Context(version_id, row.lineage_id, document, contract, job_id, definition)

    def _records(self, ctx: _Context, since: datetime, until: datetime) -> list[dict[str, Any]]:
        """The version's immutable inference records in ``[since, until)``, each re-checked to
        cite exactly this version and its pinned provenance and model."""
        out = []
        pins = ctx.pins
        for row in self.deployments.store.inferences(ctx.version_id):
            created = _when(row.created_at)
            if not since <= created < until:
                continue
            record = json.loads(row.record_json)
            if (
                record.get("workflow_version") != ctx.version_id
                or row.version_id != ctx.version_id
                or record["provenance"]["artifact_id"] != pins["artifact_id"]
                or record["provenance"]["provenance_id"] != pins["provenance_id"]
                or record["model"]["model_hash"] != pins["model_hash"]
                or record["genome_hash"] != pins["genome_hash"]
            ):
                raise MonitoringIntegrityError(
                    f"inference {row.inference_id} does not cite version {ctx.version_id}'s pins"
                )
            out.append(record | {"created_at": row.created_at})
        return out

    def _window(
        self, policy: MonitoringPolicy, since: str | None, until: str | None
    ) -> tuple[datetime, datetime]:
        end = _when(until) if until is not None else self.now().astimezone(UTC)
        start = _when(since) if since is not None else end - timedelta(seconds=policy.window_s)
        if not start < end:
            raise InvalidWindow("since must be earlier than until")
        return start, end

    # -- feedback ---------------------------------------------------------------------------
    def submit_feedback(
        self, inference_id: str, request: FeedbackRequest
    ) -> tuple[FeedbackView, bool]:
        """Bind feedback to one immutable inference record (never modified). ``bool``: new."""
        row = self.deployments.store.inference(inference_id)
        if row is None:
            raise InferenceNotFound(f"no inference {inference_id}")
        record = json.loads(row.record_json)
        ctx = self._context(row.version_id)
        contract = ctx.contract
        try:
            values = validate_inputs(contract, request.inputs)
        except InvalidInferenceRequest as exc:
            raise InvalidFeedback(f"inputs do not fit the pinned input schema: {exc}") from None
        if canonical_hash(values) != record["request_sha256"]:
            raise FeedbackMismatch("inputs are not the request this inference served")
        if record["status"] == "SUCCEEDED":
            if request.output is None or canonical_hash(request.output) != record["output_sha256"]:
                raise FeedbackMismatch("output is not the answer this inference returned")
        elif request.output is not None:
            raise FeedbackMismatch("this inference failed and returned no output")
        evaluation = None
        if request.expected is not None:
            problems = validate_answer(contract.output_schema, request.expected)
            if problems:
                name = sorted(problems)[0]
                raise InvalidFeedback(f"expected.{name}: {problems[name]}")
            rec = evaluate_prediction(
                contract.evaluation, contract.output_schema, request.expected, request.output
            )
            if not rec.ok:
                raise InvalidFeedback(f"the pinned evaluator could not judge it: {rec.detail}")
            evaluation = rec.model_dump(mode="json")
        lineage = {
            **{k: v for k, v in ctx.pins.items()},
            "promotion_id": record["provenance"]["promotion_id"],
        }
        identity = {
            "record_schema": FEEDBACK_SCHEMA,
            "inference_id": inference_id,
            "expected": request.expected,
        }
        feedback_id = "fb-" + canonical_hash(identity)[:24]
        body = {
            "record_schema": FEEDBACK_SCHEMA,
            "feedback_id": feedback_id,
            "inference_id": inference_id,
            "workflow_version": row.version_id,
            "lineage_id": row.lineage_id,
            "labelled": request.expected is not None,
            "lineage": lineage,
            "inputs": values,
            "output": request.output,
            "expected": request.expected,
            "evaluation": evaluation,
            "inference_created_at": row.created_at,
            # opaque, unauthenticated: recorded, never used for any identity or decision
            "untrusted_metadata": {"actor": request.actor, "source": dict(request.source)},
            "metadata_trusted": False,
        }
        try:
            stored, created = self.store.put_feedback(
                FeedbackRow(
                    feedback_id=feedback_id,
                    inference_id=inference_id,
                    workflow_version=row.version_id,
                    lineage_id=row.lineage_id,
                    labelled=request.expected is not None,
                    inference_created_at=row.created_at,
                    record_json=canonical_json(body),
                    received_at=self.store.clock(),
                )
            )
        except FeedbackConflict as exc:
            raise FeedbackAlreadyRecorded(str(exc), inference_id=inference_id) from None
        return _feedback_view(stored), created

    def feedback(self, feedback_id: str) -> FeedbackView:
        row = self.store.feedback(feedback_id)
        if row is None:
            raise FeedbackNotFound(f"no feedback {feedback_id}")
        return _feedback_view(row)

    def _feedback_in(
        self, ctx: _Context, records: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        ids = {r["inference_id"] for r in records}
        out = []
        for row in self.store.feedback_for_version(ctx.version_id):
            if row.inference_id not in ids:
                continue
            body = json.loads(row.record_json)
            if body["workflow_version"] != ctx.version_id or body["feedback_id"] != row.feedback_id:
                raise MonitoringIntegrityError(f"feedback {row.feedback_id} is inconsistent")
            out.append(body)
        return out

    # -- aggregates -------------------------------------------------------------------------
    def summary(
        self,
        version_id: str,
        since: str | None = None,
        until: str | None = None,
        policy_id: str | None = None,
    ) -> MonitoringSummary:
        policy, _ = self.policy(policy_id)
        ctx = self._context(version_id)
        start, end = self._window(policy, since, until)
        records = self._records(ctx, start, end)
        return _summarize(ctx, start, end, records, self._feedback_in(ctx, records))

    # -- drift ------------------------------------------------------------------------------
    def drift(
        self,
        version_id: str,
        since: str | None = None,
        until: str | None = None,
        policy_id: str | None = None,
    ) -> DriftReport:
        policy, pid = self.policy(policy_id)
        ctx = self._context(version_id)
        start, end = self._window(policy, since, until)
        records = self._records(ctx, start, end)
        return self._drift(ctx, policy, pid, start, end, records, self._feedback_in(ctx, records))

    def _reference(self, ctx: _Context) -> dict[str, Any]:
        """Reference rows (optimization + validation inputs of the champion's dataset version)
        and the champion's selection-time runs. Missing pieces are reported, never invented."""
        out: dict[str, Any] = {"rows": None, "runs": None, "unavailable": []}
        definition = ctx.source_definition
        if definition is None:
            out["unavailable"].append("source_experiment_not_in_this_store")
            return out
        allowed = set(definition.splits.rows_for(SplitRole.OPTIMIZATION, SplitUse.REPORTING))
        allowed |= set(definition.splits.rows_for(SplitRole.VALIDATION, SplitUse.REPORTING))
        data = self._dataset_bytes(ctx)
        if data is None:
            out["unavailable"].append("source_dataset_not_registered")
        else:
            names = [f.name for f in ctx.contract.input_schema.fields]
            out["rows"] = [
                {n: row[n] for n in names if row.get(n) is not None}
                for rid, row in load_rows(ctx.contract, data)
                if rid in allowed
            ]
        runs = self._champion_runs(ctx)
        if runs:
            out["runs"] = runs
        else:
            out["unavailable"].append("champion_selection_runs_not_found")
        return out

    def _dataset_bytes(self, ctx: _Context) -> bytes | None:
        if self.datasets is None:
            return None
        ds = ctx.contract.dataset
        try:
            record = self.datasets.get_version(ds.dataset_id, ds.dataset_version)
        except NotFound:
            return None
        if record.identity_hash != ds.identity_hash:  # registered, but not the pinned bytes
            raise MonitoringIntegrityError(
                f"registered {ds.dataset_id} v{ds.dataset_version} is not the version "
                f"{ctx.version_id} pins"
            )
        return self.datasets.blobs.get(ds.content_hash)

    def _champion_runs(self, ctx: _Context) -> list[dict[str, Any]]:
        """Every stored selection-time run (optimization / validation rows: the job store holds
        no test run) of exactly the champion genome, deduplicated by run id."""
        if self.jobs is None:
            return []
        job = self.jobs.store.get_job(ctx.source_job_id)
        if job is None:
            return []
        genome = ctx.pins["genome_hash"]
        runs: dict[str, dict[str, Any]] = {}
        for unit in job.units:
            for a in self.jobs.store.attempts(job.job_id, unit.strategy, unit.seed):
                if a.genome_hash != genome or a.result_json is None:
                    continue
                run = EvaluatedRun.model_validate_json(a.result_json)
                ev = run.evaluation
                quality = (
                    ev.evaluator.quality
                    if ev.evaluator is not None and ev.evaluator.quality is not None
                    else (1.0 if ev.verdict is Verdict.PASS else 0.0)
                )
                runs[run.run_id] = {
                    "run_id": run.run_id,
                    "prompt_tokens": run.execution.metrics.prompt_tokens,
                    "completion_tokens": run.execution.metrics.completion_tokens,
                    "latency_s": run.execution.budget_usage.wall_time_s,
                    "quality": quality,
                }
        return [runs[k] for k in sorted(runs)]

    def _drift(
        self,
        ctx: _Context,
        policy: MonitoringPolicy,
        policy_id: str,
        start: datetime,
        end: datetime,
        records: Sequence[Mapping[str, Any]],
        feedback: Sequence[Mapping[str, Any]],
    ) -> DriftReport:
        cfg = policy.drift
        ref = self._reference(ctx)
        observed = [f["inputs"] for f in feedback]
        features: list[dict[str, Any]] = []

        def feature(name: str, kind: str, stat: str, threshold: float, a, b) -> None:
            item: dict[str, Any] = {
                "feature": name,
                "kind": kind,
                "statistic": stat,
                "threshold": threshold,
                "reference_n": None if a is None else len(a),
                "observed_n": len(b),
                "value": None,
            }
            if a is None or not a:
                item["status"] = "UNAVAILABLE"
            elif len(b) < cfg.min_samples:
                item["status"] = "INSUFFICIENT_EVIDENCE"
            else:
                value = total_variation(a, b) if stat == "tvd" else ks_statistic(a, b)
                item["value"] = _r(value)
                item["status"] = "DRIFTED" if value > threshold else "STABLE"
            features.append(item)

        requests = len(records)  # every request this version received, served or rejected
        violations = [r["failure"].get("violations", []) for r in records if _rejected(r)]

        def schema_rate(name: str, match: Callable[[Mapping[str, Any]], bool]) -> None:
            hits = sum(1 for vs in violations if any(match(v) for v in vs))
            item: dict[str, Any] = {
                "feature": name,
                "kind": "schema",
                "statistic": "violation_rate",
                "threshold": cfg.schema_violation_rate,
                "reference_n": None,  # the reference rows satisfy the schema by construction
                "observed_n": requests,
                "violations": hits,
                "value": None,
            }
            if requests < cfg.min_samples:
                item["status"] = "INSUFFICIENT_EVIDENCE"
            else:
                item["value"] = _r(hits / requests)
                item["status"] = (
                    "DRIFTED" if hits / requests > cfg.schema_violation_rate else "STABLE"
                )
            features.append(item)

        rows = ref["rows"]
        for f in ctx.contract.input_schema.fields:
            kind = f.type.value
            ref_vals = None if rows is None else [r[f.name] for r in rows if f.name in r]
            obs_vals = [o[f.name] for o in observed if o.get(f.name) is not None]
            # schema: presence of an optional field + values outside the pinned type
            if not f.required:
                a = None if rows is None else ["1" if f.name in r else "0" for r in rows]
                b = ["1" if o.get(f.name) is not None else "0" for o in observed]
                feature(f"{f.name}.presence", "schema", "tvd", cfg.presence_delta, a, b)
            # schema: values outside the pinned type / missing required fields, observed in the
            # #30 records of requests the input schema REJECTED (feedback inputs are always
            # schema-valid, so they can never show a violation)
            schema_rate(
                f"{f.name}.type", lambda v, n=f.name: v.get("field") == n and v["code"] == "type"
            )
            if f.required:
                schema_rate(
                    f"{f.name}.missing",
                    lambda v, n=f.name: v.get("field") == n and v["code"] == "missing",
                )
            if kind == "boolean" or (
                kind in ("string", "date")
                and ref_vals is not None
                and len(set(ref_vals)) <= cfg.max_categories
                and len(set(ref_vals)) * 2 <= max(len(ref_vals), 1)
            ):
                feature(
                    f"{f.name}.distribution",
                    "categorical",
                    "tvd",
                    cfg.categorical_tvd,
                    None if ref_vals is None else [canonical_json(v) for v in ref_vals],
                    [canonical_json(v) for v in obs_vals],
                )
            elif kind in ("integer", "number"):
                feature(
                    f"{f.name}.distribution",
                    "numeric",
                    "ks",
                    cfg.numeric_ks,
                    None if ref_vals is None else [float(v) for v in ref_vals],
                    [float(v) for v in obs_vals if isinstance(v, int | float)],
                )
            elif kind == "date":
                feature(
                    f"{f.name}.distribution",
                    "numeric",
                    "ks",
                    cfg.numeric_ks,
                    None if ref_vals is None else [_ordinal(v) for v in ref_vals],
                    [_ordinal(v) for v in obs_vals if _ordinal(v) is not None],
                )
            else:  # free text or list: its length distribution
                feature(
                    f"{f.name}.length",
                    "length",
                    "ks",
                    cfg.length_ks,
                    None if ref_vals is None else [float(len(v)) for v in ref_vals],
                    [float(len(v)) for v in obs_vals if isinstance(v, str | list)],
                )
        schema_rate("input.unknown_fields", lambda v: v["code"] == "unknown")
        feature(
            "input.length",
            "length",
            "ks",
            cfg.length_ks,
            None if rows is None else [float(len(canonical_json(r))) for r in rows],
            [float(len(canonical_json(o))) for o in observed],
        )
        runs = ref["runs"]
        feature(
            "prompt_tokens",
            "tokens",
            "ks",
            cfg.token_ks,
            None if runs is None else [float(r["prompt_tokens"]) for r in runs],
            [float(r["usage"]["prompt_tokens"]) for r in records if r["usage"]["model_calls"]],
        )
        # quality degradation: labelled production quality vs the champion's selection runs
        labelled = [f for f in feedback if f["labelled"]]
        q_obs = _mean([f["evaluation"]["quality"] for f in labelled])
        q_ref = _mean([r["quality"] for r in runs]) if runs else None
        quality: dict[str, Any] = {
            "feature": "quality",
            "statistic": "mean_quality_drop",
            "threshold": cfg.quality_max_drop,
            "reference_quality": _r(q_ref),
            "observed_quality": _r(q_obs),
            "reference_n": None if runs is None else len(runs),
            "observed_n": len(labelled),
            "min_samples": cfg.min_labelled,
            "value": None,
        }
        if q_ref is None:
            quality["status"] = "UNAVAILABLE"
        elif len(labelled) < cfg.min_labelled or q_obs is None:
            quality["status"] = "INSUFFICIENT_EVIDENCE"
        else:
            drop = q_ref - q_obs
            quality["value"] = _r(drop)
            quality["status"] = "DRIFTED" if drop > cfg.quality_max_drop else "STABLE"
        drifted = sorted(f["feature"] for f in features if f["status"] == "DRIFTED")
        evidence = {
            "inference_ids": sorted(r["inference_id"] for r in records),
            "feedback_ids": sorted(f["feedback_id"] for f in feedback),
            "reference_run_ids": [] if runs is None else [r["run_id"] for r in runs],
        }
        body = {
            "record_schema": DRIFT_SCHEMA,
            "workflow_version": ctx.version_id,
            "policy_id": policy_id,
            "window": {"since": _stamp(start), "until": _stamp(end)},
            "reference": {
                "dataset_id": ctx.contract.dataset.dataset_id,
                "dataset_version": ctx.contract.dataset.dataset_version,
                "dataset_hash": ctx.contract.dataset.identity_hash,
                "splits": ["optimization", "validation"],  # test stays sealed for #25
                "rows": None if rows is None else len(rows),
                "champion_runs": None if runs is None else len(runs),
                "unavailable": ref["unavailable"],
            },
            "features": features,
            "quality": quality,
            "material": len(drifted) >= cfg.material_features,
            "drifted_features": drifted,
            "evidence": evidence | {"evidence_hash": canonical_hash(evidence)},
        }
        return DriftReport.model_validate(body)

    # -- triggers ---------------------------------------------------------------------------
    def evaluate(
        self,
        version_id: str,
        since: str | None = None,
        until: str | None = None,
        policy_id: str | None = None,
        *,
        reoptimize: bool = True,
    ) -> TriggerDecisionView:
        """Decide (once per evidence) whether this window triggers a re-optimization; a
        TRIGGERED decision then starts (or resumes) it. Never deploys anything."""
        policy, pid = self.policy(policy_id)
        ctx = self._context(version_id)
        start, end = self._window(policy, since, until)
        records = self._records(ctx, start, end)
        feedback = self._feedback_in(ctx, records)
        summary = _summarize(ctx, start, end, records, feedback)
        drift = self._drift(ctx, policy, pid, start, end, records, feedback)
        identity = {
            "record_schema": DECISION_SCHEMA,
            "workflow_version": version_id,
            "policy_id": pid,
            "inference_ids": summary.evidence["inference_ids"],
            "feedback_ids": summary.evidence["feedback_ids"],
        }
        trigger_id = "tr-" + canonical_hash(identity)[:24]
        evidence_until = max((r["created_at"] for r in records), default=_stamp(end))
        reasons, checks = _fire(policy, summary, drift, self._reference_costs(ctx))
        eligible = sorted(
            f["feedback_id"] for f in feedback if f["labelled"] and f["evaluation"] is not None
        )
        links = {
            **ctx.pins,
            "lineage_id": ctx.lineage_id,
            "summary_evidence_hash": summary.evidence["evidence_hash"],
            "drift_evidence_hash": drift.evidence["evidence_hash"],
        }

        def decide(
            prior: list[DecisionRow], frozen: set[str]
        ) -> tuple[TriggerOutcome, str, list[str]]:
            suppressed: list[dict[str, Any]] = []
            freeze: list[str] = []
            if not reasons:
                outcome = TriggerOutcome.NOT_TRIGGERED
            else:
                horizon = _when(evidence_until) - timedelta(seconds=policy.trigger.cooldown_s)
                for p in prior:
                    if p.outcome is TriggerOutcome.TRIGGERED and _when(p.window_until) > horizon:
                        suppressed.append(
                            {
                                "code": "cooldown_open_trigger",
                                "trigger_id": p.trigger_id,
                                "cooldown_s": policy.trigger.cooldown_s,
                            }
                        )
                new = [f for f in eligible if f not in frozen]
                if len(new) < policy.trigger.min_new_labelled_examples:
                    suppressed.append(
                        {
                            "code": "insufficient_new_labelled_examples",
                            "observed": len(new),
                            "minimum": policy.trigger.min_new_labelled_examples,
                        }
                    )
                if suppressed:
                    outcome = TriggerOutcome.SUPPRESSED
                else:
                    outcome = TriggerOutcome.TRIGGERED
                    freeze = new
            body = {
                "record_schema": DECISION_SCHEMA,
                "trigger_id": trigger_id,
                "workflow_version": version_id,
                "lineage_id": ctx.lineage_id,
                "policy_id": pid,
                "outcome": outcome.value,
                "reasons": reasons + suppressed,
                "checks": checks,
                "evidence": {
                    **identity,
                    "window": {"since": _stamp(start), "until": _stamp(end)},
                    "evidence_until": evidence_until,
                    "summary": summary.model_dump(mode="json", exclude={"evidence"}),
                    "drift": drift.model_dump(mode="json", exclude={"evidence"}),
                },
                "frozen_feedback_ids": freeze,
                "links": links,
            }
            return outcome, canonical_json(body), freeze

        row, created = self.store.decide(
            trigger_id, version_id, ctx.lineage_id, pid, evidence_until, decide
        )
        view = _decision_view(row, created)
        if reoptimize and view.outcome is TriggerOutcome.TRIGGERED:
            try:
                self.reoptimize(trigger_id)
            except ReoptimizationUnavailable:
                pass  # TRIGGERED stays resumable: POST .../reoptimize once a backend exists
        return view

    def _reference_costs(self, ctx: _Context) -> dict[str, float | None]:
        """The champion's selection-time p95 latency and mean cost per run. Cost only when the
        pinned registry entry prices the model (the same authority #30 inference records use)."""
        runs = self._champion_runs(ctx)
        entry = ModelEntry.model_validate(ctx.document["model"]["registry_entry"])
        cost = None
        if runs and entry.pricing is not None:
            cost = _mean(
                [entry.pricing.cost(r["prompt_tokens"], r["completion_tokens"]) for r in runs]
            )
        return {
            "p95_latency_s": percentile([r["latency_s"] for r in runs], 0.95) if runs else None,
            "mean_cost": cost,
        }

    def decision(self, trigger_id: str) -> TriggerDecisionView:
        row = self.store.decision(trigger_id)
        if row is None:
            raise TriggerNotFound(f"no trigger decision {trigger_id}")
        return _decision_view(row, False)

    def history(self, version_id: str) -> TriggerHistory:
        self.deployments._row(version_id)
        return TriggerHistory(
            workflow_version=version_id,
            decisions=[_decision_view(r, False) for r in self.store.decisions(version_id)],
        )

    # -- re-optimization --------------------------------------------------------------------
    def reoptimize(self, trigger_id: str) -> ReoptimizationView:
        """Create (or resume, or return) the trigger's new dataset version and challenger job.
        Fenced: one monitor works on a trigger at a time, and every write re-checks the claim."""
        decision = self.store.decision(trigger_id)
        if decision is None:
            raise TriggerNotFound(f"no trigger decision {trigger_id}")
        if decision.outcome is not TriggerOutcome.TRIGGERED:
            raise NotTriggered(f"{trigger_id} is {decision.outcome.value}")
        if self.jobs is None or self.jobs.runtime is None or self.datasets is None:
            raise ReoptimizationUnavailable(
                "this process has no experiment backend / dataset store to re-optimize with"
            )
        claim = self.store.claim_reoptimization(trigger_id, self.owner, self.clock(), self.lease_s)
        if claim is None:  # finished, or a live monitor holds it
            return self.challenger(trigger_id)
        try:
            self._run(decision, claim)
        except _ReoptBlocked as exc:
            self.store.advance_reoptimization(
                trigger_id,
                claim.owner,
                claim.fence,
                ReoptState.FAILED,
                detail=canonical_json({"reason": exc.reason, "message": str(exc)[:500]}),
            )
        return self.challenger(trigger_id)

    def _run(self, decision: DecisionRow, claim: ReoptRow) -> None:
        assert self.jobs is not None and self.datasets is not None
        body = json.loads(decision.decision_json)
        ctx = self._context(decision.workflow_version)
        if ctx.source_definition is None:
            raise _ReoptBlocked("source_experiment_not_found", "the champion's job is not here")
        policy, _ = self.policy(decision.policy_id)
        frozen = self.store.frozen_by(decision.trigger_id)
        if sorted(frozen) != sorted(body["frozen_feedback_ids"]):
            raise MonitoringIntegrityError(f"{decision.trigger_id}'s frozen examples changed")
        built = self._new_dataset(ctx, policy, decision.trigger_id, frozen)
        claim = self.store.advance_reoptimization(
            decision.trigger_id,
            claim.owner,
            claim.fence,
            ReoptState.DATASET_CREATED,
            dataset_id=built["dataset_id"],
            dataset_version=built["dataset_version"],
            splits_hash=built["splits_hash"],
            detail=canonical_json(built["derivation"]),
        )
        definition = self._challenger_definition(ctx, decision, built)
        job_id = "j-" + canonical_hash({"trigger_id": decision.trigger_id, "job": 1})[:24]
        existing = self.jobs.store.get_job(job_id)
        if existing is None:
            jobs = ExperimentJobs(
                self.jobs.store, self.jobs.runtime, clock=self.jobs.clock, _ids=lambda: job_id
            )
            try:
                jobs.create(definition)
            except Conflict:  # a concurrent monitor created exactly this job first
                pass
            except BackendUnavailable as exc:
                raise ReoptimizationUnavailable(str(exc)) from None
            existing = self.jobs.store.get_job(job_id)
        if existing is None or existing.definition_hash != definition.definition_hash:
            raise MonitoringIntegrityError(f"challenger job {job_id} is not this trigger's job")
        self.store.advance_reoptimization(
            decision.trigger_id, claim.owner, claim.fence, ReoptState.JOB_CREATED, job_id=job_id
        )

    def _new_dataset(
        self, ctx: _Context, policy: MonitoringPolicy, trigger_id: str, frozen: Sequence[str]
    ) -> dict[str, Any]:
        """Parent rows + the frozen labelled examples -> a NEW dataset version (the parent is
        never touched) with splits that keep every parent row in its parent role."""
        assert self.datasets is not None and ctx.source_definition is not None
        ds = ctx.contract.dataset
        try:
            parent = self.datasets.get_version(ds.dataset_id, ds.dataset_version)
        except NotFound:
            raise _ReoptBlocked(
                "source_dataset_not_registered", "the champion's dataset is not registered"
            ) from None
        if parent.identity_hash != ds.identity_hash:
            raise MonitoringIntegrityError(
                f"registered {ds.dataset_id} v{ds.dataset_version} is not the version the "
                "champion pins"
            )
        parent_splits = ctx.source_definition.splits
        if parent_splits.dataset_hash != ds.identity_hash:
            raise MonitoringIntegrityError("the champion's splits are not its dataset's")
        data = self.datasets.blobs.get(ds.content_hash)
        rows = load_rows(ctx.contract, data)
        inputs = (*ds.input_columns, *ds.context_columns)

        def key(values: Mapping[str, Any]) -> str:
            return canonical_hash({n: values[n] for n in inputs if values.get(n) is not None})

        existing = {key(row) for _, row in rows}
        examples = [json.loads(self._feedback_row(fid).record_json) for fid in sorted(frozen)]
        excluded: dict[str, str] = {}
        by_input: dict[str, list[dict[str, Any]]] = {}
        for fb in examples:
            if fb["workflow_version"] != ctx.version_id or not fb["labelled"]:
                raise MonitoringIntegrityError(f"{fb['feedback_id']} is not eligible")
            k = key(fb["inputs"])
            if k in existing:
                excluded[fb["feedback_id"]] = "duplicate_of_parent_row"  # no cross-split copy
            else:
                by_input.setdefault(k, []).append(fb)
        kept: list[dict[str, Any]] = []
        for group in by_input.values():
            labels = {canonical_hash(fb["expected"]) for fb in group}
            if len(labels) > 1:
                for fb in group:
                    excluded[fb["feedback_id"]] = "conflicting_labels"
                continue
            first, *rest = sorted(group, key=lambda fb: fb["feedback_id"])
            kept.append(first)
            for fb in rest:
                excluded[fb["feedback_id"]] = "duplicate_feedback"
        kept.sort(key=lambda fb: fb["feedback_id"])
        if not kept:
            raise _ReoptBlocked("no_new_examples", "every frozen example duplicates a parent row")
        if ds.id_column is not None and ds.column(ds.id_column).type.value != "string":
            raise _ReoptBlocked(
                "unsupported_id_column", "new rows need string ids in the id column"
            )
        columns = [c.name for c in ds.columns]
        lines = [json.dumps({c: row[c] for c in columns}, ensure_ascii=False) for _, row in rows]
        for fb in kept:
            new = {c: None for c in columns}
            new.update(fb["inputs"])
            new.update(fb["expected"])
            if ds.id_column is not None:
                new[ds.id_column] = fb["feedback_id"]
            lines.append(json.dumps({c: new[c] for c in columns}, ensure_ascii=False))
        payload = ("\n".join(lines) + "\n").encode()
        upload, _ = self.datasets.upload(
            parent.project_id, payload, "jsonl", filename=f"{ds.dataset_id}-{trigger_id}.jsonl"
        )
        record, _ = self.datasets.register(
            upload.upload_id,
            RegisterDataset(
                dataset_id=ds.dataset_id,
                name=parent.spec.name,
                input_columns=ds.input_columns,
                target_columns=ds.target_columns,
                context_columns=ds.context_columns,
                row_ids="column" if parent.row_id_source is RowIdSource.COLUMN else "generated",
                id_column=ds.id_column,
            ),
        )
        new_spec = record.spec
        if record.dataset_version <= parent.dataset_version:
            raise MonitoringIntegrityError("re-optimization must create a NEW dataset version")
        if [(c.name, c.type) for c in new_spec.columns] != [(c.name, c.type) for c in ds.columns]:
            raise _ReoptBlocked("dataset_schema_changed", "new rows change a column's type")
        new_ids = self.datasets.repo.get_row_ids(ds.dataset_id, record.dataset_version)
        parent_ids = set(self.datasets.repo.get_row_ids(ds.dataset_id, ds.dataset_version))
        added = sorted(set(new_ids) - parent_ids)
        if not parent_ids <= set(new_ids) or len(added) != len(kept):
            raise MonitoringIntegrityError("the new version does not extend its parent's rows")
        plan = parent_splits.plan or policy.new_row_splits
        assigned = assign_new_rows(added, plan)
        splits = inherit_splits(parent_splits, record.identity_hash, assigned)
        stored, _ = self.datasets.repo.put_splits(
            SplitsRecord(
                dataset_id=ds.dataset_id,
                dataset_version=record.dataset_version,
                splits_hash=splits.identity_hash,
                splits=splits,
                sizes={s.role.value: len(s.row_ids) for s in splits.splits},
                created_at=self.datasets.clock(),
            )
        )
        derivation = {
            "record_schema": REOPT_SCHEMA,
            "parent": {
                "dataset_id": ds.dataset_id,
                "dataset_version": ds.dataset_version,
                "dataset_hash": ds.identity_hash,
                "splits_hash": parent_splits.identity_hash,
            },
            "new": {
                "dataset_id": ds.dataset_id,
                "dataset_version": record.dataset_version,
                "dataset_hash": record.identity_hash,
                "content_hash": new_spec.content_hash,
                "upload_id": upload.upload_id,
                "splits_hash": stored.splits_hash,
            },
            "split_rule": "parent rows keep their parent role; new rows: seeded_hash/1 "
            f"(seed={plan.seed}, validation_bps={plan.validation_bps}, test_bps={plan.test_bps})",
            "new_rows": {role.value: list(ids) for role, ids in sorted(assigned.items())},
            "frozen_feedback_ids": sorted(frozen),
            "kept_feedback_ids": [fb["feedback_id"] for fb in kept],
            "excluded": dict(sorted(excluded.items())),
        }
        return {
            "dataset_id": ds.dataset_id,
            "dataset_version": record.dataset_version,
            "splits_hash": stored.splits_hash,
            "spec": new_spec,
            "splits": stored.splits,
            "upload_id": upload.upload_id,
            "derivation": derivation,
        }

    def _feedback_row(self, feedback_id: str) -> FeedbackRow:
        row = self.store.feedback(feedback_id)
        if row is None:
            raise MonitoringIntegrityError(f"frozen feedback {feedback_id} is missing")
        return row

    def _challenger_definition(
        self, ctx: _Context, decision: DecisionRow, built: Mapping[str, Any]
    ) -> ExperimentJobDefinition:
        """The SAME TaskContract (objective, constraints, evaluator, workflow vocabulary) and
        model authority, re-bound to the new dataset version; the SAME plan (model, budget,
        seeds, trials), always comparing strong Fixed vs Random vs ACO on equal terms."""
        assert ctx.source_definition is not None and self.jobs is not None
        source = ctx.source_definition
        contract = TaskContract.model_validate(
            ctx.contract.model_dump(mode="json")
            | {"dataset": built["spec"].model_dump(mode="json")}
        )
        plan = source.plan.model_validate(
            source.plan.model_dump(mode="json")
            | {"strategies": [s.value for s in FAIR_STRATEGIES], "dataset_manifest_hash": None}
        )
        runtime = self.jobs.runtime
        return ExperimentJobDefinition(
            contract=contract,
            splits=built["splits"],
            plan=plan,
            synthetic=bool(runtime is not None and runtime.synthetic) or source.synthetic,
            workers=source.workers,
            provenance={
                "dataset_id": built["dataset_id"],
                "dataset_version": built["dataset_version"],
                "splits_hash": built["splits_hash"],
                "upload_id": built["upload_id"],
                "reoptimization": {
                    "trigger_id": decision.trigger_id,
                    "policy_id": decision.policy_id,
                    "workflow_version": ctx.version_id,
                    "lineage_id": ctx.lineage_id,
                    **{
                        k: ctx.pins[k]
                        for k in (
                            "champion_id",
                            "artifact_id",
                            "provenance_id",
                            "experiment_id",
                            "source_job_id",
                            "contract_hash",
                            "model_hash",
                        )
                    },
                    "parent_dataset_version": ctx.contract.dataset.dataset_version,
                },
            },
        )

    def challenger(self, trigger_id: str) -> ReoptimizationView:
        decision = self.store.decision(trigger_id)
        if decision is None:
            raise TriggerNotFound(f"no trigger decision {trigger_id}")
        if decision.outcome is not TriggerOutcome.TRIGGERED:
            raise NotTriggered(f"{trigger_id} is {decision.outcome.value}")
        body = json.loads(decision.decision_json)
        row = self.store.reoptimization(trigger_id)
        job_state = None
        strategies: list[str] = []
        if row is not None and row.job_id is not None and self.jobs is not None:
            try:
                view = self.jobs.get(row.job_id)
                job_state = view.state.value
                job = self.jobs.store.get_job(row.job_id)
                assert job is not None
                definition = ExperimentJobDefinition.model_validate_json(job.definition_json)
                strategies = [s.value for s in definition.plan.strategies]
            except JobNotFound:
                job_state = None
        dataset = None
        detail = None
        if row is not None and row.detail is not None:
            parsed = json.loads(row.detail)
            if row.state is ReoptState.FAILED:
                detail = parsed.get("reason")
            else:
                dataset = parsed
        return ReoptimizationView(
            record_schema=REOPT_SCHEMA,
            trigger_id=trigger_id,
            state=None if row is None else row.state,
            workflow_version=decision.workflow_version,
            lineage_id=decision.lineage_id,
            production=body["links"],
            dataset=dataset,
            challenger_job_id=None if row is None else row.job_id,
            challenger_job_state=job_state,
            strategies=strategies,
            deployed=False,
            detail=detail,
        )


# -- helpers ------------------------------------------------------------------------------------
def _ordinal(v: Any) -> float | None:
    try:
        return float(date.fromisoformat(v).toordinal())
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def assign_new_rows(row_ids: Sequence[str], plan: SplitPlan) -> dict[SplitRole, tuple[str, ...]]:
    """Seeded assignment of ONLY the new rows (``seeded_hash/1`` ordering and fractions)."""
    ordered = sorted(row_ids, key=lambda r: (split_order_key(plan.seed, r), r))
    n = len(ordered)
    n_test = n * plan.test_bps // BPS
    n_val = n * plan.validation_bps // BPS
    out = {
        SplitRole.TEST: tuple(sorted(ordered[:n_test])),
        SplitRole.VALIDATION: tuple(sorted(ordered[n_test : n_test + n_val])),
        SplitRole.OPTIMIZATION: tuple(sorted(ordered[n_test + n_val :])),
    }
    return {role: rows for role, rows in out.items() if rows}


def inherit_splits(
    parent: DatasetSplits, dataset_hash: str, added: Mapping[SplitRole, Sequence[str]]
) -> DatasetSplits:
    """Every parent row keeps its parent role (a parent test row can never become an
    optimization or validation row); new rows join the role they were assigned."""
    splits = []
    for role in (SplitRole.OPTIMIZATION, SplitRole.VALIDATION, SplitRole.TEST):
        old = parent.split(role)
        rows = (*(old.row_ids if old else ()), *added.get(role, ()))
        if rows:
            splits.append(
                DatasetSplit(
                    split_id=old.split_id if old else role.value, role=role, row_ids=tuple(rows)
                )
            )
    return DatasetSplits(
        dataset_hash=dataset_hash, method=SplitMethod.EXPLICIT, splits=tuple(splits)
    )


def _rejected(record: Mapping[str, Any]) -> bool:
    failure = record.get("failure")
    return failure is not None and failure.get("kind") == INPUT_SCHEMA_REJECTED


def _summarize(
    ctx: _Context,
    start: datetime,
    end: datetime,
    records: Sequence[Mapping[str, Any]],
    feedback: Sequence[Mapping[str, Any]],
) -> MonitoringSummary:
    for r in records:  # never mix versions: a record of another version is an integrity error
        if r["workflow_version"] != ctx.version_id:
            raise MonitoringIntegrityError("an aggregate may only read its own version's records")
    rejected = [r for r in records if _rejected(r)]
    served = [r for r in records if not _rejected(r)]  # the workflow actually ran
    n = len(served)
    failed = [r for r in served if r["status"] == "FAILED"]
    usage = [r["usage"] for r in served]
    latency = [u["latency_s"] for u in usage]
    calls = [u["model_calls"] for u in usage]
    authoritative = bool(usage) and all(u["cost_authoritative"] for u in usage)
    costs = [u["cost"] for u in usage] if authoritative else []
    labelled = [f for f in feedback if f["labelled"]]
    qualities = [f["evaluation"]["quality"] for f in labelled]
    evidence = {
        "inference_ids": sorted(r["inference_id"] for r in records),
        "feedback_ids": sorted(f["feedback_id"] for f in feedback),
    }
    return MonitoringSummary(
        record_schema=SUMMARY_SCHEMA,
        workflow_version=ctx.version_id,
        lineage_id=ctx.lineage_id,
        window={"since": _stamp(start), "until": _stamp(end)},
        pins=ctx.pins,
        inferences={
            "count": n,
            "succeeded": n - len(failed),
            "failed": len(failed),
            "failure_rate": _r(len(failed) / n) if n else None,
            "failure_kinds": dict(sorted(Counter(r["failure"]["kind"] for r in failed).items())),
            # requests the pinned input schema rejected before any model call (#30 telemetry):
            # a client / schema signal, never a workflow failure, so outside failure_rate
            "rejected_inputs": len(rejected),
        },
        latency_s={
            "mean": _r(_mean(latency)),
            "p50": _r(percentile(latency, 0.5)),
            "p95": _r(percentile(latency, 0.95)),
        },
        model_calls={"total": sum(calls), "mean": _r(_mean(calls))},
        tokens={
            k: {"total": sum(u[k] for u in usage), "mean": _r(_mean([u[k] for u in usage]))}
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        cost={
            "authoritative": authoritative,
            "total": _r(sum(costs)) if authoritative else None,
            "mean": _r(_mean(costs)) if authoritative else None,
        },
        feedback={
            "count": len(feedback),
            "labelled": len(labelled),
            "mean_quality": _r(_mean(qualities)),
            "pass_rate": _r(_mean([1.0 if f["evaluation"]["passed"] else 0.0 for f in labelled])),
        },
        evidence=evidence | {"evidence_hash": canonical_hash(evidence)},
    )


def _fire(
    policy: MonitoringPolicy,
    summary: MonitoringSummary,
    drift: DriftReport,
    reference: Mapping[str, float | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(reasons that fired, every rule checked)``. Each rule needs its minimum sample count:
    below it the rule reports ``insufficient_evidence`` and cannot fire."""
    t = policy.trigger
    reasons: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    n = summary.inferences["count"]
    labelled = summary.feedback["labelled"]

    def check(rule: str, observed, threshold, samples, minimum, fired: bool) -> None:
        enough = samples >= minimum
        item = {
            "rule": rule,
            "observed": observed,
            "threshold": threshold,
            "samples": samples,
            "min_samples": minimum,
            "status": "FIRED" if enough and fired else "OK" if enough else "INSUFFICIENT_EVIDENCE",
        }
        checks.append(item)
        if item["status"] == "FIRED":
            reasons.append({"code": rule, **{k: v for k, v in item.items() if k != "status"}})

    if t.min_quality is not None:
        q = summary.feedback["mean_quality"]
        check(
            "quality_below_threshold",
            q,
            t.min_quality,
            labelled,
            t.min_labelled,
            q is not None and q < t.min_quality,
        )
    if t.max_failure_rate is not None:
        fr = summary.inferences["failure_rate"]
        check(
            "failure_rate_above_threshold",
            fr,
            t.max_failure_rate,
            n,
            t.min_inferences,
            fr is not None and fr > t.max_failure_rate,
        )
    if t.on_material_drift:
        data = [
            f for f in drift.features if f["status"] in ("DRIFTED", "STABLE")
        ]  # features with enough evidence
        samples = min((f["observed_n"] for f in data), default=0)
        check(
            "material_data_drift",
            drift.drifted_features,
            policy.drift.material_features,
            samples,
            policy.drift.min_samples,
            drift.material,
        )
    if t.on_quality_drift:
        q = drift.quality
        check(
            "quality_drift",
            q["value"],
            q["threshold"],
            q["observed_n"],
            q["min_samples"],
            q["status"] == "DRIFTED",
        )
    if t.max_p95_latency_ratio is not None:
        ref = reference.get("p95_latency_s")
        p95 = summary.latency_s["p95"]
        ratio = _r(p95 / ref) if ref and p95 is not None else None
        check(
            "latency_regression",
            ratio,
            t.max_p95_latency_ratio,
            n if ref else 0,
            t.min_inferences,
            ratio is not None and ratio > t.max_p95_latency_ratio,
        )
    if t.max_mean_cost_ratio is not None:
        ref_cost = reference.get("mean_cost")
        mean = summary.cost["mean"]
        ratio = _r(mean / ref_cost) if ref_cost and mean is not None else None
        check(
            "cost_regression",
            ratio,
            t.max_mean_cost_ratio,
            n if summary.cost["authoritative"] and ref_cost else 0,
            t.min_inferences,
            ratio is not None and ratio > t.max_mean_cost_ratio,
        )
    return reasons, checks


def _feedback_view(row: FeedbackRow) -> FeedbackView:
    body = json.loads(row.record_json)
    return FeedbackView(
        record_schema=body["record_schema"],
        feedback_id=row.feedback_id,
        inference_id=row.inference_id,
        workflow_version=row.workflow_version,
        lineage_id=row.lineage_id,
        labelled=row.labelled,
        lineage=body["lineage"],
        evaluation=body["evaluation"],
        inference_created_at=row.inference_created_at,
        received_at=row.received_at,
        untrusted_metadata=body["untrusted_metadata"],
        metadata_trusted=False,
    )


def _decision_view(row: DecisionRow, created: bool) -> TriggerDecisionView:
    body = json.loads(row.decision_json)
    return TriggerDecisionView(
        record_schema=body["record_schema"],
        trigger_id=row.trigger_id,
        workflow_version=row.workflow_version,
        lineage_id=row.lineage_id,
        policy_id=row.policy_id,
        outcome=row.outcome,
        reasons=body["reasons"],
        checks=body["checks"],
        evidence=body["evidence"],
        frozen_feedback_ids=body["frozen_feedback_ids"],
        links=body["links"],
        created_at=row.created_at,
        created=created,
    )


__all__ = [
    "DriftConfig",
    "FeedbackRequest",
    "MonitoringError",
    "MonitoringPolicy",
    "ProductionMonitor",
    "TriggerRules",
    "assign_new_rows",
    "inherit_splits",
    "ks_statistic",
    "percentile",
    "total_variation",
]
