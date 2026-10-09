"""Champion promotion across dataset versions (prerequisite of #31).

    dataset v1 champion  ──(same LINEAGE identity)──►  dataset v2 challenger
      validation (v2) picks ONE challenger -> the v2 test split opens once -> the challenger AND
      the current incumbent genome run on exactly those rows / trials / seeds -> #25 gate

Every model here is a TEST DOUBLE (``Rigged``: a run passes iff a rule over (genome, row) says
so). Nothing here is a benchmark result.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from core.constraints import ConstraintChecker, ConstraintLimits
from core.dataset import (
    DatasetSplit,
    DatasetSplits,
    SplitMethod,
    SplitRole,
    SplitUse,
)
from core.evaluation_spec import EvaluationSpec
from core.objective import ObjectiveMode, ObjectiveSpec
from core.task_spec import AnswerField, AnswerSchema, FieldType
from experiments.budget_ledger import ExperimentBudget
from experiments.contract_run import contract_suite
from experiments.jobs import ExperimentJobDefinition, HeldoutBinding, JobBinding, JobWorker
from experiments.optimization_experiment import Strategy
from experiments.promotion import (
    CHAMPION_SCHEMA,
    CONTEXT_SCHEMA,
    DECISION_SCHEMA,
    LEGACY_CHAMPION_SCHEMA,
    PROVENANCE_DECISION_SCHEMA,
    Decision,
    EvidenceMismatch,
    IncompatibleIncumbent,
    NotPromotable,
    ReasonCode,
    lineage_identity,
)
from experiments.synthetic import SYNTHETIC_VERSION
from ingestion.parse import sha256_bytes
from store.champions import SCHEMA as CHAMPION_STORE_SCHEMA
from store.champions import SQLiteChampionStore
from tests.test_champion_promotion import EVERYTHING, Rigged, World
from tests.test_champion_promotion import definition as v1_definition
from tests.test_contract_runtime import CAPITALS, DATA, qa_contract
from tests.test_durable_jobs import LEASE
from tests.test_optimization_experiment import LIMITS, MODEL, plan

LINEAGE = "capitals.capitals"
NEW_ROWS = [("q7", "Peru", "Lima"), ("q8", "Chile", "Santiago")]
DATA2 = (
    DATA
    + "".join(
        json.dumps(
            {
                "qid": qid,
                "question": f"Which city is the capital of {country}?",
                "passage": f"{city} is the capital of {country}. It is a large city.",
                "answer": city,
            }
        )
        + "\n"
        for qid, country, city in NEW_ROWS
    ).encode()
)
V1_TEST = ("q6",)
V2_TEST = ("q6", "q7", "q8")  # the parent test row keeps its role; the new rows are test rows
BYTES = {sha256_bytes(DATA): DATA, sha256_bytes(DATA2): DATA2}


# -- dataset v2 of the same task ----------------------------------------------------------------
def v2_contract(**changes):
    base = qa_contract()
    dataset = base.dataset.model_copy(
        update={
            "dataset_version": 2,
            "content_hash": sha256_bytes(DATA2),
            "row_count": len(CAPITALS) + len(NEW_ROWS),
        }
    )
    changes.setdefault("constraints", ConstraintLimits(**LIMITS))
    return qa_contract(data=DATA2, dataset=dataset, **changes)


def v2_definition(strategies=(Strategy.RANDOM,), *, plan_changes=None, **changes):
    contract = v2_contract(**changes)
    splits = DatasetSplits(
        dataset_hash=contract.dataset.identity_hash,
        method=SplitMethod.EXPLICIT,
        splits=(
            DatasetSplit(split_id="opt", role=SplitRole.OPTIMIZATION, row_ids=("q1", "q2", "q3")),
            DatasetSplit(split_id="val", role=SplitRole.VALIDATION, row_ids=("q4", "q5")),
            DatasetSplit(split_id="test", role=SplitRole.TEST, row_ids=V2_TEST),
        ),
    )
    return ExperimentJobDefinition(
        contract=contract,
        splits=splits,
        plan=plan(
            ExperimentBudget(max_candidate_evaluations=4),
            seeds=(0,),
            strategies=strategies,
            trials=1,
            model=MODEL,
            **(plan_changes or {}),
        ),
        synthetic=True,
    )


# -- test doubles -------------------------------------------------------------------------------
class AnyVersionRuntime:
    """TEST DOUBLE runtime: ``GateRuntime`` for ANY stored dataset version (bytes by hash)."""

    synthetic = True

    def __init__(self, backend: Rigged) -> None:
        self.backend, self.pricing, self.proof = backend, None, None
        self.heldout_binds: list[int] = []

    def _suite(self, d: ExperimentJobDefinition):
        suite, _ = contract_suite(d.contract, d.splits, BYTES[d.contract.dataset.content_hash])
        return suite

    def bind(self, d: ExperimentJobDefinition) -> JobBinding:
        return JobBinding(
            suite=self._suite(d),
            evaluate=self.backend,
            checker=ConstraintChecker(),
            evaluator_version=SYNTHETIC_VERSION,
        )

    def bind_heldout(self, d: ExperimentJobDefinition) -> HeldoutBinding:
        self.heldout_binds.append(d.contract.dataset.dataset_version)
        tasks = self._suite(d).tasks_for(SplitRole.TEST, SplitUse.PROMOTION_GATE)
        return HeldoutBinding(tasks, self.backend, SYNTHETIC_VERSION, None)


class ModelSwap(Rigged):
    """TEST DOUBLE: ``Rigged`` whose runs report another model_hash (another model binding)."""

    def __init__(self, passes, model_hash: str) -> None:
        super().__init__(passes)
        inner = self._evaluate

        def evaluate(genome, task, trial, seed):
            run = inner(genome, task, trial, seed)
            versions = run.execution.key.versions.model_copy(update={"model_hash": model_hash})
            key = run.execution.key.model_copy(update={"versions": versions})
            return run.model_copy(
                update={"execution": run.execution.model_copy(update={"key": key})}
            )

        self._evaluate = evaluate


class VWorld(World):
    """``World`` (#24 jobs + #25 promotions) whose runtime binds dataset v1 AND v2."""

    def __init__(self, root: Path, backend: Rigged, *, name: str = "p") -> None:
        super().__init__(root, backend, name=name)
        self.runtime = AnyVersionRuntime(backend)
        self.jobs.runtime = self.runtime
        self.worker = JobWorker(
            self.jobs.store,
            self.runtime,
            worker_id=name,
            lease_s=LEASE,
            clock=self.clock,
            heartbeat=False,
        )
        self.promotions.runtime = self.runtime

    def champion(self, d: ExperimentJobDefinition) -> dict[str, Any]:
        view = self.promote(self.experiment(d))
        assert view.decision is Decision.PROMOTED, view.record
        return view.record


def calls_on(backend: Rigged, rows) -> list[tuple]:
    return [k for k in backend.calls if k[1] in set(rows)]


def subjects(w: World, promotion_id: str) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {"challenger": [], "incumbent": []}
    for subject, a in w.store.attempts(promotion_id):
        out[subject].append((a.task_id, a.trial, a.run_seed))
    return {k: sorted(v) for k, v in out.items()}


# -- 1. one lineage across dataset versions -----------------------------------------------------
def test_a_v1_champion_and_a_v2_challenger_share_the_lineage(tmp_path):
    rigged = Rigged(EVERYTHING)
    w = VWorld(tmp_path, rigged)
    first = w.champion(v1_definition(strategies=(Strategy.FIXED,)))
    job = w.experiment(v2_definition())
    view = w.promote(job)  # the DEFAULT lineage: dataset_id.task_id

    rec = view.record
    assert view.lineage_id == LINEAGE and view.decision is Decision.PROMOTED, rec
    assert rec["schema"] == DECISION_SCHEMA
    assert rec["lineage_identity"]["lineage_hash"] == first["lineage_identity"]["lineage_hash"]
    for key in ("dataset_version", "dataset_hash", "splits_hash", "test_row_ids"):
        assert key not in rec["lineage_identity"]  # which rows is never lineage identity
    ctx = rec["evaluation_context"]
    assert ctx["schema"] == CONTEXT_SCHEMA
    assert (ctx["dataset_id"], ctx["dataset_version"]) == ("capitals", 2)
    assert ctx["test_row_ids"] == list(V2_TEST)
    assert ctx["dataset_hash"] == v2_contract().dataset.identity_hash
    assert first["evaluation_context"]["dataset_version"] == 1
    assert first["evaluation_context"]["test_row_ids"] == list(V1_TEST)
    # every champion still pins the dataset version / provenance that selected it
    history = w.promotions.history(LINEAGE).champions
    assert [c.version for c in history] == [1, 2]
    assert [c.record["compatibility"]["dataset_version"] for c in history] == [1, 2]
    assert [c.record["schema"] for c in history] == [CHAMPION_SCHEMA, CHAMPION_SCHEMA]
    assert len({c.record["lineage_identity"]["lineage_hash"] for c in history}) == 1
    assert {w.store.champion(c.champion_id).compat_hash for c in history} == {
        first["lineage_identity"]["lineage_hash"]
    }
    assert history[1].record["provenance"]["job_id"] == job
    assert history[1].record["provenance"]["artifact_id"] == rec["artifact"]["artifact_id"]
    w.promotions.verify(view.promotion_id)


def test_the_incumbent_is_re_evaluated_on_the_v2_rows_trials_and_seeds(tmp_path):
    box: dict[str, str] = {}

    def passes(g: str, row: str) -> bool:  # the incumbent never learned the new rows
        return not (g == box.get("incumbent") and row in ("q7", "q8"))

    rigged = Rigged(passes)
    w = VWorld(tmp_path, rigged)
    first = w.champion(v1_definition(strategies=(Strategy.FIXED,)))
    box["incumbent"] = first["challenger"]["genome_hash"]
    old_heldout = w.promotions.current(LINEAGE).record["heldout"]
    assert old_heldout["pass_rate"] == 1.0 and old_heldout["row_ids"] == list(V1_TEST)

    view = w.promote(w.experiment(v2_definition()))

    rec = view.record
    inc, ch = rec["heldout"]["incumbent"], rec["heldout"]["challenger"]
    assert ch["genome_hash"] != inc["genome_hash"] == box["incumbent"]
    # identical v2 rows / trials / seeds for both subjects, as pinned in the context
    ran = subjects(w, view.promotion_id)
    assert ran["challenger"] == ran["incumbent"]
    assert ran["incumbent"] == sorted(tuple(x) for x in rec["evaluation_context"]["seeds"])
    assert inc["row_ids"] == ch["row_ids"] == list(V2_TEST)
    # the incumbent's OLD held-out score (1.0 on q6) is never reused: it scored 1/3 on v2
    assert inc["pass_rate"] == pytest.approx(1 / 3) and inc != old_heldout
    own = {a.attempt_id for s, a in w.store.attempts(view.promotion_id) if s == "incumbent"}
    assert set(inc["attempt_ids"]) == own and not set(inc["attempt_ids"]) & set(
        old_heldout["attempt_ids"]
    )
    assert view.decision is Decision.PROMOTED and rec["comparison"]["relation"] == "better"
    # the incumbent received only v2 test targets (no v1 training / validation rows)
    inc_calls = [k for k in rigged.calls if k[0] == box["incumbent"]]
    after_v1 = inc_calls[-len(V2_TEST) :]
    assert sorted(k[1] for k in after_v1) == list(V2_TEST)
    assert w.runtime.heldout_binds == [1, 2]
    w.promotions.verify(view.promotion_id)


def test_a_v2_challenger_worse_than_the_re_evaluated_incumbent_is_rejected(tmp_path):
    box: dict[str, str] = {}

    def passes(g: str, row: str) -> bool:  # the challenger fails a NEW test row
        return g == box.get("incumbent") or row != "q8"

    rigged = Rigged(passes)
    w = VWorld(tmp_path, rigged)
    first = w.champion(v1_definition(strategies=(Strategy.FIXED,)))
    box["incumbent"] = first["challenger"]["genome_hash"]
    job = w.experiment(v2_definition())
    view = w.promote(job)
    assert view.decision is Decision.REJECTED
    assert ReasonCode.REGRESSION in view.record["reason_codes"]
    assert w.promotions.current(LINEAGE).version == 1
    # a rejected challenger never brings candidate #2 to the test split
    tested = {k[0] for k in calls_on(rigged, ("q7", "q8"))}
    assert tested == {view.challenger["genome_hash"], box["incumbent"]}
    calls = list(rigged.calls)
    assert w.promote(job) == view
    with pytest.raises(NotPromotable):
        w.promote(job, lineage="capitals.retry")
    assert rigged.calls == calls
    w.promotions.verify(view.promotion_id)


# -- 2. any change of task semantics fails closed ----------------------------------------------
QUALITY_FLOOR = ConstraintLimits(**LIMITS, minimum_quality=0.5)
OPTIONAL_PASSAGE = AnswerSchema(
    fields=(
        AnswerField(name="question", type=FieldType.STRING),
        AnswerField(name="passage", type=FieldType.STRING, required=False),
    )
)


@pytest.mark.parametrize(
    "changes",
    [
        {"input_schema": OPTIONAL_PASSAGE},
        {"evaluation": EvaluationSpec(evaluator="exact_match", config={"case_sensitive": True})},
        {
            "objective": ObjectiveSpec(mode=ObjectiveMode.MINIMIZE_LATENCY),
            "constraints": QUALITY_FLOOR,
        },
        {"constraints": ConstraintLimits(**{**LIMITS, "maximum_retries": 1})},
        {"instructions": "Name the capital city."},
    ],
    ids=["schema", "evaluator", "objective", "constraints", "instructions"],
)
def test_a_changed_task_semantics_fails_closed(tmp_path, changes):
    rigged = Rigged(EVERYTHING)
    w = VWorld(tmp_path, rigged)
    # (the objective case keeps the constraints equal: only the objective differs)
    same = {"constraints": QUALITY_FLOOR} if "objective" in changes else {}
    w.champion(v1_definition(strategies=(Strategy.FIXED,), **same))
    job = w.experiment(v2_definition(**changes))
    before = list(rigged.calls)

    with pytest.raises(IncompatibleIncumbent):
        w.promote(job)

    assert w.store.for_job(job) is None  # nothing recorded: its test split is still closed
    assert rigged.calls == before and calls_on(rigged, ("q7", "q8")) == []
    assert w.promotions.current(LINEAGE).version == 1


def test_a_changed_model_hash_fails_closed(tmp_path):
    w = VWorld(tmp_path, Rigged(EVERYTHING))
    w.champion(v1_definition(strategies=(Strategy.FIXED,)))
    other = VWorld(tmp_path, ModelSwap(EVERYTHING, "m-2"), name="q")
    job = other.experiment(v2_definition(plan_changes={"expected_model_hash": "m-2"}))
    with pytest.raises(IncompatibleIncumbent, match="model_hash"):
        other.promote(job)
    assert other.store.for_job(job) is None


# -- 3. the new test split stays sealed ---------------------------------------------------------
def test_the_v2_test_split_reaches_only_the_challenger_and_the_pinned_incumbent(tmp_path):
    rigged = Rigged(EVERYTHING)
    w = VWorld(tmp_path, rigged)
    first = w.champion(v1_definition(strategies=(Strategy.FIXED,)))
    calls = len(rigged.calls)
    job = w.experiment(v2_definition(strategies=(Strategy.FIXED, Strategy.RANDOM, Strategy.ACO)))
    experiment_rows = {k[1] for k in rigged.calls[calls:]}
    assert experiment_rows <= {"q1", "q2", "q3", "q4", "q5"}  # optimizer + validation only
    assert not experiment_rows & set(V2_TEST)

    view = w.promote(job)
    tested = {k[0] for k in calls_on(rigged, V2_TEST)[1:]}  # [0]: v1's own test run on q6
    expected = {view.challenger["genome_hash"], first["challenger"]["genome_hash"]}
    assert tested == expected  # exactly one challenger + the pinned comparison subject
    candidates = view.record["selection"]["candidates"]
    validated = [c["candidate_id"] for c in candidates if c["status"] == "VALIDATED"]
    assert validated == [view.record["selection"]["challenger"]]
    others = {c["genome_hash"] for c in candidates} - expected
    assert not others & {k[0] for k in calls_on(rigged, V2_TEST)}  # runners-up never ran there
    assert view.record["incumbent"]["champion_id"] == first["champion"]["champion_id"]


def test_mixed_test_rows_are_refused(tmp_path):
    """Evidence for a held-out row outside the pinned context can never decide the gate."""
    rigged = Rigged(EVERYTHING)
    w = VWorld(tmp_path, rigged)
    w.champion(v1_definition(strategies=(Strategy.FIXED,)))
    view = w.promote(w.experiment(v2_definition()))
    row = w.store.get(view.promotion_id)
    pinned = json.loads(row.selection_json)
    tampered = json.loads(json.dumps(pinned))
    tampered["evaluation_context"]["test_row_ids"] = ["q6"]  # the v1 rows
    from experiments.promotion import heldout_record

    d = ExperimentJobDefinition.model_validate_json(
        w.jobs.store.get_job(row.job_id).definition_json
    )
    with pytest.raises(EvidenceMismatch):
        heldout_record(d, tampered, w.store.attempts(view.promotion_id))
    tampered = json.loads(json.dumps(pinned))
    tampered["evaluation_context"]["seeds"][0][2] += 1
    with pytest.raises(EvidenceMismatch):
        heldout_record(d, tampered, w.store.attempts(view.promotion_id))


# -- 4. concurrency: a superseded incumbent is never overwritten --------------------------------
def test_a_cross_version_promotion_never_overwrites_a_newer_champion(tmp_path):
    box: dict[str, Any] = {}

    def hook(n, key):
        if key[1] == "q8" and not box.get("fired"):
            box["fired"] = True  # while the v2 gate runs, another champion is promoted
            box["b"] = box["other"].promote(box["job_b"])

    rigged = Rigged(EVERYTHING, hook=hook)
    w = VWorld(tmp_path, rigged, name="a")
    w.champion(v1_definition(strategies=(Strategy.FIXED,)))
    job_v2 = w.experiment(v2_definition())
    box["job_b"] = w.experiment(v1_definition(strategies=(Strategy.ACO,)))
    box["other"] = VWorld(tmp_path, rigged, name="b").promotions

    view = w.promote(job_v2)

    assert box["b"].decision is Decision.PROMOTED
    assert view.decision is Decision.REJECTED
    assert view.record["gate"]["decision"] == Decision.PROMOTED
    assert view.record["reason_codes"][-1] == ReasonCode.INCUMBENT_SUPERSEDED
    current = w.promotions.current(LINEAGE)
    assert current.promotion_id == box["b"].promotion_id and current.version == 2
    w.promotions.verify(view.promotion_id)


# -- 5. backward compatibility ------------------------------------------------------------------
def legacy_promotion(monkeypatch, w: World, job: str):
    """Promote ``job`` exactly as #25/#26 did (decision/2, champion/1, dataset-scoped
    compat_hash): the pin is stored without the lineage identity / evaluation context."""
    real = SQLiteChampionStore.open

    def open_(self, promotion_id, job_id, lineage_id, selection_json, *rest):
        pinned = json.loads(selection_json)
        pinned.pop("lineage", None)
        pinned.pop("evaluation_context", None)
        if pinned["incumbent"] is not None:
            for key in ("lineage_hash", "lineage_source"):
                pinned["incumbent"].pop(key, None)
        legacy = json.dumps(pinned, sort_keys=True, separators=(",", ":"))
        return real(self, promotion_id, job_id, lineage_id, legacy, *rest)

    monkeypatch.setattr(SQLiteChampionStore, "open", open_)
    try:
        return w.promote(job)
    finally:
        monkeypatch.undo()


def schema_1_store(path: Path) -> None:
    """An empty champions.sqlite3 exactly as store schema 1 created it."""
    conn = sqlite3.connect(path)
    for statement in CHAMPION_STORE_SCHEMA:
        conn.execute(
            statement.replace(
                ",\n        lineage_hash TEXT                   -- stable lineage "
                "identity (NULL: champion/1 only)",
                "",
            ).replace("        lineage_hash TEXT,                  -- NULL for a champion/1\n", "")
        )
    conn.execute("INSERT INTO meta(key, value) VALUES ('champions_schema_version', '1')")
    conn.commit()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(champions)")}
    assert "lineage_hash" not in cols
    conn.close()


def test_schema_1_records_migrate_and_a_champion_v1_lineage_crosses_versions(tmp_path, monkeypatch):
    schema_1_store(tmp_path / "champions.sqlite3")
    rigged = Rigged(EVERYTHING)
    w = VWorld(tmp_path, rigged)  # opening it migrates 1 -> 2: columns added, nothing changed
    conn = sqlite3.connect(tmp_path / "champions.sqlite3")
    assert conn.execute("SELECT value FROM meta").fetchone() == ("2",)
    conn.close()

    old = legacy_promotion(monkeypatch, w, w.experiment(v1_definition((Strategy.FIXED,))))
    assert old.record["schema"] == PROVENANCE_DECISION_SCHEMA and "lineage_identity" not in (
        old.record
    )
    champ = w.store.current(LINEAGE)
    legacy_record = json.loads(champ.record_json)
    assert legacy_record["schema"] == LEGACY_CHAMPION_SCHEMA
    assert champ.compat_hash == legacy_record["compatibility"]["compat_hash"]
    assert champ.lineage_hash is None
    w.promotions.verify(old.promotion_id)  # an old decision verifies exactly as written

    view = w.promote(w.experiment(v2_definition()))
    assert view.decision is Decision.PROMOTED, view.record
    assert view.record["incumbent"]["lineage_source"] == "derived_from_champion_v1_experiment"
    d1 = ExperimentJobDefinition.model_validate_json(
        w.jobs.store.get_job(legacy_record["provenance"]["job_id"]).definition_json
    )
    assert (
        view.record["incumbent"]["lineage_hash"] == view.record["lineage_identity"]["lineage_hash"]
    )
    assert d1.contract.dataset.dataset_version == 1
    new = w.store.current(LINEAGE)
    assert new.version == 2 and new.lineage_hash == view.record["lineage_identity"]["lineage_hash"]
    conn = sqlite3.connect(tmp_path / "champions.sqlite3")
    lineage_row = conn.execute(
        "SELECT compat_hash, lineage_hash FROM champion_lineages WHERE lineage_id=?", (LINEAGE,)
    ).fetchone()
    conn.close()
    assert lineage_row == (champ.compat_hash, new.lineage_hash)  # adopted once, legacy kept
    assert json.loads(w.store.champion(champ.champion_id).record_json) == legacy_record
    w.promotions.verify(old.promotion_id)
    w.promotions.verify(view.promotion_id)


def test_same_dataset_promotion_behaves_as_before(tmp_path):
    rigged = Rigged(EVERYTHING)
    w = VWorld(tmp_path, rigged)
    first = w.champion(v1_definition(strategies=(Strategy.FIXED,)))
    view = w.promote(w.experiment(v1_definition(strategies=(Strategy.RANDOM,))))
    rec = view.record
    assert view.decision is Decision.PROMOTED
    assert rec["evaluation_context"]["dataset_version"] == 1
    assert rec["evaluation_context"]["test_row_ids"] == list(V1_TEST)
    assert rec["heldout"]["incumbent"]["row_ids"] == list(V1_TEST)
    assert rec["identities"]["compat_hash"] == first["identities"]["compat_hash"]
    assert rec["lineage_identity"] == first["lineage_identity"]
    w.promotions.verify(view.promotion_id)


def test_the_lineage_identity_is_stable_across_versions_and_names_no_rows():
    d1, d2 = v1_definition(), v2_definition()
    body = {"run_versions": [], "synthetic": True}
    identity1 = lineage_identity(d1, _artifact(d1) | body)
    identity2 = lineage_identity(d2, _artifact(d2) | body)
    assert identity1 == identity2
    d3 = v2_definition(constraints=ConstraintLimits(**{**LIMITS, "maximum_retries": 1}))
    changed = lineage_identity(d3, _artifact(d3) | body)
    with pytest.raises(EvidenceMismatch):  # a contract that is not the pinned one is refused
        lineage_identity(d3, _artifact(d2) | body)
    assert changed["lineage_hash"] != identity1["lineage_hash"]


def _artifact(d: ExperimentJobDefinition) -> dict[str, Any]:
    """The legacy (pre-provenance) identity block an artifact carries for ``d``."""
    from core.canonical import canonical_hash

    c = d.contract
    return {
        "identity": {
            "problem": {
                "dataset_id": c.dataset.dataset_id,
                "dataset_version": c.dataset.dataset_version,
                "dataset_hash": c.dataset.identity_hash,
                "dataset_content_hash": c.dataset.content_hash,
                "splits_hash": d.splits.identity_hash,
                "task_id": c.task_id,
                "contract_version": c.contract_version,
                "task_contract_hash": c.contract_hash,
                "objective_hash": canonical_hash(c.objective.model_dump(mode="json")),
                "constraints_hash": canonical_hash(c.constraints.model_dump(mode="json")),
                "evaluation_hash": c.evaluation.identity_hash,
                "evaluator_run_version": SYNTHETIC_VERSION,
                "grammar_version": "g",
                "model": d.plan.model.model_dump(mode="json"),
                "model_config_hash": "m",
                "model_hash": d.plan.expected_model_hash,
                "prompt_template_version": None,
            }
        }
    }


# -- 6. #30: published workflow versions keep resolving; a v2 champion deploys ------------------
def test_published_versions_resolve_and_a_cross_version_champion_deploys(tmp_path, monkeypatch):
    from core.models import AllowedModels
    from runtime.prompts import PROMPT_TEMPLATE_VERSION
    from store.deployments import VersionState
    from tests.test_champion_inference import ENTRY, Env, ModelRuntime, real_versions
    from tests.test_champion_inference import definition as served_definition

    class Stamped2(Rigged):
        def __init__(self, passes) -> None:
            super().__init__(passes)
            inner = self._evaluate
            versions = real_versions(served_definition().contract)

            def evaluate(genome, task, trial, seed):
                run = inner(genome, task, trial, seed)
                key = run.execution.key.model_copy(
                    update={"versions": run.execution.key.versions.model_validate(versions)}
                )
                return run.model_copy(
                    update={"execution": run.execution.model_copy(update={"key": key})}
                )

            self._evaluate = evaluate

    class Served(ModelRuntime):
        def bind(self, d):
            suite, _ = contract_suite(d.contract, d.splits, BYTES[d.contract.dataset.content_hash])
            return JobBinding(
                suite=suite,
                evaluate=self.backend,
                checker=ConstraintChecker(),
                evaluator_version=SYNTHETIC_VERSION,
                models=AllowedModels((ENTRY,)),
            )

        def bind_heldout(self, d):
            suite, _ = contract_suite(d.contract, d.splits, BYTES[d.contract.dataset.content_hash])
            tasks = suite.tasks_for(SplitRole.TEST, SplitUse.PROMOTION_GATE)
            return HeldoutBinding(tasks, self.backend, SYNTHETIC_VERSION, None)

    env = Env(tmp_path, Stamped2(EVERYTHING))
    env.runtime = Served(env.backend)
    env.jobs.runtime = env.runtime
    env.worker = JobWorker(
        env.jobs.store, env.runtime, worker_id="p", lease_s=LEASE, clock=env.clock, heartbeat=False
    )
    env.promotions.runtime = env.runtime
    env.promote = env.promotions.promote  # type: ignore[attr-defined]

    old = legacy_promotion(monkeypatch, env, env.experiment(served_definition((Strategy.FIXED,))))
    v1 = env.publish(old.record["champion"]["champion_id"])  # a champion/1 publishes as before
    env.deploy(v1)
    doc1 = env.deployments.get(v1).document

    pins = {
        "expected_model_hash": ENTRY.model_hash,
        "expected_prompt_version": PROMPT_TEMPLATE_VERSION,
    }
    view = env.promote(env.experiment(v2_definition(plan_changes=pins)))
    assert view.decision is Decision.PROMOTED, view.record
    v2, _ = env.deployments.publish(view.record["champion"]["champion_id"])
    assert v2.document["champion"]["compat_hash"] == view.record["lineage_identity"]["lineage_hash"]
    assert v2.document["provenance"]["artifact_id"] == view.record["artifact"]["artifact_id"]
    assert v2.document["contract"]["task_contract"]["dataset"]["dataset_version"] == 2
    env.deployments.stage(v2.version_id, env.revision())
    env.deployments.promote(v2.version_id, env.revision())
    assert env.deployments.deployment(LINEAGE).production_version_id == v2.version_id
    # the old version still resolves, unchanged, and is still its champion's version
    assert env.deployments.get(v1).document == doc1
    assert env.deployments.get(v1).state is VersionState.RETIRED
    assert doc1["contract"]["task_contract"]["dataset"]["dataset_version"] == 1
    env.deployments.rollback(LINEAGE, env.revision())
    assert env.deployments.deployment(LINEAGE).production_version_id == v1
