# Dataset + Task Contract (Phase 1)

Wynk is moving from a hackathon demo with fixed benchmark task classes (A/B/C) to a platform where
a user brings a dataset and Wynk searches for the best workflow for it against an explicit
objective and explicit limits. This document describes the contract layer that makes the core
dataset-driven. Upload, inspection, registration, durable storage and splitting of user datasets
are described in [dataset_ingestion.md](dataset_ingestion.md); execution of user datasets is a
later phase (see [Deferred](#deferred)).

```
Dataset bytes (held outside the contract)
   |
   v
DatasetSpec            core/dataset.py          columns, roles, content hash, identity
   |
   v
TaskContract           core/task_contract.py    task type, instructions, input/output schema
   |-- EvaluationSpec  core/evaluation_spec.py  evaluator kind + pinned version + strict config
   |-- ObjectiveSpec   core/objective.py        what "better" means among feasible candidates
   '-- ConstraintLimits core/constraints.py     hard limits on measured behaviour
          |
          v
Workflow search        optimizers/ (unchanged: proposes genomes via the shared ConstraintChecker)
          |
          v
Execution              runtime/    consumes ExecutionTask (core/run_contract.py)
          |
          v
Evaluation             evaluation/metrics.py (new metrics) | evaluation/gate.py (legacy)
          |
          v
Optimizer feedback     optimization split only (DatasetSplits.check_feedback)
```

Every model is a frozen pydantic model with `extra="forbid"`, carries a `schema_version`, and is
validated deterministically in code. No contract is authored or amended by a model.

## DatasetSpec (`core/dataset.py`)

| Field | Meaning |
|---|---|
| `dataset_id`, `dataset_version` | slug id (never a path) + positive integer version |
| `content_hash` | sha256 of the dataset bytes, computed by whoever holds them |
| `format` | `csv`, `jsonl`, or `wynk_snapshot` (the frozen benchmark layout; legacy only) |
| `columns` | typed columns: `string`, `integer`, `number`, `boolean`, `date`, `string_list`, `json` |
| `id_column` | stable row ids (string/integer, non-nullable); `None` when ingestion generates row ids |
| `input_columns` / `context_columns` / `target_columns` | disjoint roles; inputs and targets are required |
| `row_count` | positive |
| `name`, `metadata` | descriptive only |

Validation fails closed on unknown or duplicate columns, a column in two roles, an empty input or
target role, a non-identifier column name, a malformed hash, a path-like id, or non-finite metadata.

**Identity.** `identity_hash` = canonical hash of every field **except** `name` and `metadata`.
It changes when the content hash, version, format, columns, roles, id column or row count change,
and it never depends on key order. `canonical_json()` is the full serialization (metadata included).

## Splits and the authority boundary (`core/dataset.py`)

`DatasetSplits` assigns row ids to explicit, named splits with one of three roles:

```
optimization (train) -> optimizer feedback allowed   (+ selection, reporting)
validation           -> selection / promotion allowed (+ reporting)   never optimizer feedback
test (final)         -> reporting only                                 held out
```

This table is `ALLOWED_USES`, enforced by `require_use(role, use)` (`SplitAccessError` on a
forbidden use). Exactly one optimization split is required, at most one validation and one test
split, and the row sets must be disjoint.

* **Seeded splitting** (`seeded_splits`, method `seeded_hash/1`): rows are ordered by
  `sha256("wynk-split/1:{seed}:{row_id}")`; the first `n * test_bps // 10000` become test, the next
  `n * validation_bps // 10000` validation, the rest optimization. Fractions are integer basis
  points, so sizes never depend on float rounding. The result does not depend on input order, and
  a stored seeded split is re-derived on load: a tampered assignment is rejected.
* **Explicit splitting** (method `explicit`): an assignment taken as-is, e.g. the frozen benchmark's
  split files.
* **Optimizer view**: `DatasetSplits.optimizer_view()` returns an `OptimizerSplitView` with
  `optimization_row_ids` and `validation_row_ids` and **no test field**.
* **Feedback gate**: `DatasetSplits.check_feedback(row_ids)` refuses any validation, test or unknown
  row. `tests/test_contract_legacy.py` runs the existing search harness (`run_search`) with an MMAS
  ACO whose `observe` goes through this gate, and proves for classes A and B that validation runs are
  measured but never observed, that no test row is executed or observed, and that a final-test result
  is refused without changing pheromone state.

## EvaluationSpec (`core/evaluation_spec.py`, `evaluation/metrics.py`)

Configuration and implementation are separate. `EvaluationSpec(evaluator=..., config=...)` names an
evaluator kind, pins `evaluator_version` (filled with the current version when omitted, rejected if
it differs) and validates `config` against that kind's own strict model. The config is stored
normalized (defaults filled in), so equivalent specs hash identically.

| Kind | Config | Judges |
|---|---|---|
| `exact_match` | `case_sensitive`, `normalize_whitespace` | every output field equal; score = fraction of fields |
| `classification_accuracy` | `labels` (>= 2, unique), `case_sensitive` | prediction is a configured label and equals the target |
| `token_f1` | `pass_threshold` in (0, 1] | SQuAD-normalized token F1; PASS iff F1 >= threshold |
| `json_schema_validity` | none | output satisfies the task's `output_schema` |
| `numeric_tolerance` | `absolute_tolerance`, `relative_tolerance` | `abs(p - t) <= max(abs_tol, rel_tol * abs(t))` per field |
| `legacy_field_match` | per-field benchmark `matchers`, `require_evidence` | executed by `evaluation.gate.DeterministicEvaluator` (`evaluator/mvp-2`); `wynk_snapshot` datasets only |

`evaluation.metrics.score(spec, output_schema, expected, predicted)` scores one example. A missing
or schema-invalid prediction scores 0. Implementations are looked up by kind **and** version
(`get_metric`); a mismatch raises `EvaluatorUnavailable` instead of substituting another evaluator.
No kind asks a model to judge output; the evaluator stays the external authority. A new evaluator is
one enum member, one config model and one version entry in `core/evaluation_spec.py`, plus one
registered implementation in `evaluation/metrics.py` (a test keeps the two in agreement).

## ObjectiveSpec (`core/objective.py`)

An objective is a **preference** among feasible candidates. It reads `CandidateMeasurements`, where
`None` means "not measured", never zero.

| Mode | Sort key (higher is better) | Requires |
|---|---|---|
| `maximize_quality` (default) | `(quality,)` | quality |
| `minimize_cost` | `(-mean_cost_per_example, quality)` | cost, quality, and `constraints.minimum_quality` |
| `minimize_latency` | `(-mean_latency_s, quality)` | latency, quality, and `constraints.minimum_quality` |
| `balanced` | `(w_q*quality - sum(w_m * value_m / scale_m),)` | every metric with a positive weight |

`balanced` weights (over `quality`, `cost`, `latency`, `tokens`) must be finite and non-negative,
sum to 1 within 1e-9, and give quality a positive weight. Every penalized metric needs an explicit
positive `scale` (the value that costs its full weight), so nothing is normalized implicitly. Weights
and scales are rejected for the other modes. A missing required metric raises `MissingMetric`.
Pareto ranking is not implemented; the key is a pure function of the measurements, so it can be
replaced later.

## Constraints (`core/constraints.py`)

`ConstraintChecker` (structural, pre-execution: grammar, stage rules, provable budget) is unchanged.
`ConstraintLimits` adds **hard** limits on measured behaviour:

| Limit | Compared with |
|---|---|
| `minimum_quality` | `quality` (floor) |
| `maximum_cost_per_example` | `mean_cost_per_example` |
| `maximum_mean_latency_s`, `maximum_p95_latency_s` | `mean_latency_s`, `p95_latency_s` |
| `maximum_tokens_per_example`, `maximum_model_calls`, `maximum_tool_calls`, `maximum_retries`, `maximum_wall_time_s` | the worst single example (`max_*_per_example`): per-run caps apply to every run |
| `maximum_workflow_steps` | `workflow_steps` |

`check_limits` returns a `LIMIT_VIOLATED` violation for each breach (equality is allowed) and a
`METRIC_MISSING` violation for an active limit whose measurement is absent. `from_caps` / `to_caps`
convert the legacy per-run `Caps` with the same numbers.

**Hard constraints outrank fitness.** `TaskContract.rank(measurements)` checks limits first. An
infeasible candidate's objective is never computed, and its `sort_key` `(0.0,)` is below every
feasible key `(1.0, *objective)`, whatever its weighted score would have been.

## TaskContract (`core/task_contract.py`)

`task_id`, `contract_version`, `task_type`, `instructions`, `input_schema`, `output_schema`,
`dataset`, `evaluation`, `objective` (default `maximize_quality`), `constraints` (default: none).

* `input_schema` fields must equal the dataset's input + context columns, and `output_schema` fields
  its target columns, with matching types (a `json` column cannot be mapped). A required field cannot
  sit on a nullable column.
* Task types are bounded:
  * `classification`: one string output; `classification_accuracy` or `exact_match`.
  * `question_answering`: one string output (`exact_match`, `token_f1`) or one numeric output
    (`exact_match`, `numeric_tolerance`).
  * `structured_extraction`: one or more outputs; `exact_match`, `json_schema_validity`,
    `numeric_tolerance` (all-numeric outputs), or `legacy_field_match` (`wynk_snapshot` only).
* No open-ended agent task type and no user code execution exist.
* `contract_hash` covers every field except the dataset's `name` / `metadata`.

## Experiment identity (`core/experiment.py`)

`experiment_identity(contract, splits, grammar_version=..., model=ModelConfiguration(...))` combines
dataset identity, split identity, task id + contract version + contract hash, workflow grammar
version, model-configuration hash, evaluator version, and the evaluation, objective and constraint
hashes. Everything is hashed through `core.canonical`, so dict order and descriptive metadata never
matter. `ExperimentIdentity.differences(other)` names the components that differ. Splits made for
another dataset are rejected.

## Examples

Classification (support-ticket routing):

```python
from core.dataset import ColumnSpec, DatasetSpec
from core.evaluation_spec import EvaluationSpec
from core.task_contract import TaskContract
from core.task_spec import AnswerField, AnswerSchema

tickets = DatasetSpec(
    dataset_id="support-tickets",
    dataset_version=1,
    name="Support tickets",
    content_hash="<sha256 of tickets.csv>",
    format="csv",
    columns=(
        ColumnSpec(name="id", type="string"),
        ColumnSpec(name="text", type="string"),
        ColumnSpec(name="label", type="string"),
    ),
    id_column="id",
    input_columns=("text",),
    target_columns=("label",),
    row_count=1200,
)
routing = TaskContract(
    task_id="ticket-routing",
    contract_version=1,
    task_type="classification",
    instructions="Route the support ticket to exactly one team.",
    input_schema=AnswerSchema(fields=(AnswerField(name="text", type="string"),)),
    output_schema=AnswerSchema(fields=(AnswerField(name="label", type="string"),)),
    dataset=tickets,
    evaluation=EvaluationSpec(
        evaluator="classification_accuracy", config={"labels": ["billing", "bugs", "sales"]}
    ),
)
```

Question answering (answer from a passage, balanced objective, quality floor):

```python
from core.constraints import ConstraintLimits
from core.objective import ObjectiveSpec

reading = TaskContract(
    task_id="reading-qa",
    contract_version=1,
    task_type="question_answering",
    instructions="Answer the question from the passage in a short phrase.",
    input_schema=AnswerSchema(
        fields=(
            AnswerField(name="question", type="string"),
            AnswerField(name="passage", type="string"),
        )
    ),
    output_schema=AnswerSchema(fields=(AnswerField(name="answer", type="string"),)),
    dataset=qa_dataset,  # columns qid / question (input) / passage (context) / answer (target)
    evaluation=EvaluationSpec(evaluator="token_f1", config={"pass_threshold": 0.8}),
    objective=ObjectiveSpec(
        mode="balanced",
        weights={"quality": 0.7, "cost": 0.2, "latency": 0.1},
        scales={"cost": 0.01, "latency": 2.0},
    ),
    constraints=ConstraintLimits(minimum_quality=0.75, maximum_p95_latency_s=5.0),
)
```

Both examples are exercised by `tests/contract_helpers.py` and `tests/test_contract_*.py`.

## Legacy benchmark compatibility (`benchmarks/legacy_adapter.py`)

The frozen A/B benchmark and its golden hashes are untouched. Since #20 the adapter is the ONLY way
benchmark tasks reach search, execution and evaluation - downstream code reads contracts only:

* `legacy_task_contract(spec, store)`: one `structured_extraction` contract per benchmark task, over a
  one-row `wynk_snapshot` dataset (`question` input, `snapshot_id` context, one target column per
  answer field). Each legacy task asks for different answer fields, so one contract per class could
  not have a single output schema. A class is therefore a suite of contracts sharing caps and sources.
* Evaluation is `legacy_field_match` with the task's own matchers and `require_evidence=True`, the
  same `evaluator/mvp-2` that `evaluation.gate.DeterministicEvaluator` runs. Caps become
  `ConstraintLimits.from_caps` with identical numbers. The objective is `maximize_quality`; the
  shaped fitness's small budget-headroom bonus among PASSes stays in `evaluation/fitness.py`.
* `content_hash` covers `TaskSpec.content_hash` and the snapshot bytes. Expected values never appear
  in a contract, but changing one still changes dataset identity.
* `legacy_splits()`: `splits.json` train → optimization, validation → validation, the held-out set
  (`benchmarks/heldout/`) → final test.
* `legacy_suite(task_class)`: one class as a `ContractSuite` (its `ExecutionTask`s + those splits);
  the class selects the suite and becomes its display name. `legacy_references()` hands the expected
  values to `ContractEvaluator` only. `allowed_sources` / `interaction_required` become the
  contract's `WorkflowSpec`.
* `legacy_adhoc_task(runtime_task, store)`: a question over a benchmark snapshot with no expected
  answer (`api/chat.py`), checked with `json_schema_validity`.

### Task-class audit

| Where `task_class` / A-B-C appears | Category | Phase 1 decision |
|---|---|---|
| `optimizers/`, `compiler/`, `runtime/`, `core/` (except `task_spec.py`), `evaluation/contract_eval.py`, `experiments/learning_curves.py` | optimizer / runtime / evaluator / harness | removed (#20): these read `TaskContract` only; `tests/test_contract_runtime.py` fails if they import `RuntimeTask`/`TaskSpec`/`TaskClass` or mention `task_class` |
| `core/task_spec.py` (`TaskClass`, `RuntimeTask.task_class`) | benchmark-specific, in a frozen contract | kept: it is part of the pinned `tasks.json` / `benchmark_hash`. Generic contracts never use it (it is dataset metadata in the adapter) |
| `benchmarks/` (`build.py`, `heldout.py`, `loader.py`, `legacy_adapter.py`) | benchmark-specific | kept; `legacy_adapter.py` is the single boundary |
| `memory/` (one memory file per suite name, `MemoryKey.task_class`) | benchmark experiment memory | keyed by the suite name the adapter assigns; should key on `ExperimentIdentity` later |
| `experiments/run_mvp.py`, `experiments/oss_baselines/`, `memory/warm_start.py` CLIs, `api/chat.py` | legacy drivers / demo | select a class, then go through the adapter |

## Deferred

Not implemented in Phase 1: dataset upload UI, PostgreSQL / object storage backends (the local
SQLite + filesystem implementation is in `store/`; see [dataset_ingestion.md](dataset_ingestion.md)),
background workers, wiring `ExperimentIdentity` into the run cache and workflow memory, cost
measurement in the runtime, Pareto optimization, public benchmark expansion, and deployment.
