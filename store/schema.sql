-- DRAFT shape of the future DuckDB run store (not executed in Phase 0).
-- Identity columns mirror core.results.RunKey; JSON columns hold canonical pydantic dumps.
CREATE TABLE IF NOT EXISTS tasks (
    task_id        VARCHAR PRIMARY KEY,
    task_class     VARCHAR NOT NULL,
    content_hash   VARCHAR NOT NULL,       -- TaskSpec.content_hash
    spec_json      JSON    NOT NULL
);

CREATE TABLE IF NOT EXISTS genomes (
    genome_hash    VARCHAR PRIMARY KEY,    -- Genome.genome_hash
    genome_json    JSON    NOT NULL        -- Genome.canonical_json()
);

CREATE TABLE IF NOT EXISTS runs (
    run_id                  VARCHAR PRIMARY KEY,   -- RunKey.run_id
    genome_hash             VARCHAR NOT NULL REFERENCES genomes(genome_hash),
    task_id                 VARCHAR NOT NULL REFERENCES tasks(task_id),
    trial                   INTEGER NOT NULL,
    seed                    BIGINT  NOT NULL,
    model_hash              VARCHAR NOT NULL,
    prompt_template_version VARCHAR NOT NULL,
    benchmark_hash          VARCHAR NOT NULL,
    compiler_version        VARCHAR NOT NULL,
    grammar_version         VARCHAR NOT NULL,
    answer_json             JSON,                  -- Answer incl. structured evidence
    metrics_json            JSON    NOT NULL,      -- ExecutionMetrics
    budget_usage_json       JSON    NOT NULL,      -- BudgetUsage
    stage_trace_json        JSON    NOT NULL,      -- list[StageTrace]
    failure_json            JSON,                  -- FailureInfo
    verdict                 VARCHAR NOT NULL CHECK (verdict IN ('PASS', 'FAIL', 'INFEASIBLE')),
    fitness                 DOUBLE  NOT NULL,
    evaluator_version       VARCHAR NOT NULL
);
