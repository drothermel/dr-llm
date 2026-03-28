CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL,
    status TEXT NOT NULL,
    metadata_json JSONB,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS run_parameters (
    run_id TEXT NOT NULL,
    param_key TEXT NOT NULL,
    param_value TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, param_key)
);

CREATE INDEX IF NOT EXISTS idx_run_parameters_run_id ON run_parameters(run_id);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id TEXT PRIMARY KEY,
    run_id TEXT,
    external_call_id TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms INTEGER,
    error_text TEXT,
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_calls_external_call_id
    ON llm_calls(external_call_id)
    WHERE external_call_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_llm_calls_run_id_created ON llm_calls(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_calls_provider_model_created ON llm_calls(provider, model, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_calls_status_created ON llm_calls(status, created_at);

ALTER TABLE llm_calls ALTER COLUMN latency_ms DROP NOT NULL;

CREATE TABLE IF NOT EXISTS llm_call_requests (
    call_id TEXT PRIMARY KEY,
    request_json JSONB NOT NULL,
    request_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llm_call_requests_hash ON llm_call_requests(request_hash);

CREATE TABLE IF NOT EXISTS llm_call_responses (
    call_id TEXT PRIMARY KEY,
    response_json JSONB,
    response_hash TEXT,
    output_text TEXT,
    finish_reason TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_text TEXT,
    reasoning_details_json JSONB,
    cost_total_usd DOUBLE PRECISION,
    cost_prompt_usd DOUBLE PRECISION,
    cost_completion_usd DOUBLE PRECISION,
    cost_reasoning_usd DOUBLE PRECISION,
    cost_currency TEXT,
    cost_json JSONB,
    warnings_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llm_call_responses_hash ON llm_call_responses(response_hash);

CREATE TABLE IF NOT EXISTS provider_model_catalog_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL,
    raw_json JSONB,
    error_text TEXT
);

CREATE INDEX IF NOT EXISTS idx_provider_model_catalog_snapshots_provider_fetched
    ON provider_model_catalog_snapshots(provider, fetched_at DESC);

CREATE TABLE IF NOT EXISTS provider_models_current (
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    display_name TEXT,
    context_window INTEGER,
    max_output_tokens INTEGER,
    supports_reasoning BOOLEAN,
    supports_vision BOOLEAN,
    pricing_json JSONB,
    rate_limits_json JSONB,
    source_quality TEXT NOT NULL DEFAULT 'live',
    metadata_json JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (provider, model)
);

CREATE INDEX IF NOT EXISTS idx_provider_models_current_provider_reasoning
    ON provider_models_current(provider, supports_reasoning);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT,
    artifact_type TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON artifacts(run_id);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_run_parameters_run_id') THEN
        ALTER TABLE run_parameters
            ADD CONSTRAINT fk_run_parameters_run_id
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_llm_calls_run_id') THEN
        ALTER TABLE llm_calls
            ADD CONSTRAINT fk_llm_calls_run_id
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_llm_call_requests_call_id') THEN
        ALTER TABLE llm_call_requests
            ADD CONSTRAINT fk_llm_call_requests_call_id
            FOREIGN KEY (call_id) REFERENCES llm_calls(call_id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_llm_call_responses_call_id') THEN
        ALTER TABLE llm_call_responses
            ADD CONSTRAINT fk_llm_call_responses_call_id
            FOREIGN KEY (call_id) REFERENCES llm_calls(call_id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_artifacts_run_id') THEN
        ALTER TABLE artifacts
            ADD CONSTRAINT fk_artifacts_run_id
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE;
    END IF;

END $$;
