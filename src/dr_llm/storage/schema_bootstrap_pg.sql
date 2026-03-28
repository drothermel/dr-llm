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
    supports_tools BOOLEAN,
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

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    version INTEGER NOT NULL,
    strategy_mode TEXT NOT NULL,
    metadata_json JSONB,
    last_error_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sessions_status_created ON sessions(status, created_at);

CREATE TABLE IF NOT EXISTS session_turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    UNIQUE(session_id, turn_index)
);

CREATE INDEX IF NOT EXISTS idx_session_turns_session_id_turn_index ON session_turns(session_id, turn_index);

CREATE TABLE IF NOT EXISTS session_events (
    event_seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    turn_id TEXT,
    event_type TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_session_events_session_seq ON session_events(session_id, event_seq);
CREATE INDEX IF NOT EXISTS idx_session_events_type_created ON session_events(event_type, created_at);

CREATE TABLE IF NOT EXISTS tool_calls (
    tool_call_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT,
    idempotency_key TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    status TEXT NOT NULL,
    args_json JSONB NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    worker_id TEXT,
    last_error_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    claimed_at TIMESTAMPTZ,
    lease_expires_at TIMESTAMPTZ,
    UNIQUE(idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_status_created ON tool_calls(status, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session_created ON tool_calls(session_id, created_at);

CREATE TABLE IF NOT EXISTS tool_results (
    tool_call_id TEXT PRIMARY KEY,
    result_json JSONB,
    error_json JSONB,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tool_call_dead_letters (
    dead_letter_id TEXT PRIMARY KEY,
    tool_call_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    payload_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tool_call_dead_letters_created ON tool_call_dead_letters(created_at);

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

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_session_turns_session') THEN
        ALTER TABLE session_turns
            ADD CONSTRAINT fk_session_turns_session
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_session_events_session') THEN
        ALTER TABLE session_events
            ADD CONSTRAINT fk_session_events_session
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_session_events_turn') THEN
        ALTER TABLE session_events
            ADD CONSTRAINT fk_session_events_turn
            FOREIGN KEY (turn_id) REFERENCES session_turns(turn_id) ON DELETE SET NULL;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_tool_calls_session') THEN
        ALTER TABLE tool_calls
            ADD CONSTRAINT fk_tool_calls_session
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_tool_calls_turn') THEN
        ALTER TABLE tool_calls
            ADD CONSTRAINT fk_tool_calls_turn
            FOREIGN KEY (turn_id) REFERENCES session_turns(turn_id) ON DELETE SET NULL;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_tool_results_tool_call') THEN
        ALTER TABLE tool_results
            ADD CONSTRAINT fk_tool_results_tool_call
            FOREIGN KEY (tool_call_id) REFERENCES tool_calls(tool_call_id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_tool_call_dead_letters_tool_call') THEN
        ALTER TABLE tool_call_dead_letters
            ADD CONSTRAINT fk_tool_call_dead_letters_tool_call
            FOREIGN KEY (tool_call_id) REFERENCES tool_calls(tool_call_id) ON DELETE CASCADE;
    END IF;
END $$;
