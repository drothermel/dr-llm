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
    latency_ms INTEGER NOT NULL DEFAULT 0,
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
    total_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llm_call_responses_hash ON llm_call_responses(response_hash);

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
