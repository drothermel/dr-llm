ALTER TABLE IF EXISTS provider_models_current
    DROP COLUMN IF EXISTS supports_tools;

DROP TABLE IF EXISTS tool_call_dead_letters;
DROP TABLE IF EXISTS tool_results;
DROP TABLE IF EXISTS tool_calls;
DROP TABLE IF EXISTS session_events;
DROP TABLE IF EXISTS session_turns;
DROP TABLE IF EXISTS sessions;
