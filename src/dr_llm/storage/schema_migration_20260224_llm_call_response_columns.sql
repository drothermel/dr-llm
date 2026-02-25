ALTER TABLE llm_call_responses ADD COLUMN IF NOT EXISTS reasoning_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE llm_call_responses ADD COLUMN IF NOT EXISTS reasoning_text TEXT;
ALTER TABLE llm_call_responses ADD COLUMN IF NOT EXISTS reasoning_details_json JSONB;
ALTER TABLE llm_call_responses ADD COLUMN IF NOT EXISTS cost_total_usd DOUBLE PRECISION;
ALTER TABLE llm_call_responses ADD COLUMN IF NOT EXISTS cost_prompt_usd DOUBLE PRECISION;
ALTER TABLE llm_call_responses ADD COLUMN IF NOT EXISTS cost_completion_usd DOUBLE PRECISION;
ALTER TABLE llm_call_responses ADD COLUMN IF NOT EXISTS cost_reasoning_usd DOUBLE PRECISION;
ALTER TABLE llm_call_responses ADD COLUMN IF NOT EXISTS cost_currency TEXT;
ALTER TABLE llm_call_responses ADD COLUMN IF NOT EXISTS cost_json JSONB;
ALTER TABLE llm_call_responses ADD COLUMN IF NOT EXISTS warnings_json JSONB;

CREATE INDEX IF NOT EXISTS idx_llm_call_responses_reasoning_tokens ON llm_call_responses(reasoning_tokens);
