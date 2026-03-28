DELETE FROM provider_models_current
WHERE provider = 'claude-code'
  AND model NOT LIKE 'claude-%';

DELETE FROM llm_calls
WHERE provider = 'claude-code'
  AND model NOT LIKE 'claude-%';
