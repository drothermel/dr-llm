export type ProviderStatus = {
  provider: string
  available: boolean
  missing_env_vars: string[]
  missing_executables: string[]
  supports_structured_output: boolean
}

export type ModelEntry = {
  provider: string
  model: string
  display_name: string | null
  context_window: number | null
  max_output_tokens: number | null
  control_mode: string | null
  supports_vision: boolean | null
  source_quality: string
}

export type ProviderModelsResponse = {
  provider: string
  models: ModelEntry[]
  source: 'live' | 'static' | 'error'
  error: string | null
}

export type SyncResultResponse = {
  provider: string
  success: boolean
  model_count: number
  models: ModelEntry[]
  source: 'live' | 'static' | 'error'
  error: string | null
}

export type NlLatentsFilters = {
  families: string[]
  splits: string[]
  enc_models: string[]
  budgets: string[]
  difficulties: string[]
  data_versions: string[]
}

export type NlLatentsSampleListRow = {
  sample_id: string
  family: string
  difficulty: string
  split: string
  language: string
  budget: string
  enc_model: string
  dec_model: string
  enc_model_label: string
  dec_model_label: string
  status: string
  attempt_count: number
  created_at: string
  result_state: string
  failure_category_normalized: string | null
  run_id: string | null
  prompt_config_label: string | null
}

export type NlLatentsSamplesResponse = {
  samples: NlLatentsSampleListRow[]
  total: number
  page: number
  limit: number
  total_pages: number
}

export type NlLatentsSampleDetail = NlLatentsSampleListRow & {
  config_id: string
  task_id: string
  task_data_version: string
  enc_reasoning_effort: string
  dec_reasoning_effort: string
  call_id: string
  sample_idx: number
  finish_reason: string | null
  prompt_block_ids: string[]
  prompt_block_names: string[]
  prompt_config_json: Record<string, unknown> | null
  passed: boolean | null
  failure_category: string | null
  model_provenance_source: string | null
  budget_ok: boolean | null
  actual_chars: number | null
  enc_time_s: number | null
  dec_time_s: number | null
  validation_compiles: boolean | null
  validation_pass_rate: number | null
  validation_time_seconds: number | null
  input_code: string | null
  enc_prompt: string | null
  enc_prompt_instructions: string | null
  description: string | null
  dec_system: string | null
  dec_task: string | null
  decoded_code: string | null
  error_detail: string | null
  validation_summary_json: Record<string, unknown> | null
}
