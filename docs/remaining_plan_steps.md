# Remaining Plan Steps

This document records the work intended after the provider-specific authoring
config implementation is finished.

The target end state is a hard cutover:

- Provider-local authoring configs create normalized `LlmConfig` values.
- Runtime code consumes normalized `LlmConfig` and `LlmRequest`.
- Old provider-specific runtime request/config subclasses are removed.
- `dr-llm` and `../nl_latents` use the same final public API.
- The live encoder and decoder pool demos in `../nl_latents` pass.

## 1. Finish Provider-Specific Authoring Configs

Complete the implementation described in `docs/provider_config_types.md`.

Required properties:

- Authoring configs live under provider implementation modules.
- Each config validates provider/model-family-specific constraints.
- Each config exposes `to_llm_config(registry: ProviderRegistry | None = None)`.
- Central `dr_llm.llm.config` contains only normalized runtime/storage config:
  `SamplingControls`, `LlmConfig`, parsing helpers, and request construction
  helpers.

Provider-specific implementation notes:

- Consolidate Anthropic model-family capability data before finalizing
  Anthropic config classes.
- Keep OpenRouter policy-driven from `model_policies.yml`.
- Keep Kimi Code and MiniMax narrow unless new supported model families appear.
- Do not rebuild a central generic reasoning/effort/thinking config hierarchy.

## 2. Normalize Runtime Request Plumbing

Remove old concrete request subclasses:

- `ApiLlmRequest`
- `OpenAILlmRequest`
- `KimiCodeLlmRequest`
- `HeadlessLlmRequest`
- `ApiBackedLlmRequest`

All providers and transports should accept `LlmRequest`.

Required updates:

- API transport checks `request.mode == CallMode.api`.
- Headless transport checks `request.mode == CallMode.headless`.
- OpenAI-compatible transport reads `request.sampling`.
- Google request builder reads `request.sampling`.
- Anthropic request builder reads `request.sampling`.
- Headless providers read `request.effort`, `request.reasoning`, and
  `request.messages` from normalized `LlmRequest`.

Sampling should be nested under:

```python
SamplingControls(temperature=..., top_p=...)
```

No runtime code should read `request.temperature` or `request.top_p`.

## 3. Normalize Runtime Config Plumbing

Remove old concrete config subclasses:

- `ApiLlmConfig`
- `OpenAILlmConfig`
- `KimiCodeLlmConfig`
- `HeadlessLlmConfig`

All runtime paths should accept `LlmConfig`.

Required updates:

- Pool backend uses normalized config.
- CLI query command builds `SamplingControls` and passes it through config or
  request helpers.
- Demo model catalog uses provider-local authoring configs and stores
  normalized `LlmConfig` values.
- Scripts use provider-local authoring configs for authoring and normalized
  `LlmConfig` for execution.

## 4. Fix Transport Layer Boundaries

OpenAI-compatible transport currently contains OpenAI-specific behavior around
`max_completion_tokens`.

Target design:

- OpenAI-specific model-family knowledge stays in the OpenAI provider impl.
- Generic OpenAI-compatible transport receives generic transport config.
- Transport config can carry a neutral rule such as model prefixes that require
  `max_completion_tokens`.
- OpenRouter and GLM should not import OpenAI-specific thinking helpers.

Suggested implementation:

- Add a field to `OpenAICompatConfig` such as:

```python
max_completion_token_model_prefixes: tuple[str, ...] = ()
```

- Populate it for OpenAI in the default registry.
- Leave it empty for OpenRouter and GLM.
- Have `OpenAICompatRequest.json_payload()` use that transport config field
  instead of importing OpenAI provider helpers.

## 5. Update Public Exports

Update `dr_llm.llm.__init__`:

- Export normalized `LlmConfig`, `LlmRequest`, and `SamplingControls`.
- Export provider-local authoring configs that should be part of the public API.
- Remove exports for old concrete runtime request/config subclasses.

Decide whether to export every family-specific config at top level or require
provider-module imports for advanced family configs.

## 6. Update Tests in dr-llm

Rewrite tests around the new API.

Required changes:

- Replace old request/config subclass construction with normalized
  `LlmRequest` / `LlmConfig`.
- Replace `temperature=` and `top_p=` fields with `sampling=SamplingControls(...)`.
- Update request default tests for `sampling_supported` and `sampling`.
- Update provider transport tests for normalized request mode validation.
- Update config tests to cover provider-local authoring configs and
  `to_llm_config(...)`.
- Update OpenAI tests for:
  - GPT-5 `max_completion_tokens`.
  - GPT-5.2 / GPT-5.4 sampling only with thinking off.
  - GPT-5 base minimal-thinking behavior.
- Update Google tests for budget vs level authoring configs.
- Update Anthropic tests for budget, effort, and effort-plus-budget families.
- Update OpenRouter tests to use policy-driven authoring configs.

Avoid tests that only assert re-export availability or incidental class names.

## 7. Update README and Examples

Remove documentation that describes the old request/config subclass design.

Required updates:

- Replace old README section describing `OpenAILlmRequest`, `ApiLlmRequest`,
  `KimiCodeLlmRequest`, and `HeadlessLlmRequest`.
- Show provider-specific authoring config examples.
- Show conversion to `LlmConfig`.
- Show normalized request construction.

Example direction:

```python
openai_config = OpenAIGpt52Config(
    model="gpt-5.2-mini",
    thinking_level=ThinkingLevel.OFF,
    sampling=SamplingControls(temperature=0.7, top_p=0.95),
)

config = openai_config.to_llm_config()
```

## 8. Update nl_latents

Cut `../nl_latents` over to the new `dr-llm` API.

Required updates:

- Replace old imports:
  - `ApiLlmConfig`
  - `OpenAILlmConfig`
  - `KimiCodeLlmConfig`
  - `HeadlessLlmConfig`
  - old request subclasses
- Update the LLM catalog to instantiate provider-local authoring configs.
- Store and pass normalized `LlmConfig` values after authoring.
- Replace `temperature` and `top_p` fields with `SamplingControls`.
- Remove provider-shape branching that only existed to choose old config
  subclasses.
- Update encoder and decoder request building to use normalized config fields.
- Update mock/live script defaults so demos are mock by default and live only
  with explicit `--live`.
- Update mock response mode inference to use normalized `mode`.

Important scripts:

- `scripts/demo_encoder_pool.py`
- `scripts/demo_decoder_pool.py`

## 9. Run dr-llm Quality Gate

Required from the repository instructions:

```bash
uv run ruff format
uv run ruff check --fix .
uv run ty check
uv run pytest tests/ -v -m "not integration"
./scripts/run-tests-local.sh
```

Manually fix any remaining lint/type/test failures before moving on.

## 10. Run nl_latents Quality Gate

Required from `../nl_latents` repository instructions:

```bash
uv run ruff format
uv run ruff check
uv run ty check
uv run pytest
```

Run from `../nl_latents`.

## 11. Run Live nl_latents Pool Demos

After both repos pass their quality gates, run the live demos from
`../nl_latents`.

Encoder pool:

```bash
uv run python scripts/demo_encoder_pool.py --live --llm-config-id openai/gpt-5-nano-2025-08-07/minimal
```

Decoder pool:

```bash
uv run python scripts/demo_decoder_pool.py --live --official-prompt-only --llm-config-id openai/gpt-5-nano-2025-08-07/minimal
```

Record any live API issues separately from local code failures.

## 12. Final Cleanup

Before considering the refactor complete:

- Search both repos for old type names and old field names.
- Remove dead compatibility aliases.
- Remove stale tests that only covered old transition behavior.
- Remove obsolete comments mentioning old request/config subclasses.
- Confirm no provider-specific validation remains in consumer code when it
  belongs in provider-local authoring configs.

Useful search:

```bash
rg -n "ApiLlmConfig|OpenAILlmConfig|KimiCodeLlmConfig|HeadlessLlmConfig|ApiLlmRequest|OpenAILlmRequest|KimiCodeLlmRequest|HeadlessLlmRequest|ApiBackedLlmRequest|temperature=|top_p=|\\.temperature|\\.top_p"
```
