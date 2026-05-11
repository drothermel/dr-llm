# Provider Config Types

This document describes the target authoring-config layout for the LLM provider
refactor.

The key design split is:

- `LlmConfig` is the normalized runtime and storage shape.
- Provider authoring configs are provider-local, more specific, and optimized
  for authoring-time validation and LSP hints.
- Every authoring config converts into the same `LlmConfig` shape before it is
  consumed by providers, pools, demos, or downstream repos.

## Runtime Shape

`dr_llm.llm.config.LlmConfig` should remain provider-neutral and boring:

```python
LlmConfig(
    provider=ProviderName.OPENAI,
    model="gpt-5.2-mini",
    mode=CallMode.api,
    max_tokens=1024,
    effort=EffortSpec.NA,
    reasoning=OpenAIReasoning(thinking_level=ThinkingLevel.OFF),
    sampling=SamplingControls(temperature=0.7, top_p=0.95),
)
```

Runtime code should not need to know which authoring config created that value.
Provider transports should consume `LlmRequest`, not provider-specific request
subclasses.

## Authoring Shape

Authoring configs should live with the provider implementation:

```text
src/dr_llm/llm/providers/impls/
  openai/config.py
  anthropic/config.py
  google/config.py
  openrouter/config.py
  glm/config.py
  codex/config.py
  claude_code/config.py
  kimi_code/config.py
  minimax/config.py
```

Each authoring config should:

- Be a Pydantic `BaseModel`.
- Encode only options valid for that provider/model family.
- Validate model-family constraints at construction/conversion time.
- Expose `to_llm_config(registry: ProviderRegistry | None = None) -> LlmConfig`.
- Avoid compatibility with old `OpenAILlmConfig`, `ApiLlmConfig`,
  `KimiCodeLlmConfig`, and `HeadlessLlmConfig`.

Top-level package exports may re-export the common authoring configs for
ergonomics, but implementation and validation rules should stay provider-local.

## Shared Helpers

Provider-local configs can share small helpers where useful:

- A base or helper function for `to_llm_config(...)`.
- Model-family matchers such as `matches_family(...)`.
- Provider-local capability tables.

Avoid rebuilding a central generic config hierarchy. The goal is to move
provider-specific quirks closer to the provider, not to create another global
reasoning/effort/thinking layer.

## OpenAI

Current inferred source:

- `OPENAI_THINKING_SUPPORTED_MODELS`
- `OPENAI_MINIMAL_THINKING_SUPPORTED_MODELS`
- `OPENAI_OFF_THINKING_SUPPORTED_MODELS`
- `OPENAI_GPT5_SAMPLING_SUPPORTED_MODELS`

Recommended authoring configs:

```python
OpenAILegacyConfig
OpenAIGpt5Config
OpenAIGpt51Config
OpenAIGpt52Config
OpenAIGpt53Config
OpenAIGpt54Config
```

Suggested constraints:

| Config | Model family | Thinking levels | Sampling |
| --- | --- | --- | --- |
| `OpenAILegacyConfig` | Non GPT-5 reasoning models | none | normal sampling |
| `OpenAIGpt5Config` | `gpt-5`, `gpt-5-mini`, `gpt-5-nano` | `minimal`, `low`, `medium`, `high` | no custom sampling |
| `OpenAIGpt51Config` | `gpt-5.1*` | `off`, `low`, `medium`, `high` | no custom sampling |
| `OpenAIGpt52Config` | `gpt-5.2*` | `off`, `low`, `medium`, `high` | only when thinking is `off` |
| `OpenAIGpt53Config` | `gpt-5.3*` | `off`, `low`, `medium`, `high` | no custom sampling |
| `OpenAIGpt54Config` | `gpt-5.4*` | `off`, `low`, `medium`, `high` | only when thinking is `off` |

OpenAI should also keep the transport rule that GPT-5 reasoning-era models use
`max_completion_tokens` rather than `max_tokens`.

## Anthropic

Current inferred sources:

- Anthropic capability rules
- Anthropic effort-supported models
- Anthropic budget/adaptive thinking lists

Recommended authoring configs:

```python
AnthropicLegacyConfig
AnthropicBudgetConfig
AnthropicEffortConfig
AnthropicEffortAndBudgetConfig
```

Suggested constraints:

| Config | Model family | Controls |
| --- | --- | --- |
| `AnthropicLegacyConfig` | Models without reasoning capability | no reasoning or effort |
| `AnthropicBudgetConfig` | Claude 3.7 / 4 / 4.1 / 4.5 budget-thinking families | `off`, `budget`; budget range `1024..128000` |
| `AnthropicEffortConfig` | Claude 4.6 effort families | `effort=low/medium/high`, plus `max` for Opus 4.6 |
| `AnthropicEffortAndBudgetConfig` | Opus 4.5 family | effort plus budget thinking |

Before implementing these classes, consolidate Anthropic's model-family data into
one provider-local capability table. The old shape split adaptive thinking,
budget thinking, and effort support across separate helpers, which is the
confusion this refactor should remove.

## Google

Current inferred source:

- Google capability rules with `google_budget` and `google_level` modes.

Recommended authoring configs:

```python
GoogleLegacyConfig
GoogleBudgetConfig
GoogleLevelConfig
```

Suggested constraints:

| Config | Model family | Controls |
| --- | --- | --- |
| `GoogleLegacyConfig` | Models without known reasoning capability | no reasoning |
| `GoogleBudgetConfig` | Gemini 2.5 Flash/Lite/Pro | `off`, `adaptive`, `budget`; model-specific budget ranges |
| `GoogleLevelConfig` | Gemini 3 / Gemma 4 | literal thinking levels from capability table |

Known budget ranges:

| Model family | Budget range |
| --- | --- |
| `gemini-2.5-flash-lite*` | `512..24576` |
| `gemini-2.5-flash*` | `1..24576` |
| `gemini-2.5-pro*` | `128..32768` |

Known level families:

| Model family | Levels |
| --- | --- |
| `gemini-3*` | `minimal`, `low`, `medium`, `high` |
| `gemma-4*` | `minimal`, `high` |

## GLM

Current inferred source:

- GLM capability rules for `glm-5`, `glm-4.7`, `glm-4.6`, and `glm-4.5`.

Recommended authoring configs:

```python
GlmLegacyConfig
GlmThinkingConfig
```

Suggested constraints:

| Config | Model family | Controls |
| --- | --- | --- |
| `GlmLegacyConfig` | Models without known GLM thinking support | no reasoning |
| `GlmThinkingConfig` | `glm-5*`, `glm-4.7*`, `glm-4.6*`, `glm-4.5*` | `off`, `adaptive` |

## OpenRouter

Current inferred sources:

- `openrouter/data/model_policies.yml`
- `OpenRouterModelPolicy`

Recommended authoring configs:

```python
OpenRouterNoReasoningConfig
OpenRouterToggleConfig
OpenRouterEffortConfig
```

Suggested constraints:

| Config | Policy style | Controls |
| --- | --- | --- |
| `OpenRouterNoReasoningConfig` | `request_style: none` | no reasoning |
| `OpenRouterToggleConfig` | `request_style: enabled_flag` | `enabled=True/False`, respecting `supports_disable` |
| `OpenRouterEffortConfig` | `request_style: effort` | one of `allowed_efforts` |

OpenRouter should remain policy-driven rather than class-per-model. The YAML
file already encodes exact model decisions, verification notes, disable support,
and allowed efforts.

## Codex

Current inferred source:

- Codex capability rules and thinking-family helpers.

Recommended authoring configs:

```python
CodexLegacyConfig
CodexGpt5Config
CodexGpt51Config
CodexGpt52Config
CodexGpt54Config
CodexGpt5CodexConfig
```

Suggested constraints:

| Config | Model family | Thinking levels |
| --- | --- | --- |
| `CodexLegacyConfig` | Models without known configurable thinking | none |
| `CodexGpt5Config` | `gpt-5*` base family | `minimal`, `low`, `medium`, `high`, `xhigh` |
| `CodexGpt51Config` | `gpt-5.1*` | `off`, `low`, `medium`, `high`, `xhigh` |
| `CodexGpt52Config` | `gpt-5.2*` | `off`, `low`, `medium`, `high`, `xhigh` |
| `CodexGpt54Config` | `gpt-5.4*` | `off`, `low`, `medium`, `high`, `xhigh` |
| `CodexGpt5CodexConfig` | `gpt-5-codex*`, `gpt-5.3-codex*`, etc. | `low`, `medium`, `high`, `xhigh`; no `off` unless listed |

The current helper data does not distinguish every Codex snapshot perfectly.
Use the existing support lists as the source of truth and validate with
family matching.

## Claude Code

Current inferred sources:

- Claude Code capability rules
- Anthropic effort-supported model list
- Anthropic adaptive-thinking model list

Recommended authoring configs:

```python
ClaudeCodeLegacyConfig
ClaudeCodeAdaptiveConfig
ClaudeCodeEffortConfig
```

Suggested constraints:

| Config | Model family | Controls |
| --- | --- | --- |
| `ClaudeCodeLegacyConfig` | `claude-*` models without explicit adaptive/effort support | no explicit thinking, no effort |
| `ClaudeCodeAdaptiveConfig` | Anthropic adaptive-thinking models | `thinking_level=adaptive` |
| `ClaudeCodeEffortConfig` | Anthropic effort-supported models | `effort=low/medium/high/max` as allowed |

Claude Code's model acceptance is broad (`claude-*`), but effort support is
narrow. Keep those two facts separate in the config API.

## Kimi Code

Current inferred source:

- Exact model rule for `kimi-for-coding`.

Recommended authoring config:

```python
KimiCodeConfig
```

Suggested constraints:

| Config | Model | Controls |
| --- | --- | --- |
| `KimiCodeConfig` | `kimi-for-coding` | effort plus Anthropic-style `off`, `adaptive`, or `budget`; budget range `1024..128000` |

This provider is already narrow enough that one config is sufficient.

## MiniMax

Current inferred source:

- MiniMax capability rules for `MiniMax-*`.

Recommended authoring config:

```python
MiniMaxConfig
```

Suggested constraints:

| Config | Model family | Controls |
| --- | --- | --- |
| `MiniMaxConfig` | `MiniMax-*` | effort only; explicit thinking is not supported |

This provider is also narrow enough that one config is sufficient.

## Implementation Order

1. Keep `LlmConfig`, `SamplingControls`, and `LlmRequest` normalized.
2. Move each provider's authoring configs into that provider's `config.py`.
3. Consolidate provider-local capability constants before adding many config
   classes, especially for Anthropic and Claude Code.
4. Make each authoring config validate model-family membership and option
   combinations.
5. Convert demos, tests, and `nl_latents` to instantiate provider-local
   authoring configs and call `to_llm_config(...)`.
6. Remove old config/request subclass exports and tests.

## Open Questions

- Whether to expose all family configs from `dr_llm.llm.__init__`, or only from
  provider modules plus a smaller curated top-level set.
- Whether `OpenAIGpt51Config`, `OpenAIGpt53Config`, and similar era-specific
  configs should be separate classes or aliases over a shared internal base.
- Whether snapshot-specific model fields should use broad `str` plus validators
  or narrower `Literal[...]` lists. Broad `str` handles future dated snapshots;
  literals give stronger LSP hints but age faster.
- Whether legacy/no-reasoning configs should be explicit public classes or just
  default behavior on the provider's broad config.
