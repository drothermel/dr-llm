# OpenRouter Reasoning Notes

Date: April 3, 2026

This note records only the live-tested findings for OpenRouter with:

- `openai/gpt-oss-20b`
- `openai/gpt-oss-120b`
- `deepseek/deepseek-chat-v3.1`
- `deepseek/deepseek-r1`
- `deepseek/deepseek-chat`
- `xiaomi/mimo-v2-flash`
- `nvidia/nemotron-nano-9b-v2:free`
- `nvidia/llama-3.1-nemotron-70b-instruct`
- `mistralai/mistral-small-2603`
- `mistralai/mistral-small-3.1-24b-instruct`
- `qwen/qwen3-next-80b-a3b-thinking`
- `qwen/qwen3-next-80b-a3b-instruct`
- `meta-llama/llama-3.1-8b-instruct`

The purpose was to verify:

1. which request configs are accepted
2. whether the response exposes reasoning presence/absence in a usable way

## Test Setup

Prompts used:

- `91 prime? yes/no only.`
- `221 prime? factor if no.`

Requests were sent directly to OpenRouter using the configured
`OPENROUTER_API_KEY`, not through the repo's current local validation layer.

## DeepSeek Notes

For the DeepSeek checks below, only reasoning control params were tested.
`exclude` was not part of these later checks except for the earlier
`deepseek-chat-v3.1` verification that had already been run.

## Other Family Notes

For the Xiaomi, NVIDIA, Mistral, Qwen, and Meta checks below, only these
reasoning control params were tested:

- `{"reasoning": {"enabled": true}}`
- `{"reasoning": {"enabled": false}}`

## Request Acceptance

Accepted on both `openai/gpt-oss-20b` and `openai/gpt-oss-120b`:

- `{"reasoning": {"enabled": true}}`
- `{"reasoning": {"effort": "low"}}`
- `{"reasoning": {"effort": "high"}}`
- `{"reasoning": {"effort": "high", "exclude": true}}`

Rejected on both models:

- `{"reasoning": {"effort": "none"}}`

Observed error:

```text
Reasoning is mandatory for this endpoint and cannot be disabled.
```

## Response Observations

When reasoning was enabled and not excluded:

- `usage.reasoning_tokens` was present and greater than zero
- `message.reasoning` was present
- `message.reasoning_details` was present
- the observed `reasoning_details` entries had `type="reasoning.text"`

When `exclude: true` was used:

- `usage.reasoning_tokens` still showed that reasoning happened
- `message.reasoning` was absent
- `message.reasoning_details` was absent

## Live Results

### `openai/gpt-oss-20b`

Prompt: `91 prime? yes/no only.`

- `{"reasoning": {"enabled": true}}`
  - accepted
  - `reasoning_tokens=25`
  - reasoning visible
- `{"reasoning": {"effort": "high"}}`
  - accepted
  - `reasoning_tokens=22`
  - reasoning visible
- `{"reasoning": {"effort": "none"}}`
  - rejected

Prompt: `221 prime? factor if no.`

- `{"reasoning": {"effort": "low"}}`
  - accepted
  - `reasoning_tokens=35`
  - reasoning visible
  - final answer content present
- `{"reasoning": {"effort": "high"}}`
  - accepted
  - `reasoning_tokens=76`
  - reasoning visible
  - final answer content absent with the tested `max_tokens`
- `{"reasoning": {"effort": "high", "exclude": true}}`
  - accepted
  - `reasoning_tokens=85`
  - reasoning hidden
  - final answer content absent with the tested `max_tokens`

### `openai/gpt-oss-120b`

Prompt: `91 prime? yes/no only.`

- `{"reasoning": {"enabled": true}}`
  - accepted
  - `reasoning_tokens=22`
  - reasoning visible
- `{"reasoning": {"effort": "high"}}`
  - accepted
  - `reasoning_tokens=26`
  - reasoning visible
- `{"reasoning": {"effort": "none"}}`
  - rejected

Prompt: `221 prime? factor if no.`

- `{"reasoning": {"effort": "low"}}`
  - accepted
  - `reasoning_tokens=31`
  - reasoning visible
  - final answer content present
- `{"reasoning": {"effort": "high"}}`
  - accepted
  - `reasoning_tokens=85`
  - reasoning visible
  - final answer content absent with the tested `max_tokens`
- `{"reasoning": {"effort": "high", "exclude": true}}`
  - accepted
  - `reasoning_tokens=80`
  - reasoning hidden
  - final answer content absent with the tested `max_tokens`

## Practical Takeaways

- OpenRouter accepts effort-style reasoning control for `gpt-oss`.
- OpenRouter does not allow reasoning to be disabled for the tested `gpt-oss`
  endpoints.
- `usage.reasoning_tokens` is the most reliable signal that reasoning happened.
- `exclude: true` hides reasoning content but does not hide reasoning-token
  accounting.
- Higher effort increases reasoning-token usage.
- With small `max_tokens`, higher effort can consume enough budget that the final
  answer content is absent.

## Live Verification: OpenRouter + `deepseek/deepseek-chat-v3.1`

Prompt:

- `221 prime? factor if no.`

### Request Acceptance

Accepted:

- `{"reasoning": {"enabled": true}}`
- `{"reasoning": {"enabled": false}}`
- `{"reasoning": {"enabled": true, "exclude": true}}`

### Response Results

With `{"reasoning": {"enabled": true}}`:

- `reasoning_tokens=119`
- `message.reasoning` present
- `message.reasoning_details` present with `type="reasoning.text"`
- final `content` absent with the tested `max_tokens`

With `{"reasoning": {"enabled": false}}`:

- `reasoning_tokens=0`
- no `message.reasoning`
- no `reasoning_details`
- final `content` present

With `{"reasoning": {"enabled": true, "exclude": true}}`:

- `reasoning_tokens=115`
- no visible reasoning fields
- final `content` absent with the tested `max_tokens`

### Practical Takeaways

- For `deepseek/deepseek-chat-v3.1`, OpenRouter accepts boolean on/off reasoning
  control.
- `enabled=true` clearly turns on tracked reasoning.
- `enabled=false` clearly turns off tracked reasoning.
- `exclude=true` hides visible reasoning fields but does not hide
  `reasoning_tokens`.
- As with `gpt-oss`, strong reasoning usage can consume enough output budget that
  no final answer is returned when `max_tokens` is too small.
- With `enabled=false`, the model may still produce step-by-step prose in normal
  `content`; the important distinction is that tracked reasoning is off.

## Live Verification: OpenRouter + `deepseek/deepseek-r1`

Prompt:

- `221 prime? factor if no.`

### Request Acceptance

Accepted:

- `{"reasoning": {"enabled": true}}`

Rejected:

- `{"reasoning": {"enabled": false}}`

Observed rejection:

```text
Reasoning is mandatory for this endpoint and cannot be disabled.
```

### Response Results

With `{"reasoning": {"enabled": true}}`:

- `reasoning_tokens=478`
- `message.reasoning` present
- `message.reasoning_details` present with `type="reasoning.text"`
- final `content` present

### Practical Takeaways

- `deepseek/deepseek-r1` behaves like a reasoning-only endpoint on OpenRouter.
- Reasoning can be used, but not disabled.
- This matches the expectation for a reasoning-first / reasoning-only model.

## Live Verification: OpenRouter + `deepseek/deepseek-chat`

Prompt:

- `221 prime? factor if no.`

### Request Acceptance

Accepted:

- `{"reasoning": {"enabled": true}}`
- `{"reasoning": {"enabled": false}}`

### Response Results

With `{"reasoning": {"enabled": true}}`:

- `reasoning_tokens=0`
- no `message.reasoning`
- no `reasoning_details`
- final `content` present

With `{"reasoning": {"enabled": false}}`:

- `reasoning_tokens=0`
- no `message.reasoning`
- no `reasoning_details`
- final `content` present

### Practical Takeaways

- `deepseek/deepseek-chat` behaves like a non-reasoning endpoint in OpenRouter's
  tracked reasoning sense.
- Sending `reasoning.enabled=true` does not cause tracked reasoning to appear.
- The request is accepted, but the reasoning parameter is effectively ignored for
  reasoning-token purposes.
- The model may still write step-by-step explanation in ordinary `content`, but
  that is not the same as OpenRouter-tracked reasoning.

## Live Verification: OpenRouter + `xiaomi/mimo-v2-flash`

Prompt:

- `221 prime? factor if no.`

### Request Acceptance

Accepted:

- `{"reasoning": {"enabled": true}}`
- `{"reasoning": {"enabled": false}}`

### Response Results

With `{"reasoning": {"enabled": true}}`:

- `reasoning_tokens=189`
- `message.reasoning` present
- `message.reasoning_details` present with `type="reasoning.text"`
- final `content` absent with the tested `max_tokens`

With `{"reasoning": {"enabled": false}}`:

- `reasoning_tokens=0`
- no `message.reasoning`
- no `reasoning_details`
- final `content` present

### Practical Takeaways

- `xiaomi/mimo-v2-flash` behaves like a clean hybrid reasoning model.
- `enabled=true` turns on tracked reasoning.
- `enabled=false` turns off tracked reasoning.

## Live Verification: OpenRouter + `nvidia/nemotron-nano-9b-v2:free`

Prompt:

- `221 prime? factor if no.`

### Request Acceptance

Accepted:

- `{"reasoning": {"enabled": true}}`
- `{"reasoning": {"enabled": false}}`

### Response Results

With `{"reasoning": {"enabled": true}}`:

- `reasoning_tokens=206`
- `message.reasoning` present
- `message.reasoning_details` present with `type="reasoning.text"`
- final `content` absent with the tested `max_tokens`

With `{"reasoning": {"enabled": false}}`:

- `reasoning_tokens=201`
- `message.reasoning` still present
- `message.reasoning_details` still present with `type="reasoning.text"`
- final `content` absent with the tested `max_tokens`

### Practical Takeaways

- `nvidia/nemotron-nano-9b-v2:free` advertises reasoning support, but
  `enabled=false` did not disable tracked reasoning in this test.
- For this tested route, the boolean reasoning control does not appear reliable.

## Live Verification: OpenRouter + `nvidia/llama-3.1-nemotron-70b-instruct`

Prompt:

- `221 prime? factor if no.`

### Request Acceptance

Accepted:

- `{"reasoning": {"enabled": true}}`
- `{"reasoning": {"enabled": false}}`

### Response Results

With both `enabled=true` and `enabled=false`:

- `reasoning_tokens=0`
- no `message.reasoning`
- no `reasoning_details`
- final `content` present

### Practical Takeaways

- `nvidia/llama-3.1-nemotron-70b-instruct` behaves like a non-reasoning endpoint
  in OpenRouter's tracked reasoning sense.
- The reasoning parameter is accepted but appears to be ignored.

## Live Verification: OpenRouter + `mistralai/mistral-small-2603`

Prompt:

- `221 prime? factor if no.`

### Request Acceptance

Accepted:

- `{"reasoning": {"enabled": true}}`
- `{"reasoning": {"enabled": false}}`

### Response Results

With `{"reasoning": {"enabled": true}}`:

- `reasoning_tokens=198`
- `message.reasoning` present
- `message.reasoning_details` present with `type="reasoning.text"`
- final `content` absent with the tested `max_tokens`

With `{"reasoning": {"enabled": false}}`:

- `reasoning_tokens=0`
- no `message.reasoning`
- no `reasoning_details`
- final `content` present

### Practical Takeaways

- `mistralai/mistral-small-2603` behaves like a clean hybrid reasoning model.
- `enabled=true` turns on tracked reasoning.
- `enabled=false` turns off tracked reasoning.

## Live Verification: OpenRouter + `mistralai/mistral-small-3.1-24b-instruct`

Prompt:

- `221 prime? factor if no.`

### Request Acceptance

Accepted:

- `{"reasoning": {"enabled": true}}`
- `{"reasoning": {"enabled": false}}`

### Response Results

With both `enabled=true` and `enabled=false`:

- `reasoning_tokens=0`
- no `message.reasoning`
- no `reasoning_details`
- final `content` present

### Practical Takeaways

- `mistralai/mistral-small-3.1-24b-instruct` behaves like a non-reasoning
  endpoint in OpenRouter's tracked reasoning sense.
- The reasoning parameter is accepted but appears to be ignored.

## Live Verification: OpenRouter + `qwen/qwen3-next-80b-a3b-thinking`

Prompt:

- `221 prime? factor if no.`

### Request Acceptance

Accepted:

- `{"reasoning": {"enabled": true}}`

Rejected:

- `{"reasoning": {"enabled": false}}`

Observed rejection:

```text
Reasoning is mandatory for this endpoint and cannot be disabled.
```

### Response Results

With `{"reasoning": {"enabled": true}}`:

- `reasoning_tokens=588`
- `message.reasoning` present
- `message.reasoning_details` present with `type="reasoning.text"`
- final `content` present

### Practical Takeaways

- `qwen/qwen3-next-80b-a3b-thinking` behaves like a reasoning-only endpoint.
- Reasoning can be used, but not disabled.

## Live Verification: OpenRouter + `qwen/qwen3-next-80b-a3b-instruct`

Prompt:

- `221 prime? factor if no.`

### Request Acceptance

Accepted:

- `{"reasoning": {"enabled": true}}`
- `{"reasoning": {"enabled": false}}`

### Response Results

With both `enabled=true` and `enabled=false`:

- `reasoning_tokens=0`
- no `message.reasoning`
- no `reasoning_details`
- final `content` present

### Practical Takeaways

- `qwen/qwen3-next-80b-a3b-instruct` behaves like a non-thinking / non-tracked
  reasoning endpoint.
- The reasoning parameter is accepted but appears to be ignored.

## Live Verification: OpenRouter + `meta-llama/llama-3.1-8b-instruct`

Prompt:

- `221 prime? factor if no.`

### Request Acceptance

Accepted:

- `{"reasoning": {"enabled": true}}`
- `{"reasoning": {"enabled": false}}`

### Response Results

With both `enabled=true` and `enabled=false`:

- `reasoning_tokens=0`
- no `message.reasoning`
- no `reasoning_details`
- final `content` present

### Practical Takeaways

- `meta-llama/llama-3.1-8b-instruct` behaves like a non-reasoning endpoint in
  OpenRouter's tracked reasoning sense.
- The reasoning parameter is accepted but appears to be ignored.

## Repo-Specific Note

The repo's current local validation layer does not yet allow OpenRouter
`openai/gpt-oss-*` to use the OpenAI-style reasoning path. These findings are
about actual OpenRouter behavior, verified with direct requests.
