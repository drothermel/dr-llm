# qwen/qwen3.5-flash-02-23

## Already in our CSV

- Source: `openrouter`
- Price: `$0.065` in / `$0.260` out / `$0.1138` blended
- No benchmark fields are currently populated in our local annotated CSV.

## Official sources

- Alibaba Model Studio release docs: <https://www.alibabacloud.com/help/en/model-studio/newly-released-models>
- Alibaba deep-thinking docs: <https://www.alibabacloud.com/help/doc-detail/2870973.html>
- Alibaba text generation docs: <https://www.alibabacloud.com/help/doc-detail/2841718.html>
- Hugging Face `Qwen3.5-35B-A3B` model card: <https://huggingface.co/Qwen/Qwen3.5-35B-A3B>

## Strongest current evidence

- The `Qwen3.5-35B-A3B` model card explicitly says `Qwen3.5-Flash is the hosted version corresponding to Qwen3.5-35B-A3B`.
- Alibaba Model Studio docs describe `qwen3.5-flash` as a fast, cost-effective hosted member of the Qwen3.5 family with multimodal support and built-in tool calling.
- The release docs group `qwen3.5-flash` with `qwen3.5-122b-a10b`, `qwen3.5-27b`, and `qwen3.5-35b-a3b`, and say the group offers overall performance comparable to `qwen3.5-plus`.

## Best proxy benchmark source

Until a standalone official `flash` benchmark table is located, the best quantitative proxy is:

- [`qwen3.5-35b-a3b.md`](/Users/daniellerothermel/drotherm/repos/dr-llm/info/model_specific/qwen3.5-35b-a3b.md)

That proxy is useful, but it is still an inference. Hosted product behavior may differ because of:

- built-in tools
- default context length
- reasoning mode defaults
- product-side prompting or routing behavior

## Gaps / next searches

- Search for a dedicated Model Studio page that contains numeric `flash` evaluations rather than descriptive product language.
- Search official release notes for terms like `Qwen3.5-Flash`, `benchmark`, `SWE-bench`, `Terminal Bench`, `LiveCodeBench`, `ScreenSpot`, and `AndroidWorld`.
- Search third-party leaderboard pages only after exhausting official product docs.

## Takeaway

This row is a good target if you want to understand hosted-vs-open relationships, but it is not the best first target if the goal is maximizing immediate numeric benchmark yield from primary sources.
