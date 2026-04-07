# qwen/qwen3.5-plus-02-15

## Already in our CSV

- Source: `openrouter`
- Price: `$0.260` in / `$1.560` out / `$0.5850` blended
- No benchmark fields are currently populated in our local annotated CSV.

## Official sources

- Alibaba text generation docs: <https://www.alibabacloud.com/help/en/model-studio/text-generation>
- Alibaba Model Studio release docs: <https://www.alibabacloud.com/help/en/model-studio/newly-released-models>
- Alibaba deep-thinking docs: <https://www.alibabacloud.com/help/doc-detail/2870973.html>
- Hugging Face `Qwen3.5-397B-A17B` model card: <https://huggingface.co/Qwen/Qwen3.5-397B-A17B>
- Qwen3.5 family launch post: <https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894>

## Strongest current evidence

- The `Qwen3.5-397B-A17B` model card explicitly says `Qwen3.5-Plus is the hosted version corresponding to Qwen3.5-397B-A17B`.
- Alibaba docs describe `qwen3.5-plus` as the strongest hosted Qwen3.5 multimodal model with built-in tool calling.
- The family launch post positions `Qwen3.5-Plus` as the hosted flagship while presenting a deep official benchmark table for `Qwen3.5-397B-A17B`.

## Best proxy benchmark source

Until a standalone official `plus` benchmark table is located, the best quantitative proxy is:

- [`qwen3.5-397b-a17b.md`](/Users/daniellerothermel/drotherm/repos/dr-llm/info/model_specific/qwen3.5-397b-a17b.md)

This is a stronger proxy than `flash -> 35B-A3B`, because the official correspondence is explicit and the `397B-A17B` source table is very rich. It is still a proxy, not proof of exact numerical identity.

## Gaps / next searches

- Search for an official Model Studio or Qwen page that reports standalone `qwen3.5-plus` scores.
- Search for product/system-card material describing any differences between `plus` and `397B-A17B`, especially around tools, context handling, and adaptive reasoning.
- Search benchmark-specific leaderboards only after exhausting official hosted-model docs.

## Takeaway

This is an important row even though it currently lacks direct benchmark numbers in the CSV. The open `397B-A17B` model is almost certainly the best starting point for recovering useful evaluation context around `qwen3.5-plus`.
