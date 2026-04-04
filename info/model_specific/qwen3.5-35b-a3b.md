# qwen/qwen3.5-35b-a3b

## Already in our CSV

- Source: `openrouter`
- Price: `$0.163` in / `$1.300` out / `$0.4469` blended
- Size: `3B` active / `35B` total
- Present local metrics:
  - `AA Coding Index = 30.3`
  - `AA Intelligence Index = 37.1`
  - `AA GPQA = 0.845`
  - `AA HLE = 0.197`
  - `AA SciCode = 0.377`
  - `AA IFBench = 0.725`
  - `AA LCR = 0.627`
  - `AA TerminalBench Hard = 0.265`
  - `AA TAU-2 = 0.892`
  - `HF Aggregate Score = 68.11`
  - `HF AIME 2026 = 93.33`
  - `HF GPQA = 84.2`
  - `HF HLE = 22.4`
  - `HF HMMT 2026 = 81.82`
  - `HF MMLU-Pro = 85.3`
  - `HF SWE-bench Verified = 69.2`
  - `HF TerminalBench = 40.5`

## Official sources

- Hugging Face model card: <https://huggingface.co/Qwen/Qwen3.5-35B-A3B>
- Qwen3.5 family launch post: <https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894>
- Official hosted-model hint in the model card: `Qwen3.5-Flash is the hosted version corresponding to Qwen3.5-35B-A3B`
- Alibaba Model Studio docs mentioning `qwen3.5-flash`: <https://www.alibabacloud.com/help/en/model-studio/newly-released-models>

## Additional official benchmark reporting found

### Language, coding, agents, and search

| Benchmark | Value |
| --- | ---: |
| MMLU-Redux | 93.3 |
| C-Eval | 90.2 |
| SuperGPQA | 63.4 |
| IFEval | 91.9 |
| MultiChallenge | 60.0 |
| LongBench v2 | 59.0 |
| HMMT Feb 25 | 89.0 |
| HMMT Nov 25 | 89.2 |
| LiveCodeBench v6 | 74.6 |
| CodeForces | 2028 |
| OJBench | 36.0 |
| FullStackBench en | 58.1 |
| FullStackBench zh | 55.0 |
| BFCL-V4 | 67.3 |
| VITA-Bench | 31.9 |
| DeepPlanning | 22.8 |
| HLE w/ tool | 47.4 |
| BrowseComp | 61.0 |
| BrowseComp-zh | 69.5 |
| WideSearch | 57.1 |
| Seal-0 | 41.4 |

### Multilingual

| Benchmark | Value |
| --- | ---: |
| MMMLU | 85.2 |
| MMLU-ProX | 81.0 |
| NOVA-63 | 57.1 |
| INCLUDE | 79.7 |
| Global PIQA | 86.6 |
| PolyMATH | 64.4 |
| WMT24++ | 76.3 |
| MAXIFE | 86.6 |

### Multimodal highlights

| Benchmark | Value |
| --- | ---: |
| MMMU | 81.4 |
| MathVision | 83.9 |
| RealWorldQA | 84.1 |
| OmniDocBench1.5 | 89.3 |
| OCRBench | 91.0 |

## Gaps / next searches

- Search specifically for hosted `flash` benchmark pages, if any, because the official correspondence to `35B-A3B` does not guarantee identical scores.
- Check whether there are product-doc statements about tool use, context limits, or multimodal support that make `flash` materially different from the open-weight model.
- Search for GUI-agent and search-agent result pages tied to the `flash` branding.

## Takeaway

This is probably the best open-weight proxy for studying `qwen3.5-flash`. It also has enough official benchmark coverage to justify adding several more columns to the CSV without leaving primary sources.
