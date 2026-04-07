# qwen/qwen3.5-27b

## Already in our CSV

- Source: `openrouter`
- Price: `$0.195` in / `$1.560` out / `$0.5363` blended
- Size: `27B` active / `27B` total
- Present local metrics:
  - `AA Coding Index = 34.9`
  - `AA Intelligence Index = 42.1`
  - `AA GPQA = 0.858`
  - `AA HLE = 0.222`
  - `AA SciCode = 0.395`
  - `AA IFBench = 0.756`
  - `AA LCR = 0.673`
  - `AA TerminalBench Hard = 0.326`
  - `AA TAU-2 = 0.939`
  - `HF Aggregate Score = 68.83`
  - `HF AIME 2026 = 90.83`
  - `HF GPQA = 85.5`
  - `HF HLE = 24.3`
  - `HF HMMT 2026 = 81.06`
  - `HF MMLU-Pro = 86.1`
  - `HF SWE-bench Verified = 72.4`
  - `HF TerminalBench = 41.6`

## Official sources

- Hugging Face model card: <https://huggingface.co/Qwen/Qwen3.5-27B>
- Qwen3.5 family launch post: <https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894>
- Qwen blog URL cited by the model card: <https://qwen.ai/blog?id=qwen3.5>

## Additional official benchmark reporting found

### Language, coding, agents, and search

| Benchmark | Value |
| --- | ---: |
| MMLU-Redux | 93.2 |
| C-Eval | 90.5 |
| SuperGPQA | 65.6 |
| IFEval | 95.0 |
| MultiChallenge | 60.8 |
| LongBench v2 | 60.6 |
| HMMT Feb 25 | 92.0 |
| HMMT Nov 25 | 89.8 |
| LiveCodeBench v6 | 80.7 |
| CodeForces | 1899 |
| OJBench | 40.1 |
| FullStackBench en | 60.1 |
| FullStackBench zh | 57.4 |
| BFCL-V4 | 68.5 |
| VITA-Bench | 41.9 |
| DeepPlanning | 22.6 |
| HLE w/ tool | 48.5 |
| BrowseComp | 61.0 |
| BrowseComp-zh | 62.1 |
| WideSearch | 61.1 |
| Seal-0 | 47.2 |

### Multilingual

| Benchmark | Value |
| --- | ---: |
| MMMLU | 85.9 |
| MMLU-ProX | 82.2 |
| NOVA-63 | 58.1 |
| INCLUDE | 81.6 |
| Global PIQA | 87.5 |
| PolyMATH | 71.2 |
| WMT24++ | 77.6 |
| MAXIFE | 88.0 |

### Multimodal highlights

| Benchmark | Value |
| --- | ---: |
| MMMU | 82.3 |
| MathVision | 86.0 |
| MathVista (mini) | 87.8 |
| RealWorldQA | 83.7 |
| MMBench EN-DEV-v1.1 | 92.6 |
| OCRBench | 89.4 |

## Gaps / next searches

- Check whether benchmark-specific leaderboards expose newer values than the official card.
- Search for `ScreenSpot Pro`, `OSWorld`, or GUI-agent reporting for the 27B model specifically.
- Search for any official API-product notes that distinguish the open 27B model from hosted `flash` behavior.

## Takeaway

This row is already strong in the local CSV, but the official model card still adds a large amount of missing context. It is a good candidate for expanding the CSV into a broader coding-and-agent benchmark sheet.
