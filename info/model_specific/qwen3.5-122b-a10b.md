# qwen/qwen3.5-122b-a10b

## Already in our CSV

- Source: `openrouter`
- Price: `$0.260` in / `$2.080` out / `$0.7150` blended
- Size: `10B` active / `122B` total
- Present local metrics:
  - `AA Coding Index = 34.7`
  - `AA Intelligence Index = 41.6`
  - `AA GPQA = 0.857`
  - `AA HLE = 0.234`
  - `AA SciCode = 0.42`
  - `AA IFBench = 0.757`
  - `AA LCR = 0.667`
  - `AA TerminalBench Hard = 0.311`
  - `AA TAU-2 = 0.936`
  - `HF Aggregate Score = 64.0`
  - `HF GPQA = 86.6`
  - `HF HLE = 25.3`
  - `HF MMLU-Pro = 86.7`
  - `HF SWE-bench Verified = 72.0`
  - `HF TerminalBench = 49.4`

## Official sources

- Hugging Face model card: <https://huggingface.co/Qwen/Qwen3.5-122B-A10B>
- Qwen3.5 family launch post: <https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894>
- Qwen blog URL cited by the model card: <https://qwen.ai/blog?id=qwen3.5>

## Additional official benchmark reporting found

### Language, coding, agents, and search

| Benchmark | Value |
| --- | ---: |
| MMLU-Redux | 94.0 |
| C-Eval | 91.9 |
| SuperGPQA | 67.1 |
| IFEval | 93.4 |
| MultiChallenge | 61.5 |
| LongBench v2 | 60.2 |
| HMMT Feb 25 | 91.4 |
| HMMT Nov 25 | 90.3 |
| LiveCodeBench v6 | 78.9 |
| CodeForces | 2100 |
| OJBench | 39.5 |
| FullStackBench en | 62.6 |
| FullStackBench zh | 58.7 |
| BFCL-V4 | 72.2 |
| VITA-Bench | 33.6 |
| DeepPlanning | 24.1 |
| HLE w/ tool | 47.5 |
| BrowseComp | 63.8 |
| BrowseComp-zh | 69.9 |
| WideSearch | 60.5 |
| Seal-0 | 44.1 |

### Multilingual

| Benchmark | Value |
| --- | ---: |
| MMMLU | 86.7 |
| MMLU-ProX | 82.2 |
| NOVA-63 | 58.6 |
| INCLUDE | 82.8 |
| Global PIQA | 88.4 |
| PolyMATH | 68.9 |
| WMT24++ | 78.3 |
| MAXIFE | 87.9 |

### Multimodal highlights

| Benchmark | Value |
| --- | ---: |
| MMMU | 83.9 |
| MMMU-Pro | 76.9 |
| MathVision | 86.2 |
| RealWorldQA | 85.1 |
| MMBench EN-DEV-v1.1 | 92.8 |
| OmniDocBench1.5 | 89.8 |
| OCRBench | 92.1 |
| VideoMME (with subtitles) | 87.3 |

## Gaps / next searches

- Search for official result pages for `ScreenSpot Pro`, because Hugging Face evaluation snippets suggest this model has some external eval artifacts.
- Search for more detailed coding-agent reporting such as `SecCodeBench` or `SWE-bench Multilingual`.
- Compare the card’s benchmark notes against the local CSV naming and decide whether some values can be safely normalized into existing columns.

## Takeaway

This model has broad official coverage already. It is a strong candidate if the goal is to extend the CSV with more coding, search-agent, and multimodal fields while staying close to primary sources.
