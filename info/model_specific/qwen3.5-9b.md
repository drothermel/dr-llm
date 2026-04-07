# qwen/qwen3.5-9b

## Already in our CSV

- Source: `openrouter`
- Price: `$0.050` in / `$0.150` out / `$0.0750` blended
- Size: `9B` active / `9B` total
- Present local metrics:
  - `AA Coding Index = 25.3`
  - `AA Intelligence Index = 32.4`
  - `AA GPQA = 0.806`
  - `AA HLE = 0.133`
  - `AA SciCode = 0.275`
  - `AA IFBench = 0.667`
  - `AA LCR = 0.59`
  - `AA TerminalBench Hard = 0.242`
  - `AA TAU-2 = 0.868`
  - `HF Aggregate Score = 81.98`
  - `HF AIME 2026 = 92.5`
  - `HF GPQA = 81.7`
  - `HF HMMT 2026 = 71.21`
  - `HF MMLU-Pro = 82.5`

## Official sources

- Hugging Face model card: <https://huggingface.co/Qwen/Qwen3.5-9B>
- Qwen3.5 family launch post: <https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894>
- Qwen blog URL cited by the model card: <https://qwen.ai/blog?id=qwen3.5>

## Additional official benchmark reporting found

The model card exposes much more than the current CSV row.

### Language, reasoning, coding, and agents

| Benchmark | Value |
| --- | ---: |
| MMLU-Redux | 91.1 |
| C-Eval | 88.2 |
| SuperGPQA | 58.2 |
| GPQA Diamond | 81.7 |
| IFEval | 91.5 |
| IFBench | 64.5 |
| MultiChallenge | 54.5 |
| AA-LCR | 63.0 |
| LongBench v2 | 55.2 |
| HMMT Feb 25 | 83.2 |
| HMMT Nov 25 | 82.9 |
| LiveCodeBench v6 | 65.6 |
| OJBench | 29.2 |
| BFCL-V4 | 66.1 |
| TAU2-Bench | 79.1 |
| VITA-Bench | 29.8 |
| DeepPlanning | 18.0 |

### Multilingual

| Benchmark | Value |
| --- | ---: |
| MMMLU | 81.2 |
| MMLU-ProX | 76.3 |
| NOVA-63 | 55.9 |
| INCLUDE | 75.6 |
| Global PIQA | 83.2 |
| PolyMATH | 57.3 |
| WMT24++ | 72.6 |
| MAXIFE | 83.4 |

### Multimodal / vision-language

| Benchmark | Value |
| --- | ---: |
| MMMU | 78.4 |
| MMMU-Pro | 70.1 |
| MathVision | 78.9 |
| MathVista (mini) | 85.7 |
| We-Math | 75.2 |
| DynaMath | 83.6 |
| VlmsAreBlind | 93.7 |
| RealWorldQA | 80.8 |
| MMStar | 80.0 |
| OCRBench | 89.0 |

## Why this row is a good pilot

- It already has enough local coverage to anchor matching quality.
- It is missing many metrics that the official model card does report.
- It is small and cheap, so it is a practical candidate for future direct re-evaluation if needed.

## Gaps / next searches

- Search for benchmark-specific result pages linked through Hugging Face `Eval Results`.
- Search for hosted API documentation or release notes that discuss whether the public API exposes the same reasoning/tool configuration as the open-weight card evaluation.
- Search benchmark-specific papers for methodology details on `BFCL-V4`, `TAU2-Bench`, and `DeepPlanning`.

## Takeaway

This is probably the best first enrichment target if the goal is to prove that a single Qwen3.5 row can be expanded far beyond the current AA/HF/Vantage coverage using mostly official sources.
