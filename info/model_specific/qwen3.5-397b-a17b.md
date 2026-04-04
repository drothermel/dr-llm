# qwen/qwen3.5-397b-a17b

## Already in our CSV

- Source: `openrouter`
- Price: `$0.390` in / `$2.340` out / `$0.8775` blended
- Size: `17B` active / `397B` total
- Present local metrics:
  - `AA Coding Index = 41.3`
  - `AA Intelligence Index = 45`
  - `AA GPQA = 0.893`
  - `AA HLE = 0.273`
  - `AA SciCode = 0.42`
  - `AA IFBench = 0.788`
  - `AA LCR = 0.657`
  - `AA TerminalBench Hard = 0.409`
  - `AA TAU-2 = 0.956`
  - `HF Aggregate Score = 73.57`
  - `HF AIME 2026 = 93.33`
  - `HF GPQA = 88.4`
  - `HF HLE = 28.7`
  - `HF HMMT 2026 = 87.88`
  - `HF MMLU-Pro = 87.8`
  - `HF SWE-bench Verified = 76.4`
  - `HF TerminalBench = 52.5`

## Official sources

- Hugging Face model card: <https://huggingface.co/Qwen/Qwen3.5-397B-A17B>
- Qwen3.5 family launch post: <https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894>
- Official hosted-model hint in the model card: `Qwen3.5-Plus is the hosted version corresponding to Qwen3.5-397B-A17B`
- Alibaba Model Studio text generation docs mentioning `qwen3.5-plus`: <https://www.alibabacloud.com/help/en/model-studio/text-generation>

## Additional official benchmark reporting found

This is the richest official source in the family and likely the best place to mine extra columns.

### Language, reasoning, search, and coding agent

| Benchmark | Value |
| --- | ---: |
| MMLU-Redux | 94.9 |
| SuperGPQA | 70.4 |
| C-Eval | 93.0 |
| IFEval | 92.6 |
| MultiChallenge | 67.6 |
| LongBench v2 | 63.2 |
| HLE-Verified | 37.6 |
| LiveCodeBench v6 | 83.6 |
| HMMT Feb 25 | 94.8 |
| HMMT Nov 25 | 92.7 |
| IMOAnswerBench | 80.9 |
| BFCL-V4 | 72.9 |
| VITA-Bench | 49.7 |
| DeepPlanning | 34.3 |
| Tool Decathlon | 38.3 |
| MCP-Mark | 46.1 |
| HLE w/ tool | 48.3 |
| BrowseComp | 69.0 or 78.6 depending on strategy |
| BrowseComp-zh | 70.3 |
| WideSearch | 74.0 |
| Seal-0 | 46.9 |
| SWE-bench Multilingual | 69.3 |
| SecCodeBench | 68.3 |

### Multilingual

| Benchmark | Value |
| --- | ---: |
| MMMLU | 88.5 |
| MMLU-ProX | 84.7 |
| NOVA-63 | 59.1 |
| INCLUDE | 85.6 |
| Global PIQA | 89.8 |
| PolyMATH | 73.3 |
| WMT24++ | 78.9 |
| MAXIFE | 88.2 |

### Multimodal and visual-agent reporting

| Benchmark | Value |
| --- | ---: |
| MMMU | 85.0 |
| MMMU-Pro | 79.0 |
| MathVision | 88.6 |
| MathVista (mini) | 90.3 |
| RealWorldQA | 83.9 |
| OCRBench | 93.1 |
| VideoMME (with subtitles) | 87.5 |
| ScreenSpot Pro | 65.6 |
| OSWorld-Verified | 62.2 |
| AndroidWorld | 66.8 |

## Why this is strategically important

- It is the deepest official benchmark table found in the family.
- It is also the best current public proxy for `qwen3.5-plus`.
- It adds several benchmark families that do not appear in the current CSV at all, especially `Tool Decathlon`, `MCP-Mark`, `SWE-bench Multilingual`, `SecCodeBench`, `ScreenSpot Pro`, `OSWorld-Verified`, and `AndroidWorld`.

## Gaps / next searches

- Search benchmark-specific papers or repos for exact methodology and date alignment on `MCP-Mark`, `Tool Decathlon`, and `OSWorld-Verified`.
- Search whether Qwen has published a separate technical report or system card for the hosted `plus` product.
- Decide whether to store dual-value benchmarks like `BrowseComp = 69.0/78.6` as a single text field, two columns, or a separate notes column.

## Takeaway

If the goal is "find the maximum amount of evaluation metadata from a single Qwen3.5 source," this is the strongest current candidate.
