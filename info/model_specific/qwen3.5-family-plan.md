# Qwen3.5 Exploration Plan

This plan is for building out richer evaluation coverage for any `qwen/qwen3.5*` row in
[`info/affordable_models_data_annotated.csv`](/Users/daniellerothermel/drotherm/repos/dr-llm/info/affordable_models_data_annotated.csv).

## Models currently in scope

- `qwen/qwen3.5-9b`
- `qwen/qwen3.5-flash-02-23`
- `qwen/qwen3.5-35b-a3b`
- `qwen/qwen3.5-27b`
- `qwen/qwen3.5-plus-02-15`
- `qwen/qwen3.5-122b-a10b`
- `qwen/qwen3.5-397b-a17b`

## First-pass conclusions

- The open-weight models already have unusually rich official benchmark reporting on their Hugging Face model cards.
- `qwen3.5-flash` and `qwen3.5-plus` look like hosted product variants, so public reporting is more product-doc oriented and less benchmark-table oriented.
- The fastest path to meaningful enrichment is not random web search. It is:
  1. harvest official model-card benchmark tables
  2. map hosted models to their closest open-weight sibling
  3. only then search benchmark-specific leaderboards, papers, and blogs

## What worked and what did not

### Highest-yield searches

- Official Hugging Face model cards were by far the best source for numeric benchmark coverage.
- The Alibaba Cloud / Qwen release materials were most useful for understanding model relationships, especially hosted-versus-open correspondences.
- Existing local CSV rows were useful as a starting point for gap analysis, because they made it obvious which benchmark families were already covered and which were still missing.

### Lower-yield searches

- Broad web search for generic `Qwen3.5` benchmark pages was much less useful than expected.
- I did not find an obvious central arXiv technical report that was more informative than the official model cards for the concrete model variants in this family.
- Secondary blogs and roundup posts added little numeric value compared with official cards and product docs.

## Implications for future searches

- Start with model cards before searching the wider web. For this family, model cards are primary sources, not just metadata pages.
- Treat hosted variants as product SKUs first and benchmark targets second. If a model name looks like `flash`, `plus`, or another serving-oriented tier, search for official correspondence to an open-weight sibling before looking for standalone benchmark tables.
- Use benchmark-specific searches only after official sources have been exhausted. That is the most efficient way to spend search time.
- Expect some benchmark values to depend on setup details such as tool use, context management, or Code Interpreter. Those notes matter and should be preserved.
- Do not assume every missing value is merely undiscovered. For some hosted variants, the public evidence may genuinely top out at product descriptions plus a sibling-model proxy.

## Source hierarchy

### Tier 1: official model cards

Use these first for any open-weight variant:

- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-27B`
- `Qwen/Qwen3.5-35B-A3B`
- `Qwen/Qwen3.5-122B-A10B`
- `Qwen/Qwen3.5-397B-A17B`

Why these matter:

- They already expose many metrics that are not in our CSV.
- They usually include language, coding, agent, multilingual, and multimodal results in one place.
- They are the strongest provenance for numeric values.

### Tier 2: official family / product docs

Use these to understand hosted variants and model relationships:

- Qwen3.5 launch post on Alibaba Cloud Community
- Alibaba Cloud Model Studio docs
- model release / lifecycle pages
- deep thinking and text generation docs

Why these matter:

- They describe `qwen3.5-plus` and `qwen3.5-flash`
- They clarify capabilities like built-in tools, long context, and multimodal support
- They sometimes state relationship hints such as hosted/open correspondence

### Tier 3: benchmark-specific sources

After harvesting official model-card tables, search benchmark-specific material for missing or updated entries:

- leaderboard pages
- benchmark GitHub repos
- benchmark papers and tech reports
- evaluation result pages linked from Hugging Face

High-value benchmark families to search next:

- SWE-bench Verified
- Terminal Bench 2
- LiveCodeBench
- HLE / HLE-Verified
- BFCL / tool-use benchmarks
- BrowseComp / search-agent benchmarks
- ScreenSpot Pro / OSWorld / AndroidWorld

### Tier 4: secondary release posts and roundup blogs

Use only after the above. These are helpful for context, but should not be the main numeric source unless they quote primary results clearly.

## Recommended workflow per model

1. Start from the local CSV row.
2. Pull the official model card if it exists.
3. Copy all benchmark families that are not already represented in our CSV.
4. Note naming mismatches between our CSV and source terminology.
5. Search for benchmark-specific source pages only for missing high-value metrics.
6. Record whether each candidate metric is:
   - exact numeric value
   - product claim without a number
   - inferred proxy from a sibling model

## Best bets by model

### Highest expected yield

- `qwen/qwen3.5-9b`
- `qwen/qwen3.5-27b`
- `qwen/qwen3.5-35b-a3b`
- `qwen/qwen3.5-122b-a10b`
- `qwen/qwen3.5-397b-a17b`

These have official model-card tables with substantial extra eval coverage.

### Lower-yield but still important

- `qwen/qwen3.5-flash-02-23`
- `qwen/qwen3.5-plus-02-15`

These appear to be hosted models. Current public evidence is better for capabilities and product positioning than for standalone benchmark tables.

## Suggested next annotation targets

If the goal is to improve the CSV with high-signal columns, the most promising additions are:

- `IFEval`
- `CodeForces`
- `OJBench`
- `BFCL-V4`
- `BrowseComp`
- `BrowseComp-zh`
- `WideSearch`
- `Tool Decathlon`
- `MCP-Mark`
- `SWE-bench Multilingual`
- `SecCodeBench`
- `ScreenSpot Pro`
- `OSWorld-Verified`
- `AndroidWorld`

For multimodal coverage, these are good candidates:

- `MMMU`
- `MathVision`
- `OCRBench`
- `VideoMME`
- `RealWorldQA`

## Important caveats

- Hosted variants may not be numerically identical to the closest open-weight model, even when official docs describe them as corresponding versions.
- Some Qwen sources report the same family under different setups, for example with tools enabled, with Code Interpreter enabled, or with a specific context-management strategy.
- Several official tables include benchmark-specific notes that matter for interpretation. Those notes should be preserved when values are copied into any future machine-readable dataset.
