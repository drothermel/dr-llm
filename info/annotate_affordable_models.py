from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import cast

import typer
from pydantic import BaseModel

app = typer.Typer(add_completion=False, no_args_is_help=False)

REPO_ROOT = Path("/Users/daniellerothermel/drotherm/repos/dr-llm")
INFER_EXPLORE_ROOT = Path("/Users/daniellerothermel/drotherm/repos/infer-explore")

INPUT_CSV_PATH = REPO_ROOT / "info" / "affordable_models_data.csv"
OUTPUT_CSV_PATH = REPO_ROOT / "info" / "affordable_models_data_annotated.csv"

AA_CSV_PATH = INFER_EXPLORE_ROOT / "data" / "artificial_analysis_models.csv"
HF_CSV_PATH = INFER_EXPLORE_ROOT / "data" / "huggingface_openevals_models.csv"
VANTAGE_CSV_PATH = INFER_EXPLORE_ROOT / "data" / "vantage_models.csv"

AA_METRIC_FIELDS = (
    "AA Intelligence Index",
    "AA Coding Index",
    "AA Math Index",
    "MMLU-Pro",
    "GPQA",
    "HLE",
    "LiveCodeBench",
    "SciCode",
    "MATH-500",
    "AIME",
    "AIME 2025",
    "IFBench",
    "LCR",
    "TerminalBench Hard",
    "TAU-2",
    "output_tokens_per_sec",
    "time_to_first_token_s",
)

HF_METRIC_FIELDS = (
    "aggregate_score",
    "coverage_count",
    "coverage_percent",
    "AIME 2026",
    "EvasionBench",
    "GPQA",
    "GSM8K",
    "HLE",
    "HMMT 2026",
    "MMLU-Pro",
    "OlmOCR",
    "SWE-Pro",
    "SWE-bench Verified",
    "TerminalBench",
)

VANTAGE_METRIC_FIELDS = (
    "swe_bench_resolved_pct",
    "swe_bench_cost_per_resolved",
    "skatebench_score",
    "skatebench_cost_per_test",
    "hle_pct",
)

AA_OUTPUT_FIELDS = (
    "AA Intelligence Index (AA)",
    "AA Coding Index (AA)",
    "AA Math Index (AA)",
    "AA MMLU-Pro",
    "AA GPQA",
    "AA HLE",
    "AA LiveCodeBench",
    "AA SciCode",
    "AA MATH-500",
    "AA AIME",
    "AA AIME 2025",
    "AA IFBench",
    "AA LCR",
    "AA TerminalBench Hard",
    "AA TAU-2",
)

HF_OUTPUT_FIELDS = (
    "HF Aggregate Score",
    "HF AIME 2026",
    "HF EvasionBench",
    "HF GPQA",
    "HF GSM8K",
    "HF HLE",
    "HF HMMT 2026",
    "HF MMLU-Pro",
    "HF OlmOCR",
    "HF SWE-Pro",
    "HF SWE-bench Verified",
    "HF TerminalBench",
)

VANTAGE_OUTPUT_FIELDS = (
    "Vantage SWE-bench Resolved %",
    "Vantage SkateBench Score",
    "Vantage HLE %",
)

APPENDED_FIELDS = (
    *AA_OUTPUT_FIELDS,
    *HF_OUTPUT_FIELDS,
    *VANTAGE_OUTPUT_FIELDS,
)

PROVIDER_ALIASES = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "gemini": "google",
    "meta": "meta",
    "metallama": "meta",
    "meta-llama": "meta",
    "zai": "zai",
    "zaiinc": "zai",
    "z ai": "zai",
    "moonshotai": "kimi",
    "kimi": "kimi",
    "minimax": "minimax",
    "deepseek": "deepseek",
    "mistral": "mistral",
    "mistralai": "mistral",
    "qwen": "qwen",
    "alibaba": "qwen",
    "baidu": "baidu",
    "bytedance": "bytedance",
    "stepfun": "stepfun",
    "xiaomi": "xiaomi",
    "xiaomimimo": "xiaomi",
    "nvidia": "nvidia",
}

QUALIFIER_PATTERN = re.compile(
    r"\b(preview|reasoning|nonreasoning|non-reasoning|experimental|latest|exp)\b",
    re.IGNORECASE,
)
PAREN_PATTERN = re.compile(r"\([^)]*\)")


class SourceRecord(BaseModel):
    row: dict[str, str]
    aliases: set[str]
    tokens: set[str]
    display_name: str
    provider_key: str
    completeness: int
    reasoning_variant: str | None = None


class MatchResult(BaseModel):
    record: SourceRecord | None = None
    method: str = ""
    candidate_count: int = 0
    candidate_names: list[str] = []


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(cast(dict[str, str], row)) for row in csv.DictReader(f)]


def _canonical_provider(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", value.lower())
    return PROVIDER_ALIASES.get(normalized, normalized)


def _normalize_model_text(value: str) -> str:
    lowered = value.lower()
    without_parens = PAREN_PATTERN.sub(" ", lowered)
    without_qualifiers = QUALIFIER_PATTERN.sub(" ", without_parens)
    alnum_only = re.sub(r"[^a-z0-9]+", "", without_qualifiers)
    return alnum_only


def _candidate_aliases(value: str) -> set[str]:
    pieces = {value}
    stripped = PAREN_PATTERN.sub(" ", value)
    pieces.add(stripped)
    if "/" in value:
        tail = value.split("/", 1)[1]
        pieces.add(tail)
        pieces.add(PAREN_PATTERN.sub(" ", tail))
    aliases = {_normalize_model_text(piece) for piece in pieces}
    return {alias for alias in aliases if alias}


def _model_tokens(value: str) -> set[str]:
    lowered = value.lower()
    without_parens = PAREN_PATTERN.sub(" ", lowered)
    without_qualifiers = QUALIFIER_PATTERN.sub(" ", without_parens)
    pieces = re.split(r"[^a-z0-9]+", without_qualifiers)
    return {piece for piece in pieces if piece}


def _classify_reasoning_expectation(value: str) -> str:
    lowered = value.lower().strip()
    if lowered == "no":
        return "non_reasoning"
    if lowered:
        return "reasoning"
    return "unknown"


def _classify_source_reasoning_variant(name: str) -> str | None:
    lowered = name.lower()
    if "non-reasoning" in lowered or "non reasoning" in lowered:
        return "non_reasoning"
    if "reasoning" in lowered or any(
        marker in lowered for marker in ("xhigh", "high", "medium", "low")
    ):
        return "reasoning"
    return None


def _nonempty_count(row: dict[str, str], fields: tuple[str, ...]) -> int:
    return sum(1 for field in fields if row.get(field, "").strip())


def _build_aa_records(rows: list[dict[str, str]]) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for row in rows:
        display_name = row.get("name", "")
        records.append(
            SourceRecord(
                row=row,
                aliases=_candidate_aliases(display_name)
                | _candidate_aliases(row.get("slug", "")),
                tokens=_model_tokens(display_name) | _model_tokens(row.get("slug", "")),
                display_name=display_name,
                provider_key=_canonical_provider(row.get("creator", "")),
                completeness=_nonempty_count(row, AA_METRIC_FIELDS),
                reasoning_variant=_classify_source_reasoning_variant(display_name),
            )
        )
    return records


def _build_hf_records(rows: list[dict[str, str]]) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for row in rows:
        display_name = row.get("model_name", "")
        records.append(
            SourceRecord(
                row=row,
                aliases=_candidate_aliases(display_name)
                | _candidate_aliases(row.get("model_id", "")),
                tokens=_model_tokens(display_name) | _model_tokens(row.get("model_id", "")),
                display_name=display_name,
                provider_key=_canonical_provider(row.get("provider", "")),
                completeness=_nonempty_count(row, HF_METRIC_FIELDS),
            )
        )
    return records


def _build_vantage_records(rows: list[dict[str, str]]) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for row in rows:
        display_name = row.get("name", "")
        records.append(
            SourceRecord(
                row=row,
                aliases=_candidate_aliases(display_name)
                | _candidate_aliases(row.get("key", "")),
                tokens=_model_tokens(display_name) | _model_tokens(row.get("key", "")),
                display_name=display_name,
                provider_key=_canonical_provider(row.get("company", "")),
                completeness=_nonempty_count(row, VANTAGE_METRIC_FIELDS),
            )
        )
    return records


def _match_record(
    *,
    model_name: str,
    provider_name: str,
    reasoning_hint: str,
    source_records: list[SourceRecord],
) -> MatchResult:
    query_aliases = _candidate_aliases(model_name)
    query_tokens = _model_tokens(model_name)
    query_provider = _canonical_provider(provider_name)
    query_reasoning = _classify_reasoning_expectation(reasoning_hint)

    ranked: list[tuple[tuple[int, int, int, int, int, int], SourceRecord, str]] = []
    for record in source_records:
        alias_intersection = record.aliases & query_aliases
        if alias_intersection:
            method = "exact"
            tier = 3
            closeness = max(len(alias) for alias in alias_intersection)
        else:
            prefix_closeness = 0
            for query_alias in query_aliases:
                for record_alias in record.aliases:
                    if query_alias and (
                        record_alias.startswith(query_alias)
                        or query_alias.startswith(record_alias)
                    ):
                        prefix_closeness = max(
                            prefix_closeness, min(len(query_alias), len(record_alias))
                        )
            if prefix_closeness:
                token_overlap = len(query_tokens & record.tokens)
                token_ratio = token_overlap / len(query_tokens) if query_tokens else 0.0
                if token_ratio < 0.75:
                    continue
                method = "prefix"
                tier = 2
                closeness = prefix_closeness
            else:
                contains_closeness = 0
                for query_alias in query_aliases:
                    for record_alias in record.aliases:
                        if len(query_alias) < 8 or len(record_alias) < 8:
                            continue
                        if query_alias in record_alias or record_alias in query_alias:
                            contains_closeness = max(
                                contains_closeness,
                                min(len(query_alias), len(record_alias)),
                            )
                if not contains_closeness:
                    continue
                token_overlap = len(query_tokens & record.tokens)
                token_ratio = token_overlap / len(query_tokens) if query_tokens else 0.0
                if token_ratio < 0.75:
                    continue
                method = "contains"
                tier = 1
                closeness = contains_closeness

        provider_score = 1 if query_provider and query_provider == record.provider_key else 0
        token_overlap = len(query_tokens & record.tokens)

        reasoning_score = 0
        if record.reasoning_variant is not None:
            if query_reasoning == record.reasoning_variant:
                reasoning_score = 2
            elif query_reasoning != "unknown":
                reasoning_score = -1

        score = (
            tier,
            provider_score,
            token_overlap,
            reasoning_score,
            record.completeness,
            closeness,
        )
        ranked.append((score, record, method))

    if not ranked:
        return MatchResult()

    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best_record, best_method = ranked[0]
    candidate_names = [
        record.display_name
        for score, record, _ in ranked
        if score[:2] == best_score[:2]
    ]
    return MatchResult(
        record=best_record,
        method=best_method,
        candidate_count=len(candidate_names),
        candidate_names=candidate_names,
    )


def _with_default(value: str, default: str = "") -> str:
    stripped = value.strip()
    return stripped if stripped else default


def _parse_numeric(value: str) -> float | None:
    cleaned = value.strip().replace("$", "").replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_ratio(value: float) -> str:
    return f"{value:.4f}"


def _with_ratio_column(row: dict[str, str]) -> dict[str, str]:
    output: dict[str, str] = {}
    coding_index = _parse_numeric(row.get("AA Coding Index", ""))
    input_price = _parse_numeric(row.get("$/M In", ""))
    output_price = _parse_numeric(row.get("$/M Out", ""))
    blended_price = _parse_numeric(row.get("Blended $/M", ""))
    blended_ratio = ""
    input_ratio = ""
    output_ratio = ""
    if coding_index is not None:
        if blended_price not in (None, 0.0):
            blended_ratio = _format_ratio(coding_index / blended_price)
        if input_price not in (None, 0.0):
            input_ratio = _format_ratio(coding_index / input_price)
        if output_price not in (None, 0.0):
            output_ratio = _format_ratio(coding_index / output_price)

    for key, value in row.items():
        output[key] = value
        if key == "Total Size":
            output["AA Coding Index / Blended $"] = blended_ratio
            output["AA Coding Index / Input $"] = input_ratio
            output["AA Coding Index / Output $"] = output_ratio
    return output


def _annotation_fields(
    affordable_row: dict[str, str],
    aa_match: MatchResult,
    hf_match: MatchResult,
    vantage_match: MatchResult,
) -> dict[str, str]:
    aa_row = aa_match.record.row if aa_match.record is not None else {}
    hf_row = hf_match.record.row if hf_match.record is not None else {}
    vantage_row = vantage_match.record.row if vantage_match.record is not None else {}

    annotated = {
        "AA Intelligence Index (AA)": aa_row.get("AA Intelligence Index", ""),
        "AA Coding Index (AA)": aa_row.get("AA Coding Index", ""),
        "AA Math Index (AA)": aa_row.get("AA Math Index", ""),
        "AA MMLU-Pro": aa_row.get("MMLU-Pro", ""),
        "AA GPQA": aa_row.get("GPQA", ""),
        "AA HLE": aa_row.get("HLE", ""),
        "AA LiveCodeBench": aa_row.get("LiveCodeBench", ""),
        "AA SciCode": aa_row.get("SciCode", ""),
        "AA MATH-500": aa_row.get("MATH-500", ""),
        "AA AIME": aa_row.get("AIME", ""),
        "AA AIME 2025": aa_row.get("AIME 2025", ""),
        "AA IFBench": aa_row.get("IFBench", ""),
        "AA LCR": aa_row.get("LCR", ""),
        "AA TerminalBench Hard": aa_row.get("TerminalBench Hard", ""),
        "AA TAU-2": aa_row.get("TAU-2", ""),
        "HF Aggregate Score": hf_row.get("aggregate_score", ""),
        "HF AIME 2026": hf_row.get("AIME 2026", ""),
        "HF EvasionBench": hf_row.get("EvasionBench", ""),
        "HF GPQA": hf_row.get("GPQA", ""),
        "HF GSM8K": hf_row.get("GSM8K", ""),
        "HF HLE": hf_row.get("HLE", ""),
        "HF HMMT 2026": hf_row.get("HMMT 2026", ""),
        "HF MMLU-Pro": hf_row.get("MMLU-Pro", ""),
        "HF OlmOCR": hf_row.get("OlmOCR", ""),
        "HF SWE-Pro": hf_row.get("SWE-Pro", ""),
        "HF SWE-bench Verified": hf_row.get("SWE-bench Verified", ""),
        "HF TerminalBench": hf_row.get("TerminalBench", ""),
        "Vantage SWE-bench Resolved %": vantage_row.get("swe_bench_resolved_pct", ""),
        "Vantage SkateBench Score": vantage_row.get("skatebench_score", ""),
        "Vantage HLE %": vantage_row.get("hle_pct", ""),
    }

    if not _with_default(affordable_row.get("AA Coding Index", "")):
        affordable_row["AA Coding Index"] = aa_row.get("AA Coding Index", "")
    if not _with_default(affordable_row.get("AA LiveCodeBench", "")):
        affordable_row["AA LiveCodeBench"] = aa_row.get("LiveCodeBench", "")

    return annotated


def annotate_affordable_models() -> Path:
    for path in (INPUT_CSV_PATH, AA_CSV_PATH, HF_CSV_PATH, VANTAGE_CSV_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    affordable_rows = _read_csv_rows(INPUT_CSV_PATH)
    aa_records = _build_aa_records(_read_csv_rows(AA_CSV_PATH))
    hf_records = _build_hf_records(_read_csv_rows(HF_CSV_PATH))
    vantage_records = _build_vantage_records(_read_csv_rows(VANTAGE_CSV_PATH))

    output_rows: list[dict[str, str]] = []
    aa_matches = 0
    hf_matches = 0
    vantage_matches = 0

    for affordable_row in affordable_rows:
        aa_match = _match_record(
            model_name=affordable_row["Model"],
            provider_name=affordable_row["Provider"],
            reasoning_hint=affordable_row.get("Thinking/Reasoning", ""),
            source_records=aa_records,
        )
        hf_match = _match_record(
            model_name=affordable_row["Model"],
            provider_name=affordable_row["Provider"],
            reasoning_hint=affordable_row.get("Thinking/Reasoning", ""),
            source_records=hf_records,
        )
        vantage_match = _match_record(
            model_name=affordable_row["Model"],
            provider_name=affordable_row["Provider"],
            reasoning_hint=affordable_row.get("Thinking/Reasoning", ""),
            source_records=vantage_records,
        )

        if aa_match.record is not None:
            aa_matches += 1
        if hf_match.record is not None:
            hf_matches += 1
        if vantage_match.record is not None:
            vantage_matches += 1

        annotated_row = dict(affordable_row)
        for field in ("Active Size", "Total Size"):
            if annotated_row.get(field, "").strip() == "NA":
                annotated_row[field] = ""
        annotation_fields = _annotation_fields(
            annotated_row, aa_match, hf_match, vantage_match
        )
        annotated_row = _with_ratio_column(annotated_row)
        annotated_row.update(annotation_fields)
        output_rows.append(annotated_row)

    output_fields = list(output_rows[0].keys()) if output_rows else []
    with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    typer.echo(f"Wrote {len(output_rows)} rows to {OUTPUT_CSV_PATH}")
    typer.echo(f"AA matches: {aa_matches}")
    typer.echo(f"HF matches: {hf_matches}")
    typer.echo(f"Vantage matches: {vantage_matches}")
    return OUTPUT_CSV_PATH


@app.command()
def main() -> None:
    """Annotate the affordable model CSV with benchmark data from infer-explore."""
    annotate_affordable_models()


if __name__ == "__main__":
    app()
