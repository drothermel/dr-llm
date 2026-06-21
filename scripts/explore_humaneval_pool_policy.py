#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from humaneval_pool_extract_common import (
    PoolPolicySummary,
    build_pool_policy_summary,
    default_candidate_specs,
    require_dsn,
    resolve_pool_targets,
    running_project,
)

app = typer.Typer(help="Explore HumanEval extraction policies for pool rows.")


@app.command()
def main(
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSON summary output path."),
    ] = None,
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", min=1, help="Rows to sample per pool."),
    ] = 100,
) -> None:
    summaries: list[PoolPolicySummary] = []
    for target in resolve_pool_targets(default_candidate_specs()):
        with running_project(target.project_name) as lease:
            runtime = DbRuntime(
                DbConfig(
                    dsn=require_dsn(lease.project),
                    application_name="humaneval_policy_explore",
                )
            )
            try:
                summary = build_pool_policy_summary(
                    target=target,
                    runtime=runtime,
                    sample_limit=sample_limit,
                )
                summaries.append(summary)
                typer.echo(
                    f"{summary.project_name}/{summary.pool_name}: "
                    f"rows={summary.row_count} "
                    f"key_humaneval={summary.key_human_eval_count} "
                    f"sample_humaneval={summary.text_human_eval_sample_count} "
                    f"decoder_sample={summary.decoder_candidate_sample_count} "
                    f"description_sample={summary.decoder_description_sample_count}"
                )
            finally:
                runtime.close()

    payload = [summary.model_dump(mode="json") for summary in summaries]
    if output is None:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Wrote policy summary to {output}")


if __name__ == "__main__":
    app()
