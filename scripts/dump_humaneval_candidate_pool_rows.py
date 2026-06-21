#!/usr/bin/env python3
from __future__ import annotations

import gzip
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from dr_llm.datetime_utils import UTC
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from humaneval_pool_extract_common import (
    DEFAULT_OUTPUT_ROOT,
    DumpManifest,
    DumpedPoolManifest,
    PoolTarget,
    default_candidate_specs,
    dump_file_name,
    manifest_path_for,
    require_dsn,
    resolve_pool_targets,
    row_count,
    row_to_dumped_pool_row,
    running_project,
    sample_table_name,
    stream_sample_rows,
    timestamped_output_dir,
    write_json_line,
    write_manifest,
)

app = typer.Typer(
    help="Dump candidate dr-llm pool rows for HumanEval analysis."
)


@app.command()
def main(
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Output directory. Defaults to a timestamped analysis path.",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Database fetch batch size."),
    ] = 1000,
) -> None:
    resolved_output_dir = output_dir or timestamped_output_dir(
        DEFAULT_OUTPUT_ROOT
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    manifest = DumpManifest(
        created_at=datetime.now(UTC),
        output_dir=resolved_output_dir,
    )

    dumped_pools: list[DumpedPoolManifest] = []
    for target in resolve_pool_targets(default_candidate_specs()):
        with running_project(target.project_name) as lease:
            runtime = DbRuntime(
                DbConfig(
                    dsn=require_dsn(lease.project),
                    application_name="humaneval_pool_dump",
                )
            )
            try:
                file_name = dump_file_name(
                    target.project_name, target.pool_name
                )
                path = resolved_output_dir / file_name
                table_name = sample_table_name(target.pool_schema)
                expected_count = row_count(runtime, table_name)
                typer.echo(
                    f"Dumping {target.project_name}/{target.pool_name} "
                    f"({expected_count} rows) -> {path}"
                )
                dumped_count = dump_pool_rows(
                    path=path,
                    target=target,
                    runtime=runtime,
                    batch_size=batch_size,
                )
                dumped_pools.append(
                    DumpedPoolManifest(
                        project_name=target.project_name,
                        pool_name=target.pool_name,
                        table_name=table_name,
                        file_name=file_name,
                        row_count=expected_count,
                        dumped_row_count=dumped_count,
                        pool_schema_json=target.pool_schema.model_dump(
                            mode="json"
                        ),
                        original_status=lease.original_status,
                        temporarily_started=lease.temporarily_started,
                    )
                )
                typer.echo(
                    f"Finished {target.project_name}/{target.pool_name}: "
                    f"{dumped_count} rows"
                )
            finally:
                runtime.close()

    write_manifest(
        manifest_path_for(resolved_output_dir),
        manifest.model_copy(update={"pools": dumped_pools}),
    )
    typer.echo(f"Wrote manifest to {manifest_path_for(resolved_output_dir)}")


def dump_pool_rows(
    *,
    path: Path,
    target: PoolTarget,
    runtime: DbRuntime,
    batch_size: int,
) -> int:
    table_name = sample_table_name(target.pool_schema)
    count = 0
    with gzip.open(path, mode="wb", compresslevel=6) as file:
        for row in stream_sample_rows(
            runtime,
            table_name=table_name,
            batch_size=batch_size,
        ):
            dumped = row_to_dumped_pool_row(
                project_name=target.project_name,
                pool_name=target.pool_name,
                schema=target.pool_schema,
                row=row,
            )
            write_json_line(file, dumped)
            count += 1
            if count % 50_000 == 0:
                typer.echo(f"  dumped {count} rows")
    return count


if __name__ == "__main__":
    app()
