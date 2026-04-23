from __future__ import annotations

import json

import typer
from pydantic import ValidationError

from dr_llm.cli.common import handle_cli_errors
from dr_llm.pool.admin_service import delete_pool, delete_pools_by_token
from dr_llm.pool.errors import PoolError
from dr_llm.pool.models import DeletePoolRequest, DeletePoolsByTokenRequest
from dr_llm.project.errors import ProjectError

pool_app = typer.Typer(help="Manage dr-llm pools")


@pool_app.command("destroy")
@handle_cli_errors(ProjectError, PoolError, ValidationError)
def pool_destroy(
    project_name: str = typer.Argument(..., help="Project name"),
    pool_name: str = typer.Argument(..., help="Pool name"),
    yes_really_delete_everything: bool = typer.Option(
        False,
        "--yes-really-delete-everything",
        help="Skip confirmation prompt.",
    ),
) -> None:
    if not yes_really_delete_everything:
        typer.confirm(
            "This will permanently delete the pool and all of its tables. Continue?",
            abort=True,
        )

    result = delete_pool(
        DeletePoolRequest(project_name=project_name, pool_name=pool_name)
    )
    typer.echo(
        json.dumps(
            result.model_dump(
                mode="json",
                exclude_none=True,
                exclude_computed_fields=True,
            ),
            indent=2,
        )
    )
    if not result.success:
        raise typer.Exit(1)


@pool_app.command("destroy-testish")
@handle_cli_errors(ProjectError, PoolError, ValidationError)
def pool_destroy_testish(
    project_name: str = typer.Argument(..., help="Project name"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview matching pools without deleting them.",
    ),
    yes_really_delete_everything: bool = typer.Option(
        False,
        "--yes-really-delete-everything",
        help="Skip confirmation prompt.",
    ),
) -> None:
    if not dry_run and not yes_really_delete_everything:
        typer.confirm(
            "This will permanently delete pools whose underscore-delimited "
            "tokens include test, tst, smoke, or demo. Continue?",
            abort=True,
        )

    result = delete_pools_by_token(
        DeletePoolsByTokenRequest(
            project_name=project_name,
            match_tokens=["test", "tst", "smoke", "demo"],
            dry_run=dry_run,
        )
    )
    typer.echo(
        json.dumps(
            result.model_dump(
                mode="json",
                exclude_none=True,
                exclude_computed_fields=True,
            ),
            indent=2,
        )
    )
    if not result.success:
        raise typer.Exit(1)
