from __future__ import annotations

import typer

from dr_llm.providers import build_default_registry

from . import common


def register(app: typer.Typer) -> None:
    app.command("providers")(providers)


def providers(
    json_output: bool = typer.Option(
        False,
        "--json/--no-json",
        help="Emit JSON output.",
    ),
) -> None:
    """List supported providers and whether they are available locally."""
    registry = build_default_registry()
    statuses = registry.availability_statuses()
    if json_output:
        common._emit(
            {
                "providers": [
                    status.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    )
                    for status in statuses
                ]
            }
        )
        return
    common._render_providers_table(statuses)
