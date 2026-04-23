from __future__ import annotations

import logging

import typer

from .models import models_app
from .pool import pool_app
from .project import project_app
from .providers import register as register_providers
from .query import register as register_query


def _get_root_logger() -> logging.Logger:
    return logging.getLogger()


def _configure_cli_logging() -> None:
    root_logger = _get_root_logger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return
    root_logger.setLevel(logging.INFO)


app = typer.Typer()


@app.callback()
def main() -> None:
    _configure_cli_logging()


app.add_typer(models_app, name="models")
app.add_typer(pool_app, name="pool")
app.add_typer(project_app, name="project")
register_providers(app)
register_query(app)
