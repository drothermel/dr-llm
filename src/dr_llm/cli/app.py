from __future__ import annotations

import logging

import typer

from dr_llm.artifact_projection.cli import artifact_projection_app
from dr_llm.env import load_dotenv
from dr_llm.metadata_projection.cli import metadata_projection_app
from dr_llm.streaming_log.cli import streaming_log_app

from .models import models_app
from .pool import pool_app
from .project import project_app
from .providers import register as register_providers
from .query import register as register_query

load_dotenv()


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
app.add_typer(streaming_log_app, name="streaming-log")
app.add_typer(artifact_projection_app, name="artifact-projection")
app.add_typer(metadata_projection_app, name="metadata-projection")
register_providers(app)
register_query(app)
