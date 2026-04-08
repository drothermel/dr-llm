from __future__ import annotations

import typer

from .models import models_app
from .project import project_app
from .providers import register as register_providers
from .query import register as register_query

app = typer.Typer()

app.add_typer(models_app, name="models")
app.add_typer(project_app, name="project")
register_providers(app)
register_query(app)
