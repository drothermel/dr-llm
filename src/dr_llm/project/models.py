from __future__ import annotations

from pydantic import BaseModel, ConfigDict

DB_NAME = "dr_llm"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DOCKER_IMAGE = "postgres:16"
CONTAINER_PREFIX = "dr-llm-pg-"
VOLUME_PREFIX = "dr-llm-data-"
LABEL_PREFIX = "dr-llm.project"


def container_name(project: str) -> str:
    return f"{CONTAINER_PREFIX}{project}"


def volume_name(project: str) -> str:
    return f"{VOLUME_PREFIX}{project}"


def dsn_for_port(port: int) -> str:
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:{port}/{DB_NAME}"


class ProjectInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    container_name: str
    volume_name: str
    port: int
    status: str
    dsn: str
    created_at: str | None = None
