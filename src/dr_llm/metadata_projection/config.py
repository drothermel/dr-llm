from __future__ import annotations

from os import getenv
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dr_llm.artifact_projection.config import ArtifactProjectionConfig


def _database_dsn() -> str:
    return getenv(
        "DR_LLM_METADATA_PROJECTION_DATABASE_DSN",
        getenv("DR_LLM_DATABASE_URL", "postgresql://localhost/dr_llm"),
    )


class MetadataProjectionConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DR_LLM_METADATA_PROJECTION_",
        extra="ignore",
        frozen=True,
    )

    database_dsn: str = Field(default_factory=_database_dsn)
    projection_version: str = "metadata-v1"
    durable_consumer: str = "drllm_metadata_projection_v1"
    artifact_attach_consumer: str = "drllm_metadata_artifact_attach_v1"
    fetch_batch_size: int = Field(default=10, ge=1)
    artifact_index_path: Path = Field(
        default_factory=lambda: ArtifactProjectionConfig().index_path
    )
    application_name: str = "dr_llm_metadata_projection"


__all__ = ["MetadataProjectionConfig"]
