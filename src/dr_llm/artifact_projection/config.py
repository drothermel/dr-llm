from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ArtifactProjectionConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DR_LLM_ARTIFACT_PROJECTION_",
        extra="ignore",
        frozen=True,
    )

    artifact_root: Path = Path(".dr_llm/artifacts")
    projection_version: str = "artifact-v1"
    target_shard_bytes: int = Field(default=134_217_728, ge=1)
    chunk_bytes: int = Field(default=8_388_608, ge=1)
    metadata_spill_threshold: int = Field(default=16_384, ge=0)
    durable_consumer: str = "drllm_artifact_projection_v1"
    fetch_batch_size: int = Field(default=10, ge=1)

    @property
    def layout_root(self) -> Path:
        return self.artifact_root / "layouts" / self.projection_version

    @property
    def shard_root(self) -> Path:
        return self.layout_root / "shards"

    @property
    def manifest_root(self) -> Path:
        return self.layout_root / "manifests"

    @property
    def index_path(self) -> Path:
        return self.layout_root / "index" / "artifacts.sqlite3"

    @property
    def staging_root(self) -> Path:
        return self.layout_root / "staging"


__all__ = ["ArtifactProjectionConfig"]
