from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

import dr_llm.project.neon_publish as neon_publish_module
from dr_llm.project.docker_project_metadata import ContainerStatus
from dr_llm.project.errors import ProjectError
from dr_llm.project.neon_publish import (
    NeonPublishConfig,
    PublishProcessor,
    load_neon_publish_config,
    publish_project_for_neon,
)
from dr_llm.project.project_info import ProjectInfo


def test_default_neon_publish_config_whitelists_nl_latents() -> None:
    config = load_neon_publish_config()

    assert config.version == 1
    assert config.manifest_table == "published_pool_summaries"
    assert config.published_table_names == (
        "published_pool_summaries",
        "published_nl_latents_samples",
    )
    assert len(config.pools) == 1
    [pool] = config.pools
    assert pool.source_pool == "nl_latents"
    assert pool.processor is PublishProcessor.nl_latents_samples_v1
    assert pool.summary_table == "published_nl_latents_samples"


def test_publish_project_for_neon_requires_running_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        neon_publish_module,
        "get_project",
        lambda _name: ProjectInfo(
            name="nl_latents",
            port=5502,
            status=ContainerStatus.STOPPED,
        ),
    )

    with pytest.raises(ProjectError, match="start it first"):
        publish_project_for_neon("nl_latents")


def test_neon_publish_config_rejects_unknown_keys() -> None:
    payload: dict[str, Any] = {
        "version": 1,
        "manifest_table": "published_pool_summaries",
        "pools": [],
        "unexpected": True,
    }
    with pytest.raises(ValidationError):
        NeonPublishConfig(**payload)


def test_neon_publish_config_rejects_duplicate_summary_tables() -> None:
    payload: dict[str, Any] = {
        "version": 1,
        "manifest_table": "published_pool_summaries",
        "pools": [
            {
                "source_pool": "nl_latents",
                "processor": "nl_latents_samples_v1",
                "summary_table": "published_nl_latents_samples",
            },
            {
                "source_pool": "other_pool",
                "processor": "nl_latents_samples_v1",
                "summary_table": "published_nl_latents_samples",
            },
        ],
    }
    with pytest.raises(ValidationError, match="summary tables"):
        NeonPublishConfig(**payload)
