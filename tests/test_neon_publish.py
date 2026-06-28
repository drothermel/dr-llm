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


def test_default_neon_publish_config_whitelists_publishable_projects() -> None:
    config = load_neon_publish_config()

    assert config.version == 1
    assert config.manifest_table == "published_pool_summaries"
    assert config.published_samples_table == "published_pool_samples"
    assert config.published_table_names_for("nl_latents") == (
        "published_pool_summaries",
        "published_pool_samples",
        "published_nl_latents_samples",
    )
    assert config.published_table_names_for("code_comp_v0") == (
        "published_pool_summaries",
        "published_pool_samples",
    )
    assert {project.project_name for project in config.projects} == {
        "nl_latents",
        "code_comp_v0",
        "code_comp_t1",
        "lla_v0",
        "rsi_v0",
        "icbinb_2026_01_31_attempt",
    }
    nl_latents = config.project_config("nl_latents")
    [pool] = nl_latents.pools
    assert pool.source_pool == "nl_latents"
    assert pool.processor is PublishProcessor.nl_latents_samples_v1
    assert pool.summary_table == "published_pool_samples"

    code_comp_v0 = config.project_config("code_comp_v0")
    processors = {
        pool.source_pool: pool.processor for pool in code_comp_v0.pools
    }
    assert (
        processors["direct_enc_t0"] is PublishProcessor.encoder_description_v1
    )
    assert (
        processors["official_decoder_t0"] is PublishProcessor.decoder_code_v1
    )
    assert "reexport_seed_encoder" not in processors


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
        "projects": [],
        "unexpected": True,
    }
    with pytest.raises(ValidationError):
        NeonPublishConfig(**payload)


def test_neon_publish_config_rejects_duplicate_projects() -> None:
    payload: dict[str, Any] = {
        "version": 1,
        "manifest_table": "published_pool_summaries",
        "projects": [
            {
                "project_name": "nl_latents",
                "pools": [],
            },
            {
                "project_name": "nl_latents",
                "pools": [],
            },
        ],
    }
    with pytest.raises(ValidationError, match="project names"):
        NeonPublishConfig(**payload)


def test_neon_publish_config_rejects_duplicate_project_pools() -> None:
    payload: dict[str, Any] = {
        "version": 1,
        "manifest_table": "published_pool_summaries",
        "projects": [
            {
                "project_name": "nl_latents",
                "pools": [
                    {
                        "source_pool": "nl_latents",
                        "processor": "nl_latents_samples_v1",
                    },
                    {
                        "source_pool": "nl_latents",
                        "processor": "nl_latents_samples_v1",
                    },
                ],
            }
        ],
    }
    with pytest.raises(ValidationError, match="source names"):
        NeonPublishConfig(**payload)
