from __future__ import annotations

from typing import Any, cast

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
        "published_sample_test_failures",
        "published_tasks",
        "published_nl_latents_samples",
    )
    assert config.published_table_names_for("code_comp_v0") == (
        "published_pool_summaries",
        "published_pool_samples",
        "published_sample_test_failures",
        "published_tasks",
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


class RecordingPublishConnection:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def execute(self, query: object, params: object | None = None) -> None:
        _ = params
        as_string = getattr(query, "as_string", None)
        if callable(as_string):
            self.queries.append(as_string())
            return
        self.queries.append(str(query))


def test_empty_common_samples_table_uses_lean_schema() -> None:
    conn = RecordingPublishConnection()

    neon_publish_module._create_empty_common_samples_table(
        cast(Any, conn), "published_pool_samples__build"
    )

    [query] = conn.queries
    assert "prompt_tokens integer" in query
    assert "total_cost_usd numeric" in query
    assert "error_text text" in query
    assert "validation_time_seconds double precision" in query
    assert "key_values_json" not in query
    assert "request_json jsonb" not in query
    assert "response_json jsonb" not in query
    assert "metadata_json jsonb" not in query
    assert "validation_json jsonb" not in query


def test_empty_sample_test_failures_table_stores_failed_case_payloads() -> (
    None
):
    conn = RecordingPublishConnection()

    neon_publish_module._create_empty_sample_test_failures_table(
        cast(Any, conn), "published_sample_test_failures__build"
    )

    [query] = conn.queries
    assert "case_idx integer" in query
    assert "input_json jsonb" in query
    assert "expected_json jsonb" in query
    assert "actual_json jsonb" in query
    assert "failure_json jsonb" in query


def test_encoder_common_select_projects_usage_cost_and_error_fields() -> None:
    source = neon_publish_module.PoolSource(
        project_name="code_comp_v0",
        pool=neon_publish_module.PublishedPoolConfig(
            source_pool="direct_enc_t0",
            processor=PublishProcessor.encoder_description_v1,
        ),
        source_table="pool_direct_enc_t0_samples",
        source_row_count=1,
        table_columns=frozenset(
            {
                "prompt_template_id",
                "data_sample_id",
                "llm_config_id",
                "status",
            }
        ),
    )

    query = neon_publish_module._encoder_common_select(source).as_string()

    assert "AS prompt_tokens" in query
    assert "AS completion_tokens" in query
    assert "AS total_cost_usd" in query
    assert "AS error_text" in query
    assert " AS request_json" not in query
    assert " AS response_json" not in query
    assert " AS metadata_json" not in query
    assert " AS validation_json" not in query
