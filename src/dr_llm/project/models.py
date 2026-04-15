from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from dr_llm.project.project_info import ProjectInfo

_PROJECT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class CreateProjectRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str

    @field_validator("project_name")
    @classmethod
    def _normalize_project_name(cls, value: str) -> str:
        return value.strip()

    @computed_field
    @property
    def name_is_valid(self) -> bool:
        return bool(_PROJECT_NAME_RE.match(self.project_name))


class ProjectCreationBlockReason(StrEnum):
    invalid_name = "invalid_name"
    already_exists = "already_exists"
    cooldown_active = "cooldown_active"


class ProjectCreationViolation(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: ProjectCreationBlockReason
    message: str
    project_name: str | None = None


class ProjectCreationReadiness(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: CreateProjectRequest
    existing_projects: list[ProjectInfo] = Field(default_factory=list)
    recent_project_names: list[str] = Field(default_factory=list)
    violations: list[ProjectCreationViolation] = Field(default_factory=list)

    @computed_field
    @property
    def allowed(self) -> bool:
        return not self.violations


class ProjectInspectionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    project: ProjectInfo
    pool_names: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def pool_count(self) -> int:
        return len(self.pool_names)
