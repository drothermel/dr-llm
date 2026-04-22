from __future__ import annotations

from datetime import datetime
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

    @computed_field
    @property
    def blocked_message(self) -> str | None:
        if self.allowed:
            return None
        return "\n".join(violation.message for violation in self.violations)


class ProjectPoolInspectionStatus(StrEnum):
    discovered = "discovered"
    skipped = "skipped"
    failed = "failed"


class ProjectPoolInspectionReason(StrEnum):
    project_not_running = "project_not_running"
    missing_dsn = "missing_dsn"
    connection_failed = "connection_failed"
    unexpected_error = "unexpected_error"


class ProjectPoolInspection(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: ProjectPoolInspectionStatus
    reason: ProjectPoolInspectionReason | None = None
    message: str | None = None
    pool_names: list[str] = Field(default_factory=list)
    inspected_at: datetime

    @computed_field
    @property
    def pool_count(self) -> int:
        return len(self.pool_names)


class ProjectInspectionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    project: ProjectInfo
    pool_inspection: ProjectPoolInspection

    @staticmethod
    def _format_datetime(value: datetime | None) -> str:
        if value is None:
            return "-"
        return value.isoformat(timespec="seconds")

    def to_row(self) -> dict[str, str | int | None]:
        inspection = self.pool_inspection
        inspection_status = str(inspection.status)
        inspection_detail = inspection.message or inspection_status
        if inspection.status == ProjectPoolInspectionStatus.discovered:
            pools = (
                ", ".join(inspection.pool_names) if inspection.pool_names else "(none)"
            )
        else:
            pools = "-"
        return {
            "project": self.project.name,
            "status": str(self.project.status),
            "port": self.project.port,
            "dsn": self.project.dsn,
            "project_created_at": self._format_datetime(self.project.created_at),
            "inspected_at": self._format_datetime(inspection.inspected_at),
            "inspection_status": inspection_status,
            "inspection": inspection_detail,
            "pool_count": inspection.pool_count,
            "pools": pools,
        }
