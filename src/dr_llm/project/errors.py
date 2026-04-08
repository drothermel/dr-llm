from __future__ import annotations

from dr_llm.errors import LlmPoolError


class ProjectError(LlmPoolError):
    """Base error for project operations."""


class ProjectNotFoundError(ProjectError):
    """A requested project does not exist."""


class ProjectAlreadyExistsError(ProjectError):
    """A project with the requested name already exists."""


class DockerUnavailableError(ProjectError):
    """Docker is not installed or the daemon is unavailable."""

    def __init__(self) -> None:
        super().__init__("Docker is not available. Install Docker or start the daemon.")


class DockerCommandError(ProjectError):
    """Base error for docker command failures."""


class DockerContainerNotFoundError(DockerCommandError):
    """Raised when a target container does not exist."""

    def __init__(self) -> None:
        super().__init__("Docker container not found.")


class DockerContainerNotRunningError(DockerCommandError):
    """Raised when a target container exists but is not running."""

    def __init__(self) -> None:
        super().__init__("Docker container is not running.")


class DockerContainerConflictError(DockerCommandError):
    """Raised when Docker reports a container-name conflict."""

    def __init__(self) -> None:
        super().__init__("Docker container name is already in use.")


class DockerPortAllocatedError(DockerCommandError):
    """Raised when Docker cannot bind the requested host port."""

    def __init__(self) -> None:
        super().__init__("Docker host port is already allocated.")
