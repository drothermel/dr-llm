from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ColumnType",
    "DbConfig",
    "KeyColumn",
    "PoolNotFoundError",
    "PoolProgress",
    "PoolReader",
    "PoolSchema",
    "PoolSchemaNotPersistedError",
    "PoolService",
    "PoolStore",
    "ProjectDeleteRequest",
    "ProjectDeletionResult",
    "assess_project_deletion",
    "delete_project",
]

_EXPORT_MODULES = {
    "ColumnType": "dr_llm.pool.db",
    "DbConfig": "dr_llm.pool.db",
    "KeyColumn": "dr_llm.pool.db",
    "PoolNotFoundError": "dr_llm.pool.errors",
    "PoolProgress": "dr_llm.pool.reader",
    "PoolReader": "dr_llm.pool.reader",
    "PoolSchema": "dr_llm.pool.db",
    "PoolSchemaNotPersistedError": "dr_llm.pool.errors",
    "PoolService": "dr_llm.sampling.pool_service",
    "PoolStore": "dr_llm.pool.pool_store",
    "ProjectDeleteRequest": "dr_llm.project.models",
    "ProjectDeletionResult": "dr_llm.project.models",
    "assess_project_deletion": "dr_llm.project.project_service",
    "delete_project": "dr_llm.project.project_service",
}

_EXPORT_ALIASES = {
    "ProjectDeleteRequest": "DeleteProjectRequest",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, _EXPORT_ALIASES.get(name, name))
    globals()[name] = value
    return value
