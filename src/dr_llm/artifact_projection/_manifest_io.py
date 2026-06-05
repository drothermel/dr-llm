from __future__ import annotations

import json
from pathlib import Path

from dr_llm.artifact_projection.models import ArtifactReference, ShardManifest


def load_manifest_references(path: Path) -> list[ArtifactReference]:
    references: list[ArtifactReference] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            references.append(ArtifactReference.model_validate_json(line))
    return references


def load_shard_manifest(path: Path) -> ShardManifest:
    return ShardManifest.model_validate_json(path.read_text())


def dump_json_line(value: ArtifactReference) -> str:
    return json.dumps(value.model_dump(mode="json"), sort_keys=True) + "\n"


__all__ = [
    "dump_json_line",
    "load_manifest_references",
    "load_shard_manifest",
]
