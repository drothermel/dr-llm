from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

def _load_pool_import_demo() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "demo-streaming-log-pool-import.py"
    )
    spec = importlib.util.spec_from_file_location(
        "demo_streaming_log_pool_import", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
