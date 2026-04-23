from __future__ import annotations

import importlib

app_module = importlib.import_module("dr_llm.cli.app")


class _FakeLogger:
    def __init__(self, handlers: list[object]) -> None:
        self.handlers = handlers
        self.levels: list[int] = []

    def setLevel(self, level: int) -> None:
        self.levels.append(level)


def test_configure_cli_logging_installs_basic_config_when_root_has_no_handlers(
    monkeypatch,
) -> None:
    fake_root = _FakeLogger(handlers=[])
    calls: list[tuple[int, str]] = []

    monkeypatch.setattr(app_module, "_get_root_logger", lambda: fake_root)
    monkeypatch.setattr(
        app_module.logging,
        "basicConfig",
        lambda *, level, format: calls.append((level, format)),
    )

    app_module._configure_cli_logging()

    assert calls == [(app_module.logging.INFO, "%(message)s")]
    assert fake_root.levels == []


def test_configure_cli_logging_sets_info_level_when_root_already_has_handlers(
    monkeypatch,
) -> None:
    fake_root = _FakeLogger(handlers=[object()])

    monkeypatch.setattr(app_module, "_get_root_logger", lambda: fake_root)

    app_module._configure_cli_logging()

    assert fake_root.levels == [app_module.logging.INFO]
