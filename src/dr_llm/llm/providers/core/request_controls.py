from __future__ import annotations

from typing import Protocol

from dr_llm.llm.providers.concepts.reasoning import ReasoningWarning


class ProviderRequestControls(Protocol):
    @property
    def warnings(self) -> list[ReasoningWarning]: ...


class HeadlessRequestControls(ProviderRequestControls, Protocol):
    @property
    def cli_args(self) -> list[str]: ...
