import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import json
    from datetime import datetime
    from pathlib import Path
    from time import perf_counter

    import marimo as mo

    from dr_llm.llm import (
        Message,
        OpenAIGpt5Config,
        OpenRouterEffortConfig,
        OpenRouterEffortLevel,
        OpenRouterToggleConfig,
        ThinkingLevel,
        build_default_registry,
    )
    from dr_llm.llm.providers.impls.openai.families import OpenAIModelFamily

    NOTEBOOK_PATH = Path(__file__).resolve()
    REPO_ROOT = NOTEBOOK_PATH.parents[1]


@app.cell
def _():
    MAX_TOKENS = 512

    HIT_PROVIDER_CONFIGS = {
        "OpenRouter / mimo-v2-flash": OpenRouterToggleConfig(
            model="xiaomi/mimo-v2-flash",
            reasoning_enabled=False,
            max_tokens=MAX_TOKENS,
        ),
        "OpenRouter / llama-3.3-nemotron-super-49b-v1.5": OpenRouterToggleConfig(
            model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
            reasoning_enabled=False,
            max_tokens=MAX_TOKENS,
        ),
        "OpenRouter / gpt-5-nano": OpenRouterEffortConfig(
            model="openai/gpt-5-nano",
            effort=OpenRouterEffortLevel.LOW,
            max_tokens=MAX_TOKENS,
        ),
        "OpenRouter / gpt-oss-20b": OpenRouterEffortConfig(
            model="openai/gpt-oss-20b",
            effort=OpenRouterEffortLevel.LOW,
            max_tokens=MAX_TOKENS,
        ),
        "OpenAI / gpt-5-nano": OpenAIGpt5Config(
            model=OpenAIModelFamily.GPT5_NANO,
            thinking_level=ThinkingLevel.MINIMAL,
            max_tokens=MAX_TOKENS,
        ),
    }
    return (HIT_PROVIDER_CONFIGS,)


@app.cell
def _():
    provider_reply_history, set_provider_reply_history = mo.state({})
    return provider_reply_history, set_provider_reply_history


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ## Hit Providers
    """)
    return


@app.cell(hide_code=True)
def _(HIT_PROVIDER_CONFIGS):
    provider_config_select = mo.ui.dropdown(
        options=list(HIT_PROVIDER_CONFIGS),
        value=next(iter(HIT_PROVIDER_CONFIGS)),
        allow_select_none=False,
        searchable=True,
        label="Provider config",
        full_width=True,
    )
    prompt_input = mo.ui.text_area(
        placeholder="Type a prompt to send to the selected provider...",
        rows=5,
        label="Prompt",
        full_width=True,
    )
    run_provider_button = mo.ui.run_button(
        label="Run provider",
        kind="success",
        full_width=True,
    )
    mo.vstack([provider_config_select, prompt_input, run_provider_button], gap=1)
    return prompt_input, provider_config_select, run_provider_button


@app.cell(hide_code=True)
def _(
    HIT_PROVIDER_CONFIGS,
    prompt_input,
    provider_config_select,
    provider_reply_history,
    run_provider_button,
    set_provider_reply_history,
):
    provider_result = mo.md("")

    if run_provider_button.value:
        prompt = prompt_input.value.strip()
        if not prompt:
            provider_result = mo.callout(
                "Enter prompt text before running the selected provider.",
                kind="warn",
            )
        else:
            selected_config_name = provider_config_select.value
            authoring_config = HIT_PROVIDER_CONFIGS[selected_config_name]
            registry = build_default_registry()
            try:
                llm_config = authoring_config.to_llm_config(registry)
                orchestrator = registry.get(llm_config.provider)
                request = orchestrator.build_request_from_config(
                    config=llm_config,
                    messages=[Message(role="user", content=prompt)],
                    metadata={"source": "nbs/hit_providers.py"},
                )
                started_at = perf_counter()
                response = orchestrator.generate(request)
                elapsed_ms = int((perf_counter() - started_at) * 1000)
                usage = response.usage
                cost_text = ""
                if (
                    response.cost is not None
                    and response.cost.total_cost_usd is not None
                ):
                    cost_text = f" · cost ${response.cost.total_cost_usd:.6f}"
                model_history_key = f"{llm_config.provider} / {llm_config.model}"
                current_history = provider_reply_history()
                set_provider_reply_history(
                    {
                        **current_history,
                        model_history_key: [
                            *current_history.get(model_history_key, []),
                            {
                                "request": request,
                                "response": response,
                            },
                        ],
                    }
                )
                provider_result = mo.vstack(
                    [
                        mo.md(
                            f"**{selected_config_name}** · "
                            f"`{response.provider}` / `{response.model}` · "
                            f"{elapsed_ms} ms · "
                            f"{usage.total_tokens} tokens"
                            f"{cost_text}"
                        ),
                        mo.md(response.text or ""),
                    ],
                    gap=1,
                )
            except Exception as exc:
                provider_result = mo.callout(
                    f"{type(exc).__name__}: {exc}",
                    kind="danger",
                )
            finally:
                registry.close()

    provider_result
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    ## History
    """)
    return


@app.cell(hide_code=True)
def _():
    save_history_button = mo.ui.run_button(
        label="Save history",
        kind="neutral",
        full_width=True,
    )
    save_history_button
    return (save_history_button,)


@app.cell(hide_code=True)
def _(provider_reply_history, save_history_button):
    save_history_status = mo.md("")

    if save_history_button.value:
        _history = provider_reply_history()
        if not _history:
            save_history_status = mo.callout(
                "No provider history to write yet.",
                kind="warn",
            )
        else:
            _logs_dir = REPO_ROOT / "logs"
            _logs_dir.mkdir(parents=True, exist_ok=True)
            _timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            _history_path = _logs_dir / f"nbs_hit_providers-{_timestamp}.json"
            _payload = {
                _model_name: [
                    {
                        "request": _entry["request"].model_dump(mode="json"),
                        "response": _entry["response"].model_dump(mode="json"),
                    }
                    for _entry in _entries
                ]
                for _model_name, _entries in _history.items()
            }
            _history_path.write_text(
                json.dumps(_payload, indent=2) + "\n",
                encoding="utf-8",
            )
            save_history_status = mo.callout(
                f"Wrote `{_history_path.relative_to(REPO_ROOT)}`.",
                kind="success",
            )

    save_history_status
    return


@app.cell(hide_code=True)
def _(provider_reply_history):
    history = provider_reply_history()

    if not history:
        history_output = mo.md("")
    else:
        history_lines = ["### Previous replies"]
        for _model_name, _entries in history.items():
            history_lines.append(f"\n**{_model_name}**")
            for _index, _entry in enumerate(_entries, start=1):
                _request = _entry["request"]
                _response = _entry["response"]
                _prompt = (
                    _request.messages[-1].content if _request.messages else ""
                )
                _prompt_preview = _prompt.replace("\n", " ")[:120]
                _response_preview = (_response.text or "").replace("\n", " ")[:160]
                history_lines.append(
                    f"{_index}. `{_request.model}` · {_request.provider}: "
                    f"{_prompt_preview!r} -> {_response_preview!r}"
                )
        history_output = mo.md("\n".join(history_lines))

    history_output
    return


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
