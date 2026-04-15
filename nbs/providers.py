import marimo

__generated_with = "0.23.1"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    from dr_llm.llm.request import (
        ApiLlmRequest,
        HeadlessLlmRequest,
        KimiCodeLlmRequest,
        OpenAILlmRequest,
    )
    from pydantic import BaseModel


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```python
    mo.inspect(
        obj: object,
        *,
        help: bool = False,
        methods: bool = False,
        docs: bool = True,
        private: bool = False,
        dunder: bool = False,
        sort: bool = True,
        all: bool = False,
        value: bool = True,
    )
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    request_objs = {
        "base": ApiLlmRequest,
        "headless": HeadlessLlmRequest,
        "kimi": KimiCodeLlmRequest,
        "openai": OpenAILlmRequest,
    }
    request_objs
    return (request_objs,)


@app.cell(column=1)
def _(request_objs):
    mo.inspect(
        request_objs["base"].model_fields,
        help=False,
        methods=False,
        docs=False,
        private=False,
        dunder=False,
        sort=False,
        value=True,
        all=False,
    )
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
