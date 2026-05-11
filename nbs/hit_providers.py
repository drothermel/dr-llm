import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    from pathlib import Path

    import marimo as mo

    NOTEBOOK_PATH = Path(__file__).resolve()
    REPO_ROOT = NOTEBOOK_PATH.parents[1]


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ## Hit Providers
    """)
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
