import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    from dr_llm.project.project_service import (
        inspect_projects,
    )


@app.cell
def _():
    project_summaries = inspect_projects()
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
