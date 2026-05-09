import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
