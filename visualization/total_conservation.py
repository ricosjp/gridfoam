import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib

    import polars as pl
    return pathlib, pl


@app.cell
def _(pathlib):
    dir = pathlib.Path("tests/outputs/grid/vortex/depth4/")
    file = dir / "time_vs_total_T.csv"
    return (file,)


@app.cell
def _(file, pl):
    df = pl.read_csv(file)
    df
    return (df,)


@app.cell
def _(df, pl):
    total_T0 = df["total_T"][0]
    df_normalized = df.with_columns(
        (pl.col("total_T") / total_T0).alias("total_T_normalized")
    )
    df_normalized
    return (df_normalized,)


@app.cell
def _():
    import plotly.graph_objects as go
    import plotly.io as pio
    return go, pio


@app.cell
def _(df_normalized, go, pio):
    # Set default template
    pio.templates.default = "plotly"

    # Create line plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_normalized["time"].to_list(),
            y=df_normalized["total_T_normalized"].to_list(),
            mode="lines+markers",
            name="total_T",
        )
    )
    fig.update_layout(
        xaxis_title="time",
        yaxis_title="total_T",
        title="Total T vs Time",
        width=1200,
        height=800,
    )
    fig
    return


if __name__ == "__main__":
    app.run()
