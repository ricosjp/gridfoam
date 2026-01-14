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
    dir = pathlib.Path("tests/outputs/grid/SpatialConvergenceTest/uniform/")
    file = dir / "result.csv"
    return (file,)


@app.cell
def _(file, pl):
    df = pl.read_csv(file)
    df
    return (df,)


@app.cell
def _(df, pl):
    df_with_dx = df.with_columns(
        (4.0 / (8 * (2 ** pl.col("depth")))).alias("dx")
    )
    df_with_dx
    return (df_with_dx,)


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    return go, np, pio


@app.cell
def _(df_with_dx, go, np, pio):
    # Set default template
    pio.templates.default = "plotly"

    # Extract data
    dx_values = df_with_dx["dx"].to_list()
    l_inf_values = df_with_dx["L_inf"].to_list()

    # Linear regression on log-log scale
    log_dx = np.log(dx_values)
    log_l_inf = np.log(l_inf_values)
    slope, intercept = np.polyfit(log_dx, log_l_inf, 1)

    # Generate regression line
    dx_fit = np.array(dx_values)
    l_inf_fit = np.exp(intercept) * (dx_fit ** slope)

    # Create log-log line plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dx_values,
            y=l_inf_values,
            mode="lines+markers",
            name="poisson",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dx_fit,
            y=l_inf_fit,
            mode="lines",
            name=f"回帰直線 (傾き: {slope:.3f})",
            line={"dash": "dash"},
        )
    )
    fig.update_layout(
        xaxis_title="dx",
        yaxis_title="L_inf",
        title=f"Spatial Convergence (Log-Log, Order of convergence: {slope:.3f})",
        width=1200,
        height=800,
    )
    # Set log scale for both axes
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    fig
    return


if __name__ == "__main__":
    app.run()
