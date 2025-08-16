import pathlib
from collections import defaultdict

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from pydantic import BaseModel, computed_field

# naming a layout theme for future reference
pio.templates["google"] = go.layout.Template(
    layout_colorway=[
        "#4285F4",
        "#DB4437",
        "#F4B400",
        "#0F9D58",
        "#185ABC",
        "#B31412",
        "#EA8600",
        "#137333",
        "#d2e3fc",
        "#ceead6",
    ]
)

# setting Google color palette as default
pio.templates.default = "google"


class CommitInfo(BaseModel, frozen=True):
    id: str
    time: str
    author_time: str
    dirty: bool
    project: str
    branch: str


class Stats(BaseModel, frozen=True):
    data: list[float]


class BenchmarkEntry(BaseModel, frozen=True):
    group: str
    name: str
    params: dict | None
    stats: Stats

    @computed_field
    @property
    def data(self) -> list[float]:
        return self.stats.data


class BenchmarkFile(BaseModel, frozen=True):
    commit_info: CommitInfo
    version: str
    benchmarks: list[BenchmarkEntry]

    @computed_field
    @property
    def benchmark_groups(self) -> dict[str, list[BenchmarkEntry]]:
        groups = defaultdict(list)
        for benchmark in self.benchmarks:
            groups[benchmark.group].append(benchmark)
        return groups


def load_benchmark(path: pathlib.Path) -> BenchmarkFile:
    with open(path) as f:
        return BenchmarkFile.model_validate_json(f.read())


def plot_level_vs_time(
    benchmark_files: list[BenchmarkFile], output_dir: pathlib.Path
):
    fig = go.Figure()
    fig.update_layout(
        template="google",
        xaxis_title="Level Limit",
        yaxis_title="Grid Generation Time (s)",
        title="Grid Generation Time vs. Level Limit",
        width=1200,
        height=800,
    )

    latest_benchmark_file = benchmark_files[-1]
    latest_benchmark_groups = latest_benchmark_file.benchmark_groups
    for group_name, benchmarks in latest_benchmark_groups.items():
        records = []
        for bench in benchmarks:
            if bench.params is None or "level_limit" not in bench.params:
                continue
            records.append(
                {
                    "group": group_name,
                    "level": bench.params["level_limit"],
                    "time": bench.data,
                }
            )

        df = pl.DataFrame(records)
        df = df.with_columns(
            [
                pl.col("time").list.mean().alias("mean"),
                pl.col("time").list.std(ddof=1).alias("std"),
            ]
        )
        df = df.with_columns(
            [
                (pl.col("mean") + pl.col("std")).alias("upper"),
                (pl.col("mean") - pl.col("std")).alias("lower"),
            ]
        )

        fig.add_trace(
            go.Scatter(
                x=df["level"],
                y=df["upper"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                name="upper",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["level"],
                y=df["lower"],
                mode="lines",
                fill="tonexty",
                line={"width": 0},
                showlegend=False,
                name="lower",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["level"],
                y=df["mean"],
                mode="lines+markers",
                name=group_name,
            )
        )
    fig.write_html(output_dir / "level_vs_time.html")


def plot_version_vs_time(
    benchmark_files: list[BenchmarkFile], output_dir: pathlib.Path
):
    fig = go.Figure()
    fig.update_layout(
        template="google",
        xaxis_title="Version",
        yaxis_title="Grid Generation Time (s)",
        title="Grid Generation Time for an 8-Level Grid by Version",
        width=1200,
        height=800,
    )
    records = []
    for benchmark_file in benchmark_files:
        version = benchmark_file.version
        benchmark_groups = benchmark_file.benchmark_groups

        for group_name, benchmarks in benchmark_groups.items():
            finest_grid_bench = benchmarks[-1]
            records.append(
                {
                    "group": group_name,
                    "version": version,
                    "time": finest_grid_bench.data,
                }
            )

    df = pl.DataFrame(records)
    df = df.with_columns(
        [
            pl.col("time").list.mean().alias("mean"),
            pl.col("time").list.std(ddof=1).alias("std"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("mean") + pl.col("std")).alias("upper"),
            (pl.col("mean") - pl.col("std")).alias("lower"),
        ]
    )

    for group in df.select("group").unique().to_series():
        group_df = df.filter(pl.col("group") == group)
        fig.add_trace(
            go.Scatter(
                x=group_df["version"],
                y=group_df["upper"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                name="upper",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=group_df["version"],
                y=group_df["lower"],
                mode="lines",
                fill="tonexty",
                line={"width": 0},
                showlegend=False,
                name="lower",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=group_df["version"],
                y=group_df["mean"],
                mode="lines+markers",
                name=group,
            )
        )
    fig.write_html(output_dir / "version_vs_time.html")


if __name__ == "__main__":
    benchmark_dir = pathlib.Path(
        "tests/outputs/benchmark/time/Linux-CPython-3.12-64bit"
    )
    output_dir = pathlib.Path("tests/outputs/benchmark/time")
    files = benchmark_dir.glob("*.json")

    benchmark_files = sorted(
        [load_benchmark(file) for file in files], key=lambda x: x.version
    )
    plot_level_vs_time(benchmark_files, output_dir)
    plot_version_vs_time(benchmark_files, output_dir)
