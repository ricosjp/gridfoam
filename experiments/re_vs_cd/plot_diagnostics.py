from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = ROOT / "outputs"
PARAMETERS_PATH = ROOT / "data" / "parameters.yml"
DEFAULT_OUTPUT_DIR = OUTPUTS_ROOT / "diagnostic_plots"

RE_DIR_PATTERN = re.compile(r"^re_(?P<re>.+)$")


class CaseConfig(BaseModel):
    name: str
    mesh_path: str
    magU_ref: float
    A_ref: float
    L_ref: float


class ExperimentParameters(BaseModel):
    case: list[CaseConfig]
    Re: list[float]


@dataclass(frozen=True)
class ContinuitySeries:
    time: np.ndarray
    local: np.ndarray
    global_: np.ndarray
    cumulative: np.ndarray


@dataclass(frozen=True)
class SolverFieldSeries:
    name: str
    initial: np.ndarray
    final: np.ndarray
    iterations: np.ndarray
    converged: np.ndarray


@dataclass(frozen=True)
class SolverInfoSeries:
    time: np.ndarray
    fields: tuple[SolverFieldSeries, ...]


def _load_parameters(path: Path) -> ExperimentParameters:
    with path.open() as f:
        return ExperimentParameters.model_validate(yaml.safe_load(f))


def _re_label(re_value: float) -> str:
    return f"{re_value:g}"


def _parse_re_dir(name: str) -> float | None:
    match = RE_DIR_PATTERN.match(name)
    if match is None:
        return None
    return float(match.group("re"))


def _gridfoam_run_dir(case_name: str, re_value: float) -> Path:
    return OUTPUTS_ROOT / case_name / "gridfoam" / f"re_{_re_label(re_value)}"


def _discover_re_values(case_name: str) -> list[float]:
    case_dir = OUTPUTS_ROOT / case_name / "gridfoam"
    if not case_dir.exists():
        return []

    re_values: list[float] = []
    for re_dir in sorted(case_dir.iterdir()):
        if not re_dir.is_dir():
            continue
        re_value = _parse_re_dir(re_dir.name)
        if re_value is None:
            continue
        continuity_path = re_dir / "continuity_error.csv"
        solver_path = re_dir / "solver_info.csv"
        if (continuity_path.exists() and _has_data_rows(continuity_path)) or (
            solver_path.exists() and _has_data_rows(solver_path)
        ):
            re_values.append(re_value)
    return re_values


def _read_float_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        if raw == "":
            values.append(np.nan)
        else:
            values.append(float(raw))
    return np.asarray(values, dtype=float)


def _read_int_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values: list[int] = []
    for row in rows:
        raw = row.get(key, "")
        if raw == "":
            values.append(0)
        else:
            values.append(int(float(raw)))
    return np.asarray(values, dtype=int)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _has_data_rows(path: Path) -> bool:
    return bool(_read_csv_rows(path))


def _read_continuity_series(path: Path) -> ContinuitySeries:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"No data rows found in {path}")

    return ContinuitySeries(
        time=_read_float_column(rows, "time"),
        local=_read_float_column(rows, "local"),
        global_=_read_float_column(rows, "global"),
        cumulative=_read_float_column(rows, "cumulative"),
    )


def _solver_field_names(fieldnames: list[str]) -> list[str]:
    suffixes = ("_solver", "_initial", "_final", "_iters", "_converged")
    names: list[str] = []
    for field in fieldnames:
        if field == "time":
            continue
        for suffix in suffixes:
            if field.endswith(suffix):
                names.append(field[: -len(suffix)])
                break
    return sorted(set(names))


def _read_solver_info_series(path: Path) -> SolverInfoSeries:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"No data rows found in {path}")

    fieldnames = list(rows[0].keys())
    fields: list[SolverFieldSeries] = []
    for name in _solver_field_names(fieldnames):
        fields.append(
            SolverFieldSeries(
                name=name,
                initial=_read_float_column(rows, f"{name}_initial"),
                final=_read_float_column(rows, f"{name}_final"),
                iterations=_read_int_column(rows, f"{name}_iters"),
                converged=_read_int_column(rows, f"{name}_converged"),
            )
        )

    return SolverInfoSeries(
        time=_read_float_column(rows, "time"),
        fields=tuple(fields),
    )


def _plot_continuity(ax: plt.Axes, series: ContinuitySeries) -> None:
    ax.semilogy(
        series.time,
        np.abs(series.local),
        linewidth=1.2,
        label="|local|",
    )
    ax.semilogy(
        series.time,
        np.abs(series.global_),
        linewidth=1.2,
        label="|global|",
    )
    ax.semilogy(
        series.time,
        np.abs(series.cumulative),
        linewidth=1.2,
        label="|cumulative|",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("continuity error")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend(loc="best", fontsize=8)


def _plot_solver_residuals(
    ax: plt.Axes,
    time: np.ndarray,
    field: SolverFieldSeries,
) -> None:
    initial = np.clip(field.initial, 1e-16, None)
    final = np.clip(field.final, 1e-16, None)
    ax.semilogy(
        time,
        initial,
        linewidth=1.2,
        label=f"{field.name} initial",
    )
    ax.semilogy(
        time,
        final,
        linewidth=1.2,
        linestyle="--",
        label=f"{field.name} final",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("residual")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend(loc="best", fontsize=8)


def _plot_solver_iterations(
    ax: plt.Axes,
    time: np.ndarray,
    field: SolverFieldSeries,
) -> None:
    ax.plot(time, field.iterations, linewidth=1.2, label=f"{field.name} iters")
    ax.set_xlabel("time")
    ax.set_ylabel("iterations")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="best", fontsize=8)


def _plot_run(case_name: str, re_value: float, output_dir: Path) -> Path | None:
    run_dir = _gridfoam_run_dir(case_name, re_value)
    continuity_path = run_dir / "continuity_error.csv"
    solver_path = run_dir / "solver_info.csv"
    has_continuity = continuity_path.exists() and _has_data_rows(
        continuity_path
    )
    has_solver = solver_path.exists() and _has_data_rows(solver_path)
    if not has_continuity and not has_solver:
        return None

    n_rows = 0
    if has_continuity:
        n_rows += 1
    if has_solver:
        n_rows += len(_read_solver_info_series(solver_path).fields) * 2

    if n_rows == 0:
        return None

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(7.0, 2.8 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    row = 0

    if has_continuity:
        _plot_continuity(
            ax=axes[row, 0], series=_read_continuity_series(continuity_path)
        )
        axes[row, 0].set_title("Continuity error")
        row += 1

    if has_solver:
        solver = _read_solver_info_series(solver_path)
        for field in solver.fields:
            _plot_solver_residuals(axes[row, 0], solver.time, field)
            axes[row, 0].set_title(f"{field.name} residual")
            row += 1
            _plot_solver_iterations(axes[row, 0], solver.time, field)
            axes[row, 0].set_title(f"{field.name} iterations")
            row += 1

    re_label = _re_label(re_value)
    fig.suptitle(f"{case_name} (Re={re_label})")
    output = output_dir / f"{case_name}_re_{re_label}_diagnostics.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


@dataclass(frozen=True)
class RunSummary:
    re_value: float
    final_cumulative: float | None
    final_global: float | None
    mean_u_iters: float | None
    mean_p_iters: float | None


def _read_run_summary(case_name: str, re_value: float) -> RunSummary | None:
    run_dir = _gridfoam_run_dir(case_name, re_value)
    continuity_path = run_dir / "continuity_error.csv"
    solver_path = run_dir / "solver_info.csv"

    final_cumulative: float | None = None
    final_global: float | None = None
    mean_u_iters: float | None = None
    mean_p_iters: float | None = None

    if continuity_path.exists() and _has_data_rows(continuity_path):
        continuity = _read_continuity_series(continuity_path)
        final_cumulative = float(continuity.cumulative[-1])
        final_global = float(continuity.global_[-1])

    if solver_path.exists() and _has_data_rows(solver_path):
        solver = _read_solver_info_series(solver_path)
        u_iters: list[float] = []
        for field in solver.fields:
            if field.name.startswith("U_"):
                u_iters.append(float(np.mean(field.iterations)))
            elif field.name == "p":
                mean_p_iters = float(np.mean(field.iterations))
        if u_iters:
            mean_u_iters = float(np.mean(u_iters))

    if (
        final_cumulative is None
        and final_global is None
        and mean_u_iters is None
        and mean_p_iters is None
    ):
        return None

    return RunSummary(
        re_value=re_value,
        final_cumulative=final_cumulative,
        final_global=final_global,
        mean_u_iters=mean_u_iters,
        mean_p_iters=mean_p_iters,
    )


def _plot_case_summary(case_name: str, output_dir: Path) -> Path | None:
    summaries = [
        summary
        for re_value in _discover_re_values(case_name)
        if (summary := _read_run_summary(case_name, re_value)) is not None
    ]
    if not summaries:
        return None

    summaries.sort(key=lambda item: item.re_value)
    re_values = np.asarray([item.re_value for item in summaries], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), constrained_layout=True)

    cumulative = np.asarray(
        [abs(item.final_cumulative or np.nan) for item in summaries],
        dtype=float,
    )
    global_error = np.asarray(
        [abs(item.final_global or np.nan) for item in summaries],
        dtype=float,
    )
    u_iters = np.asarray(
        [item.mean_u_iters or np.nan for item in summaries],
        dtype=float,
    )
    p_iters = np.asarray(
        [item.mean_p_iters or np.nan for item in summaries],
        dtype=float,
    )

    axes[0, 0].loglog(re_values, cumulative, marker="o", linewidth=1.5)
    axes[0, 0].set_xlabel("Re")
    axes[0, 0].set_ylabel("|final cumulative|")
    axes[0, 0].set_title("Final cumulative continuity error")
    axes[0, 0].grid(True, which="both", linestyle=":", linewidth=0.6)

    axes[0, 1].loglog(re_values, global_error, marker="o", linewidth=1.5)
    axes[0, 1].set_xlabel("Re")
    axes[0, 1].set_ylabel("|final global|")
    axes[0, 1].set_title("Final global continuity error")
    axes[0, 1].grid(True, which="both", linestyle=":", linewidth=0.6)

    axes[1, 0].semilogx(re_values, u_iters, marker="o", linewidth=1.5)
    axes[1, 0].set_xlabel("Re")
    axes[1, 0].set_ylabel("mean iterations")
    axes[1, 0].set_title("Mean U solver iterations")
    axes[1, 0].grid(True, which="both", linestyle=":", linewidth=0.6)

    axes[1, 1].semilogx(re_values, p_iters, marker="o", linewidth=1.5)
    axes[1, 1].set_xlabel("Re")
    axes[1, 1].set_ylabel("mean iterations")
    axes[1, 1].set_title("Mean p solver iterations")
    axes[1, 1].grid(True, which="both", linestyle=":", linewidth=0.6)

    fig.suptitle(f"{case_name} diagnostics summary")
    output = output_dir / f"{case_name}_diagnostics_summary.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _parse_args(case_names: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot gridfoam continuity and solver diagnostics from "
            "experiments/re_vs_cd outputs."
        )
    )
    parser.add_argument(
        "--cases",
        choices=case_names,
        nargs="+",
        default=case_names,
        help="Cases to plot.",
    )
    parser.add_argument(
        "--re",
        type=float,
        nargs="*",
        default=None,
        help="Reynolds numbers to plot. Default: all available runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            f"Directory for generated PNG files. Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Generate only per-case summary plots across Re.",
    )
    parser.add_argument(
        "--runs-only",
        action="store_true",
        help="Generate only per-run time-series plots.",
    )
    return parser.parse_args()


def main() -> None:
    parameters = _load_parameters(PARAMETERS_PATH)
    case_names = [case.name for case in parameters.case]
    args = _parse_args(case_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_runs = not args.summary_only
    plot_summary = not args.runs_only

    for case_name in args.cases:
        re_values = args.re if args.re else _discover_re_values(case_name)
        if plot_runs:
            for re_value in re_values:
                output = _plot_run(case_name, re_value, args.output_dir)
                if output is not None:
                    print(output)
        if plot_summary:
            output = _plot_case_summary(case_name, args.output_dir)
            if output is not None:
                print(output)


if __name__ == "__main__":
    main()
