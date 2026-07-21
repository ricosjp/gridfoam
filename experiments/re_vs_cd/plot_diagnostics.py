from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from openfoam_diagnostics import ensure_openfoam_diagnostic_csvs
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = ROOT / "outputs"
PARAMETERS_PATH = ROOT / "data" / "parameters.yml"
DEFAULT_OUTPUT_DIR = OUTPUTS_ROOT / "diagnostic_plots"
SOLVERS = ("gridfoam", "openfoam")
FIELD_DISPLAY_NAMES = {
    "U_0": "Ux",
    "U_1": "Uy",
    "U_2": "Uz",
}

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


def _run_dir(case_name: str, solver: str, re_value: float) -> Path:
    return OUTPUTS_ROOT / case_name / solver / f"re_{_re_label(re_value)}"


def _prepare_run_dir(run_dir: Path, solver: str) -> bool:
    if solver == "openfoam":
        return ensure_openfoam_diagnostic_csvs(run_dir)
    return True


def _has_diagnostic_data(run_dir: Path) -> bool:
    continuity_path = run_dir / "continuity_error.csv"
    solver_path = run_dir / "solver_info.csv"
    return (continuity_path.exists() and _has_data_rows(continuity_path)) or (
        solver_path.exists() and _has_data_rows(solver_path)
    )


def _discover_re_values(case_name: str, solver: str) -> list[float]:
    case_dir = OUTPUTS_ROOT / case_name / solver
    if not case_dir.exists():
        return []

    re_values: list[float] = []
    for re_dir in sorted(case_dir.iterdir()):
        if not re_dir.is_dir():
            continue
        re_value = _parse_re_dir(re_dir.name)
        if re_value is None:
            continue
        if not _prepare_run_dir(re_dir, solver):
            continue
        if _has_diagnostic_data(re_dir):
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
                name=FIELD_DISPLAY_NAMES.get(name, name),
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


def _step_indices(length: int) -> np.ndarray:
    return np.arange(1, length + 1, dtype=int)


def _plot_continuity(ax: plt.Axes, series: ContinuitySeries) -> None:
    steps = _step_indices(len(series.local))
    ax.semilogy(
        steps,
        np.abs(series.local),
        linewidth=1.2,
        label="|local|",
    )
    ax.semilogy(
        steps,
        np.abs(series.global_),
        linewidth=1.2,
        label="|global|",
    )
    ax.semilogy(
        steps,
        np.abs(series.cumulative),
        linewidth=1.2,
        label="|cumulative|",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("continuity error")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend(loc="best", fontsize=8)


def _plot_solver_residuals(
    ax: plt.Axes,
    steps: np.ndarray,
    field: SolverFieldSeries,
) -> None:
    initial = np.clip(field.initial, 1e-16, None)
    final = np.clip(field.final, 1e-16, None)
    ax.semilogy(
        steps,
        initial,
        linewidth=1.2,
        label=f"{field.name} initial",
    )
    ax.semilogy(
        steps,
        final,
        linewidth=1.2,
        linestyle="--",
        label=f"{field.name} final",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("residual")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend(loc="best", fontsize=8)


def _plot_solver_iterations(
    ax: plt.Axes,
    steps: np.ndarray,
    field: SolverFieldSeries,
) -> None:
    ax.plot(
        steps,
        field.iterations,
        linewidth=1.2,
        label=f"{field.name} iters",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("iterations")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="best", fontsize=8)


def _load_run_data(
    case_name: str,
    solver: str,
    re_value: float,
) -> tuple[ContinuitySeries | None, SolverInfoSeries | None]:
    """Load continuity and solver-info data for a single run."""
    run_dir = _run_dir(case_name, solver, re_value)
    if not _prepare_run_dir(run_dir, solver):
        return None, None

    continuity_path = run_dir / "continuity_error.csv"
    solver_path = run_dir / "solver_info.csv"

    continuity = None
    solver_info = None
    if continuity_path.exists() and _has_data_rows(continuity_path):
        continuity = _read_continuity_series(continuity_path)
    if solver_path.exists() and _has_data_rows(solver_path):
        solver_info = _read_solver_info_series(solver_path)
    return continuity, solver_info


def _plot_run(
    case_name: str,
    solver: str,
    re_value: float,
    output_dir: Path,
) -> Path | None:
    continuity, solver_info = _load_run_data(case_name, solver, re_value)
    if continuity is None and solver_info is None:
        return None

    n_rows = 0
    if continuity is not None:
        n_rows += 1
    if solver_info is not None:
        n_rows += len(solver_info.fields) * 2

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

    if continuity is not None:
        _plot_continuity(ax=axes[row, 0], series=continuity)
        axes[row, 0].set_title("Continuity error")
        row += 1

    if solver_info is not None:
        steps = _step_indices(len(solver_info.time))
        for field in solver_info.fields:
            _plot_solver_residuals(axes[row, 0], steps, field)
            axes[row, 0].set_title(f"{field.name} residual")
            row += 1
            _plot_solver_iterations(axes[row, 0], steps, field)
            axes[row, 0].set_title(f"{field.name} iterations")
            row += 1

    re_label = _re_label(re_value)
    fig.suptitle(f"{case_name} / {solver} (Re={re_label})")
    output = output_dir / f"{case_name}_{solver}_re_{re_label}_diagnostics.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _plot_comparison(
    case_name: str,
    solvers: list[str],
    re_value: float,
    output_dir: Path,
) -> Path | None:
    """Plot side-by-side comparison of solvers with shared axes for a given Re."""
    data: dict[str, tuple[ContinuitySeries | None, SolverInfoSeries | None]] = {}
    for solver in solvers:
        c, s = _load_run_data(case_name, solver, re_value)
        if c is not None or s is not None:
            data[solver] = (c, s)

    if len(data) < 2:
        return None

    has_continuity = any(d[0] is not None for d in data.values())
    all_field_names: list[str] = []
    for _, solver_info in data.values():
        if solver_info is not None:
            for f in solver_info.fields:
                if f.name not in all_field_names:
                    all_field_names.append(f.name)

    n_rows = 0
    if has_continuity:
        n_rows += 1
    n_rows += len(all_field_names) * 2

    if n_rows == 0:
        return None

    n_cols = len(data)
    solver_list = list(data.keys())
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.0 * n_cols, 2.8 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    row = 0
    if has_continuity:
        for col, solver in enumerate(solver_list):
            continuity = data[solver][0]
            if continuity is not None:
                _plot_continuity(ax=axes[row, col], series=continuity)
            axes[row, col].set_title(f"{solver} – Continuity error")
        _share_axes(axes[row, :])
        row += 1

    for field_name in all_field_names:
        for col, solver in enumerate(solver_list):
            solver_info = data[solver][1]
            field = _find_field(solver_info, field_name) if solver_info else None
            if field is not None and solver_info is not None:
                steps = _step_indices(len(solver_info.time))
                _plot_solver_residuals(axes[row, col], steps, field)
            axes[row, col].set_title(f"{solver} – {field_name} residual")
        _share_axes(axes[row, :])
        row += 1

        for col, solver in enumerate(solver_list):
            solver_info = data[solver][1]
            field = _find_field(solver_info, field_name) if solver_info else None
            if field is not None and solver_info is not None:
                steps = _step_indices(len(solver_info.time))
                _plot_solver_iterations(axes[row, col], steps, field)
            axes[row, col].set_title(f"{solver} – {field_name} iterations")
        _share_axes(axes[row, :])
        row += 1

    re_label = _re_label(re_value)
    fig.suptitle(f"{case_name} comparison (Re={re_label})")
    output = output_dir / f"{case_name}_comparison_re_{re_label}_diagnostics.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _share_axes(axes_row: np.ndarray) -> None:
    """Synchronize x/y limits across a row of axes."""
    xlims = [ax.get_xlim() for ax in axes_row if ax.lines]
    ylims = [ax.get_ylim() for ax in axes_row if ax.lines]
    if xlims:
        xmin = min(l[0] for l in xlims)
        xmax = max(l[1] for l in xlims)
        for ax in axes_row:
            ax.set_xlim(xmin, xmax)
    if ylims:
        ymin = min(l[0] for l in ylims)
        ymax = max(l[1] for l in ylims)
        for ax in axes_row:
            ax.set_ylim(ymin, ymax)


def _find_field(
    solver_info: SolverInfoSeries | None,
    name: str,
) -> SolverFieldSeries | None:
    if solver_info is None:
        return None
    for f in solver_info.fields:
        if f.name == name:
            return f
    return None


@dataclass(frozen=True)
class RunSummary:
    re_value: float
    final_cumulative: float | None
    final_global: float | None
    mean_u_iters: float | None
    mean_p_iters: float | None


def _read_run_summary(
    case_name: str,
    solver: str,
    re_value: float,
) -> RunSummary | None:
    run_dir = _run_dir(case_name, solver, re_value)
    if not _prepare_run_dir(run_dir, solver):
        return None

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
        solver_series = _read_solver_info_series(solver_path)
        u_iters: list[float] = []
        for field in solver_series.fields:
            if field.name in {"Ux", "Uy", "Uz"}:
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


def _plot_case_summary(
    case_name: str,
    solvers: list[str],
    output_dir: Path,
) -> Path | None:
    series_by_solver: dict[str, list[RunSummary]] = {}
    for solver in solvers:
        summaries = [
            summary
            for re_value in _discover_re_values(case_name, solver)
            if (summary := _read_run_summary(case_name, solver, re_value))
            is not None
        ]
        if summaries:
            summaries.sort(key=lambda item: item.re_value)
            series_by_solver[solver] = summaries

    if not series_by_solver:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), constrained_layout=True)
    markers = {"gridfoam": "o", "openfoam": "s"}

    for solver, summaries in series_by_solver.items():
        re_values = np.asarray(
            [item.re_value for item in summaries],
            dtype=float,
        )
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
        marker = markers.get(solver, "o")

        axes[0, 0].loglog(
            re_values,
            cumulative,
            marker=marker,
            linewidth=1.5,
            label=solver,
        )
        axes[0, 1].loglog(
            re_values,
            global_error,
            marker=marker,
            linewidth=1.5,
            label=solver,
        )
        axes[1, 0].semilogx(
            re_values,
            u_iters,
            marker=marker,
            linewidth=1.5,
            label=solver,
        )
        axes[1, 1].semilogx(
            re_values,
            p_iters,
            marker=marker,
            linewidth=1.5,
            label=solver,
        )

    axes[0, 0].set_xlabel("Re")
    axes[0, 0].set_ylabel("|final cumulative|")
    axes[0, 0].set_title("Final cumulative continuity error")
    axes[0, 0].grid(True, which="both", linestyle=":", linewidth=0.6)
    axes[0, 0].legend(loc="best", fontsize=8)

    axes[0, 1].set_xlabel("Re")
    axes[0, 1].set_ylabel("|final global|")
    axes[0, 1].set_title("Final global continuity error")
    axes[0, 1].grid(True, which="both", linestyle=":", linewidth=0.6)
    axes[0, 1].legend(loc="best", fontsize=8)

    axes[1, 0].set_xlabel("Re")
    axes[1, 0].set_ylabel("mean iterations")
    axes[1, 0].set_title("Mean U solver iterations")
    axes[1, 0].grid(True, which="both", linestyle=":", linewidth=0.6)
    axes[1, 0].legend(loc="best", fontsize=8)

    axes[1, 1].set_xlabel("Re")
    axes[1, 1].set_ylabel("mean iterations")
    axes[1, 1].set_title("Mean p solver iterations")
    axes[1, 1].grid(True, which="both", linestyle=":", linewidth=0.6)
    axes[1, 1].legend(loc="best", fontsize=8)

    fig.suptitle(f"{case_name} diagnostics summary")
    output = output_dir / f"{case_name}_diagnostics_summary.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _parse_args(case_names: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot continuity and solver diagnostics for gridfoam and OpenFOAM "
            "from experiments/re_vs_cd outputs."
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
        "--solvers",
        choices=list(SOLVERS),
        nargs="+",
        default=list(SOLVERS),
        help="Solvers to plot. Default: gridfoam and openfoam.",
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
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip side-by-side solver comparison plots.",
    )
    return parser.parse_args()


def main() -> None:
    parameters = _load_parameters(PARAMETERS_PATH)
    case_names = [case.name for case in parameters.case]
    args = _parse_args(case_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_runs = not args.summary_only
    plot_summary = not args.runs_only
    plot_comparison = not args.no_comparison and len(args.solvers) >= 2

    for case_name in args.cases:
        if plot_runs:
            for solver in args.solvers:
                re_values = (
                    args.re
                    if args.re
                    else _discover_re_values(case_name, solver)
                )
                for re_value in re_values:
                    output = _plot_run(
                        case_name,
                        solver,
                        re_value,
                        args.output_dir,
                    )
                    if output is not None:
                        print(output)
        if plot_comparison:
            all_re: set[float] = set()
            for solver in args.solvers:
                all_re.update(
                    args.re
                    if args.re
                    else _discover_re_values(case_name, solver)
                )
            for re_value in sorted(all_re):
                output = _plot_comparison(
                    case_name,
                    args.solvers,
                    re_value,
                    args.output_dir,
                )
                if output is not None:
                    print(output)
        if plot_summary:
            output = _plot_case_summary(
                case_name,
                args.solvers,
                args.output_dir,
            )
            if output is not None:
                print(output)


if __name__ == "__main__":
    main()
