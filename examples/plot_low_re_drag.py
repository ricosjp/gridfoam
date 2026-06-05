from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "low_re_drag_comparison.png"


def _read_rows(path: Path) -> list[tuple[float, float]]:
    if not path.exists():
        return []

    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        return [(float(row["Re"]), float(row["Cd"])) for row in rows]


def _clift_cd(re: float) -> float:
    # Clift et al. (1978), matching low_re_sphere/compare.py.
    w = math.log(re)
    if re < 0.01:
        return (24.0 / re) * (1.0 + (3.0 / 16.0) * re)
    if re < 20.0:
        return (24.0 / re) * (1.0 + 0.1315 * re ** (0.82 - 0.05 * w))
    if re < 260.0:
        return (24.0 / re) * (1.0 + 0.1935 * re**0.6305)

    raise ValueError("Re is out of validity range.")


def _sucker_brauwer_cd(re: float) -> float:
    # Sucker-Brauwer (1975), matching low_re_cylinder/compare.py.
    return (
        6.8 / re**0.89
        + 1.96 / re**0.5
        + 1.18
        - 0.0004 * re / (1 + 3.63e-7 * re * re)
    )


def _logspace(start: float, stop: float, count: int) -> list[float]:
    if start <= 0.0 or stop <= 0.0:
        raise ValueError("Log-space bounds must be positive.")
    if count < 2:
        return [start]

    log_start = math.log10(start)
    log_stop = math.log10(stop)
    return [
        10 ** (log_start + (log_stop - log_start) * i / (count - 1))
        for i in range(count)
    ]


def _plot_rows(
    ax: plt.Axes,
    rows: list[tuple[float, float]],
    *,
    label: str,
    marker: str,
) -> None:
    if not rows:
        return

    re_values, cd_values = zip(*sorted(rows), strict=True)
    ax.loglog(re_values, cd_values, marker=marker, linewidth=1.5, label=label)


def _plot_correlation(
    ax: plt.Axes,
    re_values: list[float],
    *,
    label: str,
    cd_function: Callable[[float], float],
) -> None:
    if not re_values:
        return

    xs = _logspace(min(re_values), max(re_values), 160)
    ys = [cd_function(re) for re in xs]
    ax.loglog(xs, ys, linestyle="--", color="black", linewidth=1.2, label=label)


def _case_rows(case: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    case_root = ROOT / case
    gridfoam_rows = _read_rows(case_root / "gridfoam" / "outputs" / "sweep_summary.csv")
    openfoam_rows = _read_rows(case_root / "of" / "outputs" / "sweep_summary.csv")
    return gridfoam_rows, openfoam_rows


def _plot_case(
    ax: plt.Axes,
    *,
    title: str,
    case: str,
    correlation_label: str | None = None,
    cd_function: Callable[[float], float] | None = None,
) -> None:
    gridfoam_rows, openfoam_rows = _case_rows(case)
    _plot_rows(ax, gridfoam_rows, label="gridfoam", marker="o")
    _plot_rows(ax, openfoam_rows, label="OpenFOAM", marker="s")

    if correlation_label is not None and cd_function is not None:
        all_re = [re for re, _ in gridfoam_rows + openfoam_rows]
        _plot_correlation(
            ax,
            all_re,
            label=correlation_label,
            cd_function=cd_function,
        )

    ax.set_title(title)
    ax.set_xlabel("Re")
    ax.set_ylabel("Cd")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()


def plot(output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    _plot_case(
        axes[0],
        title="Sphere",
        case="low_re_sphere",
        correlation_label="Clift et al.",
        cd_function=_clift_cd,
    )
    _plot_case(
        axes[1],
        title="Circular cylinder",
        case="low_re_cylinder",
        correlation_label="Sucker-Brauwer",
        cd_function=_sucker_brauwer_cd,
    )
    _plot_case(
        axes[2],
        title="Cube",
        case="low_re_cube",
    )

    fig.suptitle("Low-Reynolds Drag Comparison")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot low-Reynolds drag comparisons on log-log axes."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    plot(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
