"""
Plot cell count vs elapsed time from resolution benchmark CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tests.profile.benchmark_results import RESULTS_CSV, CsvRow, load_rows

DEFAULT_CSV = RESULTS_CSV.with_name("resolution_scaling-latest.csv")
DEFAULT_OUTPUT = RESULTS_CSV.with_name("cells_vs_time.png")


def _group_by_solver(rows: list[CsvRow]) -> dict[str, list[CsvRow]]:
    grouped: dict[str, list[CsvRow]] = {}
    for row in rows:
        label = f"{row['solver']} ({row['device']}, {row['n_steps']} steps)"
        grouped.setdefault(label, []).append(row)
    for solver_rows in grouped.values():
        solver_rows.sort(key=lambda row: int(row["n_cells"]))
    return grouped


def plot_cells_vs_time(
    csv_path: Path,
    output_path: Path,
    *,
    use_time_per_step: bool,
) -> None:
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"No benchmark results in {csv_path}")
    grouped = _group_by_solver(rows)

    fig, ax = plt.subplots(figsize=(8, 5))

    y_label = (
        "Total elapsed time / number of steps [s]"
        if use_time_per_step
        else "Total elapsed time [s]"
    )
    end_times: set[str] = set()
    for solver, solver_rows in grouped.items():
        x = [int(row["n_cells"]) for row in solver_rows]
        if use_time_per_step:
            y = [float(row["time_per_step_s"]) for row in solver_rows]
        else:
            y = [float(row["elapsed_s"]) for row in solver_rows]

        end_times.update(row["end_time"] for row in solver_rows)

        ax.plot(
            x,
            y,
            marker="s" if solver.startswith("openfoam") else "o",
            linewidth=1.5,
            label=solver,
        )

    ax.set_xlabel("Number of cells")
    ax.set_ylabel(y_label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    end_time_note = ", ".join(sorted(end_times))
    title = "motorBike profile: cells vs elapsed time"
    if use_time_per_step:
        title += " (per time step)"
    ax.set_title(title)
    source_note = csv_path.name
    fig.text(
        0.01,
        0.01,
        f"source: {source_note} | endTime values: {end_time_note}",
        fontsize=8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Benchmark CSV produced by benchmark_resolution.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path",
    )
    parser.add_argument(
        "--per-step",
        action="store_true",
        help="Plot elapsed time per time step instead of total elapsed time",
    )
    args = parser.parse_args()
    plot_cells_vs_time(args.csv, args.output, use_time_per_step=args.per_step)


if __name__ == "__main__":
    main()
