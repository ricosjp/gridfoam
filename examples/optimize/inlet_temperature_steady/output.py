"""CSV, history plot and final temperature slice for the example."""

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pyvista as pv

from gridfoam.core.grid.base import GridBase
from gridfoam.io.vtu import save_export_fields_as_vtu

# Fixed range for the cooled-wall case, shared with the historical example.
T_FIELD_CLIM = (0.0, 1.30)


@dataclass(frozen=True)
class OptimizationRecord:
    """One evaluation: all values correspond to the same inlet temperature."""

    step: int
    inlet_temperature: float
    outlet_mean_temperature: float
    loss: float


def write_history(history: list[OptimizationRecord], path: Path) -> None:
    """Save scalar values without retaining any autograd tensors."""
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["step", "inlet_temperature", "outlet_mean_temperature", "loss"]
        )
        writer.writerows(
            (r.step, r.inlet_temperature, r.outlet_mean_temperature, r.loss)
            for r in history
        )


def plot_history(
    history: list[OptimizationRecord], path: Path, target: float
) -> None:
    """Plot inlet and outlet temperatures versus optimization step."""
    steps = [r.step for r in history]
    fig, (ax_inlet, ax_outlet) = plt.subplots(
        2, 1, figsize=(6.0, 6.0), sharex=True
    )
    ax_inlet.plot(
        steps,
        [r.inlet_temperature for r in history],
        marker="o",
        linewidth=1.5,
        label="inlet_temperature",
    )
    ax_inlet.set_ylabel("inlet_temperature")
    ax_inlet.set_title("Temperature during optimization")
    ax_outlet.plot(
        steps,
        [r.outlet_mean_temperature for r in history],
        marker="o",
        linewidth=1.5,
        label="outlet_mean_temperature",
    )
    ax_outlet.axhline(
        target, color="C1", linestyle="--", linewidth=1.5, label="target"
    )
    ax_outlet.set_ylabel("outlet_mean_temperature")
    ax_outlet.set_xlabel("optimization step")
    for axis in (ax_inlet, ax_outlet):
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_final_fields(ugrid: pv.UnstructuredGrid, path: Path) -> None:
    """Render the cell-centered temperature on the central XY plane."""
    sliced = ugrid.slice(normal=(0.0, 0.0, 1.0), origin=ugrid.center)
    assert isinstance(sliced, pv.DataSet)
    if sliced.n_cells == 0:
        raise ValueError(f"Slice produced an empty mesh at {ugrid.center}")

    plotter = pv.Plotter(off_screen=True, window_size=[700, 380])
    try:
        plotter.renderer.enable_parallel_projection()
        plotter.add_mesh(
            sliced,
            scalars="T",
            preference="cell",
            cmap="coolwarm",
            clim=T_FIELD_CLIM,
            show_edges=True,
            edge_color="black",
            line_width=0.1,
            edge_opacity=0.08,
            scalar_bar_args={"title": "Temperature T"},
        )
        plotter.add_text("Temperature T", font_size=10)
        plotter.renderer.view_xy()
        plotter.renderer.reset_camera(bounds=sliced.bounds)
        plotter.screenshot(str(path))
    finally:
        plotter.close()


def save_results(
    grid: GridBase,
    history: list[OptimizationRecord],
    output_dir: Path,
    target: float,
) -> tuple[Path, ...]:
    """Write history and the final solved field, returning artifact paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = grid.sim_config.control.output.base_name
    csv_path = output_dir / f"{base_name}_history.csv"
    history_plot = output_dir / f"{base_name}_history.png"
    vtu_path = output_dir / f"{base_name}_final.vtu"
    fields_plot = output_dir / f"{base_name}_final_fields.png"

    write_history(history, csv_path)
    ugrid = save_export_fields_as_vtu(grid, str(vtu_path))
    plot_history(history, history_plot, target)
    plot_final_fields(ugrid, fields_plot)
    return csv_path, history_plot, vtu_path, fields_plot
