"""CSV, history plot and final velocity/pressure slices for the example."""

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pyvista as pv

from gridfoam.core.grid.base import GridBase
from gridfoam.io.vtu import save_export_fields_as_vtu

# Fixed scales keep numerical noise in this uniform flow visually negligible.
U_FIELD_CLIM = (0.0, 1.3)
P_FIELD_CLIM = (-0.01, 0.01)


@dataclass(frozen=True)
class OptimizationRecord:
    """One evaluation: all values correspond to the same inlet velocity."""

    step: int
    inlet_velocity: float
    mean_u_x: float
    loss: float


def write_history(history: list[OptimizationRecord], path: Path) -> None:
    """Save scalar values without retaining any autograd tensors."""
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "inlet_velocity", "mean_u_x", "loss"])
        writer.writerows(
            (r.step, r.inlet_velocity, r.mean_u_x, r.loss) for r in history
        )


def plot_history(
    history: list[OptimizationRecord], path: Path, target: float
) -> None:
    """Plot inlet velocity, arithmetic mean Ux and loss versus step."""
    steps = [r.step for r in history]
    fig, (ax_velocity, ax_loss) = plt.subplots(
        2, 1, figsize=(6.0, 6.0), sharex=True
    )
    for values, label, style in (
        ([r.inlet_velocity for r in history], "Inlet velocity", "o-"),
        ([r.mean_u_x for r in history], "Arithmetic mean Ux", "x--"),
    ):
        ax_velocity.plot(steps, values, style, linewidth=1.5, label=label)
    ax_velocity.axhline(target, color="C2", linestyle="--", label="Target")
    ax_velocity.set_title("Inlet velocity optimization")
    ax_velocity.set_ylabel("Velocity [m/s]")
    ax_velocity.legend()
    ax_loss.plot(steps, [r.loss for r in history], marker="o", linewidth=1.5)
    ax_loss.set_ylabel("Loss [(m/s)^2]")
    ax_loss.set_xlabel("Optimization step")
    ax_loss.set_xticks(steps)
    for axis in (ax_velocity, ax_loss):
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_final_fields(ugrid: pv.UnstructuredGrid, output_path: Path) -> None:
    """Render Ux and kinematic pressure on the central XY slice."""
    sliced = ugrid.slice(normal=(0.0, 0.0, 1.0), origin=ugrid.center)
    assert isinstance(sliced, pv.DataSet)
    if sliced.n_cells == 0:
        raise ValueError(f"Slice produced an empty mesh at {ugrid.center}")
    sliced.cell_data["U_x"] = sliced.cell_data["U"][:, 0]
    plotter = pv.Plotter(off_screen=True, shape=(2, 1), window_size=[700, 760])
    try:
        for row, (scalars, title, clim) in enumerate(
            (
                ("U_x", "Velocity Ux [m/s]", U_FIELD_CLIM),
                ("p", "Pressure p/rho [m^2/s^2]", P_FIELD_CLIM),
            )
        ):
            plotter.subplot(row, 0)
            plotter.renderer.enable_parallel_projection()
            plotter.add_mesh(
                sliced,
                scalars=scalars,
                preference="cell",
                cmap="coolwarm",
                clim=clim,
                show_edges=True,
                edge_color="black",
                line_width=0.1,
                edge_opacity=0.08,
                scalar_bar_args={"title": title},
            )
            plotter.add_text(title, font_size=10)
            plotter.renderer.view_xy()
            plotter.renderer.reset_camera(bounds=sliced.bounds)
        plotter.screenshot(str(output_path))
    finally:
        plotter.close()


def save_results(
    grid: GridBase,
    history: list[OptimizationRecord],
    output_dir: Path,
    target: float,
) -> tuple[Path, ...]:
    """Write history and the final solved fields, returning artifact paths."""
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
