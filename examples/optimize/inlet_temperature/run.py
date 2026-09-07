"""
Optimize inlet temperature to match a target outlet mean temperature.

The flow field (``U``, ``phi``) is fixed and assumed already converged.
Each optimization step solves steady advection-diffusion:

    div(phi, T) - laplacian(alpha, T) = 0

Design variable
    inlet_temperature : Dirichlet value at the inlet patch.

Objective
    (outlet_mean_temperature - target_outlet_mean_temperature) ** 2

Gradient modes (``--grad-mode``)
    adjoint : detach the Krylov solve and backprop via the implicit adjoint.
    unrolled : differentiate through BiCGSTAB iterations (comparison only).
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pyvista as pv
import torch
from jaxtyping import Float

from gridfoam.core.equation import equation
from gridfoam.core.field import (
    CellField,
    FaceField,
    get_or_create_cellfield,
    get_or_create_facefield,
)
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.name import make_field_name
from gridfoam.fv import fvm
from gridfoam.fv.flux import correct_flux
from gridfoam.io.vtu import save_export_fields_as_vtu, to_unstructured_grid
from gridfoam.meta.enums import DomainBoundaryPatch, FieldRole, SolverType
from gridfoam.runner import manual_run
from gridfoam.solvers.base import GradientMode, LinearSolver
from gridfoam.solvers.factory import create_solver

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.optimize._common import (  # noqa: E402
    configure_run_logger,
    patch_cell_mean,
    set_inlet_dirichlet,
)

TARGET_OUTLET_MEAN_TEMPERATURE = 0.004
INLET_TEMPERATURE_INIT = 2.0
N_OPT_STEPS = 100
LEARNING_RATE = 0.10
FD_EPS = 1e-3

FINAL_VTU_NAME = "inlet_temperature_final.vtu"
T_FIELD_CLIM = (0.0, 1.30)


@dataclass
class OptimizationHistory:
    """Scalar traces recorded during inlet-temperature optimization."""

    steps: list[int] = field(default_factory=list)
    inlet_temperature: list[float] = field(default_factory=list)
    outlet_mean_temperature: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)

    def append(
        self,
        step: int,
        inlet_value: float,
        outlet_value: float,
        loss_value: float,
    ) -> None:
        """Record one optimization iteration."""
        self.steps.append(step)
        self.inlet_temperature.append(inlet_value)
        self.outlet_mean_temperature.append(outlet_value)
        self.loss.append(loss_value)

    def write_csv(self, output_path: Path) -> None:
        """Write recorded traces to a CSV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "inlet_temperature",
                    "outlet_mean_temperature",
                    "loss",
                ]
            )
            for step, inlet, outlet, loss in zip(
                self.steps,
                self.inlet_temperature,
                self.outlet_mean_temperature,
                self.loss,
                strict=True,
            ):
                writer.writerow([step, inlet, outlet, loss])


def reset_scalar_field(field: CellField) -> None:
    """Clear field values without autograd history."""
    field.data = torch.zeros_like(field.data)


def solve_steady_temperature(
    T: CellField,
    phi: FaceField,
    alpha: float,
    solver: LinearSolver,
) -> None:
    """Solve steady advection-diffusion for ``T``."""
    mat = fvm.div(phi, T) - fvm.laplacian(alpha, T)
    temperature_eq = equation(T, mat)
    T.data = solver.solve(temperature_eq).solution


def outlet_temperature_loss(
    inlet_temperature: Float[torch.Tensor, ""],
    T: CellField,
    phi: FaceField,
    alpha: float,
    solver: LinearSolver,
    target_outlet_mean_temperature: float,
) -> tuple[Float[torch.Tensor, ""], Float[torch.Tensor, ""]]:
    """Return squared outlet error and the outlet mean temperature."""
    set_inlet_dirichlet(T, DomainBoundaryPatch.X_MINUS, inlet_temperature)
    solve_steady_temperature(T, phi, alpha, solver)
    outlet_mean_temperature = patch_cell_mean(T, DomainBoundaryPatch.X_PLUS)
    loss = (outlet_mean_temperature - target_outlet_mean_temperature) ** 2
    return loss, outlet_mean_temperature


def check_gradient_fd(
    inlet_temperature_init: float,
    T: CellField,
    phi: FaceField,
    alpha: float,
    solver: LinearSolver,
    target_outlet_mean_temperature: float,
    eps: float = FD_EPS,
) -> None:
    """Compare autograd and central FD gradients."""
    grid = T.grid

    def eval_loss(value: float) -> float:
        inlet = torch.tensor(
            value,
            dtype=grid.dtype,
            device=grid.device,
        )
        with torch.no_grad():
            loss, _ = outlet_temperature_loss(
                inlet,
                T,
                phi,
                alpha,
                solver,
                target_outlet_mean_temperature,
            )
            reset_scalar_field(T)
        return loss.item()

    loss_center = eval_loss(inlet_temperature_init)
    fd_grad = (
        eval_loss(inlet_temperature_init + eps)
        - eval_loss(inlet_temperature_init - eps)
    ) / (2.0 * eps)

    inlet = torch.tensor(
        inlet_temperature_init,
        dtype=grid.dtype,
        device=grid.device,
        requires_grad=True,
    )
    loss, _ = outlet_temperature_loss(
        inlet,
        T,
        phi,
        alpha,
        solver,
        target_outlet_mean_temperature,
    )
    loss.backward()
    grad = inlet.grad
    assert grad is not None
    autograd_grad = grad.item()
    reset_scalar_field(T)

    rel_err = abs(autograd_grad - fd_grad) / max(abs(fd_grad), 1e-12)
    configure_run_logger().info(
        "grad check: loss=%.4e fd=%.4e autograd=%.4e rel_err=%.2e",
        loss_center,
        fd_grad,
        autograd_grad,
        rel_err,
    )


def plot_optimization_history(
    history: OptimizationHistory,
    output_path: Path,
    target_outlet_mean_temperature: float,
) -> None:
    """Plot inlet and outlet temperatures versus optimization step."""
    fig, (ax_inlet, ax_outlet) = plt.subplots(
        2,
        1,
        figsize=(6.0, 6.0),
        sharex=True,
    )

    ax_inlet.plot(
        history.steps,
        history.inlet_temperature,
        marker="o",
        linewidth=1.5,
        label="inlet_temperature",
    )
    ax_inlet.set_ylabel("inlet_temperature")
    ax_inlet.set_title("Temperature during optimization")
    ax_inlet.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_inlet.legend()

    ax_outlet.plot(
        history.steps,
        history.outlet_mean_temperature,
        marker="o",
        linewidth=1.5,
        label="outlet_mean_temperature",
    )
    ax_outlet.axhline(
        target_outlet_mean_temperature,
        color="C1",
        linestyle="--",
        linewidth=1.5,
        label="target",
    )
    ax_outlet.set_xlabel("optimization step")
    ax_outlet.set_ylabel("outlet_mean_temperature")
    ax_outlet.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_outlet.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _slice_origin(ugrid: pv.UnstructuredGrid) -> tuple[float, float, float]:
    xmin, xmax, ymin, ymax, zmin, zmax = ugrid.bounds
    return (
        0.5 * (xmin + xmax),
        0.5 * (ymin + ymax),
        0.5 * (zmin + zmax),
    )


def plot_final_fields(
    ugrid: pv.UnstructuredGrid,
    output_path: Path,
    *,
    slice_origin: tuple[float, float, float],
) -> None:
    """Plot the final temperature field on a mid-plane slice."""
    sliced = ugrid.slice(normal=(0.0, 0.0, 1.0), origin=slice_origin)
    assert isinstance(sliced, pv.DataSet)
    if sliced.n_cells == 0:
        msg = f"Slice produced an empty mesh at origin={slice_origin}"
        raise ValueError(msg)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(
        off_screen=True,
        window_size=[700, 380],
    )
    plotter.renderer.enable_parallel_projection()
    plotter.add_mesh(
        sliced,
        scalars="T",
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path))
    plotter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize inlet temperature against a target outlet mean "
            "temperature with a fixed converged flow field."
        )
    )
    parser.add_argument(
        "--n-opt",
        type=int,
        default=N_OPT_STEPS,
        help=f"Optimization steps (default: {N_OPT_STEPS}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Optimizer learning rate (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--grad-mode",
        choices=("adjoint", "unrolled"),
        default="adjoint",
        help=(
            "How to differentiate the linear solve: implicit adjoint "
            "(default) or unrolled autograd through BiCGSTAB iterations."
        ),
    )
    parser.add_argument(
        "--check-grad",
        action="store_true",
        help="Run a finite-difference gradient check before optimization.",
    )
    return parser.parse_args()


def create_temperature_solver(
    grid: IGridBase,
    field_name: str,
    grad_mode: GradientMode,
) -> LinearSolver:
    """Create the temperature solver and configure its gradient mode."""
    solver_cfg = grid.sim_config.fvSolution.solvers[field_name]
    if grad_mode == "unrolled" and solver_cfg.method is not SolverType.BiCGSTAB:
        msg = (
            "grad_mode='unrolled' is only supported with BiCGSTAB "
            f"(configured: {solver_cfg.method.value})."
        )
        raise ValueError(msg)

    solver = create_solver(solver_cfg)
    solver.grad_mode = grad_mode
    return solver


def main() -> None:
    logger = configure_run_logger()
    args = parse_args()

    config_path = Path(__file__).resolve().parent / "config.yaml"
    grid = manual_run(config_path)

    T_name = make_field_name("T")
    t_solver_cfg = grid.sim_config.fvSolution.solvers[T_name]
    run_tag = (
        f"{t_solver_cfg.method.value}-{args.grad_mode}-{INLET_TEMPERATURE_INIT}"
    )
    output_dir = Path(grid.sim_config.control.output.output_dir) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logger(str(output_dir / "run.log"))

    logger.info("hierarchical mesh: num_cells=%d", grid.num_cells)
    logger.info(
        "temperature solver: method=%s grad_mode=%s",
        t_solver_cfg.method.value,
        args.grad_mode,
    )
    logger.info(
        "target outlet mean temperature=%.4f",
        TARGET_OUTLET_MEAN_TEMPERATURE,
    )

    U_name = make_field_name("U")
    phi_name = make_field_name("phi")

    U = get_or_create_cellfield(grid, U_name, FieldRole.LOCAL, (3,))
    T = get_or_create_cellfield(grid, T_name, FieldRole.LOCAL, ())
    phi = get_or_create_facefield(grid, phi_name, FieldRole.LOCAL, ())

    correct_flux(phi, U, update_internal=True)
    logger.info("using fixed converged flow field (U, phi)")

    alpha = grid.sim_config.properties.transport.nu
    T_solver = create_temperature_solver(grid, T_name, args.grad_mode)

    if args.check_grad:
        check_gradient_fd(
            INLET_TEMPERATURE_INIT,
            T,
            phi,
            alpha,
            T_solver,
            TARGET_OUTLET_MEAN_TEMPERATURE,
        )

    inlet_temperature = torch.tensor(
        INLET_TEMPERATURE_INIT,
        dtype=grid.dtype,
        device=grid.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([inlet_temperature], lr=args.lr)
    history = OptimizationHistory()

    logger.info(
        "optimization start (steps=%d, lr=%.3f, init=%.4f, grad_mode=%s)",
        args.n_opt,
        args.lr,
        INLET_TEMPERATURE_INIT,
        args.grad_mode,
    )
    for step in range(1, args.n_opt + 1):
        optimizer.zero_grad()
        loss, outlet_mean_temperature = outlet_temperature_loss(
            inlet_temperature,
            T,
            phi,
            alpha,
            T_solver,
            TARGET_OUTLET_MEAN_TEMPERATURE,
        )
        outlet_value = outlet_mean_temperature.item()
        loss.backward()
        optimizer.step()
        reset_scalar_field(T)

        history.append(
            step,
            inlet_temperature.item(),
            outlet_value,
            loss.item(),
        )
        logger.info(
            "step=%2d inlet_temperature=%.4f outlet_mean=%.6f loss=%.4e",
            step,
            inlet_temperature.item(),
            outlet_value,
            loss.item(),
        )

    logger.info(
        "optimization done: inlet_temperature=%.4f outlet_mean=%.6f",
        inlet_temperature.item(),
        history.outlet_mean_temperature[-1],
    )

    with torch.no_grad():
        _, final_outlet_mean = outlet_temperature_loss(
            inlet_temperature.detach(),
            T,
            phi,
            alpha,
            T_solver,
            TARGET_OUTLET_MEAN_TEMPERATURE,
        )

    ugrid = to_unstructured_grid(grid)
    save_export_fields_as_vtu(
        grid,
        str(output_dir / FINAL_VTU_NAME),
        ugrid=ugrid,
    )

    history_csv = output_dir / "inlet_temperature_history.csv"
    history_plot = output_dir / "inlet_temperature_history.png"
    fields_plot = output_dir / "inlet_temperature_final_fields.png"
    history.write_csv(history_csv)
    plot_optimization_history(
        history,
        history_plot,
        TARGET_OUTLET_MEAN_TEMPERATURE,
    )
    plot_final_fields(
        ugrid,
        fields_plot,
        slice_origin=_slice_origin(ugrid),
    )

    logger.info("saved optimization history CSV: %s", history_csv)
    logger.info("saved optimization history plot: %s", history_plot)
    logger.info("saved final field plot: %s", fields_plot)
    logger.info("saved final VTU: %s", output_dir / FINAL_VTU_NAME)
    logger.info(
        "final outlet mean temperature=%.6f (target=%.4f)",
        final_outlet_mean.item(),
        TARGET_OUTLET_MEAN_TEMPERATURE,
    )


if __name__ == "__main__":
    main()
