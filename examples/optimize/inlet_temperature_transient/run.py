"""Optimize inlet temperature through a short unrolled scalar trajectory.

The prescribed flow carries heat downstream while the walls cool it.
Each evaluation advances a few physical time steps of
ddt(T) + div(phi, T) - laplacian(alpha, T) = 0 and minimizes
(outlet_mean_temperature - target)**2. Linear solves use an implicit
adjoint; the time loop retains its autograd graph.

Run: python -m examples.optimize.inlet_temperature_transient.run --check-grad
"""

import argparse
import math
import sys
from functools import partial
from pathlib import Path

import torch

from gridfoam.core.equation import equation
from gridfoam.core.field import (
    CellField,
    FaceField,
    get_or_create_cellfield,
    get_or_create_facefield,
)
from gridfoam.core.grid.base import GridBase
from gridfoam.fv import fvm
from gridfoam.fv.flux import correct_flux
from gridfoam.meta.enums import DomainBoundaryPatch, FieldRole, SolverType
from gridfoam.runner import manual_run
from gridfoam.solvers.base import (
    GradientMode,
    LinearSolver,
    require_converged_solves,
)
from gridfoam.solvers.factory import create_solver

# Support both direct execution and python -m from the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.optimize._common import (  # noqa: E402
    configure_run_logger,
    patch_cell_mean,
    set_inlet_dirichlet,
)
from examples.optimize.inlet_temperature_steady.gradient_check import (  # noqa: E402
    check_gradient_fd,
)
from examples.optimize.inlet_temperature_transient.output import (  # noqa: E402
    OptimizationRecord,
    save_results,
)

TARGET_OUTLET_MEAN_TEMPERATURE = 0.004
INLET_TEMPERATURE_INIT = 0.2
N_OPT_STEPS = 20
N_TIME_STEPS = 8
LEARNING_RATE = 0.10


def advance_temperature(
    T: CellField,
    phi: FaceField,
    alpha: float,
    solver: LinearSolver,
    n_time: int,
) -> None:
    """Advance ``n_time`` implicit transport steps, retaining the graph."""
    with require_converged_solves((solver,)):
        for _ in range(n_time):
            matrix = fvm.ddt(T) + fvm.div(phi, T) - fvm.laplacian(alpha, T)
            result = solver.solve(equation(T, matrix))
            T.data = result.solution
            T.update_history()


def outlet_temperature_loss(
    inlet_temperature: torch.Tensor,
    *,
    T: CellField,
    phi: FaceField,
    alpha: float,
    solver: LinearSolver,
    n_time: int,
    target: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reset T, apply the inlet, advance, and return loss and outlet mean."""
    T.data = torch.zeros_like(T.data, requires_grad=False)
    T.update_history(reset=True)
    set_inlet_dirichlet(T, DomainBoundaryPatch.X_MINUS, inlet_temperature)
    advance_temperature(T, phi, alpha, solver, n_time)
    outlet_mean = patch_cell_mean(T, DomainBoundaryPatch.X_PLUS)
    return (outlet_mean - target).square(), outlet_mean


def create_temperature_solver(
    grid: GridBase, grad_mode: GradientMode
) -> LinearSolver:
    """Use YAML solver settings with the requested differentiation mode."""
    config = grid.sim_config.fvSolution.solvers["T"]
    if grad_mode == "unrolled" and config.method is not SolverType.BiCGSTAB:
        raise ValueError("--grad-mode unrolled requires BiCGSTAB")
    solver = create_solver(config)
    solver.grad_mode = grad_mode
    return solver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-opt", type=int, default=N_OPT_STEPS)
    parser.add_argument("--n-time", type=int, default=N_TIME_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--grad-mode",
        choices=("adjoint", "unrolled"),
        default="adjoint",
        help="Implicit adjoint (default) or unrolled BiCGSTAB for comparison.",
    )
    parser.add_argument(
        "--check-grad",
        action="store_true",
        help="Check the initial gradient against finite differences.",
    )
    args = parser.parse_args()
    if args.n_opt < 0:
        parser.error("--n-opt must be non-negative")
    if args.n_time < 1:
        parser.error("--n-time must be at least 1")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        parser.error("--lr must be finite and positive")
    return args


def main() -> None:
    args = parse_args()
    logger = configure_run_logger()

    # 1. Prepare the prescribed flow and the temperature equation.
    grid = manual_run(Path(__file__).with_name("config.yaml"))
    U = get_or_create_cellfield(grid, "U", FieldRole.LOCAL, (3,))
    T = get_or_create_cellfield(grid, "T", FieldRole.TRANSIENT, ())
    phi = get_or_create_facefield(grid, "phi", FieldRole.LOCAL, ())
    correct_flux(phi, U, update_internal=True)
    solver = create_temperature_solver(grid, args.grad_mode)
    evaluate = partial(
        outlet_temperature_loss,
        T=T,
        phi=phi,
        alpha=grid.sim_config.properties.transport.nu,
        solver=solver,
        n_time=args.n_time,
        target=TARGET_OUTLET_MEAN_TEMPERATURE,
    )
    logger.info(
        "prescribed uniform flow: num_cells=%d n_time=%d delta_t=%g",
        grid.num_cells,
        args.n_time,
        grid.dt,
    )

    # 2. Optimize one scalar inlet temperature using ordinary loss.backward().
    inlet_temperature = torch.nn.Parameter(
        torch.tensor(
            INLET_TEMPERATURE_INIT, dtype=grid.dtype, device=grid.device
        )
    )
    if args.check_grad:
        check_gradient_fd(evaluate, inlet_temperature)
    optimizer = torch.optim.Adam([inlet_temperature], lr=args.lr)
    history: list[OptimizationRecord] = []

    # Step 0 is the initial evaluation; step n follows n optimizer updates.
    for step in range(args.n_opt + 1):
        optimizer.zero_grad()
        with torch.set_grad_enabled(step < args.n_opt):
            loss, outlet_mean = evaluate(inlet_temperature)
        record = OptimizationRecord(
            step, inlet_temperature.item(), outlet_mean.item(), loss.item()
        )
        history.append(record)
        logger.info(
            "step=%3d inlet_temperature=%.6f outlet_mean=%.6f loss=%.6e",
            record.step,
            record.inlet_temperature,
            record.outlet_mean_temperature,
            record.loss,
        )
        if step < args.n_opt:
            loss.backward()
            optimizer.step()

    # 3. The last evaluation leaves T ready for export and matches the CSV.
    method = grid.sim_config.fvSolution.solvers["T"].method.value
    run_tag = f"{method}-{args.grad_mode}-{INLET_TEMPERATURE_INIT}"
    output_dir = Path(grid.sim_config.control.output.output_dir) / run_tag
    for path in save_results(
        grid, history, output_dir, TARGET_OUTLET_MEAN_TEMPERATURE
    ):
        logger.info("saved: %s", path)
    logger.info(
        "final outlet mean=%.8f (target=%.4f), T range=[%.8e, %.8e]",
        history[-1].outlet_mean_temperature,
        TARGET_OUTLET_MEAN_TEMPERATURE,
        T.data.min().item(),
        T.data.max().item(),
    )


if __name__ == "__main__":
    main()
