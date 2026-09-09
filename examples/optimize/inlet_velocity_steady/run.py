"""Optimize uniform inlet velocity using a converged SIMPLE implicit adjoint.

Each evaluation solves steady flow and minimizes (mean_u_x - target)**2.
Run: python -m examples.optimize.inlet_velocity_steady.run
"""

import argparse
import math
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import torch

from gridfoam.algorithms.simple import SIMPLE
from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.state import TensorState
from gridfoam.meta.enums import DomainBoundaryPatch
from gridfoam.optimize.simple import SimpleStepMap
from gridfoam.optimize.steady import steady_solve
from gridfoam.runner import manual_run

# Support both direct execution and python -m from the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.optimize._common import configure_run_logger  # noqa: E402
from examples.optimize.inlet_velocity_steady.output import (  # noqa: E402
    OptimizationRecord,
    save_results,
)

TARGET_MEAN_U_X = 1.2
INLET_VELOCITY_INIT = 0.8
N_OPT_STEPS = 4
LEARNING_RATE = 0.2


@contextmanager
def apply_inlet_velocity(
    algorithm: SIMPLE, design: TensorState
) -> Generator[None]:
    """Apply and restore the example's uniform inlet velocity."""
    patch = DomainBoundaryPatch.X_MINUS
    value = design["inlet_velocity"]
    if value.shape != (3,):
        raise ValueError("inlet_velocity must have shape (3,)")
    if (
        value.dtype != algorithm.grid.dtype
        or value.device != algorithm.grid.device
    ):
        raise ValueError("inlet_velocity must match grid dtype and device")
    original = algorithm.U.bcs.get(patch)
    if not isinstance(original, DirichletBC):
        raise ValueError("The inlet patch must have a Dirichlet velocity BC")
    algorithm.U.add_boundary_conditions({patch: DirichletBC(value)})
    try:
        yield
    finally:
        algorithm.U.add_boundary_conditions({patch: original})
        algorithm.grid.invalidate_derived_caches()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-opt", type=int, default=N_OPT_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    if args.n_opt < 0:
        parser.error("--n-opt must be non-negative")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        parser.error("--lr must be finite and positive")
    return args


def main() -> None:
    args = parse_args()
    logger = configure_run_logger()

    # 1. Prepare the SIMPLE step map and the variable inlet boundary.
    grid = manual_run(Path(__file__).with_name("config.yaml"))
    mapping = SimpleStepMap(SIMPLE(grid), apply_inlet_velocity)
    logger.info("steady SIMPLE flow: num_cells=%d", grid.num_cells)

    # 2. Optimize one scalar speed using ordinary loss.backward().
    speed = torch.nn.Parameter(
        torch.tensor(INLET_VELOCITY_INIT, dtype=grid.dtype, device=grid.device)
    )
    direction = speed.new_tensor([1.0, 0.0, 0.0])
    optimizer = torch.optim.SGD([speed], lr=args.lr)
    history: list[OptimizationRecord] = []
    state = mapping.initial_state

    # Step 0 is the initial evaluation; step n follows n optimizer updates.
    for step in range(args.n_opt + 1):
        optimizer.zero_grad()
        with torch.set_grad_enabled(step < args.n_opt):
            state = steady_solve(
                mapping,
                mapping.initial_state,
                TensorState({"inlet_velocity": speed * direction}),
            )
            mean_u_x = state["U"][:, 0].mean()
            loss = (mean_u_x - TARGET_MEAN_U_X).square()
        record = OptimizationRecord(
            step, speed.item(), mean_u_x.item(), loss.item()
        )
        history.append(record)
        logger.info(
            "step=%3d inlet_velocity=%.6f mean_u_x=%.6f loss=%.6e",
            record.step,
            record.inlet_velocity,
            record.mean_u_x,
            record.loss,
        )
        if step < args.n_opt:
            loss.backward()
            optimizer.step()

    # 3. Export the returned solution: the step map restores the caller's grid.
    # Install U and p only for output; no solver step follows this assignment.
    mapping.algorithm.U.data = state["U"].detach().clone()
    mapping.algorithm.p.data = state["p"].detach().clone()
    output_dir = Path(grid.sim_config.control.output.output_dir)
    for path in save_results(grid, history, output_dir, TARGET_MEAN_U_X):
        logger.info("saved: %s", path)
    logger.info(
        "final mean Ux=%.8f (target=%.4f)",
        history[-1].mean_u_x,
        TARGET_MEAN_U_X,
    )


if __name__ == "__main__":
    main()
