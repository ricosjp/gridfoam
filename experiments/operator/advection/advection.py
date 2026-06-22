import logging
import sys
from pathlib import Path

import yaml

from gridfoam.boundaries.factory import apply_boundary_condition_configs
from gridfoam.core.builtins import make_builtin_key
from gridfoam.core.equation import equation
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.factory import create_grid as create_grid_from_config
from gridfoam.fv import fvm
from gridfoam.fv.flux import correct_flux
from gridfoam.fv.fvm.laplacian import IGridBase
from gridfoam.io.vtu import save_export_fields_as_vtu, to_unstructured_grid
from gridfoam.meta.config import (
    GridfoamConfig,
)
from gridfoam.meta.enums import (
    FieldRole,
)
from gridfoam.solvers.factory import create_solver

LOG_FILE_NAME = "advection.log"

logger = logging.getLogger("gridfoam.experiments.operator.advection")


def create_grid(config_path: Path | None = None) -> IGridBase:
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.yml"
    with open(config_path) as f:
        raw_yaml = yaml.safe_load(f)
    config = GridfoamConfig.model_validate(raw_yaml)
    return create_grid_from_config(config)


def configure_run_logger(log_file: Path) -> None:
    formatter = logging.Formatter("%(message)s")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def initialize_U(U: CellField) -> None:
    """Set uniform velocity U=[1.0, 0.0, 0.0]."""
    U.data[:, 0] = 0.707
    U.data[:, 2] = -0.707


def initialize_T(T: CellField) -> None:
    """Set uniform initial temperature T=1 on [-1, 1] x [-1, 1]."""
    cell_centers = T.grid.cell_centers
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    z = cell_centers[:, 2]
    mask = (
        (-1.0 <= x)
        & (x <= 1.0)
        & (-1.0 <= y)
        & (y <= 1.0)
        & (1.0 <= z)
        & (z <= 3.0)
    )
    T.data[mask] = 1.0


def main() -> None:
    grid = create_grid()
    T = CellField(grid, "T", role=FieldRole.TRANSIENT, num_components=1)
    U = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
    phi = FaceField(grid, "phi", role=FieldRole.LOCAL, num_components=1)
    grid.register_builtin_field(make_builtin_key("phi"), phi)

    boundary_conditions = grid.sim_config.boundaryConditions
    if boundary_conditions is None or "T" not in boundary_conditions:
        raise ValueError("boundaryConditions.T is required for this example.")
    apply_boundary_condition_configs(T, boundary_conditions["T"])

    if boundary_conditions is None or "U" not in boundary_conditions:
        raise ValueError("boundaryConditions.U is required for this example.")
    apply_boundary_condition_configs(U, boundary_conditions["U"])

    initialize_U(U)
    correct_flux(phi, U, update_internal=True)

    initialize_T(T)
    T.update_history()

    T_solver = create_solver(grid.sim_config.fvSolution.solvers["temperature"])

    output_dir = Path(grid.sim_config.control.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logger(output_dir / LOG_FILE_NAME)
    write_interval = grid.sim_config.control.writeInterval
    ugrid = to_unstructured_grid(grid)

    n_steps = int(
        grid.sim_config.control.endTime / grid.sim_config.control.deltaT
    )

    save_export_fields_as_vtu(
        grid,
        str(output_dir / "advection_0000.vtu"),
        ugrid=ugrid,
    )
    print(
        f"step={0:4d} "
        f"min(T)={T.data.min().item():.4e} "
        f"max(T)={T.data.max().item():.4e}"
    )

    for step in range(1, n_steps + 1):
        TEqn_mat = fvm.ddt(T) + fvm.div(phi, T)
        temperature_eq = equation("temperature", T, TEqn_mat)
        T.data = T_solver.solve(temperature_eq)
        T.update_history()

        if step % write_interval == 0 or step == n_steps:
            save_export_fields_as_vtu(
                grid,
                str(output_dir / f"advection_{step:04d}.vtu"),
                ugrid=ugrid,
            )
            print(
                f"step={step:4d} "
                f"min(T)={T.data.min().item():.4e} "
                f"max(T)={T.data.max().item():.4e}"
            )


if __name__ == "__main__":
    main()
