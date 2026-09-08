import logging
import sys
from pathlib import Path

from gridfoam.core.equation import equation
from gridfoam.core.field import (
    CellField,
    get_or_create_cellfield,
    get_or_create_facefield,
)
from gridfoam.core.name import make_field_name
from gridfoam.fv import fvm
from gridfoam.fv.flux import correct_flux
from gridfoam.io.vtu import save_export_fields_as_vtu, to_unstructured_grid
from gridfoam.meta.enums import (
    FieldRole,
)
from gridfoam.runner import manual_run
from gridfoam.solvers.factory import create_solver

logger = logging.getLogger(__name__)


def configure_run_logger(log_file: Path) -> None:
    formatter = logging.Formatter("%(message)s")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def initialize_T(T: CellField) -> None:
    """Set T=1 on [-1, 1] x [-1, 1] x [1, 3]."""
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
    log_file = Path(__file__).resolve().parent / "outputs" / "diffusion.log"
    configure_run_logger(log_file)
    config_path = Path(__file__).resolve().parent / "config.yaml"
    grid = manual_run(config_path)

    output_dir = Path(grid.sim_config.control.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = grid.sim_config.control.output.base_name

    ugrid = to_unstructured_grid(grid)

    write_interval = grid.sim_config.control.writeInterval
    end_time = grid.sim_config.control.endTime
    deltaT = grid.sim_config.control.deltaT
    n_steps = int(end_time / deltaT)

    U_name = make_field_name("U")
    T_name = make_field_name("T")
    phi_name = make_field_name("phi")

    U = get_or_create_cellfield(grid, U_name, FieldRole.LOCAL, (3,))
    T = get_or_create_cellfield(grid, T_name, FieldRole.TRANSIENT, ())
    phi = get_or_create_facefield(grid, phi_name, FieldRole.LOCAL, ())

    correct_flux(phi, U, update_internal=True)
    initialize_T(T)
    T.update_history(reset=True)

    alpha = grid.sim_config.properties.transport.nu
    T_solver = create_solver(grid.sim_config.fvSolution.solvers[T_name])

    save_export_fields_as_vtu(
        grid,
        str(output_dir / f"{base_name}_0000.vtu"),
        ugrid=ugrid,
    )

    logger.info(
        f"step={0:4d} "
        f"min(T)={T.data.min().item():.4e} "
        f"max(T)={T.data.max().item():.4e}"
    )

    for step in range(1, n_steps + 1):
        TEqn_mat = fvm.ddt(T) - fvm.laplacian(alpha, T)
        temperature_eq = equation(T, TEqn_mat)
        T.data = T_solver.solve(temperature_eq).solution
        T.update_history()

        if step % write_interval == 0 or step == n_steps:
            save_export_fields_as_vtu(
                grid,
                str(output_dir / f"{base_name}_{step:04d}.vtu"),
                ugrid=ugrid,
            )
            logger.info(
                "step=%4d min(T)=%.4e max(T)=%.4e",
                step,
                T.data.min().item(),
                T.data.max().item(),
            )


if __name__ == "__main__":
    main()
