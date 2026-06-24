import logging
import sys
from pathlib import Path

import torch
import yaml

from gridfoam.algorithms.simple import SIMPLE
from gridfoam.boundaries.factory import apply_boundary_condition_configs
from gridfoam.core.field import CellField
from gridfoam.core.grid.factory import create_grid as create_grid_from_config
from gridfoam.fv.fvm.laplacian import IGridBase
from gridfoam.io.vtu import save_export_fields_as_vtu, to_unstructured_grid
from gridfoam.meta.config import (
    GridfoamConfig,
)
from gridfoam.meta.enums import (
    FieldRole,
)
from gridfoam.models.turbulence.laminar import Laminar
from gridfoam.post.forces import ForceEvaluator
from gridfoam.pre.potential_flow import PotentialFlow

LOG_FILE_NAME = "ahmed_body.log"

logger = logging.getLogger("gridfoam.examples.ahmed_body")


def create_grid(config_path: Path | None = None) -> IGridBase:
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "data" / "config.yml"
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


def main() -> None:
    torch.set_num_threads(6)
    grid = create_grid()
    U = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
    p = CellField(grid, "p", role=FieldRole.LOCAL, num_components=1)

    boundary_conditions = grid.sim_config.boundaryConditions
    if boundary_conditions is None:
        raise ValueError("boundaryConditions is required for this example.")
    apply_boundary_condition_configs(U, boundary_conditions["U"])
    apply_boundary_condition_configs(p, boundary_conditions["p"])

    PotentialFlow(grid, U, p).solve()

    turbulence = Laminar(grid=grid, nu=grid.sim_config.properties.nu)

    force_config = grid.sim_config.forceCoeff
    if force_config is None:
        raise ValueError("forceCoeff is required for this example.")
    force_evaluator = ForceEvaluator(force_config)

    algo = SIMPLE(
        grid=grid,
        U=U,
        p=p,
        turbulence=turbulence,
    )

    output_dir = Path(grid.sim_config.control.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logger(output_dir / LOG_FILE_NAME)
    write_interval = grid.sim_config.control.writeInterval
    ugrid = to_unstructured_grid(grid)

    n_steps = int(
        grid.sim_config.control.endTime / grid.sim_config.control.deltaT
    )
    for step in range(1, n_steps + 1):
        algo.step()
        if step % write_interval == 0 or step == n_steps:
            save_export_fields_as_vtu(
                grid,
                str(output_dir / f"ahmed_{step:04d}.vtu"),
                ugrid=ugrid,
            )
            max_u = torch.linalg.vector_norm(U.data, ord=2, dim=1).max().item()
            logger.info("step=%4d max|U|=%.4e", step, max_u)

    force_evaluator.evaluate(
        grid,
        time=float(n_steps),
        p=p,
        U=U,
        turbulence=turbulence,
    )
    logger.info(
        (
            "final Cd=%.6e Cd(f)=%.6e Cd(r)=%.6e "
            "Cl=%.6e Cl(f)=%.6e Cl(r)=%.6e "
            "CmPitch=%.6e CmRoll=%.6e CmYaw=%.6e "
            "Cs=%.6e Cs(f)=%.6e Cs(r)=%.6e"
        ),
        force_evaluator.history[-1].Cd.item(),
        force_evaluator.history[-1].Cd_f.item(),
        force_evaluator.history[-1].Cd_r.item(),
        force_evaluator.history[-1].Cl.item(),
        force_evaluator.history[-1].Cl_f.item(),
        force_evaluator.history[-1].Cl_r.item(),
        force_evaluator.history[-1].CmPitch.item(),
        force_evaluator.history[-1].CmRoll.item(),
        force_evaluator.history[-1].CmYaw.item(),
        force_evaluator.history[-1].Cs.item(),
        force_evaluator.history[-1].Cs_f.item(),
        force_evaluator.history[-1].Cs_r.item(),
    )
    grid.surface_mesh.save(
        output_dir / "surface_mesh.vtu",
        overwrite_features=True,
        overwrite_file=True,
    )


if __name__ == "__main__":
    main()
