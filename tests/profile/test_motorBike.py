"""Profiling motorBike simulation"""

import logging
from pathlib import Path

import pytest
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

logger = logging.getLogger(__name__)


def create_grid(config_path: Path | None = None) -> IGridBase:
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "data" / "config.yml"
    with open(config_path) as f:
        raw_yaml = yaml.safe_load(f)
    config = GridfoamConfig.model_validate(raw_yaml)
    return create_grid_from_config(config)


@pytest.mark.profile
def test_motorBike_profile():
    """Profile motorBike."""
    grid = create_grid()
    U = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
    p = CellField(grid, "p", role=FieldRole.LOCAL, num_components=1)

    boundary_conditions = grid.sim_config.boundaryConditions
    if boundary_conditions is None:
        raise ValueError("boundaryConditions is required for this example.")
    apply_boundary_condition_configs(U, boundary_conditions["U"])
    apply_boundary_condition_configs(p, boundary_conditions["p"])

    turbulence = Laminar(grid=grid, nu=0.1)

    algo = SIMPLE(
        grid=grid,
        U=U,
        p=p,
        turbulence=turbulence,
    )

    output_dir = Path(grid.sim_config.control.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
                str(output_dir / f"motorBike_{step:04d}.vtu"),
                ugrid=ugrid,
            )
            max_u = torch.linalg.vector_norm(U.data, ord=2, dim=1).max().item()
            print(f"step={step:4d} max|U|={max_u:.4e}")
