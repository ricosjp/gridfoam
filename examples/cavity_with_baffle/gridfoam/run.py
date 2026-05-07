from pathlib import Path

import torch
import yaml

from gridfoam.algorithms.piso import PISO
from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.basic.empty import EmptyBC
from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.factory import create_grid as create_grid_from_config
from gridfoam.io.vtu import save_export_fields_as_vtu, to_unstructured_grid
from gridfoam.meta.config import (
    GridfoamConfig,
)
from gridfoam.meta.enums import (
    DomainBoundaryPatch,
    FieldRole,
)
from gridfoam.models.turbulence.laminar import Laminar


def create_grid(config_path: Path | None = None) -> AxisProjectedGrid:
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "data" / "config.yml"
    with open(config_path) as f:
        raw_yaml = yaml.safe_load(f)
    config = GridfoamConfig.model_validate(raw_yaml)
    return create_grid_from_config(config)


def main() -> None:
    grid = create_grid()
    U = CellField(grid, "U", role=FieldRole.TRANSIENT, num_components=3)
    p = CellField(
        grid, "p", role=FieldRole.LOCAL, num_components=1, ref_value=0.0
    )

    turbulence = Laminar(grid=grid, nu=0.01)

    zero_u = torch.zeros((3,), dtype=grid.dtype, device=grid.device)
    lid_u = torch.tensor((1.0, 0.0, 0.0), dtype=grid.dtype, device=grid.device)
    zero_grad_p = torch.zeros((1,), dtype=grid.dtype, device=grid.device)
    U.add_boundary_conditions(
        {
            DomainBoundaryPatch.X_MINUS: DirichletBC(zero_u),
            DomainBoundaryPatch.X_PLUS: DirichletBC(zero_u),
            DomainBoundaryPatch.Y_MINUS: DirichletBC(zero_u),
            DomainBoundaryPatch.Y_PLUS: DirichletBC(lid_u),
            DomainBoundaryPatch.Z_MINUS: EmptyBC(),
            DomainBoundaryPatch.Z_PLUS: EmptyBC(),
            "_default": DirichletBC(zero_u),
        }
    )
    p.add_boundary_conditions(
        {
            DomainBoundaryPatch.X_MINUS: NeumannBC(zero_grad_p),
            DomainBoundaryPatch.X_PLUS: NeumannBC(zero_grad_p),
            DomainBoundaryPatch.Y_MINUS: NeumannBC(zero_grad_p),
            DomainBoundaryPatch.Y_PLUS: NeumannBC(zero_grad_p),
            DomainBoundaryPatch.Z_MINUS: EmptyBC(),
            DomainBoundaryPatch.Z_PLUS: EmptyBC(),
            "_default": NeumannBC(zero_grad_p),
        }
    )

    algo = PISO(
        grid=grid,
        U=U,
        p=p,
        turbulence=turbulence,
        n_correctors=2,
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
        U.update_history()
        if step % write_interval == 0 or step == n_steps:
            save_export_fields_as_vtu(
                grid,
                str(output_dir / f"cavity_flow_{step:04d}.vtu"),
                ugrid=ugrid,
            )
            max_u = torch.linalg.vector_norm(U.data, ord=2, dim=1).max().item()
            print(f"step={step:4d} max|U|={max_u:.4e}")


if __name__ == "__main__":
    main()
