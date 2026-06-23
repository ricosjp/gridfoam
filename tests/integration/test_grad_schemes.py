from __future__ import annotations

import pathlib

import torch

from gridfoam.core.field import CellField
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv import fvc
from gridfoam.meta.config import (
    ControlConfig,
    DomainConfig,
    FluxelConfig,
    GridfoamConfig,
    OutputConfig,
    PropertiesConfig,
    RefinementRegionConfig,
    SimulatorConfig,
    SolverConfig,
    fvSchemesConfig,
    fvSolutionConfig,
)
from gridfoam.meta.enums import (
    DeviceType,
    FieldRole,
    GradScheme,
    IbmType,
    PrecisionType,
    SolverType,
)


def _refined_3d_grid(grad_scheme: GradScheme) -> IGridBase:
    config = GridfoamConfig(
        fluxel=FluxelConfig(
            domain=DomainConfig(lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0]),
            root_resolution=[4, 4, 4],
            target_level=0,
            n_leaf_refinement=0,
            refinement_regions=[
                RefinementRegionConfig(
                    name="center",
                    min=[0.25, 0.25, 0.25],
                    max=[0.75, 0.75, 0.75],
                    level=1,
                )
            ],
            mesh_path=None,
            ibm_type=IbmType.AXIS_PROJECTED,
        ),
        simulator=SimulatorConfig(
            control=ControlConfig(
                deltaT=0.01,
                endTime=0.01,
                writeInterval=1,
                output=OutputConfig(
                    output_dir=pathlib.Path("/tmp/gridfoam_pytest_out"),
                    base_name="grad_scheme",
                ),
                precision=PrecisionType.FLOAT64,
            ),
            fvSchemes=fvSchemesConfig(
                gradSchemes={"default": grad_scheme},
            ),
            fvSolution=fvSolutionConfig(
                solvers={"p": SolverConfig(method=SolverType.CG)}
            ),
            boundaryConditions=None,
            properties=PropertiesConfig(nu=0.1),
            device=DeviceType.CPU,
        ),
    )
    return create_grid(config)


def _linear_scalar_field(grid: IGridBase) -> tuple[CellField, torch.Tensor]:
    field = CellField(
        grid,
        name="psi",
        role=FieldRole.LOCAL,
        num_components=1,
        export=False,
    )
    gradient = torch.tensor(
        [2.0, -3.0, 5.0], dtype=grid.dtype, device=grid.device
    )
    field.data = (grid.cell_centers @ gradient + 7.0).reshape(-1, 1)
    return field, gradient


def _interior_mask(grid: IGridBase) -> torch.Tensor:
    mask = torch.ones(grid.num_cells, dtype=torch.bool, device=grid.device)
    mask[grid.domain_bnd_owner] = False
    return mask


def test_leastsquare_grad_is_linear_exact_on_refined_internal_cells():
    grid = _refined_3d_grid(GradScheme.LEASTSQUARE)
    field, expected = _linear_scalar_field(grid)

    grad_field = fvc.grad(field)
    interior = _interior_mask(grid)

    torch.testing.assert_close(
        grad_field.data[interior],
        expected.expand_as(grad_field.data[interior]),
        atol=1e-12,
        rtol=1e-12,
    )
