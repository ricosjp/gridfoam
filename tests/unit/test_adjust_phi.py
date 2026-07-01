"""Unit tests for ``adjust_phi``."""

from __future__ import annotations

import pathlib

import torch

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.derived.inlet_outlet import InletOutletBC
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv.adjust_phi import adjust_phi
from gridfoam.meta.config import (
    BoundaryConditionConfig,
    ConditionConfig,
    ControlConfig,
    DomainConfig,
    FluxelConfig,
    GridfoamConfig,
    LaminarConfig,
    ManualAlgorithm,
    NewtonianTransportConfig,
    OutputConfig,
    PropertiesConfig,
    SimulatorConfig,
    SolverConfig,
    fvSchemesConfig,
    fvSolutionConfig,
)
from gridfoam.meta.enums import (
    AlgorithmType,
    BoundaryConditionType,
    DeviceType,
    DomainBoundaryPatch,
    IbmType,
    PrecisionType,
    SolverType,
    TransportModelType,
    TurbulenceType,
)
from gridfoam.meta.enums import FieldRole


def _channel_config(output_dir: pathlib.Path) -> GridfoamConfig:
    return GridfoamConfig(
        fluxel=FluxelConfig(
            domain=DomainConfig(
                lower=[0.0, 0.0, 0.0],
                upper=[2.0, 0.2, 0.1],
            ),
            root_resolution=[8, 2, 1],
            target_level=0,
            n_leaf_refinement=0,
            mesh_path=None,
            ibm_type=IbmType.AXIS_PROJECTED,
        ),
        simulator=SimulatorConfig(
            control=ControlConfig(
                deltaT=0.01,
                endTime=0.01,
                writeInterval=1,
                output=OutputConfig(output_dir=output_dir, base_name="adj"),
                precision=PrecisionType.FLOAT64,
            ),
            fvSchemes=fvSchemesConfig(),
            fvSolution=fvSolutionConfig(
                algorithm=ManualAlgorithm(type=AlgorithmType.MANUAL),
                solvers={"p": SolverConfig(method=SolverType.CG)},
            ),
            conditions={
                "U": ConditionConfig(
                    internal=[1.0, 0.0, 0.0],
                    boundary={
                        "inlet": BoundaryConditionConfig(
                            type=BoundaryConditionType.DIRICHLET,
                            patches=[DomainBoundaryPatch.X_MINUS.value],
                            value=[1.0, 0.0, 0.0],
                        ),
                        "outlet": BoundaryConditionConfig(
                            type=BoundaryConditionType.INLET_OUTLET,
                            patches=[DomainBoundaryPatch.X_PLUS.value],
                            value=[0.0, 0.0, 0.0],
                        ),
                    },
                ),
            },
            properties=PropertiesConfig(
                transport=NewtonianTransportConfig(
                    type=TransportModelType.NEWTONIAN,
                    nu=0.1,
                ),
                turbulence=LaminarConfig(type=TurbulenceType.LAMINAR),
            ),
            device=DeviceType.CPU,
        ),
    )


def test_adjust_phi_balances_domain_flux(tmp_path: pathlib.Path):
    grid = create_grid(_channel_config(tmp_path))
    U = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
    phi = FaceField(grid, "phi", role=FieldRole.LOCAL, num_components=1)

    inlet_mask = grid.get_domain_bnd_mask(DomainBoundaryPatch.X_MINUS)
    outlet_mask = grid.get_domain_bnd_mask(DomainBoundaryPatch.X_PLUS)
    phi.domain_bnd_data[inlet_mask, 0] = -1.0
    phi.domain_bnd_data[outlet_mask, 0] = 0.8

    U.add_boundary_conditions(
        {
            DomainBoundaryPatch.X_MINUS: DirichletBC(
                torch.tensor([1.0, 0.0, 0.0], dtype=grid.dtype, device=grid.device)
            ),
            DomainBoundaryPatch.X_PLUS: InletOutletBC(
                torch.tensor([0.0, 0.0, 0.0], dtype=grid.dtype, device=grid.device)
            ),
        }
    )

    adjust_phi(phi, U)

    total = phi.domain_bnd_data.sum().item()
    assert abs(total) < 1e-10


def test_adjust_phi_skips_scaling_when_no_adjustable_outflow(
    tmp_path: pathlib.Path,
):
    grid = create_grid(_channel_config(tmp_path))
    U = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
    phi = FaceField(grid, "phi", role=FieldRole.LOCAL, num_components=1)

    inlet_mask = grid.get_domain_bnd_mask(DomainBoundaryPatch.X_MINUS)
    outlet_mask = grid.get_domain_bnd_mask(DomainBoundaryPatch.X_PLUS)
    phi.domain_bnd_data[inlet_mask, 0] = -1.0
    phi.domain_bnd_data[outlet_mask, 0] = 0.5

    U.add_boundary_conditions(
        {
            DomainBoundaryPatch.X_MINUS: DirichletBC(
                torch.tensor([1.0, 0.0, 0.0], dtype=grid.dtype, device=grid.device)
            ),
            DomainBoundaryPatch.X_PLUS: DirichletBC(
                torch.tensor([0.0, 0.0, 0.0], dtype=grid.dtype, device=grid.device)
            ),
        }
    )

    phi_before = phi.domain_bnd_data.clone()
    adjust_phi(phi, U)
    assert torch.allclose(phi.domain_bnd_data, phi_before)
