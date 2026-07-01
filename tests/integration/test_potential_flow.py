"""Integration tests for potential-flow initialization."""

from __future__ import annotations

import pathlib

import torch

from gridfoam.core.grid.factory import create_grid
from gridfoam.fv import fvc
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
    PotentialFlowConfig,
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
    IbmType,
    PrecisionType,
    SolverType,
    TransportModelType,
    TurbulenceType,
)
from gridfoam.pre.potential_flow import PotentialFlow


def _potential_flow_config(output_dir: pathlib.Path) -> GridfoamConfig:
    return GridfoamConfig(
        fluxel=FluxelConfig(
            domain=DomainConfig(
                lower=[0.0, 0.0, 0.0],
                upper=[1.0, 1.0, 0.1],
            ),
            root_resolution=[4, 4, 1],
            target_level=0,
            n_leaf_refinement=0,
            mesh_path=None,
            ibm_type=IbmType.AXIS_PROJECTED,
        ),
        simulator=SimulatorConfig(
            control=ControlConfig(
                deltaT=1.0,
                endTime=1.0,
                writeInterval=1,
                output=OutputConfig(output_dir=output_dir, base_name="pot"),
                precision=PrecisionType.FLOAT64,
            ),
            fvSchemes=fvSchemesConfig(),
            fvSolution=fvSolutionConfig(
                algorithm=ManualAlgorithm(type=AlgorithmType.MANUAL),
                potentialFlow=PotentialFlowConfig(
                    nNonOrthogonalCorrectors=0,
                    phiRefCell=0,
                    phiRefValue=0.0,
                ),
                solvers={
                    "Phi": SolverConfig(method=SolverType.CG, tolerance=1e-8),
                },
            ),
            conditions={
                "U": ConditionConfig(
                    internal=[1.0, 0.1, 0.0],
                    boundary={
                        "walls": BoundaryConditionConfig(
                            type=BoundaryConditionType.SLIP,
                            patches=[
                                "x_minus",
                                "x_plus",
                                "y_minus",
                                "y_plus",
                            ],
                        ),
                    },
                ),
                "p": ConditionConfig(
                    internal=[0.0],
                    boundary={
                        "walls": BoundaryConditionConfig(
                            type=BoundaryConditionType.NEUMANN,
                            patches=[
                                "x_minus",
                                "x_plus",
                                "y_minus",
                                "y_plus",
                            ],
                            value=[0.0],
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


def test_potential_flow_reduces_divergence(tmp_path: pathlib.Path):
    grid = create_grid(_potential_flow_config(tmp_path))
    solver = PotentialFlow(grid)

    from gridfoam.core.field import get_or_create_facefield
    from gridfoam.meta.enums import FieldRole

    phi = get_or_create_facefield(grid, "phi", FieldRole.LOCAL, 1)
    from gridfoam.fv.flux import correct_flux

    correct_flux(phi, solver.U, update_internal=True)
    initial = torch.linalg.vector_norm(fvc.div(phi).data, ord=2).item()

    solver.solve()
    final = torch.linalg.vector_norm(fvc.div(solver.phi).data, ord=2).item()
    assert final <= initial + 1e-6
