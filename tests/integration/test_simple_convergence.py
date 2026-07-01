"""Integration tests for SIMPLE residual-based early termination."""

from __future__ import annotations

import pathlib

from gridfoam.algorithms.simple import SIMPLE
from gridfoam.core.grid.factory import create_grid
from gridfoam.meta.config import (
    BoundaryConditionConfig,
    ConditionConfig,
    ControlConfig,
    DomainConfig,
    FluxelConfig,
    GridfoamConfig,
    LaminarConfig,
    NewtonianTransportConfig,
    OutputConfig,
    PropertiesConfig,
    RelaxationFactorsConfig,
    SimulatorConfig,
    SIMPLEAlgorithm,
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


def _simple_config(output_dir: pathlib.Path) -> GridfoamConfig:
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
                endTime=100.0,
                writeInterval=1,
                output=OutputConfig(output_dir=output_dir, base_name="conv"),
                precision=PrecisionType.FLOAT64,
            ),
            fvSchemes=fvSchemesConfig(),
            fvSolution=fvSolutionConfig(
                algorithm=SIMPLEAlgorithm(
                    type=AlgorithmType.SIMPLE,
                    residualControl={"p": 1e20, "phi": 1e20},
                    relaxationFactors=RelaxationFactorsConfig(
                        equations={"U": 0.7, "p": 0.3},
                    ),
                    pRefCell=0,
                    pRefValue=0.0,
                ),
                solvers={
                    "U": SolverConfig(
                        method=SolverType.BiCGSTAB,
                        tolerance=1e-8,
                        rel_tolerance=0.1,
                    ),
                    "p": SolverConfig(
                        method=SolverType.CG,
                        tolerance=1e-8,
                        rel_tolerance=0.01,
                    ),
                },
            ),
            conditions={
                "U": ConditionConfig(
                    internal=[0.0, 0.0, 0.0],
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


def test_simple_has_converged_with_loose_residual_control(tmp_path: pathlib.Path):
    grid = create_grid(_simple_config(tmp_path))
    algo = SIMPLE(grid)
    algo.step()
    assert algo.has_converged()
