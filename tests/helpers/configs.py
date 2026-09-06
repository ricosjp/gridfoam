"""Reusable ``GridfoamConfig`` builders for tests."""

from __future__ import annotations

import pathlib

from tests.conftest import default_properties, small_gridfoam_config

from gridfoam.core.dimensions import (
    DIM_KIN_PRESSURE,
    DIM_VELOCITY,
    dimension_config,
)
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
    RefinementRegionConfig,
    RelaxationFactorsConfig,
    SIMPLEAlgorithm,
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
    GradScheme,
    IbmType,
    PrecisionType,
    SolverType,
    TransportModelType,
    TurbulenceType,
)


def refined_config(
    *,
    grad_scheme: GradScheme | None = None,
    fv_schemes: fvSchemesConfig | None = None,
    root_resolution: tuple[int, int, int] = (4, 4, 1),
    domain_upper: tuple[float, float, float] = (1.0, 1.0, 0.1),
    refinement_min: tuple[float, float, float] = (0.25, 0.25, 0.0),
    refinement_max: tuple[float, float, float] = (0.75, 0.75, 0.1),
) -> GridfoamConfig:
    """
    Axis-projected mesh with one centred refinement region.

    ``fv_schemes`` replaces the whole ``fvSchemes`` block; ``grad_scheme``
    only overrides ``gradSchemes.default`` on top of it.
    """
    config = small_gridfoam_config()
    if fv_schemes is not None:
        config = config.model_copy(
            update={
                "simulator": config.simulator.model_copy(
                    update={"fvSchemes": fv_schemes}
                )
            }
        )
    config = config.model_copy(
        update={
            "fluxel": config.fluxel.model_copy(
                update={
                    "domain": DomainConfig(
                        lower=[0.0, 0.0, 0.0],
                        upper=list(domain_upper),
                    ),
                    "root_resolution": list(root_resolution),
                    "refinement_regions": [
                        RefinementRegionConfig(
                            name="center",
                            min=list(refinement_min),
                            max=list(refinement_max),
                            level=1,
                        )
                    ],
                }
            )
        }
    )
    if grad_scheme is not None:
        config = config.model_copy(
            update={
                "simulator": config.simulator.model_copy(
                    update={
                        "fvSchemes": config.simulator.fvSchemes.model_copy(
                            update={"gradSchemes": {"default": grad_scheme}}
                        )
                    }
                )
            }
        )
    return config


def refined_3d_config(
    *,
    grad_scheme: GradScheme | None = None,
    fv_schemes: fvSchemesConfig | None = None,
) -> GridfoamConfig:
    """3-D refined mesh for hierarchy-interface scheme tests."""
    if fv_schemes is None:
        fv_schemes = fvSchemesConfig(
            gradSchemes=(
                None if grad_scheme is None else {"default": grad_scheme}
            ),
        )
    elif grad_scheme is not None:
        fv_schemes = fv_schemes.model_copy(
            update={"gradSchemes": {"default": grad_scheme}}
        )
    return GridfoamConfig(
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
            fvSchemes=fv_schemes,
            fvSolution=fvSolutionConfig(
                algorithm=ManualAlgorithm(type=AlgorithmType.MANUAL),
                solvers={"p": SolverConfig(method=SolverType.CG)},
            ),
            conditions={},
            properties=default_properties(),
            device=DeviceType.CPU,
        ),
    )


def simple_convergence_config(output_dir: pathlib.Path) -> GridfoamConfig:
    """Tiny cavity with loose residual control for convergence checks."""
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
                    dimension=dimension_config(DIM_VELOCITY),
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
                    dimension=dimension_config(DIM_KIN_PRESSURE),
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


def potential_flow_config(output_dir: pathlib.Path) -> GridfoamConfig:
    """Minimal domain for potential-flow divergence reduction checks."""
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
                    dimension=dimension_config(DIM_VELOCITY),
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
                    dimension=dimension_config(DIM_KIN_PRESSURE),
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


def channel_config(output_dir: pathlib.Path) -> GridfoamConfig:
    """Inlet/outlet channel used by ``adjust_phi`` tests."""
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
                    dimension=dimension_config(DIM_VELOCITY),
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
